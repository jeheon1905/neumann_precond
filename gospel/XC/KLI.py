import numpy as np
from itertools import product, combinations_with_replacement
from gospel.Poisson.ISF_exx_solver import create_ISF_exx_solver
from gospel.Poisson.Poisson_ISF_python import Poisson_ISF
from time import time
from gospel.util import timer


class KLI:
    r"""Calculate KLI approximated EXX potential and energy.

    V^{KLI}_{\sigma} = \frac{1}{\rho_{\sigma}(\vec{r})} \sum_{\vec{k}} \sum^{N_\sigma}{i=1}
                       |phi_{i\vec{k}\sigma}(\vec{r})|^2 \times (U_{i\vec{k}\sigma}(\vec{r})
                       + (\bar{V}^{KLI}_{i\vec{k}\sigma} - \bar{U}_{i\vec{k}\sigma})) + C.C.
    U_{i\vec{k}\sigma}(\vec{r}) = - \frac{1}{\phi_{i\vec{k}\sigma}*{\ast}(\vec{r})} \sum_{j\vec{q}}
                                    f_{j\vec{q}} \phi_{i\vec{k}\sigma}^{\ast} V_{i\vec{k}j\vec{q}\sigma}(\vec{r})
    \bar{V}^{KLI}_{i\vec{k}\sigma} = ...
    """
    def __init__(self, grid, kpts, exx_solver=None):
        st = time()
        self.__grid = grid
        self.__functional_type = 'exx'
        self.__kpts = kpts

        ## Initialize exchange solvers
        if exx_solver is None:
            assert kpts is not None
            self.__exx_solver = create_ISF_exx_solver(grid, kpts)
        else:
            self.__exx_solver = exx_solver
        print(self.__exx_solver)

#        self.__poisson_solver = create_ISF_exx_solver(grid)
        self.__poisson_solver = Poisson_ISF(grid)

        self.__Ex = None
        self.__Vx = None
        print(f"Debug : KLI init Time : {time()-st} sec")
        return


    def get_describe(self):
        s = "Functional Name: KLI approximated Exact Exchange (EXX-KLI)"
        return s


    @timer
    def compute(self, rho, eigvec, occ):
        """Compute exact exchange
        Arguments:
            rho (ndarray)     : charge density. shape=(nspin, ngpts)
            eigvec (ndarray) : eigenvectors. shape=(nspins, nkpts, nbands, ngpts)
            occ (ndarray)     : occupation number. shape=(nspins, nkpts, nbands)
        """
        ## rho.shape     : (nspins, ngpts)
        ## eigvec.shape : (nspins, nkpts) dtype=object. element's shape=(nbands, ngpts)
        ## phi_ski.shape : (nspins, nkpts * nbands, ngpts)
        ## occ.shape     : (nspins, nkpts, nbands)
        ## occ_ski.shape : (nspins, nkpts * nbands)

        if eigvec is None:
            print(f"eigvec is not defined. Set EXX-KLI potential and energy to 0.0.")
            self.__Vx = 0.0
            self.__Ex = 0.0
            return

        nspins = eigvec.shape[0]
        nkpts  = eigvec.shape[1]
        nbands = eigvec[0,0].shape[0] # Implement get_occupied_nbands() !!
        ngpts  = eigvec[0,0].shape[1]
        self.nspins, self.nkpts, self.nbands, self.ngpts = nspins, nkpts, nbands, ngpts

        phi_ski = np.empty((nspins, nkpts * nbands), dtype=object)
        for i_s, i_k, i in product(range(nspins), range(nkpts), range(nbands)):
            phi_ski[i_s, i_k * nbands + i] = eigvec[i_s, i_k][i]
        phi_ski /= np.sqrt(self.__grid.microvolume)
        occ_ski = occ.reshape(nspins, nkpts * nbands)

        ## Calculate KLI potential
        V_KLI  = np.zeros((nspins, ngpts), dtype=float)
        V_S = np.zeros((nspins, ngpts), dtype=float)
        for i_s in range(nspins):
            u_conj_U_ki  = self._compute_u_conj_U_ki(phi_ski[i_s], occ_ski[i_s]) # time-consuming
            V_S[i_s] = self._compute_V_S(rho[i_s], occ_ski[i_s], phi_ski[i_s], u_conj_U_ki)
            continue
            # u_conj_U_ki.shape = (nkpts * nbands,)---(ngpts,)
            U_bar = self.__grid.integrate(u_conj_U_ki * phi_ski[i_s]).reshape(-1)
            # U_bar.shape = (nkpts * nbands,)---(ngpts,)
            V_KLI_bar = self._compute_V_KLI_bar(occ_ski[i_s], rho, phi_ski[i_s], U_bar, u_conj_U_ki)
            # V_KLI_bar.shape = (nkpts * nbands,)
            V_KLI[i_s] = (phi_ski[i_s] * (u_conj_U_ki + phi_ski[i_s].conj() *
                             (V_KLI_bar - U_bar))).sum(0).real * 2
        del(u_conj_U_ki)
#        del(U_bar)
#        del(V_KLI_bar)
        self.__Vx = V_S
        self.__Ex = 0.0
        return

        ## V_KLI divided by rho
        asymp_tol = 1E-10
        upper_tol = abs(rho) > asymp_tol
        if upper_tol.all():
            V_KLI /= rho
        else:
            print("EXX-KLI asymptotic potential is applied.")
            V_KLI[upper_tol] /= rho[upper_tol]
            asymp_V_KLI = - self.__poisson_solver.compute_potential(rho, remove_avg=True)
            asymp_V_KLI = np.asarray(list(asymp_V_KLI))
            asymp_V_KLI /= self.__grid.integrate(rho).reshape(-1)
            V_KLI[np.logical_not(upper_tol)] = asymp_V_KLI[np.logical_not(upper_tol)]

        self.__Vx = V_KLI
        self.__Ex = self._compute_Ex(occ_ski, phi_ski, V_KLI) # 임시.
        print(f"Debug: int[rho*V_KLI] = {self.__grid.integrate(V_KLI * rho)}")
        return


    def _compute_Ex(self, occ_ski, phi_ski, V_KLI):
        Ex = 0.0
        for i_s in range(len(phi_ski)):
            for ki, qj in combinations_with_replacement(range(len(phi_ski[i_s])), 2):
                if occ_ski[i_s, ki] * occ_ski[i_s, qj] < 1E-8: continue
                print(f"Debug: ki, qj = {ki}, {qj}")
                pair_density = phi_ski[i_s, ki:ki+1].conj() * phi_ski[i_s, qj:qj+1] # shape=(1,)---(ngpts,)
                pair_density *= occ_ski[i_s, ki] * occ_ski[i_s, qj]
                kq = self.__kpts[ki % self.nkpts] - self.__kpts[qj % self.nkpts]
                V_kiqj = self.__exx_solver.compute_potential(pair_density, remove_avg=True, kq=kq).reshape(-1)
                print(f"Debug: integrate(pair_density.conj() * V_kiqj) = {self.__grid.integrate(pair_density.conj() * V_kiqj)}")
                print(f"Debug: integrate(pair_density) = {self.__grid.integrate(pair_density)}")
                print(f"Debug: integrate(V_kiqj) = {self.__grid.integrate(V_kiqj)}")
                tmp = self.__grid.integrate(pair_density.conj() * V_kiqj).real.item()
                if ki == qj:
                    Ex += tmp * 0.5
                else:
                    Ex += tmp
        return - Ex


    def __compute_Ex(self, occ_ski, phi_ski, V_KLI):
        sum_pair_density = np.zeros_like(V_KLI) # shape=(nspins, ngpts)
        for i_s in range(len(phi_ski)):
            for ki, qj in combinations_with_replacement(range(len(phi_ski[i_s])), 2):
                pair_density = occ_ski[i_s, ki] * occ_ski[i_s, qj] * phi_ski[i_s, ki].conj() * phi_ski[i_s, qj]
                if ki == qj:
                    sum_pair_density[i_s] += pair_density * 0.5
                else:
                    sum_pair_density[i_s] += pair_density
            print(f"abs(sum_pair_density.imag).sum() = {abs(sum_pair_density[i_s].imag).sum()}")
        sum_pair_density = sum_pair_density.real
        print(f"Debug: integrate(sum_pair_density) = {self.__grid.integrate(sum_pair_density)}")
        print(F"Debug: integrate(sum_pair_density*V_KLI) = {self.__grid.integrate(sum_pair_density * V_KLI)}")
        Ex = self.__grid.integrate(sum_pair_density * V_KLI).sum()
        return - Ex


    @timer
    def _compute_V_KLI_bar(self, occ_ki, rho, phi_ki, U_bar, u_conj_U_ki):
        phi_square_ki = np.array(list(map(np.real, phi_ki.conj() * phi_ki))) #shape=(nkpts * nbands, ngpts)
        U_bar_real = np.array(list(map(np.real, U_bar))) # shape=(nkpts * nbands,)
        M = self._compute_M(occ_ki, rho, phi_square_ki)
        I_M = np.identity(len(M)) - M # I_M = (I - M)
        V_S_bar = self._compute_V_S_bar(rho, occ_ki, phi_ki, phi_square_ki, u_conj_U_ki)
        V_KLI_bar = np.linalg.inv(I_M) @ (V_S_bar - U_bar_real) + U_bar_real
        return V_KLI_bar


    def _compute_M(self, occ_ki, rho, phi_square_ki):
        M = np.zeros((len(phi_square_ki), len(phi_square_ki)), dtype=float)
        for ki, qj in combinations_with_replacement(range(len(phi_square_ki)), 2):
            M[ki, qj] = self.__grid.integrate(phi_square_ki[ki] * phi_square_ki[qj] / rho)
            M[ki, qj] *= occ_ki[ki] * occ_ki[qj]
            if ki != qj:
                M[qj, ki] = M[ki, qj]
        return M


    def _compute_V_S_bar(self, rho, occ_ki, phi_ki, phi_square_ki, u_conj_U_ki):
        """
        Error. phi_square_ki : (nkpts * nband, ngpts)
               XXXXX U_bar_real : (nkpts * nbands,)
               occ_ki : (nkpts * nbands,)
               U_real : (nkpts * nbands, ngpts)
        """
        tmp = occ_ki * phi_ki * u_conj_U_ki
        tmp = np.array(list(map(np.real, tmp))).sum(0) # shape=(ngpts,)
        return self.__grid.integrate(phi_square_ki / rho * tmp).reshape(-1)


    def _compute_V_S(self, rho, occ_ki, phi_ki, u_conj_U_ki):
        V_S = (occ_ki * phi_ki * u_conj_U_ki).sum()
        print(f"Debug: V_S.shape = {V_S.shape}")

        asymp_tol = 1E-10
        upper_tol = abs(rho) > asymp_tol
        if upper_tol.all():
            V_S /= rho
        else:
            print("Slater asymptotic potential is applied.")
            V_S[upper_tol] /= rho[upper_tol]
#            asymp_V = - self.__poisson_solver.compute_potential(rho.reshape(1,-1), remove_avg=True)[0]
            asymp_V = - self.__poisson_solver.compute_potential(rho)
            asymp_V = np.asarray(list(asymp_V))
            asymp_V /= self.__grid.integrate(rho).reshape(-1)
            V_S[np.logical_not(upper_tol)] = asymp_V[np.logical_not(upper_tol)]
        print(f"Debug: V_S.shape = {V_S.shape}")
        return V_S


    @timer
    def _compute_u_conj_U_ki(self, phi_ki, occ_ki):
        nkpts = self.nkpts
        nbands = self.nbands
        ## phi_ki.shape = (nkpts * nbands,)---(ngpts,)
        u_conj_U_ki = np.empty_like(phi_ki) # shape=(nkpts * nbands,)---(ngpts,)
        for ki in range(nkpts * nbands):
            V_kiqj = np.empty_like(phi_ki) # shape=(nkpts * nbands,)---(ngpts,)
            k = ki // nbands
            for q in range(nkpts):
                kq = self.__kpts[k] - self.__kpts[q]
                q_block = slice(q * nbands, (q + 1) * nbands)
                pair_density = phi_ki[ki:ki+1].conj() * phi_ki[q_block] # shape=(nbands,)---(ngpts,)
                V_kiqj[q_block] = self.__exx_solver.compute_potential(pair_density, remove_avg=True, kq=kq)
            u_conj_U_ki[ki] = (- occ_ki[ki] * phi_ki[ki:ki+1].conj() * V_kiqj).sum(0) # shape=(ngpts,)
#        return u_conj_U_ki
        return -u_conj_U_ki


    def _get_occupied_nbands(self):
        pass

    @property
    def Vxc(self): return self.__Vx

    @property
    def Exc(self): return self.__Ex

    @property
    def functional_type(self): return self.__functional_type
