import numpy as np
from itertools import product, combinations_with_replacement
from gospel.Poisson.ISF_exx_solver import create_ISF_exx_solver
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
    def __init__(self, grid, kpts, func_type='kli', exx_solver=None):
        print(f"Slater potential Test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        st = time()
        #assert func_type in ['slater', 'exx']
        assert func_type in ['slater']
        self.__grid = grid
        self.__functional_type = func_type
        self.__kpts = kpts

        ## Initialize exchange solvers
        if exx_solver is None:
            assert kpts is not None
            self.__exx_solver = create_ISF_exx_solver(grid, kpts)
        else:
            self.__exx_solver = exx_solver
        print(self.__exx_solver)

        self.__poisson_solver = create_ISF_exx_solver(grid)
#        self.__poisson_solver = Poisson_ISF(grid)

        self.__Ex = None
        self.__Vx = None
        print(f"Debug : KLI init Time : {time()-st} sec")
        return


    def get_describe(self):
        s = "Functional Name: KLI approximated Exact Exchange (EXX-KLI)"
        return s


    def compute(self, rho, eigvec, occ):
        """Compute exact exchange
        Arguments:
            rho (ndarray)     : charge density. shape=(nspin, ngpts)
            eigvec (ndarray) : eigenvectors. shape=(nspins, nkpts, nbands, ngpts)
            occ (ndarray)     : occupation number. shape=(nspins, nkpts, nbands)
        """
        ## rho.shape     : (nspins, ngpts)
        ## eigvec.shape : (nspins, nkpts) dtype=object. element's shape=(nbands, ngpts)
        ## occ.shape     : (nspins, nkpts, nbands)

        if eigvec is None:
            print(f"'eigvec' is not defined. Set EXX-KLI potential and energy to 0.0.")
            self.__Vx = np.zeros_like(rho)
            self.__Ex = 0.0
            return

        nspins = eigvec.shape[0]
        nkpts  = eigvec.shape[1]
        nbands = eigvec[0,0].shape[0] # Implement get_occupied_nbands() !!
        ngpts  = eigvec[0,0].shape[1]
        self.nspins, self.nkpts, self.nbands, self.ngpts = nspins, nkpts, nbands, ngpts

        occ_ski = self._get_only_occupied(occ, occ) # shape=(nspins, nkpts)---(nband_occ,)
        phi_ski = self._get_only_occupied(eigvec, occ) # shape=(nspins, nkpts)---(nband_occ,)

        ## Calculate Slater potential
        V_S = np.zeros((nspins, ngpts), dtype=float) # Slater potential
        if self.__functional_type == 'exx':
            V_KLI = np.zeros((nspins, ngpts), dtype=float) # Slater potential

        for i_s in range(nspins):
            phi_ki, occ_ki = phi_ski[i_s], occ_ski[i_s]
            u_conj_U_ki = self._compute_u_conj_U_ki(phi_ki, occ_ki)
            V_S[i_s] = self._compute_V_S(rho[i_s], occ_ki, phi_ki, u_conj_U_ki)
            if self.__functional_type == 'slater':
                continue
            u_square_ki = abs(phi_ki.conj() * phi_ki)
            delta_M = self._compute_delta_M(occ_ki, u_square_ki, rho)
#            U_bar_ki = self._compute_U_bar_ki(phi_ki, u_conj_U_ki)
            V_S_bar_U_bar_ki = self._compute_V_S_bar_U_bar_ki(phi_ki, V_S[i_s], u_conj_U_ki)
            V_KLI_bar_ki = self._compute_V_KLI_bar_ki(occ_ki, rho[i_s], phi_ki,
                                                      U_bar_ki, u_conj_U_ki)
            V_KLI[i_s] = (phi_ki * (u_conj_U_ki + phi_ki.conj() *
                         (V_KLI_bar - U_bar))).sum(0).real * 2

        self.__Vx = V_S
        self.__Ex = self._compute_Ex(occ_ski, phi_ski)
        return

    def _compute_V_S_bar_U_bar_ki(self, phi_ki, V_S, u_conj_U_ki):
        """Calculate <phi_{ki}|V^{s} - U_{ki}|phi_{ki}> = \bar{V}^{s}_{ki}(r) - \bar{U}_{ki}(r)
        """
#        potential_diff = phi_ki * V_S
#        u_coeff = u_conj_U_ki * np.sqrt(mv)
#        potential_diff -= u_coeff
#        value = potential_diff @ phi_ki
        V_S_bar_U_bar_ki = (phi_ki * V_S - u_conj_U_ki * np.sqrt(mv)) @ phi_k
        
        #V_S_bar_U_bar_ki = 
        return V_S_bar_U_bar_ki
        pass


    def _compute_delta_M(self, occ_ki, u_square_ki, rho):
        """Calculate \delta_{ij} \delta_{kq} - M_{kiqj}
        """
        occ_ki_nohomo = self._get_only_occupied(occ_ki, occ_ki, count_homo=False)
        index_set = []
        for i_k in range(len(occ_ki_nohomo)):
            for i in range(len(occ_ki_nohomo[i_k])):
                index_set.append( (i_k, i) )
        delta_M = np.zeors((len(index_set), len(index_set)), dtype=float)
        for n1, ind1 in enumerate(index_set):
            i_k, i = ind1
            for n2, ind2 in enumerate(index_set):
                if n2 < n1:
                    continue
                i_q, j = ind2
                delta_M[n1, n2] = self.__grid.integrate(u_square_ki[i_k][i]
                                  * u_square_ki[i_q][j] / rho)
                delta_M[n1, n2] *= occ_ki_nohomo[i_k][i] * occ_ki_nohomo[i_q][j]
                delta_M[n2, n1] = delta[n1, n2]
        delta_M *= (self.nspins / 2)
        delta_M = np.identity(len(index_set)) - delta_M
        return delta_M


    def _compute_V_KLI_bar_ki(self, occ_ki, rho, phi_ki, U_bar_ki, u_conj_U_ki):
        pass

#    def _compute_U_bar_ki(self, phi_ki, u_conj_U_ki):
#        ## phi_ki.shape=(nkpts,)---(nbands_occ, ngpts)
#        ## u_conj_U_ki.shape=(nkpts,)---(nbands_occ, ngpts)
#        mv = self.__grid.microvolume
#        U_bar_ki = self.__grid.integrate(phi_ki * u_conj_U_ki / mv)
#        return U_bar_ki.reshape(-1)


    @timer
    def _compute_Ex(self, occ_ski, phi_ski):
        mv = self.__grid.microvolume
        energy = 0.0
        for i_s in range(self.nspins):
            for index_set in self._get_combined_index_set(occ_ski[i_s]):
                i_k, i = index_set[0]
                i_q, j = index_set[1]
                kq = self.__kpts[i_k] - self.__kpts[i_q]
                pair_density = phi_ski[i_s, i_k][i].conj() * phi_ski[i_s, i_q][j]
                pair_density /= mv
                pair_density = np.expand_dims(pair_density, axis=0) # 임시.
                V_kiqj = self.__exx_solver.compute_potential(pair_density, remove_avg=True, kq=kq)
                V_kiqj = V_kiqj[0] # 임시.
                tmp = self.__grid.integrate(phi_ski[i_s, i_k][i] * phi_ski[i_s, i_q][j].conj() * V_kiqj / mv)
                tmp *= occ_ski[i_s, i_k][i] * occ_ski[i_s, i_q][j]
                if not (i_k == i_q and i == j):
                    tmp *= 2
                energy += tmp
        energy *= self.nspins / 2
        return - 0.5 * energy


    def _get_combined_index_set(self, occ_ki):
        """Make orbital-pair index set.
        index_set = [ [(i_k1, i1), (i_k1, i1), ...], [(i_k1, i1), (i_k1, i2), ...] ]
        """
        index_set = []
        for i_k in range(len(occ_ki)):
            for i in range(len(occ_ki[i_k])):
                index_set.append( (i_k, i) )
        combined_index_set = []
        for ind in combinations_with_replacement(index_set, 2):
            combined_index_set.append(ind)
        return combined_index_set


    @timer
    def _compute_u_conj_U_ki(self, phi_ki, occ_ki):
        """
        phi_ki.shape=(nkpts,)---(nbands, ngpts)
        occ_ki.shape=(nkpts,)---(nbands_occ,)
        u_conj_U_ki.shape=(nkpts,)---(nbands_occ, ngpts)
        """
        mv = self.__grid.microvolume
        u_conj_U_ki = np.empty_like(occ_ki) # shape=(nkpts,)
        for i_k in range(len(occ_ki)):
            u_conj_U_ki[i_k] = np.zeros((len(occ_ki[i_k]), self.ngpts), dtype=float) # shape=(nbands_occ, ngpts)

        for index_set in self._get_combined_index_set(occ_ki):
            i_k, i = index_set[0]
            i_q, j = index_set[1]
            kq = self.__kpts[i_k] - self.__kpts[i_q]
            pair_density = phi_ki[i_k][i].conj() * phi_ki[i_q][j]
            pair_density /= mv
            pair_density = np.expand_dims(pair_density, axis=0) # 임시.
            V_kiqj = self.__exx_solver.compute_potential(pair_density, remove_avg=True, kq=kq)
            V_kiqj = V_kiqj[0] # 임시.
            if i_k == i_q and i == j: 
                u_conj_U_ki[i_k][i] -= occ_ki[i_q][j] * phi_ki[i_q][j].conj() * V_kiqj
            else:
                u_conj_U_ki[i_k][i] -= occ_ki[i_q][j] * phi_ki[i_q][j].conj() * V_kiqj
                u_conj_U_ki[i_q][j] -= occ_ki[i_k][i] * phi_ki[i_k][i].conj() * V_kiqj.conj() # f_ki * phi_ki.conj() * V_qjkI
        u_conj_U_ki *= (self.nspins / 2)
        u_conj_U_ki /= np.sqrt(mv)
        return u_conj_U_ki


    def _compute_V_kli(self, rho, occ_ki, phi_ki, u_conj_U_ki):
        pass


    @timer
    def _compute_V_S(self, rho, occ_ki, phi_ki, u_conj_U_ki):
        """Compute the Slater potential.

        rho.shape=(ngpts,)
        occ_ki.shape=(nkpts,)---(nbands_occ,) # dtype=object---float
        phi_ki.shape=(nkpts,)---(nbands_occ, ngpts) # dtype=object---float
        u_conj_U_ki.shape=(nkpts,)---(nbands_occ, ngpts) # dtype=object---float
        V_S.shape=(ngpts,)
        """
        V_S = np.zeros(self.ngpts, dtype=float)
        for i_k in range(len(occ_ki)):
            V_S += (occ_ki[i_k] * (phi_ki[i_k] * u_conj_U_ki[i_k]).T).sum(1)
        V_S *= (self.nspins / 2) / np.sqrt(self.__grid.microvolume)

        V_S /= rho
#         ## Replace numerically unstable points with asymptotic potentials. ##
#         asymp_tol = 1E-10
#         upper_tol = abs(rho) > asymp_tol
#         if upper_tol.all():
#             V_S /= rho
#         else:
#             print("Slater asymptotic potential is applied.")
#             V_S[upper_tol] /= rho[upper_tol]
# #            asymp_V = - self.__poisson_solver.compute_potential(rho.reshape(1,-1), remove_avg=True)[0]
#             asymp_V = - self.__poisson_solver.compute_potential(rho)
#             asymp_V = np.asarray(list(asymp_V))
#             asymp_V /= self.__grid.integrate(rho).reshape(-1)
#             V_S[np.logical_not(upper_tol)] = asymp_V[np.logical_not(upper_tol)]

        V_S *= (2.0 / self.nspins)
        return V_S


#    def _get_only_occupied(self, val, occ, criteria=1E-7):
    def _get_only_occupied(self, val, occ, count_homo=True, criteria=1E-7):
        """
        val_ski.shape=(nspins, nkpts)---(nbands_occ, ngpts) # dtype=(object)---(float)
        """
        val_ski = np.empty((self.nspins, self.nkpts), dtype=object)
        for i_s, i_k in product(range(self.nspins), range(self.nkpts)):
            if count_homo:
                val_ski[i_s, i_k] = val[i_s, i_k][occ[i_s, i_k] > criteria]
            else:
                val_ski[i_s, i_k] = val[i_s, i_k][occ[i_s, i_k] > criteria][:-1]
        return val_ski


    @property
    def Vxc(self): return self.__Vx

    @property
    def Exc(self): return self.__Ex

    @property
    def functional_type(self): return self.__functional_type
