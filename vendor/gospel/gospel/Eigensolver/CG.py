"""This module has been deprecated."""
from gospel.util import timer
from gospel.Eigensolver.Eigensolver import Eigensolver
from scipy.sparse.linalg import eigsh, lobpcg
from itertools import product
import numpy as np
import copy
#import multiprocessing as mp
from multiprocessing import Pool


class CG(Eigensolver):
    """
    Payne M. C., Teter M. P., Allan D. C., Rev. Mod. Phys., 64, 4, 1045-1097 (1992)
    """
    def __init__(self, maxiter=25, convg_tol=1e-5, relative_tol=1e-3, orthogonal_to_prev=False, orthogonal_to_all=False):
        super().__init__()
        self._type                = 'cg'
        assert type(maxiter)==int, "maxiter type should be int"
        self._maxiter            = maxiter
        self._convg_tol          = convg_tol
        self.__relative_tol       = relative_tol
        self.__orthogonal_to_prev = orthogonal_to_prev
        self.__orthogonal_to_all  = orthogonal_to_all
        return


    def __str__(self):
        s = str()
        s += '\n======================== [ Eigensolver(CG) ] =========================='
        s += '\n* type               : {}'.format(self._type)
        s += '\n* convg_tol          : {}'.format(self._convg_tol)
        s += '\n* maxiter            : {}'.format(self._maxiter)
        s += '\n* relative_tol       : {}'.format(self.__relative_tol)
        s += '\n* orthogonal_to_prev : {}'.format(self.__orthogonal_to_prev)
        s += '\n* orthogonal_to_all  : {}'.format(self.__orthogonal_to_all)
        s += '\n=======================================================================\n'
        return s


    @timer
    def diagonalize(self, hamiltonian, i_iter=None):
        nspins   = hamiltonian.nspins
        nibzkpts = hamiltonian.nibzkpts
        nbands   = hamiltonian.nbands
        ngpts    = hamiltonian.ngpts

        ## Set starting_vector and starting_value
        if self._starting_vector is None:
            self._starting_vector = np.empty((nspins, nibzkpts), dtype=object)
            ## Randomize initial vectors and values
            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                if np.all(np.asarray(hamiltonian.kpts[i_k]) == 0):
                    self._starting_vector[i_s, i_k] = np.random.random((nbands, ngpts))
                else:
                    self._starting_vector[i_s, i_k] = np.random.random((nbands, ngpts))*(1+1j)
                ## orthonormalize starting_vector
                self._starting_vector[i_s, i_k] = np.linalg.qr( self._starting_vector[i_s, i_k].conj().T )[0].T[:nbands]

        ## For first few iterations, operate eigsh diagonalization. (CG's convergence performance is sensitive to guess's quality)
        first_eigsh_iter = 1
        if i_iter is not None and i_iter < first_eigsh_iter:
            return self.first_diagonalize(hamiltonian)

        eigval = np.empty((nspins, nibzkpts, nbands), dtype=float)
        eigvec = np.empty((nspins, nibzkpts), dtype=object)

        ## diagonalization
        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            val, vec = self.cg_iter(hamiltonian[i_s, i_k], self._starting_vector[i_s, i_k].T, nbands)

            ## sort
            indices = val.argsort()
            val, vec = val[indices], np.asarray(vec[:,indices])
            eigval[i_s, i_k], eigvec[i_s, i_k] = val[:nbands], vec[:,:nbands].T

            ## set starting_vector
            self._starting_vector[i_s, i_k] = eigvec[i_s, i_k]

        return eigval, eigvec


    def cg_iter(self, H, eigvec, nbands):
        Hpsi    = 0; phi  = 0; zeta      = 0
        eta     = 0; Hphi = 0;
        debug  = False

        ret_eigvec = copy.deepcopy(eigvec)
        ret_eigval = np.zeros(nbands)

        for i_band in range(nbands):
            psi = copy.deepcopy(eigvec[:,i_band])

            ## Orthogonalize starting eigenfunctions
            if i_band > 0:
                if self.__orthogonal_to_prev:
                    psi = self.orthogonalize(psi, eigvec, i_band)
                else:
                    psi = self.orthogonalize(psi, ret_eigvec, i_band)
                psi /= np.sqrt((psi.conj() @ psi).real)

            ## Calculate H|psi>
            Hpsi = H @ psi

            ## Calculate starting eigenvalue: eigval = <psi|H|psi>
            eigval = (psi.conj() @ Hpsi).real
            old_energy = copy.deepcopy(eigval)
            first_delta_e = 0

            ## Start CG iterations for each band
            for m in range(self._maxiter):

                ## Eq.5.10
                zeta = Hpsi - eigval * psi

#                ## Eq.5.12
#                if self.__orthogonal_to_prev:
#                    zeta = self.orthogonalize(g, eigvec, i_band, self.__orthogonal_to_all)
#                else:
#                    zeta = self.orthogonalize(g, ret_eigvec, i_band, self.__orthogonal_to_all)

                ## Eq.5.17
                ## Using inverse preconditioner
                if self._preconditioner is not None:
                    eta = self._preconditioner() @ zeta
                else:
                    eta = copy.deepcopy(zeta)

                ## Eq.5.18
                # This needs to be done before the orthogonalization_single call, as psi is not guaranted 
                # to be orthogonal to the other bands here
                eta -= (psi.conj() @ eta) * psi

                ## Orthogonalize against previous or all bands.
                if self.__orthogonal_to_prev:
                    eta = self.orthogonalize(eta, eigvec, i_band, self.__orthogonal_to_all)
                else:
                    eta = self.orthogonalize(eta, ret_eigvec, i_band, self.__orthogonal_to_all)

                eta_zeta = eta.conj() @ zeta
                if m == 0:
                    ## Eq.5.20
                    eta_zeta_prev = eta_zeta
                    phi  = copy.deepcopy(eta)
                else:
                    ## Eq.5.20
                    gamma = eta_zeta / eta_zeta_prev
                    eta_zeta_prev = eta_zeta

                    ## Eq.5.19
                    phi = gamma * phi + eta

                    ## Eq.5.21
                    # phi is not normalized here, phi_nrm is calculated further down.
                    phi -= (psi.conj() @ phi) * psi

                Hphi = H @ phi 

                ## Energy minimization (Eq.5.23 to Eq.5.38)
                phi_nrm = np.sqrt((phi.conj() @ phi).real) # calculate norm of phi
                ## Eq. 5.31
                dE_dtheta   = 2 * (psi.conj() @ Hphi).real / phi_nrm
                dE2_d2theta = 2 * (eigval - (phi.conj() @ Hphi).real / phi_nrm**2)

                ## Eq.5.37
                theta = np.arctan(2 * dE_dtheta/dE2_d2theta) * 0.5
                stheta = np.sin(theta)
                ctheta = np.cos(theta)
                es_1 = dE2_d2theta * (0.5 - stheta**2) + (2*dE_dtheta)*2*stheta*ctheta
                stheta2 = np.sin(theta + np.pi*0.5)
                ctheta2 = np.cos(theta + np.pi*0.5)
                es_2 = dE2_d2theta * (0.5 - stheta2**2) + (2*dE_dtheta)*2*stheta2*ctheta2

                ## Choose the minimum solutions
                if es_2 < es_1:
                    theta = theta + 0.5 * np.pi
                    cos_theta = ctheta2
                    sin_theta = stheta2 / phi_nrm
                else:
                    cos_theta = ctheta
                    sin_theta = stheta / phi_nrm

                ## Eq.5.38 (update psi, Hpsi)
                psi  = cos_theta * psi  + sin_theta * phi
                Hpsi = cos_theta * Hpsi + sin_theta * Hphi

                ## Calculate residue
                eigval = (psi.conj() @ Hpsi).real
                residue = np.linalg.norm(Hpsi - eigval * psi)

                ## Check convergence for relative energy tol
                if m == 0:
                    first_delta_e = abs(eigval - old_energy)
                else: # m > 0:
                    if abs(eigval - old_energy) < first_delta_e * self.__relative_tol:
                        if debug: print('break with relative_tol(={})'.format(self.__relative_tol))
                        break
                old_energy = copy.deepcopy(eigval)

                ## Check convergence for residue
                if debug: print('residue = ', residue)
                if residue < self._convg_tol:
                    if debug:
                        print('[i_band={}] break with {} iters ({} < {})'.format(i_band+1, m+1, residue, self._convg_tol))
                    break
                if m == self._maxiter and debug:
                    print('[i_band={}] no break'.format(i_band))

            ## Orthogonalize psi
            if self.__orthogonal_to_prev:
                psi = self.orthogonalize(psi, eigvec, i_band)
            else:
                psi = self.orthogonalize(psi, ret_eigvec, i_band)
            psi /= np.sqrt(psi.conj() @ psi)

            ## Update an eigen vector
            ret_eigvec[:, i_band] = copy.deepcopy(psi)
            ret_eigval[i_band] =  copy.deepcopy(eigval)
        return ret_eigval, ret_eigvec


    def orthogonalize(self, f, psi, i_band=None, orthogonal_to_all=False):
        nbands = psi.shape[-1]
        f_ = copy.deepcopy(f)
        if orthogonal_to_all:
            mask = np.delete(np.arange(nbands), i_band)
            f_ -= np.sum((f @ psi[:, mask].conj()) * psi[:, mask], axis=-1)
        else:
            f_ -= np.sum((f @ psi[:, :i_band].conj()) * psi[:, :i_band], axis=-1)
        return f_


    @timer
    def first_diagonalize(self, hamiltonian, i_iter=None):
        maxiter = 20
        diag_type = 'eigsh' #'lobpcg', 'eigh'

        nspins   = hamiltonian.nspins
        nibzkpts = hamiltonian.nibzkpts
        nbands   = hamiltonian.nbands
        ngpts    = hamiltonian.ngpts

        eigval = np.empty((nspins, nibzkpts, nbands), dtype=float)
        eigvec = np.empty((nspins, nibzkpts), dtype=object)

        for i_s, i_k in product(range(nspins), range(nibzkpts)):
            if diag_type == 'eigsh':
                print('first diagonalization with eigsh')
                val, vec = eigsh(hamiltonian[i_s, i_k], nbands, which='SA',
                                 tol=self._convg_tol)#, v0=self._starting_vector[i_s, i_k])
            if diag_type == 'lobpcg':
                print('first diagonalization with lobpcg')
                val, vec = lobpcg(hamiltonian[i_s, i_k],
                                  self._starting_vector[i_s, i_k].T,
                                  tol=self._convg_tol, maxiter=maxiter, largest=False)
            if diag_type == 'eigh':
                print('first diagonalization with eigh')
                val, vec = eigh(hamiltonian[i_s, i_k].todense())
            ## sort
            indices = val.argsort()
            val, vec = val[indices], np.asarray(vec[:,indices])
            eigval[i_s, i_k], eigvec[i_s, i_k] = val[:nbands], vec[:,:nbands].T
            ## set starting_vector
            self._starting_vector[i_s, i_k] = eigvec[i_s, i_k]
        return eigval, eigvec


if __name__=="__main__":
    import scipy.sparse as sparse
    from ase.build import bulk
    from gospel.Grid import Grid
    from gospel.Eigensolver.Preconditioner import Preconditioner
    from gospel.Eigensolver.Scipy import Scipy
    from gospel.Poisson.Poisson_ISF import Poisson_ISF

    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    grid = Grid(atoms, gpts=[4,4,4], spacing=1.) 
    poisson = Poisson_ISF(grid)

    preconditioner = Preconditioner(grid, None, None)
#    preconditioner = Preconditioner(grid, poisson, 'laplacian')

    nbands = grid.ngpts-1
    nspins = 1
    nibzkpts = 1

    class hamiltonian:
        def __init__(self, nspins, nibzkpts, nbands):
            self.nspins = nspins
            self.nibzkpts = nibzkpts
            self.nbansd = nbands
            self.ngpts = grid.ngpts
            self.kpts = [(0,0,1)]
            self.H = np.empty((1,1), dtype=object)
#            M = np.random.rand(self.ngpts, self.ngpts)
            M = np.random.rand(self.ngpts, self.ngpts) + np.random.rand(self.ngpts, self.ngpts) * 1j
            self.H[0,0] = sparse.csr_matrix(M + M.conj().T)


        def __getitem__(self, indx):
            i_s, i_k = indx
            if i_s >= self.nspins or i_k >= self.nibzkpts:
                raise IndexError
            else:
                return self.H[i_s, i_k]

    H = hamiltonian(nspins, nibzkpts, nbands)


    print('=========================[ CG ]================================')
    cg = CG(maxiter=100, convg_tol=1e-7, relative_tol=1e-3, orthogonal_to_prev=False, orthogonal_to_all=False)
    cg.preconditioner = preconditioner
    eigval, eigvec = cg.diagonalize(H)
    print('eigval = ', np.round(eigval,6))
#    print('eigvec = ', eigvec)
#    print(eigvec[0,0].T.conj() @ eigvec[0,0])
    print('===============================================================')

    print('=========================[ LOBPCG ]============================')
    lobpcg = Scipy(maxiter=25, convg_tol=1e-5)
    lobpcg.preconditioner= preconditioner
    eigval, eigvec = lobpcg.diagonalize(H)
    print('eigval = ', np.round(eigval,6))
#    print('eigvec = ', eigvec)
#    print(eigvec[0,0].T.conj() @ eigvec[0,0])
    print('===============================================================')
