"""This module has been deprecated."""
import numpy as np
from itertools import product
from scipy.sparse.linalg import eigsh, lobpcg
from numpy.linalg import eigh
from gospel.Poisson.Poisson_solver import Poisson_solver
from gospel.util import timer
from gospel.Eigensolver.Eigensolver import Eigensolver


class Scipy(Eigensolver):
    def __init__(self, maxiter=20, scipy_type="lobpcg", verbosity=0):
        super().__init__()
        self._type = "scipy"
        self.__scipy_type = scipy_type  # 'lobpcg' or 'eigsh' or 'eigh'
        assert type(maxiter)==int, "maxiter type should be int"
        self._maxiter = maxiter
        self.__verbosity = verbosity
        return

    def __str__(self):
        s = str()
        s += f"\n======================= [ Eigensolver(Scipy) ] ======================="
        s += f"\n* scipy_type   : {self.__scipy_type}"
        if self.__scipy_type == "lobpcg":
            s += f"\n* maxiter      : {self._maxiter}"
            s += f"\n* verbosity    : {self.__verbosity}"
        s += f"\n======================================================================\n"
        return str(s)

    @timer
    def diagonalize(self, hamiltonian, convg_tol=None, i_scf=None):
        self._initialize_guess(hamiltonian)

        ## set convergence tolerance
        if convg_tol is None:
            # convg_tol = hamiltonian.ngpts * np.sqrt(1e-15)
            convg_tol = hamiltonian.ngpts * 1e-10
            print(f"convg_tol is set to {convg_tol}")
        else:
            convg_tol = self._convg_tol

        ## Solve eigenvalue problem
        solver = {"eigsh": eigsh, "lobpcg": lobpcg, "eigh": eigh}
        eigval = np.zeros_like(self._starting_value)
        eigvec = np.zeros_like(self._starting_vector)
        for i_s, i_k in product(range(eigvec.shape[0]), range(eigvec.shape[1])):
            if self.__scipy_type == "eigsh":
                assert S is None
                solve_options = {
                    "A": hamiltonian[i_s, i_k],
                    "k": hamiltonian.nbands,
                    "which": "SA",
                    "tol": convg_tol,
                    "v0": self._starting_vector[i_s, i_k][0],
                }
            elif self.__scipy_type == "lobpcg":
                solve_options = {
                    "A": hamiltonian[i_s, i_k],
                    "X": self._starting_vector[i_s, i_k],
                    "B": hamiltonian.get_S(),
                    "M": self._preconditioner(),
                    "tol": convg_tol,
                    "maxiter": self._maxiter,
                    "largest": False,
                    "verbosityLevel": self.__verbosity,
                }
            elif self.__scipy_type == "eigh":
                solve_options = {
                    "a": hamiltonian[i_s, i_k].todense(),
                    "b": hamiltonian.get_S(),
                }
            else:
                assert False, f"{self.__scipy_type} is not supported scipy_type."
            val, vec = solver[self.__scipy_type](**solve_options)

            ## sort
            nbands = hamiltonian.nbands
            indices = val.argsort()
            val, vec = val[indices][:nbands], np.asarray(vec[:, indices])[:, :nbands]
            eigval[i_s, i_k] = val
            eigvec[i_s, i_k] = vec.T  ## memory copy.. (not eifficient code)

            ## set starting_vector
            self._starting_value[i_s, i_k] = val
            self._starting_vector[i_s, i_k] = vec
        return eigval, eigvec
