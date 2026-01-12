from itertools import product
import copy
import torch
import numpy as np
from gospel.FdOperators import gradient
from gospel.Pseudopotential.KB_form import KB_form
from gospel.Pseudopotential.Pseudopotential import Pseudopotential
from gospel.util import Timer
# Hermitian of P (projector) is named PH thus ParallelHelper is used as itself in this file
from gospel.ParallelHelper import ParallelHelper
from gospel.LinearOperator import LinearOperator


class NCPP(Pseudopotential):
    """Norm-conserving Pseudopotential. Explanation is same with Pseudopotential class.

    :type  use_dense_proj: bool
    :param use_dense_proj:
        whether use dense or sparse for KB projectors
    """

    def __init__(
        self,
        grid,
        upf,
        NLCC=True,
        use_comp=True,
        filtering=False,
        num_near_cell=1,
        kpts=[[0, 0, 0]],
        use_dense_proj=False,
        device = torch.device("cpu")
    ):
        super().__init__(grid, upf, NLCC, use_comp, filtering, num_near_cell, kpts, device)

        assert upf.pseudo_type == "NC"
        self._has_nonlocal = False if sum(upf.num_projs.values()) == 0 else True
        self.__use_dense_proj = use_dense_proj
        self.device = device

        if self._has_nonlocal:
            self.__KB_form = KB_form(grid, upf, num_near_cell, kpts, use_dense_proj, device=device)
            print(self.__KB_form)
            self._V_NL_cache = {}  # cache for nonlocal operator
        else:
            self.__KB_form = None

    def __str__(self):
        upf_files = "\n\t".join(self._upf.filenames)
        info = (
            f"\n========================= [ Pseudopotential ] ========================="
            f"\n* type           : NCPP"
            f"\n* upf            : \n\t{upf_files}"
            f"\n* NLCC           : {self._NLCC}"
            f"\n* use_comp       : {self._use_comp}"
            f"\n* filtering      : {self._filtering}"
            f"\n* use_dense_proj : {self.__use_dense_proj}"
            f"\n=======================================================================\n"
        )
        return info

    @Timer.timeit
    def calc_nonlocal_energy(self, occ, eigvec):
        r"""Calculate nonlocal energy from KB projectors.

        :type  occ: torch.Tensor
        :param occ:
            array of occupation numbers, shape=(nspins, nibzkpts, nbands)
        :type  eigvec: torch.Tensor
        :param eigvec:
            array of eigenvectors, shape=(nspins, nibzkpts)---(nbands, ngpts)

        :rtype: float
        :return:
            external energies from nonlocal potentials, respectively.

        ( Eq )
        E_{ext} = \sum_{\sigma} \sum_{\vec{k}}^{N_k} \sum_{i}^{occ} < \phi_{i, \vec{k}, \sigma} | \hat{V}^{NL} | \phi_{i, \vec{k}, \sigma} >
        """
        E_NL = torch.zeros(1, dtype=eigvec[0,0].dtype, device=eigvec[0,0].device)

        if self._has_nonlocal:
            nspins, nibzkpts, nbands = occ.shape
            list_i_bands = ParallelHelper.split(torch.arange(nbands))
            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                occ_split = ParallelHelper.split(occ[i_s,i_k])
                indices= (abs(occ_split)>1e-7).to(eigvec[i_s,i_k].device)
                val = torch.sum(eigvec[i_s,i_k][indices].conj().T*self.get_nonlocal_op(i_k) ( eigvec[i_s,i_k][indices].T ), dim=0 )
                val = torch.sum(occ_split[indices]*val )
                if torch.is_complex(val):
                    assert torch.all( abs(val.imag) < 1e-7 )
                E_NL += val.real


#                for idx, i_band in enumerate(list_i_bands) :
#                #for i in range(nbands):
#                    if abs(occ[i_s, i_k, i_band]) < 1e-7:
#                        continue
#                    ev = eigvec[i_s, i_k][idx]
#                    val = ev.conj() @ self.get_nonlocal_op(i_k, )(ev)
#                    if val.dtype in [torch.complex64, torch.complex128]:
#                        assert abs(val.imag) < 1e-7
#                    E_NL += occ[i_s, i_k, i_band] * val.real
            E_NL = ParallelHelper.all_reduce(E_NL)
        return E_NL.item()

    @Timer.timeit
    def calc_forces(
        self,
        density,
        potential=None,
        deriv_density=False,
        deriv_comp=True,
    ):
        """Calculate atomic forces from nuclei. :math:`F = F_{NN} + F_{local} + F_{NL}`

        :type  density: gospel.Density
        :param density:
            It contains density, eigvec, and occ.
        :type  potential: torch.Tensor
        :param potential:
            poisson potential. (only used when use_comp==True)
        :type  deriv_density: bool
        :param deriv_density:
            derivatives of density or potential in the calculation of F_loc.
        :type  deriv_comp: bool
        :param deriv_comp:
            derivatives of compensation charge or potential in the calculation of F_comp.

        :rtype: torch.Tensor
        :return:
            atomic forces, shape=(natoms, 3)
        """
        if self._has_nonlocal:
            F_NL = self.calc_nonlocal_forces(density.eigvec, density.occ)
        else:
            F_NL = 0.0

        if self._use_comp:
            F_loc = self._compensation.calc_short_forces(
                density.get_density().sum(0),
                deriv_density=deriv_density,
            )
            F_comp = self._compensation.get_forces_correction(potential, deriv_comp)
            F = F_NL + F_loc + F_comp.cpu()
            F_NN = 0.0  # debug
        else:
            F_loc = self.calc_local_forces(
                density.get_density().sum(0),
                deriv_density=deriv_density,
            ).cpu()
            F_NN = self.calc_ion_ion_forces()
            F = F_NL + F_loc + F_NN
            F_comp = 0.0  # debug

        from pprint import pprint

        pprint(f"Debug:")
        pprint(f"  F_NN   = \n{F_NN}")
        pprint(f"  F_NL   = \n{F_NL}")
        pprint(f"  F_loc  = \n{F_loc}")
        pprint(f"  F_comp = \n{F_comp}")
        pprint(f"  F      = \n{F}")
        return F

    @Timer.timeit
    def calc_nonlocal_forces(self, eigvec, occ, deriv_orbital=True):
        r"""Calculate atomic forces from nonlocal potential.

        :type  eigvec: np.ndarray[torch.Tensor]
        :param eigvec:
            array of eigenvectors, shape=(nspins, nibzkpts)---(nbands, ngpts)
        :type  occ: torch.Tensor
        :param occ:
            array of occupation number, shape=(nspins, nibzkpts, nbands)
        :type  deriv_orbital: bool, optional
        :param deriv_orbital:
            derivatives of orbital or potential potential, defaults to True

        F_{NL} = -2 Re \sum_{s,k,i} f_{ski} <\psi_{ski} | V_{NL} | \nabla \psi_{ski} >
        """
        # assert eigvec[i_s, i_k].device==self.device
        # print(f"!!!!!!!!!!!! Debug: eigvec[0, 0].shape in calc_nonlocal_forces: {eigvec[0, 0].shape}")
        assert ParallelHelper.global_size == 1, "Parallelization is not supported 'calc_nonlocal_forces()' yet."
        assert occ.device==self.device, "device info should be matched"
        symbols = self._atoms.get_chemical_symbols()
        F = torch.zeros((len(symbols), 3), dtype=torch.float64, device=occ.device)
        nspins, nibzkpts = eigvec.shape
        nbands = len(eigvec[0, 0])
        ngpts = self._grid.ngpts


        if deriv_orbital:
            ## Calculate derivatives of orbital
            d_eigvec = np.zeros_like(
                eigvec
            )  # shape=(nspins, nibzkpts)---(nbands, 3, ngpts)
            for i_s, i_k in product(range(nspins), range(nibzkpts)):
                d_eigvec[i_s, i_k] = torch.zeros(
                    (nbands, 3, ngpts), dtype=eigvec[i_s, i_k].dtype, device=self.device
                )
                for i in range(nbands):
                    if abs(occ[i_s, i_k, i]) < 1e-7: continue
#                        d_eigvec[i_s, i_k][i] = torch.zeros(
#                            3, ngpts, dtype=torch.float64
#                        )
#                        continue
                    d_eigvec[i_s, i_k][i] = gradient(self._grid, eigvec[i_s, i_k][i])

            ## Calculate atomic forces
            for i_atom, i_sym in enumerate(symbols):
                D = self.__KB_form.get_D_matrix(i_atom)
                if len(D) == 0:  # if no projectors for this atom
                    continue
                for i_s, i_k in product(range(nspins), range(nibzkpts)):
                    P = self.__KB_form.get_KB_proj_op(i_k, i_atom)
                    #P = self.get_KB_proj(i_k, idx)
                    D = D.to(dtype=P.dtype, device=self.device)
                    for i in range(nbands):
                        _occ = occ[i_s, i_k, i]
                        if abs(_occ) < 1e-7:
                            continue
                        # now eigenvector and its graident are moved to cpu and computed which is super inefficient 
                        # TODO: KB proj operation on GPU
						# DONE: find details in get_KB_proj_op operator (not very efficient but anyway it works)
                        dpsi = d_eigvec[i_s, i_k][i]
                        _vec = eigvec[i_s, i_k][i]
                        #dpsi = d_eigvec[i_s, i_k][i].cpu()
                        #_vec = eigvec[i_s, i_k][i].cpu()
                        #_occ = _occ.cpu()

                        psi_P_D = (P @ _vec.T).conj().T @ D  # Warning: _vec.shape=(ngpts,) TODO: replace _vec.T
                        for axis in range(3):
                            val = psi_P_D @ (P @ dpsi[axis])
                            F[i_atom][axis] -= 2 * _occ * val.real
        else:  # derivate nonlocal potential instead of orbital
            raise NotImplementedError
        return F.cpu() # now positions of atoms are updated on CPU therefore force should be moved into CPU memory

    def get_nonlocal_matrix(self, i_k):
        """Get nonlocal matrix"""
        from scipy.sparse import csr_matrix

        assert self._has_nonlocal
        P = self.__KB_form.get_KB_proj(i_k)
        D = self.__KB_form.get_D_matrix(as_sparse=True)
        if isinstance(P, torch.Tensor):
            if P.is_sparse:
                D = self.__KB_form.get_D_matrix(as_sparse=False)
                return P.conj().T @ D.to(P.device) @ P
            else:
                raise NotImplementedError    
            #return (P.conj().T @ D.to(P.device) @ P).to_sparse()
        return csr_matrix(P.getH() @ D @ P)

    def get_nonlocal_op(
        self,
        i_k: int,
        dtype: torch.dtype = None,
    ) -> LinearOperator:
        """Return Pseudopotential's nonlocal operator

        Args:
            i_k (int): index of list of k-points
            dtype (torch.dtype, optional): data type of the nonlocal operator
        Returns:
            gospel.LinearOperator: nonlocal operator of the pseudopotential
        """
        # Set dtype to default if not specified
        if dtype is None:
            is_gamma = np.all(np.array(self._kpts[i_k]) == 0)
            dtype = torch.float64 if is_gamma else torch.complex128

        key = (i_k, dtype, self.__use_dense_proj)
        if key in self._V_NL_cache:
            return self._V_NL_cache[key]

        if self.__use_dense_proj:
            P_I = [p.to(self.device, dtype) for p in self.__KB_form.KB_proj[i_k]]
            D_I = [
                self.__KB_form.get_D_matrix(i_atom).to(self.device, dtype)
                for i_atom in range(len(self._atoms))
            ]
            I_I = [mask.to(self.device) for mask in self.__KB_form.isinside]

            def matvec(x):
                retval = torch.zeros_like(x)
                for P, D, I in zip(P_I, D_I, I_I):
                    retval[I] += P.conj().T @ (D @ (P @ x[I]))
                return retval
        else:
            P = self.__KB_form.KB_proj[i_k].to(self.device, dtype)
            PH = self.__KB_form.KB_proj_H[i_k].to(self.device, dtype)
            D = self.__KB_form.get_D_matrix().to(self.device, dtype)
            matvec = lambda x: PH @ (D @ (P @ x))

        op = LinearOperator(
            shape=(self._grid.ngpts, self._grid.ngpts),
            matvec=matvec,
            dtype=dtype,
            name="nonlocal_op",
        )
        self._V_NL_cache[key] = op
        return op

    @property
    def KB_form(self):
        return self.__KB_form

    @property
    def use_dense_proj(self):
        return self.__use_dense_proj

# @torch.jit.script
# def operate_nonlocal_projector(P, D, I, x):
#     return P.conj().T @ (D @ (P @ x[I]))