from itertools import product
import time
import torch
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import block_diag
from scipy.interpolate import interp1d

from gospel.special_functions import Y_lm_real_batch
from gospel.LinearOperator import LinearOperator
from gospel.util import Timer
np.set_printoptions(threshold=1e10)


class KB_form:
    """Provide pseudopotential's projectors with Kleinman-Bylander form.

    :type  grid: gospel.Grid
    :param grid:
        Grid object
    :type  upf: gospel.Pseudopotential.UPF
    :param upf:
        UPF object. It read UPF files and ready to use.
    :type  num_near_cell: int
    :param num_near_cell:
        the number of near cells to consider when PBC
    :type  kpts: list or ndarray
    :param kpts:
        list of k-points, e.g., [[0,0,0], [1,0,0], ...]
    :type  use_dense_proj: bool, optional
    :param use_dense_proj:
        whether use dense or sparse for KB projectors, defaults to False
    """
    @Timer.timeit
    def __init__(
        self, grid, upf, num_near_cell=1, kpts=[[0, 0, 0]], use_dense_proj=False, device=torch.device('cpu')
    ):
        self.__grid = grid
        self.__upf = upf
        self.__num_near_cell = num_near_cell
        self.__kpts = kpts
        self.__use_dense_proj = use_dense_proj
        self.device = device

        self.__num_projs = None  ## calculated inside self.calc_D_matrix()
        self.__D_matrix = torch.from_numpy(self.calc_D_matrix(grid.atoms, upf))
        if use_dense_proj:
            self.__KB_proj, self.__isinside = self.calc_KB_proj2(grid, upf, kpts)
        else:  # use sparse projectors
            self.__KB_proj = [P.to_sparse_csr() for P in self.calc_KB_proj2(grid, upf, kpts)]
            self.__KB_proj_H = [P.t()._conj_physical() for P in self.__KB_proj]
        assert self.__num_projs is not None
        return

    def __str__(self):
        s = str()
        s += f"\n========================== [ KB_projector ] =========================="
        s += f"\n* number of atoms : {len(self.__grid.atoms)}"
        s += f"\n* use_dense_proj  : {self.__use_dense_proj}"
        s += f"\n symbol | natoms | nproj each"
        symb_list = self.__grid.atoms.get_chemical_symbols()
        for symb, repeat_num in zip(*np.unique(symb_list, return_counts=True)):
            i_atom = symb_list.index(symb)
            s += f"\n{symb:>{7}} | {repeat_num:>{6}} | {self.__num_projs[i_atom]:>{10}}"
        s += f"\n======================================================================\n"
        return s

    def calc_D_matrix(self, atoms, upf):
        """Calculate D matrix.

        :type  atoms: ase.Atoms
        :param atoms:
            Atoms class object
        :type  upf: gospel.Pseudopotential.UPF
        :param upf:
            UPF class object
        :rtype: list[torch.Tensor]
        :return:
            list of D matrices

        'list_D' is a list or numpy arrays.
        'list_D[i_atom][combined index]' ; i_atom is atom index, and combined index is the index that combines angular momuentum(l) and magnetic momentum(m) indices.

                           (l/m)
                          ______
                         |      |
        list_D[i_atom] = |  Di  | (l/m)  : (l/m)-by-(l/m) matrix
                         |______|
                  _                                 _
                 |            ___________            |
                 |           |           |           |
                 |   _____   |           |           |
        list_D = |  |     |  |    D1     |           |
                 |  | D0  |  |           |           |
                 |  |_____|, |___________|, ...      |
                 |_                                 _|

        : (l/m) represents the combination of l and m.
                ex) (l/m) = (0/0), (1/-1), (1/0), (1/1), (2/-2), ...
        """
        ## Make list of D matrices of each atom
        list_D = list()
        for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
            num_proj = upf[i_sym].num_proj
            if num_proj == 0:
                D_matrix = np.zeros((0, 0), dtype=float)
            else:
                dij = np.array(upf[i_sym].dij).reshape(num_proj, num_proj)
                L = upf[i_sym].angmom  # len(L) == num_proj
                nLM = sum([2 * l + 1 for l in L])
                D_matrix = np.zeros((nLM, nLM), dtype=float)
                start_m_i = 0
                for i, l_i in enumerate(L):
                    stop_m_i = start_m_i + 2 * l_i + 1
                    start_m_j = 0
                    for j, l_j in enumerate(L):
                        stop_m_j = start_m_j + 2 * l_j + 1
                        if l_i == l_j:
                            dm = stop_m_i - start_m_i
                            D_matrix[start_m_i:stop_m_i, start_m_j:stop_m_j] = (
                                np.eye(dm) * dij[i, j]
                            )
                        else:
                            assert D_matrix[i, j] == 0.0
                        start_m_j = stop_m_j
                    start_m_i = stop_m_i
            list_D.append(D_matrix)

        ## Calculate the number of projectors for each atom.
        self.__num_projs = np.array(list(map(len, list_D)))
        return block_diag(*list_D)

    def calc_KB_proj_per_atom2(self, grid, atom_position, upf, kpts):
    #def calc_KB_proj_per_atom2(self, grid, atom_position, upf, kpts, as_sparse=False):
        """Calculate projectors of an atom for kpts

        :rtype: (np.ndarray, np.ndarray)
        :return:
            (KB-projectors, indices); KB projectors and correspoing indices
            KB-projectors.shape=(nkpts, nLM, nnz)
            indices.shape=(ngpts,)
        """
        atoms = grid.atoms

        ## Under PBC, the number of near cells to consider potentials coming in.
        num_near = [self.__num_near_cell if grid.get_pbc()[i] else 0 for i in range(3)]

        nLM = sum([2 * el + 1 for el in upf.angmom])
        is_inside = torch.full((grid.ngpts,), False, device=self.device)

        ## make template of projector. shape=(nkpts,)---(nLM, ngpts)
        projector = np.empty(len(kpts), dtype=object)
        for i_k, kpt in enumerate(kpts):
            is_gamma = np.all(np.array(kpt) == 0)
            dtype = torch.float64 if is_gamma else torch.complex128
            projector[i_k] = torch.zeros(nLM, grid.ngpts, dtype=dtype, device=self.device)

        ##################################################################
        kpts = torch.tensor(kpts, dtype=torch.float64, device=self.device)
        cell = grid.get_cell().to(self.device)
        i_xyzs = torch.tensor(
            list(
                product(
                    range(-num_near[0], num_near[0] + 1),
                    range(-num_near[1], num_near[1] + 1),
                    range(-num_near[2], num_near[2] + 1),
                )
            ),
            dtype=cell.dtype,
            device=self.device,
        )
        center_point = torch.from_numpy(grid.center_point).to(self.device)
        points = grid.points.to(self.device)
        ##################################################################

        is_batch_i_xyz = True
        # is_batch_i_xyz = False
        if is_batch_i_xyz:
            shifts = (i_xyzs @ cell).unsqueeze(dim=1)
            nshifts = len(shifts)
            del i_xyzs
            R_a = atom_position
            R_a_batch = R_a.repeat(nshifts, 1, 1) + shifts
            del shifts

            d_R_a_R_c = (R_a_batch - center_point).norm(dim=-1) # |R_a - R_c|

            start_l = 0
            for i_proj in range(upf.num_proj):
                el = upf.angmom[i_proj]
                stop_l = start_l + 2 * el + 1
                cutoff_radius = upf.beta_cutoff[i_proj]

                cubic_interp = interp1d(
                    upf.R,
                    upf.beta[i_proj],
                    kind="cubic",
                    bounds_error=False,
                    fill_value=0.0,
                )

                is_overlap_cell = d_R_a_R_c <= grid.get_radius() + cutoff_radius
                R_a_batch_i = R_a_batch[is_overlap_cell.squeeze(dim=1)]
                nshifts_i = len(R_a_batch_i)

                points_batch = points.repeat(nshifts_i, 1, 1)  # shape=(nshifts_i, ngpts, 3)

                r_R_a = points_batch - R_a_batch_i  # shape=(nshifts_i, ngpts, 3)
                d_r_R_a = r_R_a.norm(dim=-1)
                del points_batch

                mask = d_r_R_a < cutoff_radius
                is_inside += mask.sum(dim=0, dtype=bool).squeeze()

                beta = torch.from_numpy(cubic_interp(d_r_R_a[mask].cpu())).to(self.device)

                for i_k, kpt in enumerate(kpts):

                    is_gamma = (kpt == 0).all()
                    y_lm = Y_lm_real_batch(r_R_a[mask], el)
                    if not is_gamma:
                        proj_tmp = torch.zeros(2 * el + 1, nshifts_i, grid.ngpts, dtype=torch.complex128, device=self.device)

                        e_ikr = torch.exp(1j * (r_R_a[mask] @ kpt))
                        tmp = e_ikr * beta.to(torch.complex128) * y_lm.to(torch.complex128)
                        proj_tmp[:, mask] = tmp
                    else:
                        proj_tmp = torch.zeros(2 * el + 1, nshifts_i, grid.ngpts, dtype=torch.float64, device=self.device)
                        proj_tmp[:, mask] = beta * y_lm  # shape=(2 * el + 1, nnz)

                    proj_tmp = proj_tmp.sum(dim=1)  # shape=(2 * el + 1, ngpts)
                    projector[i_k][start_l:stop_l] += proj_tmp

                start_l = stop_l

            ## only get non-zero values
            for i_k in range(len(kpts)):
                projector[i_k] = projector[i_k][:, is_inside]
            projector *= np.sqrt(grid.microvolume)

            return projector, is_inside

        st = time.time()
        interp_time = 0.0
        ylm_time = 0.0
        chk_time1 = 0.0
        ## interpolate to grid
        start_l = 0
        for i_proj in range(upf.num_proj):
            el = upf.angmom[i_proj]
            stop_l = start_l + 2 * el + 1
            cutoff_radius = upf.beta_cutoff[i_proj]

            st_interp = time.time()
            cubic_interp = interp1d(
                upf.R,
                upf.beta[i_proj],
                kind="cubic",
                bounds_error=False,
                fill_value=0.0,
            )
            interp_time += time.time() - st_interp

            for i_xyz in i_xyzs:
                N_c = i_xyz @ cell  # cell vector
                R_a = atom_position + N_c
                st_chk1 = time.time()
                # d_R_a_R_c = (R_a - center_point).norm()  # |R_a - R_c|
                d_R_a_R_c = (R_a - center_point).pow(2).sum(dim=0).sqrt() # |R_a - R_c|
                chk_time1 += time.time() - st_chk1
                if grid.get_radius() + cutoff_radius < d_R_a_R_c:
                    continue

                ## calculate indices inside the cutoff radius
                r_R_a = points - R_a
                st_chk1 = time.time()
                # d_r_R_a = r_R_a.norm(dim=1)  # |r - R_a|, slower -> replaced to the line below
                d_r_R_a = r_R_a.pow(2).sum(dim=1).sqrt()  # |r - R_a|
                chk_time1 += time.time() - st_chk1
                _is_inside = d_r_R_a < cutoff_radius
                is_inside += _is_inside
                r_inside = r_R_a[_is_inside]
                d_inside = d_r_R_a[_is_inside]

                ## calculate beta(r) * Y_lm(r) * e^{ikr}
                st_interp = time.time()
                beta = torch.from_numpy(cubic_interp(d_inside.cpu())).to(self.device)
                interp_time += time.time() - st_interp

                st_ylm = time.time()
                for i_k, kpt in enumerate(kpts):
                    is_gamma = (kpt == 0).all()
                    y_lm = Y_lm_real_batch(r_inside, el)
                    if not is_gamma:
                        e_ikr = torch.exp(1j * (points[_is_inside] - N_c) @ kpt.to(dtype=torch.complex128))
                        projector[i_k][start_l:stop_l, _is_inside] += (
                            beta * e_ikr
                        ) * y_lm
                    else:
                        projector[i_k][start_l:stop_l, _is_inside] += beta * y_lm
                ylm_time += time.time() - st_ylm
            start_l = stop_l
        print(f"        L internal time: {time.time() - st} sec, interp_time: {interp_time} sec")
        print(f"               ylm_time: {ylm_time} sec")
        print(f"               chk_time1: {chk_time1} sec")

        ## only get non-zero values
        for i_k in range(len(kpts)):
            projector[i_k] = projector[i_k][:, is_inside]
        projector *= np.sqrt(grid.microvolume)

        return projector, is_inside

    @Timer.timeit
    def calc_KB_proj2(self, grid, upf, kpts):
        """Make list of KB-projectors and list of corresponding grid indices

        :type  grid: gospel.Grid
        :param grid:
            Grid class object
        :type  upf: gospel.Pseudopotential.UPF
        :param upf:
            UPF class object
        :type  kpts: list or np.ndarray
        :param kpts:
            list of k-points

        :rtype: (list[torch.Tensor], list[torch.Tensor])
        :return:
            list of projectors and corresponding indices
        """
        atoms = grid.atoms
        list_KB_proj_kpt = [[] for i_k in range(len(kpts))]
        list_isinside = []

        from gospel.ParallelHelper import ParallelHelper as PH
        from ase.data import chemical_symbols

        st = time.time()
        list_all_atomic_numbers = PH.split(torch.from_numpy(atoms.get_atomic_numbers()), return_all=True)
        num_atoms_per_rank = [len(atomic_numbers) for atomic_numbers in list_all_atomic_numbers]
        atomic_numbers = list_all_atomic_numbers[PH.rank]
        positions = PH.split(torch.from_numpy(atoms.get_positions()), dim=0).to(self.device)
        chktime1 = time.time() - st

        st = time.time()
        for i_atom, atomic_number in enumerate(atomic_numbers):
            sym = chemical_symbols[atomic_number]
            
            P, I = self.calc_KB_proj_per_atom2(
                grid, positions[i_atom], upf[sym], kpts
            )
            I = I.to(self.device)
            list_isinside.append(I)
            for i_k in range(len(kpts)):
                list_KB_proj_kpt[i_k].append(P[i_k].to(self.device))
        chktime2 = time.time() - st

        st = time.time()
        if self.device != torch.device('cpu') and PH.global_size>1:
            #assert as_sparse==False, "If use_cuda is True, use_dense_proj should be True"
            # list_atoms = PH.split(torch.from_numpy(atoms.get_atomic_numbers()), return_all=True)
            # list_atoms_size = [len(atoms) for atoms in atomic_numbers]
            proj_sizes = torch.tensor([ tuple(P.size()) for P in list_KB_proj_kpt[0]], device=self.device, dtype=torch.int64)
            total_proj_sizes = PH.merge(proj_sizes, dim=0, list_size=num_atoms_per_rank)
            assert len(total_proj_sizes)==len(atoms), f"KB_form.calc_KB_proj2 {len(total_proj_sizes)}=={len(atoms)}"
            all_list_KB_proj_kpt = []
    
            # broadcast list_KB_proj_kpt
            for i_k, KB_proj in enumerate(list_KB_proj_kpt):
                all_list_KB_proj_kpt.append( [ torch.empty(tuple(size), device=self.device, dtype=list_KB_proj_kpt[i_k][0].dtype) for size in total_proj_sizes ])
                idx=0
                for rank in range(PH.global_size):
                    for i_atom in range(num_atoms_per_rank[rank]):
                        if PH.rank==rank:
                            assert all_list_KB_proj_kpt[i_k][idx].size()==KB_proj[i_atom].size()
                            all_list_KB_proj_kpt[i_k][idx]=KB_proj[i_atom].contiguous()
                        all_list_KB_proj_kpt[i_k][idx] = PH.broadcast(all_list_KB_proj_kpt[i_k][idx], rank) 
                        #PH.broadcast(all_list_KB_proj_kpt[-1][idx], rank, async_op=True) 
                        idx+=1
    #            assert idx ==  sum([proj_size.size(0) for proj_size in total_proj_sizes ] ), f"{idx}!={sum([proj_size.size(0) for proj_size in total_proj_sizes ] )}\n{num_proj_per_rank}"
            # broadcast list_isinside
            idx=0
            all_list_isinside = torch.empty( (len(atoms), grid.ngpts ), device=self.device, dtype=torch.bool) 
            for rank in range(PH.global_size):
                if PH.rank==rank:
                    all_list_isinside[idx: (idx+ len(list_all_atomic_numbers[rank] ) ) ] =  torch.stack(list_isinside)
    
                all_list_isinside[idx: (idx+len(list_all_atomic_numbers[rank]) )  ] = PH.broadcast(all_list_isinside[idx: (idx+len(list_all_atomic_numbers[rank]) )  ], rank) 
                #PH.broadcast(all_list_isinside[idx: (idx+len(list_all_atomic_numbers[rank]) )  ], rank, async_op=True) 
                idx+=len(list_all_atomic_numbers[rank])

            
            list_KB_proj_kpt = all_list_KB_proj_kpt # i_k, i_atom, i_proj, nonzero
            list_isinside = [ I for I in  all_list_isinside ] # i_atom, ngpt
        chktime3 = time.time() - st
        print(f"Debug: * Part time of calc_KB_proj2: (1) {chktime1} sec, (2) {chktime2} sec, (3) {chktime3} sec")


        if self.__use_dense_proj==False:
            assert not self.__use_dense_proj
            ## build sparse matrices and change list to array
            nonzero_positions = [ is_inside.nonzero().squeeze(-1) for is_inside in list_isinside ] # nonzero_positions : int (i_atom, nonzero)
                                                                                                   # list_isinside: bool  (i_atom, ngpt)
            # build list_indices                                                                                                   
            list_indices=[]                                                                                                   
            dim0 = 0
            for i_atom, KB_proj in enumerate(list_KB_proj_kpt[0]):
                indices = nonzero_positions[i_atom].repeat(KB_proj.size(0) ) # repeat index as the number of proj
                indices = torch.stack([torch.arange(dim0, dim0+KB_proj.size(0), dtype=indices.dtype, device=indices.device).repeat(KB_proj.size(1),1).T.reshape(-1), indices], dim=0)
                list_indices.append(indices)
                dim0+=KB_proj.size(0)
            list_indices = torch.cat(list_indices, dim=-1)                

            # build list_values & construct sparse tensor 
            for i_k in range(len(kpts)):
                list_values = [ KB_proj.reshape(-1)   for KB_proj in list_KB_proj_kpt[i_k] ]
                list_KB_proj_kpt[i_k] = torch.sparse_coo_tensor( list_indices, torch.cat(list_values), (dim0, grid.ngpts ) ).coalesce()
                
            nonzero_ratio = (
                len(list_KB_proj_kpt[0].values()) / list_KB_proj_kpt[0].numel()
            )
            print(f"KB projector non-zero ratio: {nonzero_ratio * 100} %")
            return list_KB_proj_kpt
        return list_KB_proj_kpt, list_isinside

    def get_D_matrix(self, idx=None, as_sparse=False):
        """Get D_matrix for idx-th atom."""
        if idx is None:
            D = self.__D_matrix
        else:
            if idx >= len(self.__num_projs):
                raise IndexError
            tmp = slice(self.__num_projs[:idx].sum(), self.__num_projs[: idx + 1].sum())
            D = self.__D_matrix[tmp, tmp]
        if as_sparse:
            return D.to_sparse_csr().to(self.device)
        else:
            return D.to(self.device)

    def get_KB_proj(self, i_k=None, idx=None):
        """Get projector for idx-th atom or i_k-th k-point.

        :type  i_k: int or None, optional
        :param i_k:
            index of k-point
        :type  idx: int or None, optional
        :param idx:
            index of atom

        :rtype: np.ndarray or sparse.csr_matrix
        :return:
            projector for idx-th atom or i_k-th k-point.
        """
        if i_k is None and idx is None:
            return self.__KB_proj
        elif i_k is not None and idx is None:
            if i_k >= len(self.__KB_proj):
                raise IndexError
            return self.__KB_proj[i_k]
        elif i_k is None and idx is not None:
            raise NotImplementedError
        else:  # i_k is not None and idx is not None
            if 0 > i_k or i_k >= len(self.__KB_proj):
                raise IndexError
            if 0 > idx or idx >= len(self.__grid.atoms):
                raise IndexError
            if self.__use_dense_proj:
                KB_proj_ka = self.__KB_proj[i_k][idx]
            else:
                tmp = slice(
                    self.__num_projs[:idx].sum(), self.__num_projs[: idx + 1].sum()
                )
                if isinstance(self.__KB_proj[i_k], torch.Tensor):
                    KB_proj_ka = self.__KB_proj[i_k].to_dense()[tmp].to(self.__KB_proj[i_k].device)
                    #KB_proj_ka = self.__KB_proj[i_k].cpu().to_dense()[tmp].to(self.__KB_proj[i_k].device)

                    # this original code is no longer working
                    #KB_proj_ka = torch_to_scipy_sparse(self.__KB_proj[i_k])[tmp]
                    #KB_proj_ka = scipy_to_torch_sparse(KB_proj_ka).to(KB_proj_ka.device)
                else:
                    KB_proj_ka = self.__KB_proj[i_k][tmp]
            return KB_proj_ka

    def get_KB_proj_op(self, i_k, idx):
        """Return an operator of KB-projector of an atom.

        :type  i_k: int
        :param i_k:
            index of k-point
        :type  idx: int
        :param idx:
            index of atom
        """
        P = self.get_KB_proj(i_k, idx)
        nproj = self.__num_projs[idx]
        ngpts = self.__grid.ngpts
        shape = (nproj, ngpts)
        if self.__use_dense_proj:
            I = self.__isinside[idx]
            # op = lambda x: P @ x[I]
            op = lambda x: P.to(x.device) @ x[I.to(x.device)]  # data copy & matmul
        else:
            # op = lambda x: P @ x
            op = lambda x: P.to(x.device) @ x   # data copy & matmul
        op = LinearOperator(shape=shape, matvec=op, dtype=P.dtype)
        return op

    @property
    def use_dense_proj(self):
        return self.__use_dense_proj

    @property
    def KB_proj(self):
        """Return KB projectors"""
        return self.__KB_proj

    @property
    def KB_proj_H(self):
        """Return complex conjugate of KB projectors"""
        return self.__KB_proj_H

    @property
    def isinside(self):
        """Return array[bool] whether each grid is near each atom"""
        return self.__isinside

    # def calc_KB_proj_per_atom(self, grid, atom_position, upf, kpt, nz_tol=1e-5):
    #     """Calculate a KB projector matrix for an atom.

    #     projector = < beta * Y_{lm} | \hat{r} >

    #                         N
    #                    ___________
    #               =   |           |
    #                   |    bY     | (l/m)
    #                   |___________|

    #              , where N is the number of grid points, and (l/m) is the combination indices of l and m.

    #     For k-points,

    #     projector = < beta * Y_{lm} | e^{i\vec{k} \dot \vec{r}} \hat{r} >

    #                              N                                               N
    #                         ___________            N                        ___________
    #                        |           |     ____________                  |           |
    #               =  (l/m) |    bY     |  * |____e_ikr___| 1    =    (l/m) | bY * e_ikr|
    #                        |___________|                                   |___________|

    #     """
    #     ## Under PBC, the number of near cells to consider potentials coming in.
    #     atoms = grid.atoms
    #     num_near = [self.__num_near_cell if grid.get_pbc()[i] else 0 for i in range(3)]
    #     is_gamma = np.all(np.array(kpt) == 0)
    #     dtype = float if is_gamma else complex

    #     ## Check whether 'num_near_cell' is enough or not.
    #     n1, n2, n3 = (num_near + np.ones(3, dtype=int)) * grid.get_pbc()
    #     for i_proj in range(upf.num_proj):
    #         cutoff_radius = upf.beta_cutoff[i_proj]
    #         for i_xyz in [
    #             (n1, 0, 0),
    #             (-n1, 0, 0),
    #             (0, n2, 0),
    #             (0, -n2, 0),
    #             (0, 0, n3),
    #             (0, 0, -n3),
    #         ]:
    #             if i_xyz == (0, 0, 0):
    #                 continue
    #             R_a = atom_position + i_xyz @ atoms.get_cell()  # atom position vector
    #             d_r_R_a = np.linalg.norm(grid.points - R_a, axis=1)  # |r - R_a|
    #             if True in (d_r_R_a < cutoff_radius):
    #                 print(
    #                     f"WARNING!! method<KBproj.calc_KB_proj_per_atom()>: 'num_near_cell'(={self.__num_near_cell}) is not sufficent to interpolate the effects of nearing cells."
    #                 )

    #     l_m_combined_size = sum([2 * el + 1 for el in upf.angmom])
    #     projector = sparse.lil_matrix((l_m_combined_size, grid.ngpts), dtype=dtype)

    #     ## Represent projectors on grid using interpoation.
    #     el_m_idx = 0  # enumerate of double for loop
    #     for i_proj in range(upf.num_proj):
    #         el = upf.angmom[i_proj]
    #         cutoff_radius = upf.beta_cutoff[i_proj]
    #         cubic_interp = interp1d(
    #             upf.R,
    #             upf.beta[i_proj],
    #             kind="cubic",
    #             bounds_error=False,
    #             fill_value=0.0,
    #         )
    #         for m in range(-el, el + 1):
    #             projector_tmp = np.zeros(grid.ngpts, dtype=dtype)
    #             for i_xyz in product(
    #                 range(-num_near[0], num_near[0] + 1),
    #                 range(-num_near[1], num_near[1] + 1),
    #                 range(-num_near[2], num_near[2] + 1),
    #             ):
    #                 N_c = i_xyz @ atoms.get_cell()  # cell vector
    #                 R_a = np.asarray(atom_position) + N_c
    #                 d_R_a_R_c = np.linalg.norm(R_a - grid.center_point)  # |R_a - R_c|
    #                 if grid.get_radius() + cutoff_radius < d_R_a_R_c:
    #                     continue

    #                 ## calculate indices inside the cutoff radius
    #                 r_R_a = grid.points - R_a
    #                 d_r_R_a = np.linalg.norm(r_R_a, axis=1)  # |r - R_a|
    #                 idx_in = np.where(d_r_R_a < cutoff_radius)[0]

    #                 ## calculate beta(r) * Y_lm(r) * e^{ikr}
    #                 beta_on_grid = np.zeros(grid.ngpts)
    #                 beta_on_grid[idx_in] = cubic_interp(d_r_R_a[idx_in])
    #                 y_lm = np.zeros(grid.ngpts, dtype=float)
    #                 y_lm[idx_in] = Y_lm(r_R_a[idx_in], el, m)
    #                 if is_gamma:
    #                     projector_tmp[idx_in] += beta_on_grid[idx_in] * y_lm[idx_in]
    #                 else:
    #                     e_ikr = np.exp(1j * (grid.points[idx_in] - N_c) @ kpt)
    #                     projector_tmp[idx_in] += (
    #                         beta_on_grid[idx_in] * y_lm[idx_in] * e_ikr
    #                     )
    #             projector_tmp *= np.sqrt(grid.microvolume)

    #             ## cut out non-zero values smaller that nz_tol
    #             nz_indices = np.abs(projector_tmp) > nz_tol
    #             projector[el_m_idx, nz_indices] = projector_tmp[nz_indices]
    #             el_m_idx += 1
    #     return projector

    # def calc_KB_proj(self, grid, upf, kpts):
    #     """Make list of projectors for each k-point.

    #     Args)
    #     grid (gospel.Grid)
    #     upf (ase.Pseudopotential.UPF)
    #     kpts (np.ndarray)

    #     Return (list of sparese matrices): list of KB projectors of each k-point.
    #     """
    #     atoms = grid.atoms
    #     list_KB_proj_kpt = []
    #     for kpt in kpts:
    #         KB_proj_atom = []
    #         for i_atom, i_sym in enumerate(atoms.get_chemical_symbols()):
    #             if upf[i_sym].num_proj == 0:
    #                 continue
    #             KB_proj_atom.append(
    #                 self.calc_KB_proj_per_atom(
    #                     grid, atoms.get_positions()[i_atom], upf[i_sym], kpt
    #                 )
    #             )
    #         if KB_proj_atom:
    #             KB_proj_atom = sparse.vstack(KB_proj_atom).tocsr()
    #             non_zero_ratio = KB_proj_atom.nnz / np.prod(KB_proj_atom.shape) * 100
    #             print(f"KB projector non-zero ratio : {non_zero_ratio} %")
    #         list_KB_proj_kpt.append(KB_proj_atom)
    #     return list_KB_proj_kpt


def clustering(atoms, cutoff_radius, choice_method="max_random", seed=0):
    """Cluster non-overlapping atoms in a group. This method has randomness.

    :type  atoms: ase.Atoms
    :param atoms:
        ase Atoms object containing position information
    :type  cutoff_radius: dict
    :param cutoff_radius:
        cutoff radius of each atom
    :type  choice_method: str, optional
    :param choice_method:
        method of how to pick the reference atoms, defaults to 'max_random'
    :type  seed: int, optional
    :param seed:
        numpy random seed, defaults to None

    :rtype: list[torch.Tensor]
    :return:
        list of each group of atom indices
    """
    assert choice_method in ["random", "max_first", "max_random"]
    np.random.seed(seed)
    positions = torch.from_numpy(atoms.get_positions())
    dist_mat = torch.cdist(positions, positions)

    cutoff_mat = torch.zeros_like(dist_mat)
    print(f"Debug: cutoff_radius={cutoff_radius}")
    cutoff_radius = [max(cutoff_radius[sym]) for sym in atoms.get_chemical_symbols()]
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i == j:
                cutoff_mat[i, j] = 0.0
            else:
                cutoff_mat[i, j] = cutoff_radius[i] + cutoff_radius[j]
    print(f"Debug: cutoff_mat = {cutoff_mat}")
    outer_mat = dist_mat > cutoff_mat

    groups = []
    pick_list2 = list(np.arange(len(atoms)))
    while True:
        if len(pick_list2) == 0:
            break
        group = []
        pick_list = torch.tensor(pick_list2)
        while True:
            if len(pick_list) == 0:
                break

            if choice_method == "random":
                pick = int(np.random.choice(pick_list, 1))
            else:
                tmp = outer_mat[pick_list][:, pick_list].sum(0)
                tmp2 = torch.where(tmp == tmp.max())[0]  # can be multiple candidates
                if choice_method == "max_first":
                    # just pick first element from maximum True list
                    pick = pick_list[tmp2[0].item()]
                else:  # "max_random"
                    # randomly pick from maximum True list
                    pick = pick_list[int(np.random.choice(tmp2, 1))]

            group.append(pick)
            is_outer = outer_mat[pick][pick_list]
            pick_list = pick_list[is_outer]

        group.sort()
        group = torch.tensor(group)
        groups.append(group)

        for i in group:
            pick_list2.remove(i.item())
    return groups


if __name__ == "__main__":
    from ase import Atoms
    from ase.units import Bohr
    from gospel.Grid import Grid
    from gospel.Pseudopotential.UPF import UPF

    print("===============================================================")
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.set_positions(atoms.get_positions() / Bohr)
    atoms.center(vacuum=4.0)
    grid = Grid(atoms, spacing=0.2)
    pp_files = [
        "/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/H_ONCV_PBE-1.2.upf",
        "/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/O_ONCV_PBE-1.2.upf",
    ]
    upf = UPF(pp_files)
    kbproj = KB_form(grid, upf, use_dense_proj=False)
    print("===============================================================")

    print("===============================================================")
    print("Li atom")
    cell = [10, 10, 10]
    gpts = [50, 50, 50]
    print("gpts = ", gpts)
    print("cell = ", cell)
    print("spacing = ", np.array(cell) / np.array(gpts), " (ang)")
    atoms = Atoms("Li", cell=cell)
    # atoms = Atoms("Li2", cell=cell)
    grid = Grid(atoms, gpts)
    atoms.set_positions([np.array(cell) / 2])  # Centered on cell
    # atoms.set_positions([[0,0,0],[0,0,1]])  # Centered on cell
    print("atoms position : ", atoms.get_positions())
    pp_files = ["/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/Li_ONCV_PBE-1.2.upf"]
    upf = UPF(pp_files)
    # kpts = [[0,0,0]]
    scaled_kpt = np.array([[0.5, 0, 0], [0.25, 0, 0]])
    unscaled_kpt = (2 * np.pi) * scaled_kpt @ np.linalg.inv(atoms.get_cell())
    kpts = unscaled_kpt
    kbproj = KB_form(grid, upf, kpts=kpts)
    print("============================ End ==============================")
