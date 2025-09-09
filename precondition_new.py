from collections import OrderedDict
import torch
from typing import Union, Optional
import numpy as np

from gospel.LinearOperator import LinearOperator
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.Poisson.ISF_solver import ISF_solver
from gospel.util import to_cuda, Timer
from gospel.precision import to_DP, to_SP, to_HP, to_BF16

# TODO: Remove MergePreconditioner
# TODO: Remove useless code comments
# TODO: Replace use_cuda argument with device argument for consistency

def create_preconditioner(precond_type=None, grid=None, use_cuda=False, options={}):
    """Create Preconditioner object from input parameters.

    :type  precond_type: str or None, optional
    :param precond_type:
        type of preconditioner, available type=[None, "jacobi", "filter", "poisson", "gapp", "shift-and-invert"]
    :type  grid: gospel.Grid or None, optional
    :param grid:
        Grid object

    *Example*
    >>> grid = Grid(atoms, gpts=(40, 40, 40))
    >>> precond = create_preconditioner("poisson", grid)
    """
    # if precond_type == "jacobi":
    #     return PreJacobi()
    if isinstance(precond_type, list):
        assert all(
            [isinstance(item, tuple) for item in precond_type]
        ), "If the type of precond_type is list, its item should be tuple "
        return MergePreconditioner(precond_type, grid, use_cuda)
    elif isinstance(precond_type, OrderedDict):
        precond_type = [ (key,*value) for key, value in precond_type.items() ]
        assert all(
            [isinstance(item, tuple) for item in precond_type]
        ), "If the type of precond_type is OrderedDict, its value should be tuple "
        return MergePreconditioner(precond_type, grid, use_cuda)

    if precond_type == "poisson":
        return PrePoisson(grid, use_cuda=use_cuda, **options)
    elif precond_type == "gapp":
        ## Reference DOI: 10.1021/acs.jpca.2c09111
        if grid.get_pbc().sum() == 0:
            t_sample = [[1.38], [2.55], 11.96]
        elif grid.get_pbc().sum() == 1:
            t_sample = [[1.31], [2.3], 6.38]
        elif grid.get_pbc().sum() == 2:
            t_sample = [[1.48], [2.55], 8.39]
        else:
            # 3D-PBC
            t_sample = [[1.63], [2.73], 7.89]
        return PrePoisson(grid, t_sample=t_sample, use_cuda=use_cuda, **options)
    elif precond_type == "filter":
        return PreFilter(grid, use_cuda=use_cuda)
    elif precond_type == "shift-and-invert":
        return PreShiftAndInvert(grid, use_cuda=use_cuda, **options)
    elif precond_type == "neumann":
        return PreNeumann(grid, use_cuda=use_cuda, **options)
    elif precond_type is None:
        return Preconditioner(None, use_cuda)
    else:
        raise NotImplementedError(f"{precond_type} is not available type.")


class Preconditioner:
    """parent class of preconditioners to accelerate convergence of iterative diagonalization.

    :type  precond_type: str or None, optional
    :param precond_type:
        available precondition types, defaults to None
        available precond_type
        - 'jacobi': inverse of diagonal elements of Hamiltonian matrix
        - 'poisson': solution of poisson equation of guess eigenvectors
        - 'filter': filtering using neighboring points
        - 'shift-and-invert': shift-and-invert preconditioning
    :type  use_cuda: bool, optional
    :param use_cuda:
        whether to use cuda, defaults to False
    """

    def __init__(self, precond_type, use_cuda=False, fp="DP"):
        assert precond_type in [
            "jacobi",
            "poisson",
            "filter",
            "GKP",
            "shift-and-invert",
            "merge",
            "neumann",
            None,
        ], f"{precond_type} is not supported precondition type."
        self._precond_type = precond_type
        self._device = PH.get_device()
        self._use_cuda = use_cuda
        self._fp = fp
        # TODO: 'num_called' is deprecated. (remove the related codes)
        self.num_called = 0  # this count number of calls in one scf step
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : {self._precond_type}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        # no preconditioning
        return residue

    def __call__(self, residue, H=None, eigval=None, i_iter=None):
        # FP type conversion
        fp_type = residue.dtype
        _dtype = {"DP": to_DP, "SP": to_SP, "HP": to_HP, "BF16": to_BF16}[self._fp](
            fp_type
        )
        residue = residue.to(_dtype)
        if eigval is not None:
            eigval = eigval.to(_dtype)

        # preconditioning
        # if self._precond_type == "shift-and-invert":
        if self._precond_type in ["shift-and-invert", "neumann"]:
            residue = self.call(residue, H, eigval)
        elif self._precond_type == "merge":
            residue = self.call(residue, H, eigval, i_iter)
        else:
            residue = self.call(residue)
        self.num_called += 1
        return residue.to(fp_type)

    def reset_num_called(self):
        self.num_called = 0


class MergePreconditioner(Preconditioner):
    def __init__(self, list_precond_types, grid, use_cuda=False):
        super().__init__("merge", use_cuda)
        self.preconditioners = []
        self.iterations = [[], []]
        # for precond_type, num_iter in list_precond_types:
        #     self.preconditioners.append( create_preconditioner(precond_type, grid, use_cuda) )
        #     self.iterations.append(num_iter)
        for precond_type, options, num_iter in list_precond_types:
            self.preconditioners.append(
                create_preconditioner(precond_type, grid, use_cuda, options)
            )

            assert (
                type(num_iter) == int or type(num_iter) == list
            ), "maxiter type should be int or list of int"
            if type(num_iter) == list:
                assert (
                    len(num_iter) == 2
                ), "if list is given for maxiter, its length should be 2"
                self.iterations[0].append(num_iter[0])
                self.iterations[1].append(num_iter[1])
            else:
                self.iterations[0].append(num_iter)
                self.iterations[1].append(num_iter)

        self.iterations = np.cumsum(np.array(self.iterations), axis=-1)

    def call(self, residue, H, eigval, i_scf):
        iterations = self.iterations[1] if i_scf != 0 else self.iterations[0]
        try:
            idx = iterations.tolist().index(
                min(iterations[(iterations - self.num_called) > 0])
            )
        except ValueError:
            idx = -1
            # return residue

        return self.preconditioners[idx](residue, H, eigval, i_scf)


class GKP(Preconditioner):
    """Gaussian Kernel Preconditioner"""

    def __init__(self, grid, t_sample1, t_sample2, nextra, use_cuda=False):
        super().__init__("GKP", use_cuda)
        self.__grid = grid
        self.__solver1 = ISF_solver(grid, t_sample1, fp="DP", device=self._device)
        self.__solver2 = ISF_solver(grid, t_sample2, fp="DP", device=self._device)
        self.__nextra = nextra  # the number of extra states
        self.linear_op = None
        return

    def call(self, residue):
        if self.linear_op is None:
            ps1 = self.__solver1
            ps2 = self.__solver2

            def f(x):
                x = x.T  # shape=(nbands, ngpts)
                x1 = x[: -self.__nextra]
                x2 = x[-self.__nextra :]
                retval = torch.zeros_like(x)
                if x.dtype in [torch.complex64, torch.complex128]:
                    retval[: -self.__nextra] = (
                        ps1.compute_potential(x1.real)
                        + ps1.compute_potential(x1.imag) * 1j
                    )
                    retval[-self.__nextra :] = (
                        ps2.compute_potential(x2.real)
                        + ps2.compute_potential(x2.imag) * 1j
                    )
                elif x.dtype in [torch.float32, torch.float64]:
                    # retval = ps.compute_potential(x)
                    retval[: -self.__nextra] = ps1.compute_potential(x1)
                    retval[-self.__nextra :] = ps2.compute_potential(x2)
                else:
                    raise TypeError(f"x.dtype={x.dtype} is inappropriate.")
                return 2.0 * retval.T

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return self.linear_op @ residue


class PrePoisson(Preconditioner):
    """Poisson Preconditioner

    :type  grid: gospel.Grid or None, optional
    :param grid:
        Grid class object, defaults to None
    :type  t_sample: list or None
    :param t_sample:
        list of t values and weights and t_delta
    :type  use_cuda: bool
    :param use_cuda:
        using cuda device
    :type  fp: str
    :param fp:
        floating-point precision, choices=['DP', 'SP', 'MP'])
    """

    def __init__(self, grid, t_sample=None, use_cuda=False, fp="DP"):
        super().__init__("poisson", use_cuda, fp)
        self.__grid = grid
        self.solver = ISF_solver(
            grid, t_sample, fp=fp, device=self._device, use_MP=(not fp == "DP")
        )
        self.linear_op = None
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : poisson"
        s += f"\n* fp    : {self._fp}"
        s += self.solver.__str__()
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        """Poisson filter preconditioning.

        :rtype: LinearOperator
        :return:
            precondition operator
        """
        if self.linear_op is None:
            # NOTE: batch_compute_potential2 is fastest on GPU.
            # kernel = self.solver.compute_potential
            # kernel = self.solver.batch_compute_potential
            kernel = self.solver.batch_compute_potential2

            def f(x):
                # x = x.T  # x.T.shape=(nbands, ngpts)
                # x = x.T.contiguous()  # x.T.shape=(nbands, ngpts)
                if x.is_complex():
                    retval = kernel(x.real) + kernel(x.imag) * 1j
                else:
                    retval = kernel(x)
                # return 2.0 * retval.T
                # return (2.0 * retval.T).contiguous()
                return (2.0 * retval).contiguous()  # TODO: Fix the coefficient (/2pi?)

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return self.linear_op @ residue


# class PreJacobi(Preconditioner):
#    """Jacobi Preconditioner"""
#
#    def __init__(self):
#        super().__init__("jacobi")
#        self.linear_op = None
#        return
#
#    def call(self, residue, H):
#        """Jacobi preconditioning. Inverse of diagonal elements
#
#        :type  H: torch.Tensor or LinearOperator
#        :param H:
#            Hamiltonian operator
#
#        :rtype: LinearOperator or None
#        :return:
#            precondition operator
#        """
#        if hasattr(H, "diagonal"):
#            f = lambda x: (1.0 / H.diagonal()) * x
#            return LinearOperator(H.shape, f, dtype=H.dtype)
#        else:
#            print(
#                "Warning: Preconditioner is not used. ('diagonal' is not defined in H.)"
#            )
#            return None


class PreFilter(Preconditioner):
    """Low-pass filter preconditioner"""

    def __init__(self, grid, alpha=0.5, use_cuda=False):
        super().__init__("filter", use_cuda)
        self.__grid = grid
        self.__alpha = alpha
        self.__filter = self.make_filter(grid)
        self.__kernel = "sparse"  # type of kernel, options=["sparse", "conv"]
        # convolution version (conv) will also be implemented.
        self.__filter = to_cuda(self.__filter, self._device)
        self.linear_op = None
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type  : filter"
        s += f"\n* alpha : {self.__alpha}"
        s += f"\n* kernel: {self.__kernel}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue):
        """Low-pass filter preconditioning.

        :rtype: LinearOperator
        :return:
            precondition operator
        """
        if self.linear_op is None:

            def f(x):
                if x.dtype in [torch.complex64, torch.complex128]:
                    retval = self.__filter @ x.real + self.__filter @ x.imag * 1j
                elif x.dtype in [torch.float32, torch.float64]:
                    retval = self.__filter @ x
                else:
                    raise TypeError(f"x.dtype={x.dtype} is inappropriate.")
                return retval

            shape = (self.__grid.ngpts, self.__grid.ngpts)
            self.linear_op = LinearOperator(shape, f)
        return self.linear_op @ residue

    def make_filter(self, grid):
        from scipy.sparse import identity, kron
        from gospel.util import scipy_to_torch_sparse

        fx = self.make_filter_axis(grid, 0)
        fy = self.make_filter_axis(grid, 1)
        fz = self.make_filter_axis(grid, 2)
        Ix = identity(grid.gpts[0])
        Iy = identity(grid.gpts[1])
        Iz = identity(grid.gpts[2])
        flter = kron(fx, kron(Iy, Iz)) + kron(Ix, kron(fy, Iz)) + kron(kron(Ix, Iy), fz)
        return scipy_to_torch_sparse(flter)

    def make_filter_axis(self, grid, axis):
        from scipy.sparse import diags
        import numpy as np

        pbc = grid.get_pbc()[axis]
        gpt = grid.gpts[axis]
        mat = diags(np.ones(gpt - 1), 1)
        if pbc:
            mat += diags(np.ones(1), gpt - 1)
        mat += mat.T
        mat *= (1 - self.__alpha) / 6
        mat += diags(np.ones(gpt) * self.__alpha / 3, 0)
        return mat


class PreShiftAndInvert(Preconditioner):
    def __init__(
        self,
        grid,
        use_cuda=False,
        rtol=0.25,
        max_iter=5,
        correction_scale=0.1,
        no_shift_thr=10.0,
        # inner_precond=None,
        # inner_precond="Neumann",############added Neumann inner precond
        fp="DP",
        verbosityLevel=0,
        locking=True,
        # order="None" ##added
        inner_precond="gapp",
        options={},  # for inner preconditioner
    ):
        super().__init__("shift-and-invert", use_cuda, fp)
        # self.prev_solution = None
        self.rtol = rtol
        self.max_iter = max_iter
        self.correction_scale = correction_scale
        self.no_shift_thr = no_shift_thr
        self.inner_precond = inner_precond
        self.verbosityLevel = verbosityLevel
        self.locking = locking

        # self.fp = fp  # TODO: Implementation
        # self.order = order #added
        
        print("inner_precond = ", inner_precond, "pcg = ", max_iter ) ## added
        # self.__precond_for_pcg = create_preconditioner(inner_precond, grid, use_cuda)
        # NOTE: Here, add 'fp' to create preconditioner (jeheon)

        #self.__precond_for_pcg = create_preconditioner(
        #        inner_precond, grid, use_cuda, options={"fp": fp},  original
        #)
 
        self.inner_precond = inner_precond
        assert inner_precond in ["gapp","poisson","neumann", None], (
            "inner_precond should be gapp, poisson, neumann, or None"
        )
        self.__precond_for_pcg = create_preconditioner(inner_precond, grid, use_cuda, options={"fp": fp, **options})
        # if inner_precond == "Neumann":
        #     self.__precond_for_pcg = create_preconditioner(
        #         inner_precond, grid, use_cuda, options={"fp": fp,
        #                                                 "order" : order, # added
        #                                                 },
        #     )
        # else:
        #     self.__precond_for_pcg = create_preconditioner(
        #         inner_precond, grid, use_cuda, options={"fp": fp,
        #                                                 },
        #     )
        return

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type             : {self._precond_type}"
        s += f"\n* rtol             : {self.rtol}"
        s += f"\n* max_iter         : {self.max_iter}"
        s += f"\n* correction_scale : {self.correction_scale}"
        s += f"\n* no_shift_thr     : {self.no_shift_thr}"
        s += f"\n* fp               : {self._fp}"
        s += f"\n* verbosityLevel   : {self.verbosityLevel}"
        s += f"\n* locking          : {self.locking}"
        s += f"\n* inner_precond    : {self.inner_precond}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(
        self,
        residue: torch.Tensor,
        H: Union[torch.Tensor, LinearOperator],
        eigval: torch.Tensor,
    ) -> torch.Tensor:
        residue = pcg_solve(
            H,
            eigval,
            residue,
            # x0= (None if self.prev_solution is None else self.prev_solution),
            x0=None,
            preconditioner=self.__precond_for_pcg,
            rtol=self.rtol,
            max_iter=self.max_iter,
            correction_scale=self.correction_scale,
            no_shift_thr=self.no_shift_thr,
            verbosityLevel=self.verbosityLevel,
            locking=self.locking,
        )
        return residue


# TODO: make it deprecated
def blockPrint(func):
    import sys, os

    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, "w")
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result

    return wrapper


def pcg_solve(
    H: Union[torch.Tensor, LinearOperator],
    eigval: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    preconditioner: Optional[Union[torch.Tensor, LinearOperator]] = None,
    rtol: float = 0.25,
    max_iter: int = 300,
    correction_scale: float = 0.1,
    no_shift_thr: float = 10.0,
    verbosityLevel: int = 0,
    timing: bool = True,
    locking: bool = True,
):
    """
    Solve the linear equation, (H - eigval) x = b.

    Parameters
    ----------
    H : torch.Tensor or LinearOperator
        The Hamiltonian operator, usually a sparse matrix.
    eigval : torch.Tensor
        Eigenvalues. Shape: (nbands,).
    b : torch.Tensor
        Residual vectors. Shape: (ngpts, nbands).
    x0 : torch.Tensor, optional
        Initial solution vectors. Shape: (ngpts, nbands).
    preconditioner : torch.Tensor or LinearOperator, optional
        Preconditioner.
    rtol : float, optional
        Relative tolerance for convergence (tol = rtol * norm of b). Default is 0.25.
    max_iter : int, optional
        Maximum number of iterations. Default is 300.
    correction_scale : float, optional
        Scale of perturbation on eigenvalues (for numerical stability). Default is 0.1.
    no_shift_thr : float, optional
        Threshold for not shifting to states with large residues. Default is 10.0.
    verbosityLevel : int, optional
        Controls the verbosity of intermediate solver output. Default is 0.
    timing : bool, optional
        Whether to measure timing using Timer.track.
    locking: bool, optional
        If True, use advanced indexing (masking) to update only unconverged columns.
        If False, update full tensors using whole-tensor operations with multiplicative masks.

    Returns
    -------
    torch.Tensor
        The solution vector with shape matching that of x0.
    """
    device = PH.get_device()
    # if preconditioner is not None:
    #     preconditioner = blockPrint(preconditioner)

    # Initialization
    if x0 is None:
        x0 = b.clone()
        if verbosityLevel > 0:
            print("_pcg_sparse_solve: x0 is initialized to b.")

    # Set shift value
    # TODO: make perturb optional
    with Timer.track("ISI. set shift value", timing, False):
        b_norm = b.norm(dim=0, keepdim=True)  # shape: (1, nbands)
        perturb = -(b_norm.conj() * b_norm)
        eigval = eigval + correction_scale * perturb
        # no shift to states with large residues
        eigval[abs(perturb) > no_shift_thr] = 0.0

    # preconditioning of x0 (when x0=b)
    with Timer.track("ISI. precond", timing, False):
        x0[:, :] = preconditioner(x0, H, eigval)  # memory issue

    # Compute residual and initialize search direction
    with Timer.track("ISI. (H - e)x", timing, False):
        r = b - (H @ x0 - eigval * x0)

    if verbosityLevel > 0:
        cumBlockSize = b.size(1)  # count the number of H operations
        print(f"(cg iter=1): res norm={r.norm(dim=0)}")

    # Initialize search direction
    if preconditioner is None:
        p = r.clone()
        z = r  # shallow copy
    else:
        with Timer.track("ISI. precond", timing, False):
            z = preconditioner(r, H, eigval)
        p = z.clone()

    with Timer.track("ISI. r.T @ z", timing, False):
        rzold = torch.sum(r.conj() * z, dim=0, keepdim=True)

    # Initialize convergence mask for each column
    is_convg = torch.zeros(b.size(1), dtype=torch.bool, device=device)

    for i in range(2, max_iter + 1):
        # Determine the active (non-converged) column indices
        if locking:
            active = Ellipsis if (~is_convg).all() else torch.where(~is_convg)[0]
        else:
            active = Ellipsis

        # Compute step size
        with Timer.track("ISI. (H - e)x", timing, False):
            Ap = H @ p[:, active] - eigval[:, active] * p[:, active]
        with Timer.track("ISI. calc alpha", timing, False):
            denom = torch.sum(p[:, active].conj() * Ap, dim=0, keepdim=True)
            alpha = rzold[:, active] / denom

        if verbosityLevel > 0:
            cumBlockSize += p.shape[1]

        # Update solution and residual
        x0[:, active] += alpha * p[:, active]
        r[:, active] -= alpha * Ap
        del Ap  # for memory efficiency

        if preconditioner is not None:
            with Timer.track("ISI. precond", timing, False):
                z[:, active] = preconditioner(r[:, active], H, eigval[:, active])

        with Timer.track("ISI. r.T @ z", timing, False):
            rznew = torch.sum(r[:, active].conj() * z[:, active], dim=0, keepdim=True)

        # Check convergence
        with Timer.track("ISI. norm", timing, False):
            if preconditioner is None:
                r_norm = rznew.real.sqrt()  # because r and z are equal.
            else:
                r_norm = r[:, active].norm(dim=0)

        active_convg = r_norm < (rtol * b_norm[0, active])
        is_convg[active] = active_convg

        # # NOTE: 대각화 수렴이 진행될수록, b_norm이 매우 작아져, r_norm이 거의 수렴되지 않음을 새롭게 발견.
        # print(f"Debug: r_norm={r_norm}")
        # print(f"Debug: b_norm={b_norm}")
        # print(f"Debug: active_convg={active_convg}")
        # print(f"Debug: is_convg={is_convg}")

        if active_convg.all():
            if verbosityLevel > 0:
                rzold[:, active] = rznew
            break

        # Update search direction
        with Timer.track("ISI. update search direction", timing, False):
            beta = rznew / rzold[:, active]
            # TODO: p는 not converged에 대한 값은 저장하고 있을 필요가 없음. 더 효율적으로 구현 가능할 듯.
            p[:, active] = z[:, active] + beta * p[:, active]
            rzold[:, active] = rznew

        if verbosityLevel > 0:
            remaining = (~is_convg).sum().item()
            print(f"(cg iter={i}): {remaining} remaining res norm={r_norm}")

    if verbosityLevel > 0:
        if preconditioner is None:
            full_r_norm = rzold.real.sqrt()
        else:
            full_r_norm = r.norm(dim=0)

        if is_convg.all():
            print(
                f"* PCG converged!\n"
                f"  - final iteration: {i}\n"
                f"  - res norm: {full_r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
        else:
            print(
                f"* PCG not converged.\n"
                f"  - final iteration: {i}\n"
                f"  - is_convg: {is_convg}\n"
                f"  - # of remaining: {(~is_convg).sum().item()}\n"
                f"  - res norm: {full_r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
    return -x0


# TODO: remove this function (deprecated)
def _pcg_sparse_solve(
    H,
    eigval,
    b,
    x0=None,
    preconditioner=None,
    rtol=0.25,
    max_iter=300,
    correction_scale=0.1,
    no_shift_thr=10.0,
    verbosityLevel=0,
):
    """
    Solve the linear equation, (H - eigval) x = b.

    Parameters
    ----------
    :type H: torch.Tensor or LinearOperator
    :param H:
        The Hamiltonian opertor, usually a sparse matrix.
    :type eigval: torch.Tensor
    :param eigval:
        Eigenvalues. shape=(nbands,)
    :type b: torch.Tensor
    :param b:
        residual vectors. shape=(ngpts, nbands)
    :type x0: torch.Tensor, optional
    :param x0:
        initial for solution vectors with shape=(ngpts, nbands).
    :type preconditioner: torch.Tensor or LinearOperator, optional
    :param preconditioner:
        Preconditioner
    :type rtol: float, optional
    :param rtol:
        relative tolerance for convergence (tol=rtol * norm of b). The default is "rtol=1e-1".
    :type max_iter: int, optional
    :param max_iter:
        the number of maximum iteration. The default is "max_iter=300".
    :type correction_scale: float, optional
    :param correction_scale:
        the scale of perturbation on eigenvalues (for numerical stability). The default is "correction_scale=1.0".
    :type verbosityLevel: int, optional
    :param verbosityLevel:
        print intermediate solver output.  The default is "verbosityLevel=0".

    :rtype: torch.Tensor
    :return:
        solution vectors. shape=(ngpts, nbands)
    """
    ## Initialization
    if x0 is None:
        x0 = b.clone()
        if verbosityLevel > 0:
            print("_pcg_sparse_solve: x0 is initialized to b.")

    cumBlockSize = 0  # count the number of H operations
    b_norm = b.norm(dim=0, keepdim=True)

    perturb = -(b_norm.conj() * b_norm)
    eigval = eigval + correction_scale * perturb
    eigval[abs(perturb) > no_shift_thr] = 0.0  # no shift to states with large residues

    if preconditioner is not None:
        preconditioner = blockPrint(preconditioner)

    x0[:, :] = preconditioner(x0)  # memory issue

    ## Compute residual and initialize search direction
    r = b - (H @ x0 - eigval * x0)
    cumBlockSize += b.size(1)
    if verbosityLevel > 0:
        print(f"(cg iter=1): res norm={r.norm(dim=0)}")
    if preconditioner is None:
        p = r.clone()
        z = r  # shallow copy
    else:
        z = preconditioner(r)
        p = z.clone()

    rzold = torch.sum(r.conj() * z, dim=0, keepdim=True)

    is_convg = torch.zeros(b.size(1), dtype=torch.bool, device=b.device)
    for i in range(2, max_iter + 1):
        ## Compute step size
        Ap = H @ p[:, ~is_convg] - eigval[:, ~is_convg] * p[:, ~is_convg]
        alpha = rzold[:, ~is_convg] / torch.sum(
            p[:, ~is_convg].conj() * Ap, dim=0, keepdim=True
        )
        cumBlockSize += (~is_convg).sum()

        ## Update solution and residual
        x0[:, ~is_convg] = x0[:, ~is_convg] + alpha * p[:, ~is_convg]
        r[:, ~is_convg] = r[:, ~is_convg] - alpha * Ap
        del Ap  # for memory efficiency
        if preconditioner is not None:
            z[:, ~is_convg] = preconditioner(r[:, ~is_convg])
        rznew = torch.sum(r[:, ~is_convg].conj() * z[:, ~is_convg], dim=0, keepdim=True)

        ## Check convergence
        r_norm = (
            rznew.real.sqrt() if preconditioner is None else r[:, ~is_convg].norm(dim=0)
        )
        is_convg2 = r_norm < rtol * b_norm[0][~is_convg]
        if torch.all(is_convg2):
            is_convg[~is_convg] = is_convg2
            break
        if verbosityLevel > 0:
            print(
                f"(cg iter={i}): {len(is_convg) - is_convg.sum()} remaining res norm={r_norm}"
            )

        ## Update search direction
        beta = rznew / rzold[:, ~is_convg]
        p[:, ~is_convg] = z[:, ~is_convg] + beta * p[:, ~is_convg]
        rzold[:, ~is_convg] = rznew
        is_convg[~is_convg] = is_convg2

    if verbosityLevel > 0:
        r_norm = rzold.real.sqrt() if preconditioner is None else r.norm(dim=0)
        if torch.all(is_convg):
            print(
                f"* PCG converged!\n"
                f"  - final iteration: {i}\n"
                f"  - res norm: {r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
        else:
            print(
                f"* PCG not converged.\n"
                f"  - final iteration: {i}\n"
                f"  - is_convg: {is_convg}\n"
                f"  - # of remaining: {len(is_convg) - is_convg.sum()}\n"
                f"  - res norm: {r_norm}\n"
                f"  - Cumulative # of blocks: {cumBlockSize}\n"
            )
    return -x0


class PreNeumann(Preconditioner):
    def __init__(
        self,
        grid,
        use_cuda=False,
        order=3,
        correction_scale=0.1,
        no_shift_thr=10.0,
        fp="DP",
        max_order=20,
        error_cutoff=-0.4,
        verbosity=True,
    ):
        super().__init__("neumann", use_cuda, fp)

        assert order == "dynamic" or (type(order)==int and order>=0), "order should be int >=0 or 'dynamic'"
        self.order = order
        self.grid = grid
        self.correction_scale = correction_scale
        self.no_shift_thr = no_shift_thr
        self.max_order = int(max_order)
        self.error_cutoff = error_cutoff
        self.verbosity = verbosity

        self.gapp = create_preconditioner("gapp", grid, use_cuda)

    def __str__(self):
        s = str()
        s += "\n========================= [ Preconditioner ] ========================"
        s += f"\n* type             : {self._precond_type}"
        s += f"\n* fp               : {self._fp}"
        s += f"\n* order            : {self.order}"
        s += f"\n* correction_scale : {self.correction_scale}"
        s += f"\n* no_shift_thr     : {self.no_shift_thr}"
        s += f"\n* verbosity        : {self.verbosity}"
        s += f"\n* max_order        : {self.max_order}"
        s += f"\n* error_cutoff     : {self.error_cutoff}"
        s += "\n=====================================================================\n"
        return str(s)

    def call(self, residue, H, eigval):
        INV_4PI = 0.25 / np.pi

        is_needed_residue_norm = (self.order == "dynamic" or self.verbosity or self.correction_scale != 0.0)
        residue_norm = residue.norm(dim=0, keepdim=True) if is_needed_residue_norm else None

        # Modify shift values
        if self.correction_scale != 0.0:
            perturb = -(residue_norm.conj() * residue_norm)
            eigval = eigval + self.correction_scale * perturb
            eigval[abs(perturb) > self.no_shift_thr] = 0.0  # no shift to states with large residues

        # keep the original residue shape
        if residue.ndim == 1:
            residue = residue.unsqueeze(0)

        # preconditioned_result = self.gapp(residue).div_(FOUR_PI)
        preconditioned_result = self.gapp(residue).mul_(INV_4PI)

        # cutoff setting and dynamic order
        if self.order != 0:
            neumann_term = preconditioned_result.clone()

            if self.order == "dynamic":
                # check accuracy ## 이전 계산이 아니라 나머지 일때 키기 cutoff 만 보려면 꺼야함
                H_minus_eigval_vec = H @ preconditioned_result
                H_minus_eigval_vec -= eigval * preconditioned_result
                pre_error = torch.norm(H_minus_eigval_vec - residue, dim=0) / residue_norm

                if self.verbosity:
                    print("test_error (using order = 0) = ", torch.log10(pre_error))

                for order in range(1, self.max_order + 1): ##########정확도 평가가 들어가는 것만 이렇게 표현
                    # Compute the next Neumann series and accumulate the result
                    if order == 1:
                        # here we can save one H operation
                        # neumann_term -= self.gapp(H_minus_eigval_vec).div_(FOUR_PI)
                        neumann_term -= self.gapp(H_minus_eigval_vec).mul_(INV_4PI)
                    else:
                        H_minus_eigval_vec = H @ neumann_term
                        H_minus_eigval_vec -= eigval * neumann_term
                        # neumann_term -= self.gapp(H_minus_eigval_vec).div_(FOUR_PI)
                        neumann_term -= self.gapp(H_minus_eigval_vec).mul_(INV_4PI)
                    preconditioned_result += neumann_term

                    # Error calculation
                    H_minus_eigval_vec = H @ preconditioned_result
                    H_minus_eigval_vec -= eigval * preconditioned_result
                    error = torch.norm(H_minus_eigval_vec.sub_(residue), dim=0) / residue_norm

                    if pre_error.sum() > error.sum():
                        error_log = torch.log10(error)
                        if order == self.max_order:
                            if self.verbosity:
                                print(f"Preconditioned diagonalization error(log10) = {error_log} "
                                      f"(using order(high_error_cutoff) = {order})")
                            break
                        elif self.error_cutoff >= torch.max(error_log):
                            if self.verbosity:
                                print(f"Preconditioned diagonalization error(log10) = {error_log} "
                                    f"(using order(low_order_cutoff) = {order})")
                            break
                        else:
                            pre_error = error
                            continue
                    else:
                        preconditioned_result -= neumann_term
                        if self.verbosity:
                            print(f"Preconditioned diagonalization error(log10)= {torch.log10(pre_error)} "
                                f"(using order(pre<now_break) = {order - 1})")
                        break
            else: 
                for order in range(1, self.order + 1):
                    # Compute the next Neumann series and accumulate the result
                    neumann_term -= self.gapp(
                        H @ neumann_term - eigval * neumann_term
                    # ).div_(FOUR_PI)
                    ).mul_(INV_4PI)
                    preconditioned_result += neumann_term

                if self.verbosity:
                    diff = (H @ preconditioned_result - eigval * preconditioned_result) - residue
                    error = torch.norm(diff, dim=0) / residue_norm
                    print(f"error (using order = {order}) = ", torch.log10(error))

            if self.verbosity:
                # inversion accuracy
                # print("Neumann result:", torch.linalg.norm(preconditioned_result, axis=0))
                # print("(H-eI)@precond result:", torch.linalg.norm( H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result , axis=0))
                # print("(H-eI)@precond - residue result:", torch.linalg.norm( H@preconditioned_result - eigval.reshape(1,-1)*preconditioned_result - residue , axis=0))
                print("res norm (Neumann): ", torch.linalg.norm(H @ preconditioned_result - eigval * preconditioned_result, axis=0))

        return preconditioned_result


if __name__ == "__main__":
    from ase import Atoms
    from gospel.Grid import Grid

    grid = Grid(Atoms(cell=[3, 4, 5]), gpts=(100, 100, 100))
    precond = PreFilter(grid)
    print(precond)
    flter = precond.make_filter(grid)
    print(flter)

    nbands = 10
    R = torch.randn(grid.ngpts, nbands, dtype=torch.float64)
    R = precond(R)
