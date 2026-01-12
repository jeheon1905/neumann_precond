# TODO: make the ./utils folder and split the code into separate files placed in ./utils.
import time
import functools
from typing import Dict, Optional, Union
from contextlib import contextmanager
import numpy as np
from scipy.special import sph_harm
import scipy.sparse as sparse
import torch
import random

from gospel.ParallelHelper import ParallelHelper as PH


class Timer:
    """
    A global Timer class that allows cumulative timing for different labeled operations.

    Usage:
    ------
    Timer.start("Subspace Projection")
    # ... some computation ...
    Timer.stop("Subspace Projection")

    Timer.start("Another Task")
    # ... another computation ...
    Timer.stop("Another Task")

    Timer.print_total("Subspace Projection")  # Prints total cumulative time for the label
    Timer.print_all()  # Prints all recorded timers
    """

    # e.g.,
    # _records[label] = {
    #     "start": Optional[float],  # start time or None
    #     "total": float,            # sum of elapsed times
    #     "count": int,              # how many times timed
    # }
    _records: Dict[str, Dict[str, Optional[float]]] = {}

    @classmethod
    def start(cls, label: str, enable_timing: bool = True):
        """
        Start the timer for a given label.

        Parameters
        ----------
        label : str
            The name of the operation being timed.
        enable_timing : bool, optional
            If False, the timer will not record any time (default: True).
        """
        if not enable_timing:
            return

        PH.synchronize()  # Synchronize all devices

        if label not in cls._records:
            cls._records[label] = {"start": None, "total": 0.0, "count": 0}

        if cls._records[label]["start"] is not None:
            return  # Ignore redundant starts

        cls._records[label]["start"] = time.time()

    @classmethod
    def stop(
        cls, label: str, enable_timing: bool = True, verbosity: bool = True
    ):
        """
        Stop the timer for a given label and accumulate elapsed time.

        Parameters
        ----------
        label : str
            The name of the operation being timed.
        enable_timing : bool, optional
            If False, the timer will not record any time (default: True).
        verbosity : bool, optional
            If False, prevents printing of elapsed time (default: True).
        """
        if (
            not enable_timing
            or label not in cls._records
            or cls._records[label]["start"] is None
        ):
            return

        PH.synchronize()  # Synchronize all devices

        elapsed = time.time() - cls._records[label]["start"]
        cls._records[label]["start"] = None  # Reset start time
        cls._records[label]["total"] += elapsed
        cls._records[label]["count"] += 1

        if verbosity:
            print(f"[Time: {label}]: {elapsed:.6f} s")

    @classmethod
    def print_total(cls, label: str):
        """
        Print the total accumulated time for a given label.

        Parameters
        ----------
        label : str
            The name of the operation being queried.
        """
        if label in cls._records:
            print(f"[{label}] Cumulative time: {cls._records[label]['total']:.6f} s")

    @classmethod
    def print_all(cls):
        """
        Print total cumulative times for all recorded timers.
        """
        if not cls._records:
            print("[Timer] No records found.")
            return

        for label, data in cls._records.items():
            print(f"[{label}] Total cumulative time: {data['total']:.6f} s")

    @classmethod
    def print_summary(cls, sort_by: str = "total", descending: bool = True):
        """
        Print a summary of all recorded labels, showing total and count.
        Allows sorting by total or count.

        Parameters
        ----------
        sort_by : str, optional
            The key to sort by ('total' or 'count'), default: 'total'.
        descending : bool, optional
            Sort order, default: True (descending).
        """
        if not cls._records:
            print("[Timer] No records to summarize.")
            return

        valid_keys = ("total", "count")
        if sort_by not in valid_keys:
            print(f"[Timer] Invalid sort_by='{sort_by}'. Using 'total'.")
            sort_by = "total"

        # Build a list of (label, total, count)
        items = []
        for label, data in cls._records.items():
            total_time = data["total"]
            c = data["count"]
            items.append((label, total_time, c))

        # Determine index for sorting
        # (label=0, total=1, count=2)
        sort_index = 1 if sort_by == "total" else 2

        # Sort
        items.sort(key=lambda x: x[sort_index], reverse=descending)

        # Print
        print("\n======================== Timer Summary ========================")
        print(f"Sorted by '{sort_by}', descending={descending}:")
        print(f"{'Label':40s} | {'Total(s)':>12} | {'Count':>5}")
        print("-" * 64)
        for label, total_time, c in items:
            print(f"{label:40s} | {total_time:12.4f} | {c:5d}")
        print("-" * 64 + "\n")

    @classmethod
    def reset(cls):
        """
        Reset all timing records.
        """
        cls._records.clear()
        print("[Timer] All records have been reset.")

    @classmethod
    @contextmanager
    def track(
        cls, label: str, enable_timing: bool = True, verbosity: bool = True
    ):
        """
        Context manager that allows using `with Timer.track(label)`.

        Example:
        --------
        with Timer.track("Subspace Projection"):
            # Some computation...
            time.sleep(1.5)
        """
        cls.start(label, enable_timing)
        yield  # Execution of "with" block happens here
        cls.stop(label, enable_timing, verbosity)

    def __init__(
        self,
        label: str,
        device=None,
        enable_timing: bool = True,
        verbosity: bool = True,
    ):
        """
        Allow `with Timer(label)` usage.
        Example:
        --------
        with Timer("Some Task"):
            # Computation
        """
        self.label = label
        self.device = device
        self.enable_timing = enable_timing
        self.verbosity = verbosity

    def __enter__(self):
        Timer.start(self.label, self.enable_timing)

    def __exit__(self, exc_type, exc_value, traceback):
        Timer.stop(self.label, self.enable_timing, self.verbosity)

    # FIXME: no synchronization when using timeit now
    @classmethod
    def timeit(cls, _label: Optional[Union[str, callable]] = None):
        """
        A decorator to measure execution time using Timer class methods.

        Usage:
        ------
        @Timer.timeit              # Uses function's __qualname__ as label
        def my_func():
            ...

        @Timer.timeit("Custom Label")  # Uses "Custom Label" as label
        def another_func():
            ...
        """
        # Without arguments, _label is a callable object; use its __qualname__ instead.
        if callable(_label):
            func = _label
            label = func.__qualname__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cls.start(label)
                result = func(*args, **kwargs)
                cls.stop(label, verbosity=True)
                return result

            return wrapper
        else:

            def decorator(func):
                label = _label if _label is not None else func.__qualname__

                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    cls.start(label)
                    result = func(*args, **kwargs)
                    cls.stop(label, verbosity=True)
                    return result

                return wrapper

            return decorator


# WARNING: deprecated
def timer(func):
    """Check time.

    Usage)
    >>> @timer
    >>> def method1(...):
    >>>     ...
    >>>     return
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(f"Elapsed time[{func.__name__}]: {end - start} sec", flush=True)
        print(f"Elapsed time[{func.__qualname__}]: {end - start} sec", flush=True)
        return result

    return wrapper


def vprint(*args, verbosity: bool = True):
    """Print messages only if verbosity is enabled."""
    if verbosity:
        print(*args)


# Custom Decorator function
# from  https://stackoverflow.com/questions/49210801/python3-pass-lists-to-function-with-functools-lru-cache
def toTuple(function):
    def wrapper(*args, **kwargs):
        args = [tuple(x.reshape(-1)) if type(x) == np.ndarray else x for x in args]
        args = [tuple(x) if type(x) == list else x for x in args]
        for key, values in kwargs.items():
            if type(values) == np.ndarray:
                kwargs[key] = tuple(values.reshape(-1))
            elif type(values) == list:
                kwargs[key] = tuple(values)

        result = function(*args, **kwargs)
        # result = tuple(result) if type(result) == list else result
        return result

    return wrapper


# NOTE: deprecated (use gospel.special_functions.Y_lm_real_batch instead)
def Y_lm_torch(point, l, m, tol=1e-7):
    """torch version of Y_lm function"""
    rxy = (point[:, 0] ** 2 + point[:, 1] ** 2).sqrt()
    phi = torch.arctan(
        torch.where(
            (point[:, 2] == 0) * (rxy == 0) == True,
            1.0,
            torch.nan_to_num(rxy / point[:, 2]),
        )
    )
    phi = torch.where(point[:, 2] < 0, phi + torch.pi, phi)

    theta = torch.arccos(
        torch.where(
            (point[:, 0] == 0) * (rxy == 0) == True,
            1.0,
            torch.nan_to_num(point[:, 0] / rxy),
        )
    )
    theta = torch.where((point[:, 1] < 0) == True, -theta + 2 * torch.pi, theta)

    ## sph_harm function in scipy does not support GPU calculation.
    device = point.device
    theta, phi = theta.cpu(), phi.cpu()

    ## Make Real Spherical harmonics
    if m > 0:
        real_sph_harm = (-1) ** m * np.sqrt(2) * sph_harm(m, l, theta, phi).real
    elif m < 0:
        real_sph_harm = (-1) ** m * np.sqrt(2) * sph_harm(abs(m), l, theta, phi).imag
    else:  # m == 0
        real_sph_harm = sph_harm(m, l, theta, phi).real
    return real_sph_harm.to(device)


# NOTE: deprecated (use gospel.special_functions.Y_lm_real_batch instead)
def Y_lm(point, l, m, tol=1e-7):
    """
    Calculate Real Spherical harmonics.

    :type  point: np.ndarray
    :param point:
        grid points, shape=(ngpts, 3)
    :type  l: int
    :param l:
        angular momentum quantum number
    :type  m: int
    :param m:
        magnetic momentum quantum number, |m| <= l

    :rtype: np.ndarray
    :return:
        real spherical harmonics values, shape=(ngpts,)

    phi   : Polar coordinate [0,pi]
    theta : Azimuthal coordinate [0,2*pi]

            z
            |
            |
            |_______ y
            /
           /
          x

    phi   = arctan( \frac{\sqrt{x^2+y^2}}{z} )
    theta = arccos( \frac{x}{\sqrt{x^2+y^2}} )
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # ex)
        # >>> np.nan_to_num( [nan, -inf, inf, nan] )
        # >>> array([0.00000000e+000, -1.79769313e+308, 1.79769313e+308, 0.00000000e+000])
        rxy = np.sqrt(point[:, 0] ** 2 + point[:, 1] ** 2)
        phi = np.arctan(
            np.where(
                (point[:, 2] == 0) * (rxy == 0) == True,
                1.0,
                np.nan_to_num(rxy / point[:, 2]),
            )
        )
        phi = np.where(point[:, 2] < 0, phi + np.pi, phi)
        theta = np.arccos(
            np.where(
                (point[:, 0] == 0) * (rxy == 0) == True,
                1.0,
                np.nan_to_num(point[:, 0] / rxy),
            )
        )
        theta = np.where((point[:, 1] < 0) == True, -theta + 2 * np.pi, theta)

    ## Make Real Spherical harmonics
    if m > 0:
        real_sph_harm = (-1) ** m * np.sqrt(2) * sph_harm(m, l, theta, phi).real
        return real_sph_harm
    elif m < 0:
        real_sph_harm = (-1) ** m * np.sqrt(2) * sph_harm(abs(m), l, theta, phi).imag
        return real_sph_harm
    else:  # m == 0
        return sph_harm(m, l, theta, phi).real


def tensordot(a, b, axes=2):
    """multiplication of sparse matrix and dense tensor. (modification of np.tensordot)

    :type  a: np.ndarray or scipy.sparse.spmatrix
    :type  b: np.ndarray or scipy.sparse.spmatrix
    :type  axes: int or tuple

    :rtype: np.ndarray
    """
    if isinstance(a, sparse.spmatrix) or isinstance(a, np.ndarray):
        assert isinstance(b, sparse.spmatrix) or isinstance(b, np.ndarray)
        _transpose = np.transpose
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        if not (a.layout == b.layout == torch.strided):
            raise NotImplementedError("torch sparse tensor is not supported.")
        _transpose = torch.permute

    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    # a, b = asarray(a), asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    if isinstance(a, sparse.spmatrix):
        if np.all(newaxes_a == [1, 0]):
            at = a.T.reshape(newshape_a)
        elif np.all(newaxes_a == [0, 1]):
            at = a.reshape(newshape_a)
    else:
        # at = a.transpose(newaxes_a).reshape(newshape_a)
        at = _transpose(a, newaxes_a).reshape(newshape_a)
    if isinstance(b, sparse.spmatrix):
        if np.all(newaxes_b == [1, 0]):
            bt = b.T.reshape(newshape_b)
        elif np.all(newaxes_b == [0, 1]):
            bt = b.reshape(newshape_b)
    else:
        # bt = b.transpose(newaxes_b).reshape(newshape_b)
        bt = _transpose(b, newaxes_b).reshape(newshape_b)

    res = at @ bt
    return res.reshape(olda + oldb)


def torch_to_scipy_sparse(A_torch):
    """convert torch.sparse to scipy.sparse

    :type  A_torch: torch.Tensor
    :param A_torch:
        torch sparse layout matrix (sparse_csr or sparse_coo)

    :rtype: scipy.sparse._csr.csr_matrix or scipy.sparse._coo.coo_matrix
    :return:
        scipy sparse type matrix
    """
    assert isinstance(A_torch, torch.Tensor)
    if A_torch.layout == torch.sparse_csr:
        A_scipy = sparse.csr_matrix(
            (
                A_torch.values().numpy(),
                A_torch.col_indices().numpy(),
                A_torch.crow_indices().numpy(),
            ),
            shape=np.asarray(A_torch.size()),
        )
    elif A_torch.layout == torch.sparse_coo:
        row, col = A_torch.indices()
        A_scipy = sparse.coo_matrix(
            (A_torch.values().cpu().numpy(), (row.cpu(), col.cpu())),
            shape=np.asarray(A_torch.size()),
        )
        pass
    else:
        raise NotImplementedError
    return A_scipy


def scipy_to_torch_sparse(A_scipy):
    """convert scipy.sparse to torch.sparse

    :type  A_scipy: scipy.sparse._csr.csr_matrix or scipy.sparse._coo.coo_matrix
    :param A_scipy:
        scipy sparse type matrix

    :rtype: torch.Tensor
    :return:
        torch sparse type tensor
    """
    if isinstance(A_scipy, sparse.csr_matrix):
        A_torch = torch.sparse_csr_tensor(
            A_scipy.indptr,
            A_scipy.indices,
            A_scipy.data,
            A_scipy.shape,
        )
    elif isinstance(A_scipy, sparse.coo_matrix):
        A_torch = torch.sparse_coo_tensor(
            np.vstack((A_scipy.row, A_scipy.col)),
            A_scipy.data,
            A_scipy.shape,
        )
    else:
        raise NotImplementedError(f"type(A_scipy) = {type(A_scipy)}")
    return A_torch


def torch_diag_sparse(inp, dtype=None):
    """convert dense Tensor to diagonal CSR sparse tensor

    :type  inp: torch.Tensor
    :param inp:
        diagonal elements, shape=(N,)
    :type  dtype: torch.dtype, optional
    :param dtype:
        data type

    :rtype: torch.Tensor (layout=torch.sparse_csr)
    :return:
        diagonal CSR sparse tensor, shape=(N, N)
    """
    if isinstance(input, torch.Tensor):
        raise TypeError(f"expected torch.Tensor (got {type(input)})")
    indices = torch.arange(len(inp), device=inp.device)
    indices = torch.vstack((indices, indices))
    output = torch.sparse_coo_tensor(indices, inp, (len(inp), len(inp)), dtype=dtype)
    return output.to_sparse_csr()


def to_cuda(inp, device=None, dtype=None):
    """Returns a copy object of torch.sparse_csr in device

    :type  inp: torch.Tensor
    :param inp:
        torch sparse csr tensor
    """
    assert isinstance(inp, torch.Tensor)
    if device is None:
        device = inp.device
    else:
        assert isinstance(device, torch.device)

    # assert inp.layout == torch.sparse_csr, f"{inp.layout}, {type(inp)}"
    if inp.layout == torch.sparse_csr:
        output = torch.sparse_csr_tensor(
            inp.crow_indices().to(device),
            inp.col_indices().to(device),
            inp.values().to(device),
            inp.size(),
            dtype=dtype,
            device=device,
        )
    elif inp.layout == torch.sparse_csc:
        output = torch.sparse_csc_tensor(
            inp.ccol_indices().to(device),
            inp.row_indices().to(device),
            inp.values().to(device),
            inp.size(),
            dtype=dtype,
            device=device,
        )
    else:
        raise NotImplementedError
    return output


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for torch, numpy, and random.

    Args:
        seed (int): Base seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False  # reproducibility
    torch.backends.cudnn.deterministic = True  # reproducibility


if __name__ == "__main__":
    # Y_lm test
    ngpts = 100
    point = np.random.randn(ngpts, 3)
    point[abs(point) < 0.5] = 0.0
    print(point)
    l, m = 1, -1

    ylm = Y_lm(point, l, m)
    ylm_test = Y_lm_torch(torch.from_numpy(point), l, m)
    print(f"sum(diff)={abs(ylm - ylm_test.numpy()).sum()}")

    # Timer test
    Timer.start("My Task")
    time.sleep(1)
    Timer.stop("My Task")

    Timer.start("My Task")
    time.sleep(2)
    Timer.stop("My Task")

    Timer.print_total("My Task")  # Outputs: 3 seconds total
    Timer.print_summary()
