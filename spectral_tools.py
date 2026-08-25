"""Matrix-free spectral analysis of the Neumann error-propagation operator.

This module is *analysis only*. It never modifies the production preconditioner
path; it only consumes the exact code-level actions exposed by
``precondition.PreNeumann`` (``apply_P``) and the GOSPEL Hamiltonian
(``H @ X``).

Operator under analysis, for a single state with corrected shift ``eps_t``:

    E x = x - P (H x - eps_t x),      P x = (1 / 4 pi) GAPP(x)

which is exactly the map realised by the production recurrence

    H_minus_eigval_vec = H @ neumann_term - eigval_active * neumann_term
    neumann_term      -= self.gapp(H_minus_eigval_vec).mul_(INV_4PI)

so that ``a_{n+1} = E a_n``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
# Operator
# ----------------------------------------------------------------------
class ErrorPropagationOperator:
    """Matrix-free ``E = I - P (H - eps_t I)`` for one state.

    :param H: GOSPEL Hamiltonian LinearOperator supporting ``H @ X`` with
        ``X`` of shape ``(ngpts, k)``.
    :param apply_P: exact code-level action of ``P`` (``PreNeumann.apply_P``).
    :param shift: corrected shift ``eps_t`` (python float / 0-dim tensor)
        captured from the production code path.
    """

    def __init__(self, H, apply_P: Callable, shift):
        self.H = H
        self.apply_P = apply_P
        self.shift = float(shift)
        self.nmatvec = 0

    def _apply_real(self, x: torch.Tensor) -> torch.Tensor:
        """Apply E to a real column block ``(ngpts, k)``."""
        Mx = self.H @ x - self.shift * x
        return x - self.apply_P(Mx)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply E to a column block ``(ngpts, k)``, real or complex.

        E is a real-linear operator, so a complex vector is handled by
        applying it to the real and imaginary parts separately. This avoids
        relying on complex support inside the Hamiltonian.
        """
        if x.ndim == 1:
            x = x.unsqueeze(1)
        self.nmatvec += x.shape[1]
        if x.is_complex():
            re = self._apply_real(x.real.contiguous())
            im = self._apply_real(x.imag.contiguous())
            return torch.complex(re, im)
        return self._apply_real(x.contiguous())


# ----------------------------------------------------------------------
# Arnoldi
# ----------------------------------------------------------------------
@dataclass
class ArnoldiResult:
    m: int  # Krylov dimension actually built
    Hm: torch.Tensor  # (m+1, m) upper Hessenberg
    Q: torch.Tensor  # (ngpts, m+1) orthonormal basis
    breakdown: bool
    ortho_error: float  # max |Q^T Q - I| over the built basis
    beta0: float  # norm of the starting vector


def arnoldi(
    apply_A: Callable,
    v0: torch.Tensor,
    m: int,
    reorth: bool = True,
    breakdown_tol: float = 1e-13,
    check_ortho: bool = True,
    project: Optional[Callable] = None,
) -> ArnoldiResult:
    """Modified Gram-Schmidt Arnoldi with one reorthogonalisation sweep.

    ``v0`` has shape ``(ngpts, 1)``. Basis vectors are never mutated in place
    after they are stored.

    ``project`` confines the Krylov space to an invariant subspace, and is
    REQUIRED when ``apply_A`` has a null space that is not of interest.
    ``A = Pi N(E) Pi`` annihilates ``span(X)``: each application leaves a
    ~1e-16 absolute component there, and because the normalisation ``w / beta``
    divides by an increasingly small ``beta`` as Arnoldi resolves the small
    singular directions, that leak is amplified until Arnoldi returns a
    spurious Ritz value at 0 and hence an infinite eigenvalue ratio.
    """
    assert v0.ndim == 2 and v0.shape[1] == 1, f"v0 must be (ngpts, 1), got {tuple(v0.shape)}"
    n = v0.shape[0]
    dtype = v0.dtype
    device = v0.device

    if project is not None:
        v0 = project(v0)
        if v0.ndim == 1:
            v0 = v0.unsqueeze(1)
    beta0 = float(torch.linalg.vector_norm(v0).item())
    assert beta0 > 0.0, "starting vector is zero"

    Q = torch.zeros(n, m + 1, dtype=dtype, device=device)
    Hm = torch.zeros(m + 1, m, dtype=dtype, device=device)
    Q[:, 0:1] = v0 / beta0

    j_built = 0
    breakdown = False
    for j in range(m):
        w = apply_A(Q[:, j : j + 1]).reshape(-1).clone()
        # modified Gram-Schmidt
        for i in range(j + 1):
            h = torch.dot(Q[:, i].conj(), w)
            Hm[i, j] = Hm[i, j] + h
            w = w - h * Q[:, i]
        if reorth:
            for i in range(j + 1):
                h = torch.dot(Q[:, i].conj(), w)
                Hm[i, j] = Hm[i, j] + h
                w = w - h * Q[:, i]
        if project is not None:
            # AFTER Gram-Schmidt, immediately before the normalisation.  Projecting
            # earlier is useless: MGS shrinks ||w|| to beta, and dividing by a small
            # beta amplifies whatever null-space residue survives by ||w|| / beta.
            # Measured on a dense 1000-point case: the leak grows 1e-17 -> 1e-5 over
            # 60 steps when projected before MGS, and stays at 1e-16 when projected
            # here.  Q spans an invariant subspace, so this does not disturb the
            # already-enforced orthogonality against Q.
            w = project(w).reshape(-1)
        beta = torch.linalg.vector_norm(w)
        Hm[j + 1, j] = beta
        j_built = j + 1
        if float(beta.item()) < breakdown_tol * max(1.0, float(Hm[: j + 1, j].abs().max())):
            breakdown = True
            break
        Q[:, j + 1 : j + 2] = (w / beta).unsqueeze(1)

    # number of *valid* basis vectors: on breakdown the (j+1)-th was not built
    n_basis = j_built if breakdown else j_built + 1

    ortho_error = float("nan")
    if check_ortho:
        Qb = Q[:, :n_basis]
        G = Qb.T.conj() @ Qb
        G = G - torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        ortho_error = float(G.abs().max().item())

    return ArnoldiResult(
        m=j_built,
        Hm=Hm[: j_built + 1, :j_built].clone(),
        Q=Q[:, :n_basis].clone(),
        breakdown=breakdown,
        ortho_error=ortho_error,
        beta0=beta0,
    )


def ritz_pairs(res: ArnoldiResult):
    """Eigenpairs of the projected operator ``H_m = Q_m^H E Q_m``.

    Returns ``(theta, Y)`` with ``theta`` complex eigenvalues (numpy) and
    ``Y`` the corresponding eigenvectors of ``H_m`` (numpy, columns).
    """
    Hm_sq = res.Hm[: res.m, : res.m].cpu().numpy()
    theta, Y = np.linalg.eig(Hm_sq)
    return theta, Y


def explicit_ritz_residual(
    res: ArnoldiResult, theta: complex, y: np.ndarray, apply_A: Callable
) -> tuple[float, torch.Tensor]:
    """Explicit ``||E v - theta v|| / ||v||`` for one Ritz pair.

    ``v = Q_m y`` is formed and E is applied to it directly; no use is made of
    the Arnoldi relation, so this is an independent check.
    """
    Qm = res.Q[:, : res.m]
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(Qm.device)
    if np.iscomplexobj(y) or abs(theta.imag) > 0.0:
        v = torch.complex(Qm @ y_t.real.to(Qm.dtype), Qm @ y_t.imag.to(Qm.dtype))
        th = torch.tensor(theta, dtype=v.dtype, device=v.device)
    else:
        v = (Qm @ y_t.to(Qm.dtype)).unsqueeze(1)
        th = torch.tensor(float(theta.real), dtype=v.dtype, device=v.device)
    if v.ndim == 1:
        v = v.unsqueeze(1)
    r = apply_A(v) - th * v
    nv = torch.linalg.vector_norm(v)
    return float((torch.linalg.vector_norm(r) / nv).item()), v


def arnoldi_residual_estimate(res: ArnoldiResult, Y: np.ndarray) -> np.ndarray:
    """Cheap Arnoldi residual bound ``|h_{m+1,m}| * |y[m-1]|`` for each pair."""
    h_last = float(res.Hm[res.m, res.m - 1].item()) if res.Hm.shape[0] > res.m else 0.0
    return np.abs(h_last * Y[-1, :])


# ----------------------------------------------------------------------
# Spectrum summary
# ----------------------------------------------------------------------
def summarize_spectrum(theta: np.ndarray, imag_tol: float = 1e-10) -> dict:
    """Extremal quantities of a Ritz set, with a real/complex verdict."""
    mag = np.abs(theta)
    max_abs_imag = float(np.max(np.abs(theta.imag))) if theta.size else 0.0
    scale = max(1.0, float(np.max(mag)) if theta.size else 1.0)
    is_real = max_abs_imag <= imag_tol * scale
    out = {
        "n_ritz": int(theta.size),
        "rho": float(np.max(mag)) if theta.size else float("nan"),
        "max_abs_imag": max_abs_imag,
        "numerically_real": bool(is_real),
        "idx_largest_magnitude": int(np.argmax(mag)) if theta.size else -1,
        "idx_most_negative": int(np.argmin(theta.real)) if theta.size else -1,
        "idx_most_positive": int(np.argmax(theta.real)) if theta.size else -1,
        "lambda_min_real": float(np.min(theta.real)) if theta.size else float("nan"),
        "lambda_max_real": float(np.max(theta.real)) if theta.size else float("nan"),
    }
    return out


def damping_factor(lam: complex) -> float:
    """Spectral damping factor of the implemented two-point averaging."""
    return abs(0.5 * (1.0 + lam))


# ----------------------------------------------------------------------
# eta_N (Phase 4 diagnostic)
# ----------------------------------------------------------------------
def eta_from_terms(a_norms: list[float], a_sum_norms: list[float]) -> dict:
    """``eta_N^NP = ||a_{N+1}|| / ||a_0||`` and ``eta_N^DNP = ||a_N + a_{N+1}|| / (2 ||a_0||)``."""
    a0 = a_norms[0]
    return {
        "eta_NP": [a_norms[n + 1] / a0 for n in range(len(a_norms) - 1)],
        "eta_DNP": [s / (2.0 * a0) for s in a_sum_norms],
    }


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_json(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)


def append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    import csv
    import os

    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def memory_estimate_bytes(ngpts: int, m: int, itemsize: int = 8) -> int:
    """Krylov basis ``(m+1) * ngpts`` plus a few work vectors."""
    return int((m + 1 + 4) * ngpts * itemsize)


# ----------------------------------------------------------------------
# Deflated analysis (JCTC revision, second pass)
#
# Davidson orthogonalises the preconditioned block against the current
# subspace X before using it:   R <- R - X (X^H R).
# The unstable eigendirections of E = I - P(H - eps_t I) are exactly the
# eigenstates below eps_t, which span(X) already represents, so the part of
# the Neumann error that grows is removed before it can act.  Every quantity
# below therefore measures the *deflated* object.
# ----------------------------------------------------------------------
class Deflator:
    """Orthogonal projector Pi = I - X X^H onto the complement of span(X)."""

    def __init__(self, X: torch.Tensor):
        self.X = X
        self.k = X.shape[1]

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim == 1:
            v = v.unsqueeze(1)
        return v - self.X @ (self.X.transpose(0, 1).conj() @ v)

    def orthonormality_error(self) -> float:
        """max |X^H X - I|; the deflation is only as good as X is orthonormal."""
        G = self.X.transpose(0, 1).conj() @ self.X
        G = G - torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return float(G.abs().max().item())


class DeflatedOperator:
    """Pi E Pi, matrix-free."""

    def __init__(self, E, deflator: Deflator):
        self.E = E
        self.Pi = deflator
        self.nmatvec = 0

    def __call__(self, v):
        self.nmatvec += 1
        return self.Pi(self.E(self.Pi(v)))


def pcg(apply_A, apply_M, b, tol=1e-10, maxiter=300):
    """Preconditioned CG for A x = b with A SPD on the working subspace.

    Returns (x, history, status). Stops and reports if negative curvature is
    met - that is itself a meaningful result (it means the deflated operator
    is not positive definite, i.e. span(X) does not yet contain all states
    below the shift).
    """
    x = torch.zeros_like(b)
    r = b.clone()
    z = apply_M(r)
    p = z.clone()
    rz = float((r * z).sum().item())
    bn = float(torch.linalg.vector_norm(b).item())
    hist, status = [], "max_iter"
    for it in range(maxiter):
        Ap = apply_A(p)
        pAp = float((p * Ap).sum().item())
        if pAp <= 0.0:
            return x, hist, f"negative_curvature(it={it}, pAp={pAp:.3e})"
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        rn = float(torch.linalg.vector_norm(r).item()) / bn
        hist.append(rn)
        if rn < tol:
            status = "converged"
            break
        z = apply_M(r)
        rz_new = float((r * z).sum().item())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, hist, status


def direction_agreement(u: torch.Tensor, t: torch.Tensor) -> dict:
    """How well does u point along t?  (Davidson normalises, so only the
    direction matters.)"""
    nu = float(torch.linalg.vector_norm(u).item())
    nt = float(torch.linalg.vector_norm(t).item())
    if nu == 0.0 or nt == 0.0:
        return {"cos": float("nan"), "angle_deg": float("nan"),
                "rel_err_normalised": float("nan")}
    c = float((u * t).sum().item()) / (nu * nt)
    c = max(-1.0, min(1.0, c))
    d = u / nu - (t / nt) * (1.0 if c >= 0 else -1.0)
    return {
        "cos": c,
        "angle_deg": float(np.degrees(np.arccos(abs(c)))),
        "rel_err_normalised": float(torch.linalg.vector_norm(d).item()),
    }


# ----------------------------------------------------------------------
# Adjoints and singular values
# ----------------------------------------------------------------------
# ``P`` and ``H`` are symmetric to machine precision (verified: relative
# <v,Au>-<u,Av> error 4e-16 for both), so
#
#     E   = I - P M        =>    E^T = I - M P
#
# i.e. the adjoint is obtained by swapping the order of the two factors.
# ``E`` itself is NOT symmetric (asymmetry 0.16-0.28), which is why the
# eigenvalue ratio is only a lower bound on the true condition number and the
# singular values have to be computed separately.
class AdjointErrorPropagationOperator:
    """Matrix-free ``E^T = I - M P`` for one state."""

    def __init__(self, E: "ErrorPropagationOperator"):
        self.E = E
        self.nmatvec = 0

    def _apply_real(self, x: torch.Tensor) -> torch.Tensor:
        Px = self.E.apply_P(x)
        return x - (self.E.H @ Px - self.E.shift * Px)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(1)
        self.nmatvec += x.shape[1]
        if x.is_complex():
            return torch.complex(self._apply_real(x.real.contiguous()),
                                 self._apply_real(x.imag.contiguous()))
        return self._apply_real(x.contiguous())


def neumann_operator(apply_E: Callable, N: int, weight: float) -> Callable:
    """Order-``N`` preconditioned operator ``P_N M = I - E^N ((1-w) I + w E)``.

    ``weight = 1`` reproduces the undamped ``I - E^(N+1)``; ``weight = 0.5`` the
    production Damped-NP.  Costs ``N + 1`` applications of ``apply_E``.
    The same routine applied to ``E^T`` yields ``(P_N M)^T`` because powers of a
    single operator commute.
    """
    def f(v):
        w = v
        for _ in range(N):
            w = apply_E(w)
        return v - ((1.0 - weight) * w + weight * apply_E(w))
    return f


@dataclass
class LanczosResult:
    k: int                 # tridiagonal size actually built
    alpha: np.ndarray      # diagonal, length k
    beta: np.ndarray       # off-diagonal, length k (beta[k-1] is the residual norm)
    breakdown: bool
    ortho_error: float


def lanczos(apply_B: Callable, v0: torch.Tensor, m: int,
            reorth: bool = True, breakdown_tol: float = 1e-13,
            project: Optional[Callable] = None) -> LanczosResult:
    """Symmetric Lanczos with full reorthogonalisation.

    ``apply_B`` MUST be symmetric for the residual bound of
    :func:`lanczos_ritz` to hold.  Used here with ``B = A^T A``.

    ``project`` restricts the Krylov space to an invariant subspace.  It is
    required whenever ``B`` has a null space that is not of interest: rounding
    reintroduces null-space components at the 1e-16 level, reorthogonalisation
    against the Lanczos basis does not remove them, and Lanczos then converges
    to the spurious zero eigenvalue instead of the true smallest one.  For
    ``A = Pi N(E) Pi`` the null space is ``span(X)`` (dimension nbands), so
    ``project = Pi`` is needed; without it ``theta_min`` collapses to ~1e-16
    and the condition number is reported as infinite.
    """
    if v0.ndim == 1:
        v0 = v0.unsqueeze(1)
    n = v0.shape[0]
    Q = torch.zeros(n, m + 1, dtype=v0.dtype, device=v0.device)
    if project is not None:
        v0 = project(v0)
        if v0.ndim == 1:
            v0 = v0.unsqueeze(1)
    Q[:, 0] = (v0 / torch.linalg.vector_norm(v0)).squeeze(1)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    breakdown = False
    k = m
    for j in range(m):
        w = apply_B(Q[:, j:j + 1]).squeeze(1)
        a = float(torch.dot(Q[:, j], w).item())
        alpha[j] = a
        w = w - a * Q[:, j]
        if j > 0:
            w = w - beta[j - 1] * Q[:, j - 1]
        if reorth:                     # twice-is-enough
            for _ in range(2):
                w = w - Q[:, :j + 1] @ (Q[:, :j + 1].transpose(0, 1) @ w)
        if project is not None:        # after reorthogonalisation, before normalising
            pw = project(w)
            w = pw.squeeze(1) if pw.ndim == 2 else pw
        b = float(torch.linalg.vector_norm(w).item())
        if b <= breakdown_tol * max(1.0, abs(a)):
            breakdown = True
            k = j + 1
            break
        beta[j] = b
        Q[:, j + 1] = w / b
    G = Q[:, :k].transpose(0, 1) @ Q[:, :k]
    G = G - torch.eye(k, dtype=G.dtype, device=G.device)
    return LanczosResult(k=k, alpha=alpha[:k], beta=beta[:k], breakdown=breakdown,
                         ortho_error=float(G.abs().max().item()))


def lanczos_ritz(res: LanczosResult):
    """Ritz values of the tridiagonal and the classical bound ``beta_k |s_k|``.

    For a symmetric operator every Ritz pair satisfies ``|lambda - theta| <=
    beta_k |s_k|`` for some true eigenvalue ``lambda``; combined with the
    interlacing property (``lambda_min <= theta_min`` and ``theta_max <=
    lambda_max``) this brackets the extremal eigenvalues rigorously.
    """
    k = res.k
    T = np.diag(res.alpha)
    if k > 1:
        off = res.beta[:k - 1]
        T = T + np.diag(off, 1) + np.diag(off, -1)
    theta, S = np.linalg.eigh(T)
    resid = 0.0 if res.breakdown else abs(float(res.beta[k - 1]))
    bound = resid * np.abs(S[-1, :])
    return theta, bound


def singular_value_bounds(apply_A: Callable, apply_At: Callable, v0: torch.Tensor,
                          m: int, **kw) -> dict:
    """Rigorous two-sided bounds on ``sigma_max``, ``sigma_min`` and ``kappa(A)``.

    Runs Lanczos on the symmetric PSD operator ``B = A^T A`` whose eigenvalues
    are the squared singular values of ``A``.  Unlike Arnoldi on the
    non-symmetric ``A``, this carries a rigorous error bound, so a small
    computed extremal value can be told apart from an unconverged one.

    Interlacing gives ``lambda_min <= theta_min`` and ``theta_max <=
    lambda_max``; the residual bound gives the other side.  Hence

        sigma_min in [sqrt(max(theta_min - b, 0)), sqrt(theta_min)]
        sigma_max in [sqrt(theta_max),             sqrt(theta_max + b)]

    and ``kappa_lower = sqrt(theta_max / theta_min)`` holds unconditionally.
    """
    def B(x):
        return apply_At(apply_A(x))

    res = lanczos(B, v0, m, **kw)
    theta, bound = lanczos_ritz(res)
    th_lo, b_lo = float(theta[0]), float(bound[0])
    th_hi, b_hi = float(theta[-1]), float(bound[-1])
    smin_lo = math.sqrt(max(th_lo - b_lo, 0.0))
    smin_hi = math.sqrt(max(th_lo, 0.0))
    smax_lo = math.sqrt(max(th_hi, 0.0))
    smax_hi = math.sqrt(max(th_hi + b_hi, 0.0))
    k_lo = smax_lo / smin_hi if smin_hi > 0 else float("inf")
    k_hi = smax_hi / smin_lo if smin_lo > 0 else float("inf")
    return {
        "m": res.k, "breakdown": res.breakdown, "ortho_error": res.ortho_error,
        "theta_min": th_lo, "theta_min_bound": b_lo,
        "theta_max": th_hi, "theta_max_bound": b_hi,
        "sigma_min_lo": smin_lo, "sigma_min_hi": smin_hi,
        "sigma_max_lo": smax_lo, "sigma_max_hi": smax_hi,
        "kappa_lower": k_lo, "kappa_upper": k_hi,
        "conclusive": bool(np.isfinite(k_hi)),
    }


def neumann_precond(apply_E: Callable, apply_P: Callable, N: int, weight: float) -> Callable:
    """The PRECONDITIONER itself, ``P_N = [sum_{n=0}^{N-1} E^n + w E^N] P``.

    This is what the production recurrence accumulates (``p_bar_N = p_{N-1} +
    w a_N`` with ``a_0 = P gamma``, ``a_{n+1} = E a_n``), as opposed to
    :func:`neumann_operator`, which builds the preconditioned operator
    ``P_N M = I - E^N ((1-w) I + w E)``.  The two are related by ``P_N M``, and
    they differ as soon as a projector is inserted between the factors, which is
    exactly the Jacobi-Davidson form ``Pi P_N Pi M Pi``.

    Costs ``N`` applications of ``apply_E`` plus one of ``apply_P``.
    """
    def f(v):
        a = apply_P(v)                      # a_0 = P v
        s = torch.zeros_like(a)
        for _ in range(N):
            s = s + a
            a = apply_E(a)
        return s + weight * a
    return f


def neumann_precond_adjoint(apply_Et: Callable, apply_P: Callable, N: int,
                            weight: float) -> Callable:
    """``P_N^T = P [sum_{n=0}^{N-1} (E^T)^n + w (E^T)^N]`` (``P`` is symmetric)."""
    def f(v):
        a = v
        s = torch.zeros_like(v)
        for _ in range(N):
            s = s + a
            a = apply_Et(a)
        return apply_P(s + weight * a)
    return f


def neumann_error_operator(apply_E: Callable, N: int, weight: float) -> Callable:
    """Error operator of the order-``N`` scheme, ``G_N = E^N ((1-w) I + w E)``.

    ``P_N M = I - G_N``, so ``||G_N||`` is the contraction factor actually achieved
    at order ``N``: below 1 the scheme contracts, above 1 it does not.  ``w = 1``
    gives the undamped ``E^(N+1)``; ``w = 0.5`` the production Damped-NP
    ``0.5 E^N (I + E)``.

    Unlike the eigenvalue expression ``max |0.5 (1+lam) lam^N|`` over spec(Pi E Pi),
    this is the operator the implementation applies: the recurrence runs in the FULL
    space, so ``Pi G_N Pi`` contains ``Pi E^N Pi``, not ``(Pi E Pi)^N``.  The two
    differ because Pi and E do not commute, and the difference is what the
    eigenvalue estimate misses.
    """
    def f(v):
        w = v
        for _ in range(N):
            w = apply_E(w)
        return (1.0 - weight) * w + weight * apply_E(w)
    return f
