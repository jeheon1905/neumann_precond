"""Unit tests for spectral_tools (no GOSPEL / no Hamiltonian required)."""
import numpy as np
import torch

import spectral_tools as st

torch.manual_seed(0)
DT = torch.float64


def _mk_operator(A):
    def apply_A(x):
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.is_complex():
            return torch.complex(A @ x.real, A @ x.imag)
        return A @ x
    return apply_A


def test_arnoldi_matches_dense_eigenvalues():
    """Full-dimension Arnoldi must reproduce the dense spectrum."""
    n = 40
    g = torch.Generator().manual_seed(7)
    A = torch.randn(n, n, generator=g, dtype=DT) / np.sqrt(n)
    apply_A = _mk_operator(A)
    v0 = torch.randn(n, 1, generator=g, dtype=DT)

    res = st.arnoldi(apply_A, v0, n)
    theta, Y = st.ritz_pairs(res)
    dense = np.linalg.eigvals(A.numpy())

    got = np.sort_complex(np.round(theta, 8))
    exp = np.sort_complex(np.round(dense, 8))
    err = np.max(np.abs(np.sort(np.abs(got)) - np.sort(np.abs(exp))))
    assert err < 1e-8, err
    assert res.ortho_error < 1e-12, res.ortho_error
    print(f"  arnoldi vs dense: max |mag| diff = {err:.3e}, ortho = {res.ortho_error:.3e}")


def test_ritz_residual_small():
    """Explicit Ritz residuals of a converged (full-dim) run are ~machine eps."""
    n = 30
    g = torch.Generator().manual_seed(11)
    A = torch.randn(n, n, generator=g, dtype=DT) / np.sqrt(n)
    apply_A = _mk_operator(A)
    v0 = torch.randn(n, 1, generator=g, dtype=DT)
    res = st.arnoldi(apply_A, v0, n)
    theta, Y = st.ritz_pairs(res)
    worst = 0.0
    for j in range(len(theta)):
        r, _ = st.explicit_ritz_residual(res, complex(theta[j]), Y[:, j], apply_A)
        worst = max(worst, r)
    assert worst < 1e-9, worst
    print(f"  worst explicit Ritz residual = {worst:.3e}")


def test_linearity_of_wrapper():
    """ErrorPropagationOperator is linear given linear H and P."""
    n = 25
    g = torch.Generator().manual_seed(3)
    Hm = torch.randn(n, n, generator=g, dtype=DT)
    Hm = 0.5 * (Hm + Hm.T)
    Pm = torch.randn(n, n, generator=g, dtype=DT) / n

    class _H:
        def __matmul__(self, x):
            return Hm @ x

    E = st.ErrorPropagationOperator(_H(), lambda x: Pm @ x, shift=0.37)
    x = torch.randn(n, 1, generator=g, dtype=DT)
    y = torch.randn(n, 1, generator=g, dtype=DT)
    a, b = 1.3, -0.7
    lhs = E(a * x + b * y)
    rhs = a * E(x) + b * E(y)
    d = float(torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(rhs))
    assert d < 1e-13, d
    print(f"  linearity delta = {d:.3e}")


def test_recurrence_equivalence_dense():
    """a_{n+1} = E a_n reproduced by the operator wrapper."""
    n = 25
    g = torch.Generator().manual_seed(5)
    Hm = torch.randn(n, n, generator=g, dtype=DT)
    Hm = 0.5 * (Hm + Hm.T)
    Pm = torch.randn(n, n, generator=g, dtype=DT) / n
    shift = 0.21

    class _H:
        def __matmul__(self, x):
            return Hm @ x

    E = st.ErrorPropagationOperator(_H(), lambda x: Pm @ x, shift=shift)

    # emulate the production recurrence literally
    gamma = torch.randn(n, 1, generator=g, dtype=DT)
    a = Pm @ gamma
    terms = [a.clone()]
    for _ in range(4):
        a = a - Pm @ (Hm @ a - shift * a)
        terms.append(a.clone())

    for k in range(4):
        pred = E(terms[k])
        d = float(
            torch.linalg.vector_norm(terms[k + 1] - pred)
            / torch.linalg.vector_norm(terms[k + 1])
        )
        assert d < 1e-13, (k, d)
    print("  recurrence equivalence OK for n=0..3")


def test_summary_and_damping():
    theta = np.array([0.9, -0.98, 0.1 + 0.0j, -0.2])
    s = st.summarize_spectrum(theta)
    assert s["numerically_real"]
    assert abs(s["rho"] - 0.98) < 1e-12
    assert abs(s["lambda_min_real"] + 0.98) < 1e-12
    assert abs(st.damping_factor(-1.0)) < 1e-15
    assert abs(st.damping_factor(1.0) - 1.0) < 1e-15
    print("  summary/damping OK")




# ----------------------------------------------------------------------
# adjoint / singular values  (dense reference)
# ----------------------------------------------------------------------
def _toy(n=60, seed=3):
    """SPD P and symmetric M, mimicking the real structure of the problem."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    P = A @ A.transpose(0, 1) / n + 0.5 * torch.eye(n, dtype=torch.float64)
    B = torch.randn(n, n, generator=g, dtype=torch.float64)
    M = (B + B.transpose(0, 1)) / (2 * n ** 0.5)
    return P, M


class _E:
    """E = I - P M with the same interface as ErrorPropagationOperator."""
    def __init__(self, P, M):
        self.H, self.apply_P, self.shift = M, (lambda x: P @ x), 0.0
    def __call__(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return x - self.apply_P(self.H @ x)


def test_adjoint_matches_transpose():
    P, M = _toy()
    E = _E(P, M)
    Et = st.AdjointErrorPropagationOperator(E)
    n = P.shape[0]
    I = torch.eye(n, dtype=torch.float64)
    assert torch.allclose(E(I).transpose(0, 1), Et(I), atol=1e-11)


def test_neumann_operator_matches_closed_form():
    P, M = _toy()
    E = _E(P, M)
    n = P.shape[0]
    I = torch.eye(n, dtype=torch.float64)
    Ed = E(I)
    for N, w in [(0, 1.0), (3, 1.0), (0, 0.5), (4, 0.5)]:
        got = st.neumann_operator(E, N, w)(I)
        want = I - torch.linalg.matrix_power(Ed, N) @ ((1 - w) * I + w * Ed)
        assert torch.allclose(got, want, atol=1e-10), (N, w)
        # the adjoint routine must reproduce the transpose
        gotT = st.neumann_operator(st.AdjointErrorPropagationOperator(E), N, w)(I)
        assert torch.allclose(gotT, want.transpose(0, 1), atol=1e-10), (N, w)


def test_singular_value_bounds_bracket_the_truth():
    P, M = _toy()
    E = _E(P, M)
    n = P.shape[0]
    I = torch.eye(n, dtype=torch.float64)
    A = st.neumann_operator(E, 2, 0.5)
    At = st.neumann_operator(st.AdjointErrorPropagationOperator(E), 2, 0.5)
    Ad = A(I).numpy()
    sv = np.linalg.svd(Ad, compute_uv=False)
    g = torch.Generator().manual_seed(0)
    v0 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    for m in [15, n]:
        r = st.singular_value_bounds(A, At, v0, m)
        assert r["sigma_min_lo"] <= sv.min() + 1e-8
        assert r["sigma_max_hi"] >= sv.max() - 1e-8
        assert r["kappa_lower"] <= sv.max() / sv.min() * (1 + 1e-8)
    # a full-dimension run must be tight
    r = st.singular_value_bounds(A, At, v0, n)
    assert r["conclusive"]
    assert abs(r["kappa_upper"] / (sv.max() / sv.min()) - 1) < 1e-4


def test_lanczos_bound_is_valid_when_unconverged():
    """The whole point: a too-small m must still bracket, never mislead."""
    P, M = _toy(n=80, seed=7)
    E = _E(P, M)
    n = P.shape[0]
    A = st.neumann_operator(E, 1, 0.5)
    At = st.neumann_operator(st.AdjointErrorPropagationOperator(E), 1, 0.5)
    sv = np.linalg.svd(A(torch.eye(n, dtype=torch.float64)).numpy(), compute_uv=False)
    g = torch.Generator().manual_seed(1)
    v0 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    for m in [5, 10, 20, 40]:
        r = st.singular_value_bounds(A, At, v0, m)
        assert r["sigma_min_lo"] - 1e-9 <= sv.min() <= r["sigma_min_hi"] + 1e-9, m
        assert r["sigma_max_lo"] - 1e-9 <= sv.max() <= r["sigma_max_hi"] + 1e-9, m


def test_neumann_precond_times_M_matches_operator():
    """P_N composed with M must reproduce the closed form P_N M."""
    P, M = _toy()
    E = _E(P, M)
    n = P.shape[0]
    I = torch.eye(n, dtype=torch.float64)
    aP = E.apply_P
    for N, w in [(0, 1.0), (0, 0.5), (1, 0.5), (4, 0.5), (5, 1.0)]:
        PN = st.neumann_precond(E, aP, N, w)
        got = PN(M @ I)                                  # P_N M
        want = st.neumann_operator(E, N, w)(I)
        assert torch.allclose(got, want, atol=1e-9), (N, w, (got - want).abs().max())
        # adjoint
        PNt = st.neumann_precond_adjoint(st.AdjointErrorPropagationOperator(E), aP, N, w)
        assert torch.allclose(PNt(I), PN(I).transpose(0, 1), atol=1e-10), (N, w)


def test_jd_form_differs_from_implemented():
    """Pi P_N Pi M Pi is NOT Pi P_N M Pi -- the inserted projector matters."""
    P, M = _toy()
    E = _E(P, M)
    n = P.shape[0]
    g = torch.Generator().manual_seed(4)
    Xr = torch.linalg.qr(torch.randn(n, 5, generator=g, dtype=torch.float64))[0]
    Pi = st.Deflator(Xr)
    I = torch.eye(n, dtype=torch.float64)
    N, w = 3, 0.5
    PN = st.neumann_precond(E, E.apply_P, N, w)
    impl = Pi(st.neumann_operator(E, N, w)(Pi(I)))
    jd = Pi(PN(Pi(M @ Pi(I))))
    assert (impl - jd).abs().max() > 1e-6, "the two forms happen to coincide"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"* {fn.__name__}")
        fn()
    print(f"\nAll {len(fns)} spectral_tools tests passed.")
