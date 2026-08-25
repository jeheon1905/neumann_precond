# Draft response to reviewers — spectral-analysis items

Covers R2.1, R2.2, R2.3, R2.5, R2.6 and the spectral half of R2.4. R1 (DIIS) and the
RMM-DIIS half of R2.4 are handled separately.

**Method.** Everything below is measured on the four benchmark systems through the unmodified
production code path, at the first preconditioner call where every analysed state's residual
norm is below 1e-3, using the Davidson Ritz value itself as the shift. Individual eigenvalues
`λ(ΠEΠ)` and the spectral radius `ρ` come from Arnoldi at Krylov dimension 80; the extremal
Ritz values are stable to 5–6 significant figures with respect to the Krylov dimension
(m = 20 → 40 → 60) and to the starting vector, and every one of the 310 Ritz sets computed in
this work is numerically real (`max|Im λ| = 0`), as the deflated operator is not guaranteed to
be — `ΠEΠ` is non-symmetric, and unlike `E` it is not similar to a symmetric matrix.

Because Arnoldi on a non-symmetric operator carries no error bound, each `ρ` was independently
bracketed through `ρ ≤ ‖Aⁿ‖^{1/n}` up to n = 20, with `‖Aⁿ‖` obtained by Lanczos on the
symmetric operator `(Aⁿ)ᵀ(Aⁿ)`; the simple case n = 1 is useless here because `‖ΠEΠ‖` itself
exceeds 1 in seven of the eight state/system combinations (1.02–1.28). These certified bounds
lie above the corresponding Arnoldi estimates in all eight cases, by 0.1–6.0 %. **Statements
that the expansion converges rest on the certified bound; the Arnoldi values are used to
explain the mechanism.** Re-measuring the HOMO of the three molecular systems at three Davidson
iterations under both shift conventions changes nothing at the two later snapshots (agreement
to three decimals). Full tables are in the SI.

---

## R2.1 — Convergence condition and validity of the expansion

> We thank the reviewer for this suggestion. Carrying it out was more informative than we
> anticipated: the convergence condition **as usually stated cannot be satisfied** by any
> positive-definite `X` for a shifted interior eigenvalue problem, while the condition that is
> actually relevant inside Block Davidson **is** satisfied over most of the occupied manifold,
> and we can now say exactly where.

**1. ρ(E) > 1 is structural, not a deficiency of P.** `P` is SPD, so `PM` is similar to the
symmetric `P^{1/2}MP^{1/2}`, which is congruent to `M`; by Sylvester's law of inertia the
number of negative eigenvalues of `PM` equals the number of states below `ε̃`, independently of
the quality of `P`. For any state but the lowest, `M` is indefinite by construction, so some
`λ(E) > 1`. Measured: **ρ(E) = 1.00–1.14 at the lowest state and 1.46–1.83 at the HOMO**. For
B12 the count of states below `ε̃` is 0 / 131 / 248 / 295 for the lowest / middle / slowest /
HOMO state, in exact agreement with the inertia argument.

**2. The operator that acts is the deflated one.** Block Davidson orthogonalises the
preconditioned block against its subspace `X`, so what acts is `ΠEΠ`, `Π = I − XXᴴ`. We
measured how much of each divergent eigenvector survives that projection:

| origin of divergence | λ | ‖Πv‖/‖v‖ | removed |
|---|---:|---:|---:|
| near-singular shift | ≈ 1 | 0.0065 | 99.4 % |
| indefiniteness of M | > 1 | 0.039–0.113 | 88.7–96.1 % |
| over-correction by P | < −1 | 0.996–1.000 | 0.0–0.4 % |

The first two are removed by deflation; the third is not, and is the mode the ½ damping
targets (R2.2).

**3. The condition reduces to one inequality, and it is a condition on P.** With the first two
sources removed, what survives is the classical Neumann requirement on the remaining spectrum,

    mu_max( Pi P M Pi ) < 2 ,

i.e. **the preconditioner must not over-correct by more than a factor two on any direction the
solver actually generates**. This is a statement about `P`, not about the material: it is
violated by a single vacuum/grid mode of the GAPP kernel. The measurements bear that out —
across three chemically unrelated molecular systems the condition fails below the *same* shift
to within 0.6 % (−0.886, −0.889, −0.892 Ha), and it is never violated in the periodic system,
whose GAPP branch differs (Figure R1). The boundary therefore transfers between systems; for
the parameterisation used here it leaves **73–80 % of the occupied manifold** in the
convergent regime, and all of MAPbI3.

| system | `ρ(ΠEΠ)` lowest | HOMO | certified `ρ(ΠEΠ) ≤` (HOMO) |
|---|---:|---:|---:|
| MAPbI3 | **0.809** | 0.927 | **0.949** |
| C60_4 | 1.073 | 0.925 | **0.981** |
| water_128 | 1.101 | 0.925 | **0.949** |
| B12 | 1.136 | 0.931 | **0.977** |

For the remaining bands the `λ < −1` mode survives deflation, and damping handles it (R2.2).

**4. Practical order: N ≈ 4–6.** The spectral analysis fixes *whether* the expansion
converges and *why* damping is needed; it does not by itself select an order, and we make no
such claim. In the archived benchmark the Damped-NP iteration count is monotone non-increasing
over the whole tested range N = 0…10 and saturates rather than turning over (B12, N = 4…10:
27, 23, 21, 20, 19, 19, 19). Each additional order costs exactly one Hamiltonian and one GAPP
application, so the optimum is where the marginal gain stops paying for the marginal cost.
That places it at **N ≈ 4–6**, the range used in this work, and it is the existing timing
results that document it. What the spectral analysis adds is the reason the gain saturates:
over the bands satisfying point 3 the modes surviving deflation have `|λ| ≈ 0.93` or below,
so the
deflated error `‖Π E^{N+1} a₀‖` falls steeply over the first few orders and then flattens.

---

## R2.2 — Theoretical justification of the damping

**Is ½ heuristic?** No. The damped mode factor is `q_N(λ) = ½(1+λ)λᴺ`; a general weight α gives
`(1−α+αλ)λᴺ`, which annihilates a mode exactly when `α = 1/(1−λ)`. For **λ = −1** — the
boundary `μ = 2` of the convergence region, i.e. exactly where the expansion starts to fail —
this is **α = ½**. The measured optimum per system is 0.468 / 0.476 / 0.482, matching
`1/(1−λ_dom)` to three decimals; ½ is 2–7 % above it and requires no knowledge of `λ_dom`.

**Alternative weights.** Requiring the offending mode to remain contracting for all N ≤ 20
gives an admissible window **α ∈ [0.43, 0.51]**, which contains ½ for every system.

**What the damping actually buys.** Mode-wise, the undamped factor `λ^{N+1}` grows with order
wherever `|λ_dom| > 1`, while the damped factor carries the prefactor `|½(1+λ_dom)|` = 0.037–0.068,
a **15–27× suppression** of exactly the mode that diverges. A direct measurement of the
error-operator norm on the deflated subspace confirms this without any eigenvalue assumption:
at the lowest state of the three molecular systems the undamped norm grows from 1.26–1.28 at
N = 0 to 2.47–4.41 at N = 10, while the damped one falls to 0.44–0.54 (SI Table S2). Away
from the bottom of the manifold, where the condition of point 3 already holds, the two schemes
agree to within a few per cent.

**Relation to polynomial preconditioning and Chebyshev filtering.** `P̄_N M = I − q_N(λ)` with
`q_N` of degree N+1: the Neumann preconditioner **is** a polynomial preconditioner and damping
is a change of its coefficients. Against the optimal shifted-Chebyshev polynomial of the same
degree, evaluated on the measured spectrum of `ΠPMΠ`, Damped-NP is within a factor **1.2–4.0**
(most cases 1.6–2.5) — while requiring **no spectral bounds at all**. Those bounds are the real
cost: the measured interval varies with state and iteration ([0.069, 1.82] for B12/HOMO against
[0.191, 1.50] for MAPbI3/lowest), so Chebyshev would need a per-state estimate and degrades
when it is wrong.

---

## R2.3 — Oscillatory behaviour and the MAPbI3 exception

The reviewer's conjecture that this relates to the spectrum of `I − X⁻¹M` is correct, and the
mechanism is a single mode. Undamped, the mode factor is `λ^{N+1}`: for **λ < 0 the sign
alternates with N** — the odd–even oscillation of Figure 1 — and for **|λ| > 1 the magnitude
also grows**, so the expansion oscillates *and* degrades with order. Deflation cannot help:
this is a vacuum/grid mode, of which only 0–0.4 % is removed by Π.

| system | λ_min(ΠEΠ) | λ_max(ΠEΠ) | ρ(ΠEΠ) set by | undamped behaviour |
|---|---:|---:|---|---|
| B12 | **−1.1365** | +0.9280 | negative mode | oscillates, diverges with order |
| water_128 | **−1.1010** | +0.9219 | negative mode | oscillates, diverges with order |
| C60_4 | **−1.0733** | +0.9221 | negative mode | oscillates, diverges with order |
| **MAPbI3** | −0.4995 | **+0.8091** | positive mode | oscillates weakly, **converges** |

MAPbI3 is the only system whose offending mode lies above −1, so `|λ|^{N+1}` decays: the
oscillation is weak and there is nothing to diverge. Consistently its damping factor is 0.25
rather than 0.037–0.068, so damping suppresses a *useful* term and costs a few iterations at low
order — which is what the benchmark shows. The same exception appears independently in its
operator norm (‖ΠEΠ‖ = 0.87, the only one below 1) and its certificate (the only lowest state
with `ρ < 1`). Across the occupied manifold `λ_min` varies smoothly with the shift
(`dλ_min/dε̃ ≈ 0.50`); the fraction of bands affected is 20–27 % in the three molecular
systems and **0 %** in MAPbI3.

---

## R2.4 — Relation to other eigensolver strategies

**Jacobi–Davidson.** The exact deflated correction used as our reference, `Π(H − ε̃I)Π t = Πγ`,
*is* the Jacobi–Davidson correction equation, so the comparison is direct. JD solves that
deflated, symmetric positive-definite equation with an inner Krylov loop — 44–64 PCG
applications with global inner products and a stopping test. Damped-NP applies a fixed-degree
polynomial in the *undeflated* `M`, with no inner products and no convergence test, and lets
the outer Davidson orthogonalisation supply the deflation afterwards. Where the projector sits
relative to the preconditioner is immaterial here because `span(X)` occupies only 0.02–0.08 %
of the grid (nbands 300–768 against ngpts 0.87–1.6 M). We state plainly that this is **not** a
rate advantage: at the measured `ρ(ΠEΠ) ≈ 0.93` the series would need ≈ 230 applications to
match what PCG reaches in 44–64. The advantage is the absence of global reductions and of an
inner stopping criterion, which is what makes the scheme cheap on GPUs and at scale.

**Complementarity.** Damped-NP is a preconditioner, not an eigensolver: it needs only the
residual and the current Ritz value, uses no inner products and no stopping test, and is
therefore attachable to any solver that supplies those two quantities.

**Other eigensolvers.** [RMM-DIIS results to be inserted — separate experiments.]

---

## R2.5 — The corrected shift

`ε̃ = ε − c‖γ‖²` with c = 0.1. Using B12, where converged reference eigenvalues are available,
we measured the smallest c that keeps the shift below the target eigenvalue across six orders
of magnitude in ‖γ‖²: **c_min = 0.058–0.072**, so c = 0.1 carries a 1.4–1.7× margin.
Performance is insensitive within that margin; below c_min the shift crosses the eigenvalue
and the preconditioner amplifies rather than corrects. The zero-shift fallback (‖γ‖² > 10)
never triggers in the benchmark runs.

---

## R2.6 — Order 0 and linear cost

At N = 0 the damped operator reduces to `½ΠPMΠ`, i.e. a single GAPP application, and
`P̄_0 M = I − E` by construction. Each additional order costs exactly one Hamiltonian plus one
GAPP application on the residual block, so the cost is exactly linear in N. Both statements
follow from the closed form and need no measurement.

---

## Limitations disclosed in the SI

1. Asymptotic convergence in N is established for the bands satisfying the condition of R2.1
   point 3. For the remaining 20–27 % `ρ(ΠEΠ) = 1.07–1.14 > 1`; damping postpones the growth
   beyond the practical order range rather than removing it.
2. The analysis assumes a solver that maintains and orthogonalises against a subspace.
3. `ρ(ΠEΠ)` governs the deflated recursion `(ΠEΠ)ⁿ`, whereas the implemented scheme runs the
   recursion in the full space and projects once, giving `Π Eⁿ Π`. Π and E do not commute, so
   the two differ at high order; the practical range N ≤ 10 is where they stay close, and the
   direct `‖G_N‖` measurements of SI Table S2 rather than `ρ` alone are what certify the
   damped scheme there. No quantitative model linking a single state's spectral data to the
   aggregate Davidson iteration count is claimed.
4. The manifold sample is uniform in band index, not in energy, and does not include the
   virtual states (20 % of the Davidson block), which sit at higher ε̃ where reversals are most
   likely; the reversal fraction is therefore a lower bound.
5. Lower ends of the singular-value brackets behind every `ρ` certificate are exact by
   interlacing; upper ends rely on the Lanczos residual bound, which certifies proximity to an
   eigenvalue rather than to the extremal one. Convergence of those ends is verified
   numerically (m-independence, dense cross-check, re-measurement at three Davidson
   iterations) rather than proved. Likewise the real spectrum of `ΠEΠ` is verified in every
   Ritz set computed here, not proved; the `ρ` bound does not depend on it, since it uses only
   the symmetric singular-value machinery and holds for a complex spectrum as well.
