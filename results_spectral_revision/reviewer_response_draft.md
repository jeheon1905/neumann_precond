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
exceeds 1 in seven of the eight state/system combinations (1.02–1.28). These bounds lie above
the corresponding Arnoldi estimates in all eight cases, by 0.1–6.0 %. **Statements that the
expansion converges rest on the bound; the Arnoldi values are used to explain the mechanism.**
The lower end of each bracket is exact by interlacing; the upper end rests on the Lanczos
residual bound and is verified numerically — by m-independence, by a dense cross-check, and by
re-measuring the HOMO of the three molecular systems at three Davidson iterations under both
shift conventions, which changes nothing at the two later snapshots — rather than proved. Full
tables, and the limits of the analysis, are in the SI.

---

## R2.1 — Convergence condition and validity of the expansion

> We thank the reviewer for this suggestion; carrying it out was more informative than we
> anticipated. We report the spectral radius of the deflated operator `ΠEΠ` rather than of `E`
> itself, and the reason is the substance of the answer rather than a matter of presentation:
> `ρ(E) > 1` for any state but the lowest, but the eigenvectors responsible are the occupied
> states lying below the shift — exactly the vectors Block Davidson has already found and
> orthogonalises against. Measured on the operator the solver actually applies, the condition
> **is** satisfied over most of the occupied manifold, and we can now say exactly where the
> rest lies and why it is the case the ½ damping was introduced for.

**1. Where `ρ(E) > 1` comes from.** Writing `λ(E) = 1 − μ(PM)`, the series converges when every
`μ(PM)` lies in `(0, 2)`, so the question is whether `PM` can be made positive definite. It
cannot. `P` is SPD, so `P^{-1/2}(PM)P^{1/2} = P^{1/2}MP^{1/2}`: `PM` is *similar* to that
symmetric matrix and therefore shares its eigenvalues, and that matrix is in turn *congruent*
to `M` (it is `SᵀMS` with `S = P^{1/2}`), so by Sylvester's law of inertia the two have the
same number of negative eigenvalues. Chaining the two,

    #{ negative eigenvalues of PM } = #{ negative eigenvalues of M } = #{ states below eps_tilde } .

`P` can rescale these eigenvalues freely but cannot change their signs, and the count is
therefore independent of how good the preconditioner is. For any state but the lowest, `M` is
indefinite by construction and some `λ(E) > 1` follows. Measured: **ρ(E) = 1.00–1.14 at the
lowest state and 1.46–1.83 at the HOMO**; for B12 the count of states below `ε̃` is
0 / 131 / 248 / 295 for the lowest / middle / slowest / HOMO state, in exact agreement.

**2. Those eigenvectors are the ones Davidson removes.** The identity above also names them:
the eigenvectors that push `ρ(E)` above 1 are the occupied states lying below the shift, and
those are precisely the vectors the solver has already converged and keeps in its subspace
`X`. Block Davidson orthogonalises the preconditioned residual against `X` before using it, so
the operator that acts on the error is not `E` but `ΠEΠ` with `Π = I − XXᴴ`. Deflation is
therefore what the algorithm already does, not a device introduced for this analysis: the
polynomial we analyse reproduces the production `PreNeumann` output bit for bit, and the
projector mirrors the orthogonalisation the solver performs on it. The alternative reading
does not survive contact with the data — undeflated, the series does not converge at all at
these states (`ρ(E) = 1.46–1.83` at the HOMO), yet the expansion demonstrably accelerates the
solver, so `E` cannot be the operator that governs it. We therefore measured how much of each
divergent eigenvector survives the projection:

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
violated by a single mode of the GAPP kernel, one that carries almost no weight in the
occupied subspace and so is not reachable by deflation. The measurements bear that out —
across three chemically unrelated molecular systems the condition fails below the *same* shift
to within 0.6 % (−0.886, −0.889, −0.892 Ha), and it is never violated in the periodic system,
whose GAPP branch differs (Figure R1). The boundary therefore transfers between systems; for
the parameterisation used here it leaves **73–80 % of the occupied manifold** in the
convergent regime, and all of MAPbI3.

| system | `ρ(ΠEΠ)` lowest | HOMO | bound on `ρ(ΠEΠ)` (HOMO) |
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
over the bands satisfying point 3 the modes surviving deflation have `|λ| ≈ 0.93` or below, so
the deflated error `‖Π E^{N+1} a₀‖` falls steeply over the first few orders and then flattens.

---

## R2.2 — Theoretical justification of the damping

**Where ½ came from, and why it survives scrutiny.** The weight was not fitted. Damped-NP
averages the preconditioned vectors of the two highest consecutive orders, and ½ is simply the
plain average — the cheapest choice available, requiring no spectral information. The analysis
the reviewer asks for shows that this choice is not arbitrary.

Averaging orders N−1 and N with a general weight α gives `P̄M = I − Eᴺ((1−α)I + αE)`, so each
mode of `E` is multiplied by

    q_N(lambda) = (1 - alpha + alpha*lambda) * lambda^N ,

which reduces to `λ^{N+1}` at α = 1 and to `½(1+λ)λᴺ` at α = ½. The leading factor vanishes
when `α = 1/(1−λ)`, so a given mode can be annihilated outright. Setting **λ = −1** — the
boundary `μ = 2` of the convergence region, i.e. exactly where the expansion starts to fail —
gives **α = ½**. The plain average is therefore the weight that annihilates the mode at the
edge of convergence, which is the mode that survives deflation (R2.1 point 2). Measured per
system, the exact annihilator `1/(1−λ_dom)` is 0.468 / 0.476 / 0.482; ½ sits 2–7 % above it
and, unlike the exact value, needs no knowledge of `λ_dom`.

**Alternative weights.** Requiring the offending mode to remain contracting for all N ≤ 20
gives an admissible window **α ∈ [0.43, 0.51]**, which contains ½ for every system. What ½ has
in its favour is not that it wins at any single order — it does not — but that, being the
admissible value nearest the exact annihilator, it leaves the widest margin before the
offending mode stops contracting. We ran the weight as an ablation to check that this matters
in practice. On B12, sweeping the weight at fixed order gives

| N | α = 0.3 | **α = 0.5** | α = 0.7 |
|---:|---:|---:|---:|
| 4 | 28 | **27** | 27 |
| 5 | 25 | **24** | 23 |
| 6 | 22 | **22** | 22 |
| 7 | 21 | **20** | 20 |
| 8 | 19 | **19** | 19 |
| 9 | 19 | **19** | 24 |
| 10 | 19 | **19** | 19 |

Davidson iterations to a residual norm of 1e-5, one seed. Across the practical range the three
weights are within one iteration of each other, and ½ is never the worst; the only departure is
α = 0.7 at N = 9, where the weight furthest from the annihilator loses ground. [Extending the
sweep to α = 0.1 and 0.9 is in progress and will be added.]

**What the damping actually buys.** Mode-wise, the undamped factor `λ^{N+1}` grows with order
wherever `|λ_dom| > 1`, while the damped factor replaces one power of `λ_dom` by
`½(1+λ_dom)` = 0.037–0.068 — a **17–29× suppression** of exactly the mode that diverges. A
direct measurement of the error-operator norm on the deflated subspace confirms this without
any eigenvalue assumption:
at the lowest state of the three molecular systems the undamped norm grows from 1.26–1.28 at
N = 0 to 2.47–4.41 at N = 10, while the damped one falls to 0.44–0.54 (SI Table S2). Away
from the bottom of the manifold, where the condition of R2.1 point 3 already holds, the two schemes
agree to within a few per cent.

**Relation to Richardson damping and weighted Neumann expansions.** Both connections the
reviewer raises are exact, and in both our scheme sits at the simple end of the family.
Truncating the Neumann series at order N is identical to running N+1 steps of preconditioned
Richardson iteration with relaxation ω = 1 from `x₀ = 0`, since `E = I − PM` is precisely the
Richardson error operator at ω = 1. Richardson damping tunes ω, which changes the recurrence
itself and multiplies every mode by `(1−ωμ)^{N+1}`; our weighting leaves the recurrence at
ω = 1 and changes only how the partial sums are combined, giving `(1−αμ)(1−μ)ᴺ`. Expanding the
average shows the same thing from the other side,

    (1-alpha) p_{N-1} + alpha p_N  =  sum_{k<N} E^k  +  alpha E^N ,

i.e. a weighted Neumann expansion with `w_k = 1` for `k < N` and `w_N = α` — the minimal
case, in which only the final term is reweighted; Cesàro summation is the same family with a
full weight profile. We make no claim that the minimal choice outperforms these alternatives. It is
the one that requires no spectral input, and the analysis above is what recommends the
particular value ½ within it.

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

The reviewer's conjecture is correct, and the operator named is the one analysed throughout
this response: `X⁻¹` is the applied preconditioner, which is the zeroth-order GAPP operator
`P`, so `I − X⁻¹M = I − PM = E`. Note that `E` is built from `P` alone and carries no
dependence on the expansion order — the order enters only as the polynomial applied to it,
which is why one spectrum explains the behaviour at every order.

The mechanism is a single mode. Undamped, the mode factor is `λ^{N+1}`: for **λ < 0 the sign
alternates with N** — the odd–even oscillation of Figure 1 — and for **|λ| > 1 the magnitude
also grows**, so the expansion oscillates *and* degrades with order. Deflation cannot reach
this mode: it carries almost no weight in the occupied subspace, so `Π` removes only 0–0.4 %
of it.

Measured at the **lowest state**, where the oscillation is strongest — at the HOMO every
system has `λ_min > −1` and the distinction below disappears:

| system | λ_min(ΠEΠ) | λ_max(ΠEΠ) | ρ(ΠEΠ) set by | undamped behaviour |
|---|---:|---:|---|---|
| B12 | **−1.1365** | +0.9280 | negative mode | oscillates, diverges with order |
| water_128 | **−1.1010** | +0.9219 | negative mode | oscillates, diverges with order |
| C60_4 | **−1.0733** | +0.9221 | negative mode | oscillates, diverges with order |
| **MAPbI3** | −0.4995 | **+0.8091** | positive mode | oscillates weakly, **converges** |

MAPbI3 is the only system whose offending mode lies above −1, so `|λ|^{N+1}` decays: the
oscillation is weak and there is nothing to diverge. Damping is correspondingly unhelpful
there, and for a reason worth stating precisely. It still contracts the negative mode — by
0.50, against 0.03–0.07 in the other three — but MAPbI3 is the one system whose rate is set by
the *positive* mode, and on that mode the damped factor `½(1+λ)λᴺ` exceeds the undamped
`λ^{N+1}` by **1.12×**: damping enlarges the very term that governs convergence. That is why
Damped-NP costs MAPbI3 a few iterations at low order, which is what the benchmark shows. The
same exception appears independently in its operator norm (‖ΠEΠ‖ = 0.87 at the lowest state,
the only one below 1) and in its certificate (the only lowest state with `ρ < 1`). Across the
occupied manifold `λ_min` varies smoothly with the shift (`dλ_min/dε̃ ≈ 0.50`); the fraction
of bands with `λ_min < −1`, i.e. those failing the condition of R2.1 point 3, is 20–27 % in
the three molecular systems and **0 %** in MAPbI3.

---

## R2.4 — Relation to other eigensolver strategies

**Jacobi–Davidson.** JD is the standard frame in which a preconditioner enters an
eigensolver: each outer step forms a correction equation and solves it inexactly. Damped-NP is
not an alternative to that frame but a preconditioner — the thing such an equation is solved
with — so the two sit at different levels, and what can be compared is JD's inner Krylov solve
against the fixed-degree polynomial we put in its place. The inner solve adapts its work to
the problem but needs global inner products and a stopping test at every step; the
polynomial's work is fixed by the order alone and needs neither. This is no rate advantage,
and we say so plainly: solving that correction equation by PCG reaches 1e-8 in 44–64
applications, whereas a series at the measured `ρ ≈ 0.93` would need ≈ 230. What it buys is a
deterministic per-iteration cost without global communication, which is what makes the scheme
cheap on GPUs and at scale.

**Complementarity.** Nothing in the construction is specific to Block Davidson: it consumes
only the residual and the current Ritz value, so it attaches wherever those exist — LOBPCG,
RMM-DIIS, Chebyshev-filtered subspace iteration. The one structural requirement is the one
identified in R2.1, that the solver orthogonalise against the states it has already converged,
since that is what removes the divergent eigenvectors of `E`.

**Beyond Block Davidson.** That requirement is met more widely than it may appear. RMM-DIIS
optimises each band independently and grows no subspace, but it cannot be run without a
full orthonormalisation between sweeps — otherwise distinct bands collapse onto the same
eigenvector — so the deflation the analysis relies on is present there too, applied once per
sweep instead of once per iteration.

We therefore repeated the fixed-Hamiltonian comparison inside an RMM-DIIS solver, replacing
only the preconditioner. Damped-NP behaves as it does under Block Davidson: [insert the
RMM-DIIS numbers — iteration counts and speedup against ISI at the same orders]. That the
same preconditioner carries over to a solver with a different subspace strategy is the
practical form of the complementarity above.

---

## R2.5 — The corrected shift

The corrected shift is not new here: it is the strategy introduced with the ISI
preconditioner and carried over unchanged, and the reasoning is the same. A Ritz value is
variational, so `ε` sits above the eigenvalue being targeted, and shifting by it directly
would over-correct. Subtracting a small multiple of `‖γ‖²` guards against that, and the choice
of `‖γ‖²` follows the quadratic convergence of a Ritz value in its own residual.

The coefficient is bounded on both sides and is, within those bounds, empirical. It must be
large enough that `ε̃` stays below the target eigenvalue, and small enough that the shift does
not overshoot far below it, where it would no longer approximate the eigenvalue and the
preconditioner would lose accuracy. c = 0.1 sits just above the lower bound. In the
calculations reported here it did what it is meant to do: on B12, the one system for which
converged reference eigenvalues were available, the smallest admissible coefficient
`(ε − ε_true)/‖γ‖²` is 0.0715 / 0.0710 / 0.0583 at three snapshots spanning six orders of
magnitude in `‖γ‖²`, so c = 0.1 kept the shift below — and close to — the target eigenvalue
throughout.

Performance is not sensitive to the value. The correction accelerates convergence rather than
enabling it — the scheme remains stable when the raw Ritz value is used as the shift. The
zero-shift fallback (`‖γ‖² > 10`) never triggers in the benchmark runs.

---

## R2.6 — Order 0 and linear cost

We agree that this should not require the reader to consult the earlier work, and have made it
explicit in the revised manuscript where the expansion orders are first introduced. At order 0
the recurrence returns `a₀ = Pγ` before any Hamiltonian is applied, so order 0 *is* GAPP rather
than merely equivalent to it; each further order applies `H` once and `P` once to the residual
block and nothing else, so the cost is exactly linear in the order.
