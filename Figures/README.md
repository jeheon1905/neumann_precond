# Figures — Damped-NP (order=5) vs ISI

Per-iteration diagonalization / preconditioning time for **Damped-NP (order=5)**
and **ISI** preconditioners across four benchmark systems:
(H₂O)₁₂₈, C₆₀ tetramer, MAPbI₃, Vitamin B₁₂.

All timings reported here come from a **fixed-Hamiltonian diagonalization**
benchmark **without the locking scheme** (i.e. no SCF update of the
Hamiltonian between outer steps, and no converged eigenpairs are locked out
of subsequent iterations).

## Figure caption

> **Per-iteration cost of the Damped-NP (order=5) and ISI preconditioners
> under fixed-Hamiltonian diagonalization (no locking).**
> Diagonalization time (solid, circles) and preconditioning time
> (dashed, squares) are plotted versus the outer diagonalization iteration
> index for four benchmark systems: **(a)** (H₂O)₁₂₈, **(b)** C₆₀ tetramer,
> **(c)** MAPbI₃, and **(d)** Vitamin B₁₂. The Hamiltonian is held fixed
> across iterations and the locking scheme is disabled, so every outer step
> operates on the full subspace and the per-iteration cost reflects the
> intrinsic behavior of each preconditioner. Colors distinguish the
> preconditioner (blue: **Damped-NP (order=5)**; orange: **ISI**) and line
> styles distinguish the timed quantity (solid: Diag. time; dashed: Precond.
> time). The first iteration carries no preconditioning step and the last
> iteration is a partial / convergence step, both of which appear as the dips
> at the curve endpoints. Across all four systems, Damped-NP exhibits an
> essentially constant per-iteration cost (steady-state σ ≲ 0.006 s for both
> Diag. and Precond. time), whereas ISI shows a noticeably larger variation
> driven by its inner iterative solve — the standard-deviation ratio
> σ(ISI)/σ(Damped-NP) ranges from ~5× (Vitamin B₁₂) to >100× (MAPbI₃ Precond.
> time); see the table below for the full breakdown.

## Files

| File | Purpose |
| --- | --- |
| `order5_np_damped_isi_per_iteration_time.csv` | Raw per-iteration timings (input). |
| `plot_order5_damped_isi_per_iter.py` | Generates the 2×2 figure. |
| `order5_np_damped_isi_per_iteration_time.png` / `.pdf` | Output figure. |
| `stats_order5_damped_isi.py` | Computes mean / std summary. |
| `order5_np_damped_isi_per_iter_std.csv` | Summary table (mean, std). |

## Environment

The project README recommends a dedicated `neumann_precond` conda environment,
but for plotting/statistics only `matplotlib` is required. The figures here
were produced with the existing `ase-autoresearch-pilot` environment
(`matplotlib 3.10.9`). Either works:

```bash
conda activate ase-autoresearch-pilot   # or: neumann_precond
```

The scripts use only `csv`, `math`, `pathlib`, and `matplotlib` — no `pandas`
or `numpy` required.

## Reproducing the figure

```bash
cd Figures
python plot_order5_damped_isi_per_iter.py
```

Outputs `order5_np_damped_isi_per_iteration_time.{png,pdf}`.

**Plot details**

- 2×2 layout: (a) (H₂O)₁₂₈, (b) C₆₀ tetramer, (c) MAPbI₃, (d) Vitamin B₁₂.
- x-axis: diagonalization iteration index.
- y-axis: time (s) — single axis covering both Diag. and Precond. time.
- Color encodes method (Damped-NP = blue, ISI = orange);
  line style encodes quantity (solid = Diag., dashed = Precond.).
- Single shared legend at the bottom.
- The first and last iterations have empty `Preconditioning time` in the CSV
  (no precond applied on iter 1; last iter is a convergence/partial step).
  They are kept in the plot for completeness, which produces the visible dips
  at the curve endpoints.

## Reproducing the std analysis

```bash
cd Figures
python stats_order5_damped_isi.py
```

Outputs a console table and `order5_np_damped_isi_per_iter_std.csv`. Two
regions are reported:

- **all** — every iteration in the CSV.
- **steady** — excludes the 1st and last iter (rows with empty Precond.),
  i.e. only steady-state iterations are used. Recommended for stability
  comparison since the boundary iters have anomalously short timings.

### Std summary with ISI / Damped-NP ratio (steady-state)

**Table caption.**
> Per-iteration cost of the Damped-NP (order=5) and ISI preconditioners under
> fixed-Hamiltonian diagonalization without locking, summarized over the
> steady-state region (first and last iteration excluded, as their
> preconditioning entries are empty). For each of the four benchmark
> systems — (H₂O)₁₂₈, C₆₀ tetramer, MAPbI₃, Vitamin B₁₂ — the mean and sample
> standard deviation (μ ± σ, in seconds) of the diagonalization time and the
> preconditioning time are reported for both methods. The third row of each
> system block gives the standard-deviation ratio σ(ISI)/σ(Damped-NP) for
> each timed quantity, which quantifies the relative variability of the two
> preconditioners. Across all four systems Damped-NP shows an essentially
> constant per-iteration cost (σ ≲ 0.006 s for both quantities), while ISI's
> σ is 5×–~100× larger; the largest disparity occurs for MAPbI₃ (Precond.
> time, ~106×), and the smallest for Vitamin B₁₂ (~5×).

| System | Method | Diag. time μ ± σ (s) | Precond. time μ ± σ (s) |
| --- | --- | --- | --- |
| (H₂O)₁₂₈ | Damped-NP | 1.4621 ± **0.0023** | 0.9880 ± **0.0013** |
|  | ISI | 1.6409 ± **0.0450** | 1.1608 ± **0.0439** |
|  | σ ratio (ISI / Damped-NP) | **19.4×** | **34.2×** |
| C₆₀ tetramer | Damped-NP | 1.0687 ± **0.0055** | 0.7841 ± **0.0047** |
|  | ISI | 1.1997 ± **0.0341** | 0.9076 ± **0.0328** |
|  | σ ratio (ISI / Damped-NP) | **6.2×** | **7.0×** |
| MAPbI₃ | Damped-NP | 1.7463 ± **0.0034** | 1.2467 ± **0.0016** |
|  | ISI | 1.8627 ± **0.1984** | 1.2907 ± **0.1725** |
|  | σ ratio (ISI / Damped-NP) | **57.6×** | **105.9×** |
| Vitamin B₁₂ | Damped-NP | 0.8840 ± **0.0052** | 0.6653 ± **0.0050** |
|  | ISI | 1.0360 ± **0.0279** | 0.8120 ± **0.0265** |
|  | σ ratio (ISI / Damped-NP) | **5.4×** | **5.3×** |

### Interpretation

- Across all four systems and for both Diag. and Precond. time,
  **Damped-NP exhibits substantially smaller per-iteration variability than
  ISI** — the standard-deviation ratio ranges from ~5× (Vitamin B₁₂) to
  >100× (MAPbI₃ Precond).
- Damped-NP's per-iteration cost is essentially constant
  (σ ≲ 0.006 s in all four systems), reflecting the deterministic nature of
  a fixed-order Neumann expansion.
- ISI's larger std stems from its inner iterative solve, whose cost varies
  per outer step.
- The "all-iterations" view inflates std (e.g. Damped-NP Diag σ ≈ 0.4 s on
  H₂O) because the 1st and last iter have anomalously short timings; this is
  why the steady-state region is preferred for comparing intrinsic
  variability.
