# Reproducing the paper's figures and tables

Five configuration files reproduce every measured figure and table in
*"Efficient Preconditioning for Iterative Diagonalization via Neumann Series
Expansion of the Inverse of a Shifted Hamiltonian"*.

Run each with

```bash
python repeat_test.py --config configs/paper/<config>.yaml
```

## What produces what

| Config | Mode | Produces |
|---|---|---|
| `config.fixed.timing.yaml`   | fixed | **Figure 1(a,b)**, **Table S1** |
| `config.fixed.history.yaml`  | fixed | **Figure 2** |
| `config.fixed.periter.yaml`  | fixed | **Figure S1**, **Table S2** |
| `config.scf.nolocking.yaml`  | scf   | **Figure 3** (blue), **Table S3(a)** |
| `config.scf.locking.yaml`    | scf   | **Figure 3** (red), **Table S3(b)** |

## Why five, and not fewer

Three axes force the split.

**Preconditioner (NP / Damped-NP / ISI) does *not* force a split.**
`averaged_sum` is expanded combinatorially by `repeat_test.py`
(`for avg_sum in cfg.averaged_sum_list`) and the results are separated by the
`avgsum=0` / `avgsum=1` component of the output path, so
`averaged_sum: [false, true]` yields NP and Damped-NP from one run. ISI comes
along in the same sweep because `preconds` lists `shift-and-invert`.

**Locking does force a split.** `repeat_test.py` reads it as a scalar
(`cfg.locking = bool(dav.get("locking", False))`), not as a sweep axis, so the
two SCF conditions are two configs.

**Instrumentation forces a split.** Two flags decide what a run can report:

| Flag | Effect | Cost |
|---|---|---|
| `do_retHistory` | `ParallelDavidson` appends `eigval.to("cpu")` and `residue.to("cpu")` every outer iteration; `test.py` saves them as `history.pt` | a GPU→CPU copy per iteration |
| `verbosity` | `Timer.stop` prints `[Time: <label>]: <t> s` for every timed block | stdout per iteration |

On the fixed-Hamiltonian path `Timer` is always on — `test.py` calls
`davidson(..., timing=True)` directly — and `timing`, not `verbosity`, is what
gates `PH.synchronize()`. The aggregate **Timer Summary** is therefore printed
even at `verbosity: 0`, which is why iteration counts and preconditioning
totals are still available from the timing runs:

```
======================== Timer Summary ========================
Label                                    |     Total(s) | Count
davidson                                 |      32.4275 |     1
Diag. Iter.                              |      32.4166 |    31
Preconditioning                          |      23.8850 |    29
```

What the summary cannot give is a *series*: per-iteration means and standard
deviations (Table S2) need the individual `[Time: ...]` lines, hence
`verbosity: 1` in `config.fixed.periter.yaml`. Conversely the timing runs keep
both flags off so that wall times are not inflated.

`config.fixed.periter.yaml` runs order 5 only, so the extra cost is small.

The SCF configs also set `verbosity: 1`, for a different reason. Table S3's
first column, *Total Diag. iter.*, cannot come from the Timer at all: in SCF
mode gospel reaches `ParallelDavidson` through the `Eigensolver` class, whose
`solve_options` carries no `timing` key, so `davidson` runs with
`timing=False` and no `Diag. Iter.` record is ever created. The count is taken
from the per-iteration `i_iter=` markers instead, which `vprint` emits only at
`verbosity >= 1` — a `verbosity: 0` SCF run yields **zero** markers. Because
`timing` is false on that path, verbosity adds stdout and nothing else; the
measured difference against matched `verbosity: 0` runs was within noise
(37.63 vs 37.75 s, 35.62 vs 35.91 s, 60.24 vs 60.38 s).

## Turning runs into the published numbers

| Output | Where the numbers are |
|---|---|
| Figure 1, Table S1 | `<results_root>/calculation_summary_fixed.txt` |
| Figure 3, Table S3 | `<results_root>/calculation_summary_scf.txt` |
| Figure 2 | `history/**/median/history.pt` → `plot_convg_history.py --filepath <...> --plot residual` |
| Figure S1, Table S2 | `extract_per_iteration_time.py` → `Figures/plot_order5_damped_isi_per_iter.py`, `Figures/stats_order5_damped_isi.py` |

```bash
python extract_per_iteration_time.py \
    --results_root results_paper/fixed.periter \
    --order 5 --out Figures/order5_np_damped_isi_per_iteration_time.csv
```

`extract_per_iteration_time.py` reproduces the published CSV exactly when
pointed at the archived runs (1465 rows, zero numeric differences), so the same
invocation against a new run yields a drop-in replacement.

## Checking the plumbing without a GPU

`configs/smoke/` holds three miniature versions of these configs that run on a
CPU in about ten minutes.  They produce no publishable number; they verify that
a change still yields the values the figures and tables are built from.  See
`configs/smoke/README.md`.

Note that `calculation_summary_{fixed,scf}.txt` are append-only: delete
`results_paper/` before re-running a config, or the new rows are added
alongside the old ones.

## Settings taken from the paper

| Quantity | Value | Where |
|---|---|---|
| Systems | (H₂O)₁₂₈, C₆₀ tetramer, MAPbI₃ (2×2×2, 3D PBC), Vitamin B₁₂ | `systems.selected` |
| Grid spacing | 0.2 | `spacing` |
| Electronic temperature | 0 K | `temperature` |
| Virtual orbitals | +20 % | `virtual_factor: 1.2` |
| Fixed-Hamiltonian convergence | residual norm < 1e-5 | `diagonalization.fixed.tol` |
| SCF convergence | density change < 1e-5 | `scf_density_tol` |
| Mixing | Pulay, on the potential | `scf_mixing: potential` |
| ISI inner PCG iterations | 5 | `pcg_neumann` |
| Damping weight | α = 1/2 | `weight: [0.5]` |
| Repeats | median of 3 | `runs_per_combo: 3`, `seed: [0, 1, 2]` |

The inner eigensolver tolerance inside the SCF loop is adaptive
(0.1 × density difference, `vendor/gospel/gospel/scf.py`) and is deliberately
not set in these configs.

## `averaged_sum` must stay explicit

Every config states `averaged_sum` even where the current default would do.
The default flipped from `False` to `True` in `cc0d2e9` (2026-01-29), and the
`--averaged_sum` CLI flag itself only appeared in `5614bf2` (2026-01-27).
Configs written before those commits do not record which preconditioner they
ran, so their NP/Damped-NP identity lives in the code version rather than in
the config. Keeping the key explicit avoids repeating that.

## Note on comparing against the published numbers

The published measurements were taken on a single NVIDIA GH200 Grace Hopper
Superchip. Absolute times on other hardware will differ; iteration counts
should not.

One numerical difference is worth recording. `765711d` changed the
diagonalization tolerance of the *first* SCF iteration from 1.0 to 0.1
(`vendor/gospel/gospel/scf.py`). The published SCF runs predate that commit,
so SCF iteration counts obtained with the current code may differ slightly
from Table S3. Fixed-Hamiltonian runs are unaffected — they never enter the
SCF loop.
