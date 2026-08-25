# CPU smoke test

Three miniature configs that exercise the same code paths as
`configs/paper/` in about ten minutes on a CPU. They produce **no publishable
number** — the system, grid and convergence settings are all deliberately
cheap. Their purpose is to check that a change still yields the values the
paper's figures and tables are built from.

```bash
rm -rf results_smoke                     # the summaries are append-only
python repeat_test.py --config configs/smoke/config.fixed.timing.yaml
python repeat_test.py --config configs/smoke/config.fixed.periter.yaml
python repeat_test.py --config configs/smoke/config.scf.nolocking.yaml
```

| Config | Mirrors | Checks |
|---|---|---|
| `config.fixed.timing.yaml`  | `paper/config.fixed.timing.yaml`  | NP and Damped-NP stay distinguishable in `calculation_summary_fixed.txt` |
| `config.fixed.periter.yaml` | `paper/config.fixed.periter.yaml` | per-iteration `[Time: ...]` lines exist and `extract_per_iteration_time.py` parses them |
| `config.scf.nolocking.yaml` | `paper/config.scf.nolocking.yaml` | `total_diag_iter` reaches `calculation_summary_scf.txt` |

`config.fixed.history.yaml` and `config.scf.locking.yaml` differ from the ones
above only in `do_retHistory` and `locking`, so they are not duplicated here.

## What to look for

**Fixed timing** — `results_smoke/fixed.timing/calculation_summary_fixed.txt`
must contain four `neumann` rows whose `averaged_sum` differs:

```
solver_type = "neumann"  order = 0  averaged_sum = false  diag_iter_count = 50
solver_type = "neumann"  order = 0  averaged_sum = true   diag_iter_count = 50
solver_type = "neumann"  order = 2  averaged_sum = false  diag_iter_count = 24
solver_type = "neumann"  order = 2  averaged_sum = true   diag_iter_count = 27
```

If `averaged_sum` is missing, the NP and Damped-NP rows of Table S1 collapse
onto each other.

**Per-iteration** —

```bash
python extract_per_iteration_time.py \
    --results_root results_smoke/fixed.periter --order 2 --out /tmp/smoke.csv
```

must report an `[ OK ]` line for both `Damped NP` and `ISI` with a non-zero
preconditioning count. Zero rows means the `[Time: ...]` lines are gone, i.e.
`verbosity` was dropped.

**SCF** — `results_smoke/scf.nolocking/calculation_summary_scf.txt` must carry
a non-null `total_diag_iter`. Cross-check it against the log:

```bash
grep -c "i_iter=" results_smoke/scf.nolocking/logs/**/scf.log
```

A `null` here means the `i_iter=` markers are absent, which happens as soon as
the SCF config drops to `verbosity: 0`.

## Reference values

Measured on 4 CPU threads, torch 2.2.0+cpu. Timings vary; the counts should
not, since the seed is fixed.

| Run | Combos | Wall time |
|---|---|---|
| fixed.timing  | 5 | ~4 min |
| fixed.periter | 2 | ~1 min |
| scf.nolocking | 3 | ~6 min |

SCF counts: ISI `total_diag_iter = 106`, order 0 → `194`, order 2 → `118`
(`scf_iter_count` 27 / 26 / 27).

## Caveat

`calculation_summary_{fixed,scf}.txt` are **append-only**. Re-running a config
without deleting `results_smoke/` first adds a second set of rows rather than
replacing the old ones — the same applies to `results_paper/`.
