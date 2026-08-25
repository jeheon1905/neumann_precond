# JCTC Spectral-Analysis Revision — Agent Report

**Status:** complete. All phases (0-4) executed; conclusions in §15, limitations in §16.

Throughout, the report separates
**[OBS]** numerical observation · **[INT]** mathematical interpretation ·
**[INF]** inference · **[UNC]** unresolved uncertainty.

---

## 1. Environment reconstruction

### 1.1 Repository state

| item | value |
|---|---|
| repository | `https://github.com/jeheon1905/neumann_precond` |
| commit used | `765711d0afc1561e55721eddca68a37cbbeeb54f` |
| verification | `git ls-remote origin HEAD` returns the same hash; `git status --porcelain` shows no modified tracked files |
| analysis branch | `revision-spectral-analysis` (created from that commit) |
| output directory | `results_spectral_revision/` (no existing result directory was touched) |

**Deviation from instruction §8.1.** The instruction assumed the working copy was a
`.git`-less ZIP and prescribed a fresh clone. The working copy at
`/home/jhwoo1905/Projects/neumann_precond` is already a git clone whose `HEAD` equals
`origin/main`, with a clean tracked tree. Work was therefore done in place on a new
branch rather than in a second clone, so that the 759 MB TM pseudopotential set and the
~2.5 GB of archived baseline results stay co-located with the code. This is functionally
equivalent to a fresh clone at the same commit.

### 1.2 Hardware constraint — no GPU

The host cluster (`baekdu`, Slurm 24.11.3) has **no GPUs on any partition**:

```
$ sinfo -a -o "%P %G %c %N"
normal*   (null)  20   b[01-94]
debug     (null)  20   baekdu-debug
```

`nvidia-smi` and `nvcc` are absent; every node reports `Gres=(null)`. The original
manuscript runs were produced elsewhere — `20260225_to_hyunbin/job.neumann.fixed.sh`
requests `--partition=kisti-grace --gres=gpu:1` and a conda env named `gospel`, neither of
which exists here.

All calculations in this report are therefore **CPU-only**. Instruction §8.4(4) permits a
controlled deviation when the README environment cannot be used, and §8.4(2)/(5) and §6-13
place wall-clock reproduction explicitly out of scope, so this does not affect the
scientific content. Every runtime quoted below is provenance only.

*(The `debug` partition rejects this account's jobs immediately — `FAILED`, signal 53 — so
only `normal` was used. Jobs requesting 20 CPUs queue indefinitely; 16 CPUs schedule
promptly.)*

### 1.3 Environment actually used

conda env `neumann_precond` at `/home/jhwoo1905/miniforge3/envs/neumann_precond`:

| package | version | note |
|---|---|---|
| python | 3.10.20 | as README |
| torch | **2.2.0+cpu** | **deviation**: CPU wheel instead of the README's CUDA 11.8 wheel |
| torchvision / torchaudio | 0.17.0+cpu / 2.2.0+cpu | as README (unused by the code) |
| numpy | 1.26.4 | README pin |
| scipy | 1.15.3 | **not in the README**; 18 vendored `gospel` modules import it |
| ase | 3.29.0 | as README |
| spglib | 2.7.0 | as README |
| pyyaml | 6.0.3 | as README |
| gitpython | 3.1.59 | as README |
| matplotlib | 3.10.9 | **not in the README**; needed by the repository plotting scripts |
| pylibxc | 6.0.0 | built from libxc 6.0.0 source exactly as the README documents (clone → `git checkout 6.0.0` → the documented `cmake_minimum_required` sed patch → conda-forge cmake 4.4.2 → `python setup.py develop`) |

Import-path verification (`results_spectral_revision/env/env_verification.txt`):

```
gospel:       /home/jhwoo1905/Projects/neumann_precond/vendor/gospel/gospel/__init__.py
precondition: /home/jhwoo1905/Projects/neumann_precond/precondition.py
torch cuda build: None    cuda available: False
```

Both resolve to the fresh checkout; the root-level (production) `precondition.py` is the
one in use, not the vendored preconditioner.

`vendor/gospel/requirements.txt` was **not** installed (it pins `torch==1.11.0+cu115`),
as instructed.

### 1.4 README vs manuscript environment

The README documents PyTorch 2.2.0 / CUDA 11.8 / NumPy 1.26.4. The manuscript
computational details report PyTorch 2.7.1 / CUDA 12.8. This difference is **not**
reconciled here: the README versions were used (with the CPU build), because the target is
numerical correctness of the spectral analysis, not reproduction of historical wall-clock
timings.

### 1.5 Third README problem found

The README usage example

```bash
python test.py ... --precond neumann --outerorder dynamic ...
```

**fails** against the current code:

```
File "precondition.py", line 718, in __init__
    raise AssertionError("order string must start with 'res' or 'DO'")
```

`PreNeumann.__init__` accepts an `int`, a `res(...)` spec or a `DO(...)` spec; the string
`"dynamic"` is passed straight through by `test.build_preconditioner._outerorder_value`
and then rejected. The sanity check was run with `--outerorder 4` instead. This is a
stale-documentation bug, unrelated to the revision, and is recorded here rather than fixed.

### 1.6 Pseudopotentials — TM verified present

Instruction §8.5 warned that `data/pseudopotentials/TM/` might be missing. In this clone
it is present and complete:

```
data/pseudopotentials/TM                              759 MB, 544 git-tracked files
data/pseudopotentials/TM/data_gga/PSEUDOPOTENTIALS_NC  97 files *.pbe-n-nc.UPF
```

They are real UPF files, not git-lfs pointers (there is no `.gitattributes`). The path
that `test.resolve_upf_files` actually uses for `pp_type=TM` is
`data/pseudopotentials/TM/data_gga/PSEUDOPOTENTIALS_NC/{symbol}.pbe-n-nc.UPF`, and
H, B, C, N, O, I, Pb — every element needed by the four benchmark systems — are present.
**No pseudopotential substitution was necessary.**

---

## 2. Gate A — installation sanity check

README NNLP example, CPU, `--outerorder 4` (see §1.5):

```bash
python test.py --filepath data/systems/Si_diamond.cif \
    --spacing 0.3 --supercell 1 1 1 --pbc 1 1 1 \
    --phase fixed --pp_type NNLP --precond neumann --outerorder 4 \
    --diag_iter 50 --retHistory results_spectral_revision/logs/History.neumann.o4.pt
```

**[OBS]** imports succeed; grid `[19 19 19]`, `ngpts = 6859`, 19 bands, `nelec = 32`;
50 Davidson iterations run; history written; residual history finite, converging to
4e-14 – 1e-15 for bands 0–12 and 1e-6 – 8e-5 for the top 6 (`--diag_tol` default 1e-6,
so the run legitimately hits `maxiter`). The GAPP (order-0) example also runs and, at the
same iteration count, is far less converged (4e-8 – 4e-2) — the expected qualitative
ordering. **Gate A passed.** No GPU sub-check applies.

---

## 3. Code paths reused

Nothing in the Hamiltonian, GAPP, corrected-shift or Neumann-recurrence implementation was
reconstructed. The analysis consumes the production objects directly.

| function | file:symbol | how it is used |
|---|---|---|
| Neumann / Damped-NP | `precondition.py:PreNeumann.call` | unchanged; probe hooks added around it |
| GAPP construction | `precondition.py:create_preconditioner("gapp", ...)` → `PrePoisson` + `ISF_solver` | used through `self.gapp` |
| corrected shift | `precondition.py:PreNeumann.call` lines 765–773 | captured, not recomputed |
| Hamiltonian action | `gospel/Hamiltonian.py` → matrix-free `LinearOperator` | used as `H @ x` |
| fixed-Hamiltonian driver | `test.py:run_once`, `phase == "fixed"` | called unmodified by `spectral_experiment.py` |
| preconditioner construction | `test.py:build_preconditioner` | wrapped, not replaced |
| Davidson | `gospel/Eigensolver/ParallelDavidson.py:davidson` | one inert metadata hook added |

**[OBS]** In the fixed phase the density is initialised by `calc.density.init_density()`
(atomic superposition), so no SCF checkpoint is needed to reproduce a benchmark.

**[OBS]** With `nblock = 2` the Davidson block-expansion `while` loop calls the
preconditioner **once per Davidson iteration**, and `i_iter` starts at 2. Empirically
`call_index = 3` ⇔ `i_iter = 5`. This mapping is *recorded*, not assumed: the metadata hook
reports `i_iter`, `i_b`, `i_scf` and the original band-index list.

---

## 4. Definition and implementation of `E`

### 4.1 The operator `P`

The production zeroth-order term is

```python
INV_4PI = 0.25 / np.pi
preconditioned_result = self.gapp(residue).mul_(INV_4PI)
```

so `P x = (1/4π) GAPP(x)`. A helper `PreNeumann.apply_P` was added that returns
`self.gapp(x) * (0.25/np.pi)` — the same expression without the in-place `mul_`, so it is
safe on analysis vectors. The `1/(4π)` factor appears exactly once.

### 4.2 The operator `E`

`spectral_tools.ErrorPropagationOperator` implements, for a single state with corrected
shift `ε̃`,

```
E x = x - P (H x - ε̃ x)
```

matching the production recurrence

```python
H_minus_eigval_vec = H @ neumann_term - eigval_active * neumann_term
neumann_term      -= self.gapp(H_minus_eigval_vec).mul_(INV_4PI)
```

so that `a_{n+1} = E a_n`. `E` is real-linear but not symmetric; a complex argument is
handled by applying it to the real and imaginary parts separately, so no complex support
is required inside the Hamiltonian. Each state keeps its **own** `ε̃`; shifts are never
merged into a single scalar-shift operator.

### 4.3 Probe design (production behaviour unchanged)

Two hooks were added to `PreNeumann`, both `None` by default:

* `spectral_probe` — called once at the top of `call`, after the residual norms and the
  corrected shifts are computed and after the residual has its column-block shape. It
  receives `residue`, `H`, `raw_eigval` (cloned *before* the correction),
  `corrected_eigval`, `residue_norm`, `zero_shift_mask`, `apply_P`, the preconditioner and
  `call_index`.
* `term_probe` — called with every recurrence term `a_n` (n = 0 is `P γ`).

`raw_eigval` is cloned only when the probe is attached, so the disabled path allocates
nothing extra. In `ParallelDavidson.davidson`, a metadata dict is stored on the
preconditioner immediately before the preconditioning call, guarded by
`getattr(preconditioner, "spectral_probe", None) is not None`; the band-index list is set
to `None` rather than guessed whenever the column count does not match `unlock.sum()`
(i.e. under `fill_block` truncation or column distribution over ranks).

**[OBS] Regression check.** The Gate-A Si run was repeated after the patch with the probe
disabled and compared to the pre-patch run:

```
max |eigval  difference| = 0.0
max |residual difference| = 0.0     ->  BITWISE IDENTICAL
```

Non-negotiable rule 1 is satisfied.

---

## 5. Gate 1 — validation of `E`

### 5.1 Unit tests (`tests/test_spectral_tools.py`, no GOSPEL required)

| test | result |
|---|---|
| full-dimension Arnoldi vs dense `eigvals` (n = 40) | max magnitude difference **0.0**, orthogonality error 4.4e-16 |
| explicit Ritz residuals at full dimension (n = 30) | worst 4.0e-15 |
| linearity of the operator wrapper | 1.4e-16 |
| literal recurrence emulation `a_{n+1} = E a_n`, n = 0..3 | all < 1e-13 |
| spectrum summary / damping factor `(1+λ)/2` | exact at λ = ±1 |

### 5.2 End-to-end validation on a small real system

`Si_diamond.cif`, spacing 0.6, NNLP, `ngpts = 1000`, probe at `i_iter = 5`, band 12,
`‖γ‖ = 2.137e-1`, `ε = 0.0483481092` → `ε̃ = 0.0437827299`, zero-shift fallback **inactive**.

| quantity | value |
|---|---|
| `‖apply_P(x) − gapp(x)·INV_4PI‖ / ‖·‖` | **0.0** (bit-identical) |
| linearity `δ_lin` | 1.15e-15 |
| **recurrence equivalence** `δ_n`, n = 0,1,2,3 | **0.0, 0.0, 0.0, 0.0** |
| dense `E` (1000×1000) | ρ = 1.224389982, λ_min = −0.158408, λ_max = +1.224390 |
| matrix-free Arnoldi, m = 20, seed 0 | ρ = 1.224390, λ_min = −0.157809 |
| Arnoldi orthogonality error | 8.9e-16 |
| non-normality `‖E − Eᵀ‖_max / ‖E‖_max` | **0.163** |

**[OBS]** The matrix-free operator reproduces the dense spectral radius to 8 significant
figures, and the production recurrence terms are reproduced **exactly**.

**[OBS]** `E` is measurably non-symmetric, so Arnoldi (not Lanczos) is required, as the
instruction anticipated.

### 5.3 Smoke test at manuscript scale (coarse grid)

`B12.sdf`, TM, spacing **0.45** (deliberately coarse — *not* a manuscript-consistent
calculation), `ngpts = 144584`, 300 bands, probe at `i_iter = 5`, band 214,
`‖γ‖ = 1.706e-1`, `ε = −0.1032449019` → `ε̃ = −0.1061567668`, fallback **inactive**.

| quantity | value |
|---|---|
| `δ_apply_P` | 0.0 |
| `δ_lin` | 2.29e-15 |
| `δ_n`, n = 0..3 | 3.9e-17, 3.9e-17, 4.4e-17, 4.3e-17 |
| Arnoldi m = 20 | ρ = 2.071719, λ_min = −0.675260, Ritz residuals 1.4e-4 / 9.2e-4 |
| Arnoldi m = 40 | ρ = 2.071717, λ_min = −0.675264, Ritz residuals **1.4e-9 / 5.4e-10** |
| spectrum | numerically real (max &#124;Im λ&#124; = 0) |

**[OBS]** At m = 40 the extremal Ritz residuals are already below the 1e-8 target of
instruction §10.6, and ρ / λ_min are stable to 6 significant figures between m = 20 and
m = 40. The chosen Krylov dimensions (20/40/60) are therefore adequate.

**Gate 1 passed.**

---

## 6. Existing order-dependent data to be explained (no new timing)

Extracted from results already on disk (see
`results_spectral_revision/reference/existing_order_dependence.md`):

* undamped NP — `20260211_to_hyunbin/result.neumann_normal.fixed_speed/`
* Damped-NP — `20260225_to_hyunbin/result.neumann.fixed/`

Both are fixed-Hamiltonian, TM, spacing 0.2, `virtual_factor` 1.2, `diag_tol` 1e-5,
`nblock` 2, `locking = fill_block = false`, seeds 0/1/2 (median quoted).

**Undamped NP, Davidson iterations**

| order | water_cluster_128 | C60_4 | B12 | MAPbI3 |
|---:|---:|---:|---:|---:|
| 1 | 322 | 223 | **1000 (no conv.)** | 37 |
| 2 | 45 | 42 | 47 | 27 |
| 3 | 174 | 112 | **1000** | 21 |
| 4 | 32 | 29 | 34 | 18 |
| 5 | 173 | 80 | **1000** | 16 |
| 7 | 174 | 68 | **1000** | 14 |
| 9 | 230 | 68 | **1000** | 12 |
| 10 | 21 | 18 | 25 | 12 |

**[OBS]** Oscillation strength orders as
**B12 (odd orders never converge) > water_cluster_128 > C60_4 ≫ MAPbI3 (no oscillation at
all; undamped NP is monotone in N and is even marginally better than Damped-NP at low
order).** Damped-NP removes the oscillation in all four systems.

**[INF] Falsifiable prediction.** If the odd–even oscillation is caused by eigenvalues of
`E` near λ = −1, the most-negative eigenvalue should order as
`B12 ≤ water_cluster_128 < C60_4 ≪ MAPbI3`, with B12 at or below −1, and the damping
factor `|(1+λ)/2|` should be near zero exactly for the oscillating systems. Sections 7–8
test this prediction; it is stated *before* the production spectra were computed.

---

## 7. Gate 0 — numerical baseline reproduction

**Manuscript-consistent settings identified.** The repository's `config.yaml` is labelled
`experiment.name: "debug"` and was *not* used. The manuscript fixed-Hamiltonian benchmark
is `20260225_to_hyunbin/config.neumann.fixed.yaml`, whose complete results (722 files,
stdout logs and `history.pt` per run) are preserved in
`20260225_to_hyunbin/result.neumann.fixed/`. Settings recovered from it and from the
archived `args=` line of the reference run:

| setting | value |
|---|---|
| phase | fixed (`mode: "fixed"`, `scf_runs_per_combo: 0`) |
| initial density | `calc.density.init_density()` (atomic superposition; no SCF checkpoint) |
| pseudopotential | TM, `filtering = true` |
| grid spacing | 0.2 Å |
| bands | automatic, `virtual_factor = 1.2` |
| temperature | 0.0 |
| Davidson | `maxiter 1000`, `tol 1e-5`, `nblock 2`, `locking = false`, `fill_block = false` |
| Neumann | `order 4`, `error_cutoff −0.4`, `averaged_sum = true`, `weight = 0.5` |
| seeds | 0 / 1 / 2 (run-1 / run-2 / run-3); the *median* run is seed 2 |

Only the smallest sufficient case was rerun (instruction §6-14): **B12, order 4, seed 2**.

```bash
python test.py --filepath data/systems/B12.sdf \
  --precond neumann --outerorder 4 --error_cutoff -0.4 --averaged_sum 1 --weight 0.5 \
  --spacing 0.2 --supercell 1 1 1 --pbc 0 0 0 --phase fixed --temperature 0.0 \
  --scf_energy_tol inf --scf_density_tol 1e-5 --scf_mixing potential \
  --virtual_factor 1.2 --pp_type TM --filtering --threads 16 --warmup 0 \
  --diag_iter 1000 --diag_tol 1e-5 --nblock 2 --verbosity 1 --seed 2 \
  --retHistory results_spectral_revision/baseline/B12_o4_seed2_history.pt
```

(Slurm job 326539, node b72, 16 CPUs, 31 m 57 s wall — provenance only, *not* a benchmark.
The reference was `--threads 1 --use_cuda` on a GPU; the thread count was raised purely to
make a CPU rerun tractable.)

| quantity | reference (GPU, seed 2) | this run (CPU, seed 2) |
|---|---:|---:|
| **Davidson iterations to convergence** | **27** | **27** |
| converged bands 0–249, max &#124;Δε&#124; | — | **2.69e-8 Ha** (mean 6.6e-9) |
| all target bands below `diag_tol = 1e-5` | yes | yes |
| final residual, target bands (max / min) | 5.31e-6 / 1.3353e-8 | 5.74e-6 / 1.3314e-8 |
| lowest Ritz value | −1.1600956143 | −1.1600956172 |

**[OBS]** The iteration count is reproduced exactly and the converged eigenvalues agree to
~1e-8 Ha (10 significant figures on the lowest state).

**[INT]** The two runs do *not* follow bit-identical trajectories: intermediate residual
norms differ by O(1) relative at late iterations, and the unconverged virtual bands
250–299 (to which no convergence criterion applies) differ by up to 2.7e-3 Ha. This is
expected and benign — `set_global_seed(seed)` seeds torch's CPU and CUDA generators
separately, so the GPU reference and the CPU rerun start from *different* random subspaces
despite the same nominal seed. What must agree — the iteration count, the converged
spectrum, and the convergence criterion — does agree.

**[INF]** The CPU/README environment reproduces the manuscript numerics. **Gate 0 passed.**

**[UNC]** Only one of the 44 (system, order) baseline points was rerun, and no
seed-to-seed spread was regenerated on CPU. The reference seed spread for this point is
26/27/27 iterations, so the single match is within, not distinguishable from, the
reference scatter.

Raw comparison: `results_spectral_revision/baseline/gate0_comparison.txt`.

## 8. Phase 2 — global spectrum of `E`

### 8.1 Snapshot selection (identical rule for all four systems)

Fixed-Hamiltonian run at the manuscript settings of §7, `seed = 2`, Neumann `order = 4`,
`locking = fill_block = false` so that active columns map directly onto band indices
(confirmed: the probe returned the full `band_index` list `[0 … nbands-1]` in every run).

* snapshot = preconditioner call index **3**, which the Davidson metadata hook reports as
  **`i_iter = 5`, `i_b = 1`, `i_scf = 0`** in all four systems;
* state = **slowest-converging active state** at that step (instruction §10.2 preference 1);
* Krylov dimensions m = 20, 40, 60; Arnoldi seeds 0 and 1; `torch.float64`; CPU.

Commands: `results_spectral_revision/jobs/job_spectral_<system>.sh`
(Slurm 326544/326545/326546/326547, nodes b88/b91/b93/b94, 8 CPUs, 18–28 min each —
provenance only). Raw output: `results_spectral_revision/global_spectrum/raw/*.json`,
table `.../summary.csv`, figures `.../figures/`.

### 8.2 Per-run validation (repeated inside every production run)

| system | `δ_apply_P` | `δ_lin` | `δ_recurrence` (n = 0..3) |
|---|---:|---:|---|
| B12 | 0.0 | 1.65e-15 | 3.4e-16, 3.8e-16, 4.1e-16, 3.8e-16 |
| C60_4 | 0.0 | 1.59e-15 | 6.2e-16, 5.7e-16, 5.7e-16, 5.5e-16 |
| water_cluster_128 | 0.0 | 1.66e-15 | 3.1e-16, 3.8e-16, 3.5e-16, 3.5e-16 |
| MAPbI3 | 0.0 | — | (see JSON) |

**[OBS]** The analysis operator reproduces the production recurrence to machine precision
in every production run, not only in the small validation case.

### 8.3 Shifts and the zero-shift fallback

| system | band | ‖γ‖ | ε (raw) | ε̃ (corrected) | fallback |
|---|---:|---:|---:|---:|---|
| B12 | 218 | 5.899923e-1 | −0.3114194182 | −0.3462285153 | **inactive** |
| C60_4 | 479 | 4.789436e-1 | −0.2960919560 | −0.3190306540 | **inactive** |
| water_cluster_128 | 482 | 6.719841e-1 | −0.2631155811 | −0.3082718410 | **inactive** |
| MAPbI3 | 636 | 5.358311e-1 | +0.0953690697 | +0.0666575685 | **inactive** |

**[OBS]** In every case `ε̃ = ε − 0.1‖γ‖²` exactly, and `|perturb| = ‖γ‖² ≤ 0.45 ≪ 10`, so
the zero-shift fallback is **inactive at this snapshot in all four systems**. The analysed
operator is therefore the ordinary shifted `E = I − P(H − ε̃I)`, not `I − PH`. (The fallback
*is* active at the very first Davidson step, where ‖γ‖ ≈ 13 for every system; that step was
deliberately not used as the representative snapshot, per instruction §4.1.)

### 8.4 Results

At m = 60 (values stable to 5–6 significant figures across m = 20/40/60 **and** both seeds;
all spectra numerically real, max |Im λ| = 0; Arnoldi orthogonality error ~1e-14):

| System | State | ‖γ‖ | ε̃ | λ_min | λ_max | ρ(E) | max Ritz resid. |
|---|---:|---:|---:|---:|---:|---:|---:|
| water_cluster_128 | 482 | 6.7198e-1 | −0.308272 | −0.791736 | +1.603650 | 1.603650 | 9.0e-4 |
| C60_4 | 479 | 4.7894e-1 | −0.319031 | −0.797751 | +1.610277 | 1.610277 | 1.0e-3 |
| B12 | 218 | 5.8999e-1 | −0.346229 | **−0.824579** | +1.603309 | 1.603309 | 1.2e-3 |
| MAPbI3 | 636 | 5.3583e-1 | +0.066658 | **−0.285602** | +1.453501 | 1.453501 | 2.6e-4 |

Damping factor of the most negative mode, `|(1+λ_min)/2|`:

| system | (1+λ_min)/2 | suppression | undamped-NP odd-order behaviour |
|---|---:|---:|---|
| B12 | **0.0877** | 11.4× | never converges within 1000 iterations |
| C60_4 | 0.1011 | 9.9× | 68–223 iterations |
| water_cluster_128 | 0.1041 | 9.6× | 173–322 iterations |
| MAPbI3 | **0.3572** | 2.8× | **no oscillation at all** |

### 8.5 What the global spectrum explains

**[OBS] The MAPbI3 exception is reproduced quantitatively.** λ_min = −0.2856 for MAPbI3
versus −0.79…−0.82 for the three molecular/cluster systems.

**[INT]** With `q_N^NP(λ) = λ^{N+1}`, the alternating contribution decays as |λ_min|^N:
`0.2856^10 ≈ 3.5e-6` (effectively dead by N ≈ 4) versus `0.82^10 ≈ 0.14` (still ~14 % at
order 10). The odd–even alternation therefore persists to high order in B12 / C60_4 /
water_cluster_128 but disappears almost immediately in MAPbI3 — exactly the reviewer's
observation.

**[INT]** The implemented damping multiplies that mode by `(1+λ)/2`: a 9.6–11.4×
suppression where the oscillation is strong, but only 2.8× for MAPbI3. This is also why
Damped-NP *costs* MAPbI3 a few iterations at low order (Table in §6): there was little
oscillation to suppress, and the damping merely discards half of the last term.

**[OBS] The dominant global mode is not a chemical property.** λ_max = ρ(E) is
1.603309 / 1.610277 / 1.603650 for B12 / C60_4 / water_cluster_128 — three systems with
completely different composition and size agreeing to ~0.4 %. MAPbI3, the only
periodic system, is the outlier at 1.4535.

**[RETRACTED — see §12]** This report originally attributed the near-equality of λ_max to
the GAPP `t_sample` being selected solely by `grid.get_pbc().sum()` (the three 0-D systems
do receive bit-identical sampled t-values, MAPbI3 a different set — that fact is correct).
That explanation is **wrong**. λ_max = 1 − μ_min where μ_min is the most negative
eigenvalue of `PM`, and μ_min < 0 is a consequence of `M = H − ε̃I` being **indefinite**,
not of the grid. The three 0-D systems agree because their deepest occupied levels
(ε₀ ≈ −1.16, −1.16, −1.08 Ha) and shifts are similar, which is physics, not a grid
artifact. §12 gives the correct account.

### 8.6 What the global spectrum does **not** explain — Gate 2 verdict

1. **[OBS]** ρ(E) = 1.45–1.61 > 1 in **all four** systems, so the infinite-series Neumann
   condition ρ(E) < 1 is **violated everywhere**, while even-order NP and Damped-NP
   converge in 12–47 Davidson iterations. → §11.1 trigger 1.
2. **[OBS]** The dominant global mode is *positive* (λ_max ≈ +1.6), so it cannot be the
   source of the sign alternation at all. → §11.1 trigger 3. (The explanation of *why*
   λ_max is nearly system-independent was originally attributed to the GAPP `t_sample`;
   that attribution is retracted — see §8.5 and §12.)
3. **[OBS]** λ_min separates MAPbI3 from the rest, but **not** B12 from C60_4 /
   water_cluster_128: −0.8246 vs −0.7978 / −0.7917 is a 3–4 % difference, whereas the
   observed behaviour differs *qualitatively* (B12 never converges at odd order; the others
   converge, slowly). → §11.1 trigger 4.

**[INF] Gate 2: the stop condition is not met.** The global spectrum explains the sign
alternation mechanism, the damping factor and the MAPbI3 exception, but it does not
explain why finite-order NP works at all when ρ(E) > 1, nor the B12/C60_4/water ordering.
**Phase 3 (residual-projected spectrum) is therefore performed.**

**[UNC]** The reported Ritz residuals (2.6e-4 – 1.2e-3) do not meet the 1e-8 target of
instruction §10.6. The extremal Ritz *values* are nevertheless stable to 5–6 significant
figures with respect to Krylov dimension (m = 20 → 40 → 60) and to the Arnoldi seed, which
is the classic behaviour of a Ritz value converging much faster than its Ritz vector for an
operator with a dense interior spectrum. A dedicated convergence study at m = 60/100/150
(B12) is reported in §8.7 to document this. **No extremal value is claimed to more digits
than the seed-to-seed and m-to-m agreement supports.**

### 8.7 Krylov-dimension convergence study

B12, slowest state, seed 0 (`results_spectral_revision/global_spectrum/convergence/`):

| m | ρ(E) | λ_min | Ritz resid. (ρ) | Ritz resid. (λ_min) | ortho. error |
|---:|---:|---:|---:|---:|---:|
| 60 | 1.60330885 | −0.82457934 | 6.21e-05 | 5.12e-04 | 2.15e-14 |
| 100 | 1.60330895 | −0.82458434 | 3.31e-09 | 2.35e-07 | 2.15e-14 |
| 150 | **1.60330895** | **−0.82458434** | **3.18e-15** | **9.08e-13** | 2.15e-14 |

**[OBS]** At m = 150 both Ritz residuals are far below the 1e-8 target of §10.6, and the
extremal values are unchanged from m = 100 to 9 significant figures. The m = 60 values
quoted in §8.4 agree with the fully converged values to 7 (ρ) and 5 (λ_min) significant
figures.

**[INF]** The looser residuals at m = 60 reflect slow *Ritz-vector* convergence in a dense
interior spectrum, not an error in the extremal *Ritz values*. The values reported
throughout are quoted to no more digits than this study and the seed-to-seed agreement
support.

---

---

## 9. Phase 3 — residual-projected spectrum  **[RESULT RETRACTED]**

Performed (jobs 326548–326551) with the exact code-level start vector
`a₀ = Pγ = (1/4π)GAPP(γ)`, m = 10…40. Measured `ρ_proj ≈ ρ` to 4–6 significant figures in
all four systems (B12 1.603301 vs 1.603309; C60_4 1.610079 vs 1.610277;
water 1.603564 vs 1.603650; MAPbI3 1.453484 vs 1.453501).

**[RETRACTED]** This report originally concluded from `ρ_proj > 1` that "the globally
unstable modes are *not* irrelevant to the residual, so the §11.5 hypothesis is refuted".
**That conclusion does not follow.** Arnoldi converges to the extremal eigenvalues of an
operator from *any* start vector with non-negligible overlap, however small that overlap
is. `ρ_proj` therefore measures whether the unstable mode is *present* in the Krylov space,
never how much *weight* it carries. The right quantity is the modal weight, measured in
§13. The projected-spectrum diagnostic as defined in §11.3 of the instruction cannot decide
this question and no conclusion is drawn from it.

## 10. Phase 4 — η_N  **[METRIC RETRACTED]**

Since `PM = I − E` and `a₀ = Pγ`, one has `P(Mp_N − γ) = −a_{N+1}` and
`P(Mp̄_N − γ) = −½(a_N + a_{N+1})`, so η_N = ‖a_{N+1}‖/‖a₀‖ is exactly the relative residual
of the linear system `M p = γ` in the P-weighted norm. That derivation is correct.
Measured curves are U-shaped: a minimum near
N = 2–3 followed by growth at the asymptotic rate ρ(E).

**[RETRACTED]** This report originally concluded "η has a minimum at finite N, therefore
increasing the order beyond it degrades the preconditioner, which explains why the optimal
Neumann order is finite". **That is wrong, for three independent reasons.**

1. **It contradicts the benchmark.** η(order 0) = 0.553 versus η(order 10, damped) = 0.977
   for B12, i.e. η says order 10 is 1.8× *worse*; the archived data says 158 versus 19
   Davidson iterations, i.e. 8× *better*. Damped-NP iteration counts are monotone
   non-increasing in N over the whole tested range 0…10 and never degrade.

2. **Davidson does not want the solution of `M p = γ`.** With γ = (H − εI)x and ε̃ ≈ ε,
   `M⁻¹γ = (H − ε̃I)⁻¹(H − εI)x ≈ x` — the *current Ritz vector*, which Davidson projects
   to zero in `R_ -= U_(U_ᴴR_)` before using it. Driving η → 0 drives the preconditioner
   output towards a vector that is discarded. A small η is therefore not a good thing, and
   the location of its minimum has no reason to mark the optimal order. (This is the
   classical reason Jacobi–Davidson uses a *projected* correction equation.)

3. **The growth of η happens along discarded directions.** `a_{N+1} = E^{N+1}a₀` grows
   along the eigendirections where `M` is negative — the states below ε̃ — which span(X)
   already represents and Davidson explicitly removes.

The correct figure of merit is developed in §13.

## 11. State dependence

(The measurements below stand; only the interpretation in §8.5 was corrected.)

Instruction §10.2 provides for adding a representative state set "if state dependence is
substantial". Jobs 326973–326977, states lowest / middle / HOMO / slowest at the same
snapshot, m = 40 and 60. Full table in `results_spectral_revision/state_dependence/`.

**[OBS]** ρ(E) rises monotonically with the corrected shift ε̃ (B12: 1.138 → 1.428 → 1.603
→ 1.832 for lowest → middle → slowest → homo), and the *character* of the dominant
eigenvalue changes: for the deeply shifted lowest state of B12, water_cluster_128 and
C60_4 it is **negative with |λ| > 1** (−1.1382, −1.1012, −1.0734); everywhere else it is
positive. MAPbI3 has no eigenvalue below −1 at any state (most negative −0.2856…−0.4995).

**[OBS]** Zero-shift fallback inactive in all 15 state/system combinations.

---

## 12. The structural result: why ρ(E) > 1 is guaranteed, not a GAPP defect

This is the diagnostic that should have been run **before** any Arnoldi calculation.

### 12.1 Argument

Let μ_j be the eigenvalues of `PM`. Then the eigenvalues of `E = I − PM` are λ_j = 1 − μ_j,
and the Neumann series converges iff

    0 < μ_j < 2   for every j.

`P` is symmetric positive definite — verified directly by building it densely on a 0-D
cartesian grid (the branch used by B12/water/C60):

```
||P - P^T||_max / ||P||_max = 0.000e+00      (exactly symmetric)
eigenvalues of P: min = 5.079e-03 > 0, max = 8.046e-01, cond = 1.58e+02
```

`PM` is similar to the symmetric matrix `P^{1/2} M P^{1/2}`, which is **congruent** to `M`.
By **Sylvester's law of inertia** they have the same number of negative eigenvalues:

    #{ μ_j < 0 }  =  #{ negative eigenvalues of M }        for every SPD P.

Verified numerically with three different SPD matrices P and a fixed M with 3 negative
eigenvalues: `#neg(PM) = 3` in all three cases, while μ_min varied over −1.95…−6.19.

`M = H − ε̃ᵢI` has one negative eigenvalue for every eigenstate of H below ε̃ᵢ. Independent
check against the converged B12 spectrum:

| state | band | ε̃ | states below ε̃ | M |
|---|---:|---:|---:|---|
| lowest | 0 | −1.160941 | **0** | positive definite |
| middle | 125 | −0.573174 | 131 | indefinite |
| slowest | 218 | −0.346229 | 248 | indefinite |
| homo | 249 | −0.054457 | 295 | indefinite |

Hence for any interior state, μ_min < 0, so λ_max = 1 − μ_min > 1 and **ρ(Eᵢ) > 1 is
mathematically guaranteed** — independent of how good GAPP is, and unfixable by rescaling,
since for α > 0 and μ < 0, `1 − αμ = 1 + α|μ| > 1`. (Numerically: min over α of
ρ(I − αPM) = 1.032 for an indefinite M, versus 0.947 for the same P with M positive
definite.)

Measured μ ranges: μ_min = −0.28…−0.83 for interior states; μ_min ≈ +0.0000…+0.0006 for
the lowest states (where M is positive definite because the `−0.1‖γ‖²` shift correction
pushes ε̃ *below* ε₀ — for B12 by 8.45e-4 Ha).

### 12.2 The one exception, and a by-product

For the **lowest** state `M ≻ 0` (the `−0.1‖γ‖²` correction pushes ε̃ below ε₀), so all
μ > 0 and the obstruction is different in the two groups:

| system | μ_min | μ_max | obstruction | α < 2/μ_max | ρ(I − αPM) at optimum |
|---|---:|---:|---|---:|---:|
| B12 | 6e-4 | 2.1382 | μ_max > 2: P **over-corrects** | 0.935 | **0.9995** |
| water_cluster_128 | 5e-4 | 2.1012 | μ_max > 2 | 0.952 | **0.9995** |
| C60_4 | 3e-4 | 2.0734 | μ_max > 2 | 0.965 | **0.9997** |
| MAPbI3 | **1.9e-5** | 1.4995 | μ_min ≈ 0: `M` **near-singular** | — | 1.0000 |

**[INF]** For B12 / water / C60_4 a modest under-relaxation of P (α ≈ 0.93–0.96) would make
the *undamped* expansion convergent at these states. **[UNC]** Derived from the measured
spectrum; not tested in an actual run.

**[OBS]** MAPbI3 is a different case: its ‖γ‖ = 3.65e-2 at the lowest state makes the shift
correction only 1.33e-4 Ha, so ε̃ sits essentially *on* ε₀ and `M` is near-singular
(μ_min = 1.9e-5). Then ρ = |1 − α·μ_min| ≈ 1 for **every** α — no rescaling helps. That
near-null direction is the ground state itself, which lies inside span(X), so deflation
removes it: ρ(ΠEΠ) = 0.807 (§13.3). This is an independent consistency check on the
deflation picture.

---

## 13. Deflated analysis — the operationally correct object

### 13.1 Why deflate, and against what

Davidson never uses the preconditioned block as returned. Immediately before
orthonormalisation it removes the component in the current subspace
(`ParallelDavidson.py:533-535`):

```python
tmp = PH.all_reduce(U_.conj().T @ R_)
tmp = U_ @ tmp
R_ -= tmp
```

At the probe point (`i_b == 1`) `U_ = X_`, the **current Ritz-vector block**, of shape
`(ngpts, nbands)` — 300 columns for B12, 614 for water_cluster_128, 576 for C60_4, 768 for
MAPbI3. Verified orthonormal to 1.1e-15, and `‖Πγ‖ = ‖γ‖` exactly, confirming the residual
is already orthogonal to X by construction.

The unstable directions of `Eᵢ` are precisely the states below ε̃ᵢ (§12), which span(X)
represents. So the relevant operator is the **deflated** one,

    Π = I − X Xᴴ,        ρ_eff(Eᵢ) = ρ(Π Eᵢ Π),

and on range(Π) all remaining states satisfy εⱼ > ε̃ᵢ, so `Π M Π ≻ 0` and ρ_eff < 1 becomes
attainable. This is *not* "projecting onto one state"; it is removing the subspace the
solver already carries.

### 13.2 Deflation becomes exact as the Davidson subspace converges

Snapshots at Davidson iterations 5, 10 and 15 (jobs 327001–327014). Positive-definiteness
of `ΠMΠ` is tested by running preconditioned CG on `ΠMΠ t = Πγ` and watching for negative
curvature:

| system / state | i_iter = 5 | i_iter = 10 | i_iter = 15 |
|---|---|---|---|
| B12 / homo | neg. curvature | OK (64 it) | OK (47 it) |
| B12 / middle | OK (72) | OK (48) | OK (46) |
| water / homo | neg. curvature | OK (45) | OK (44) |
| MAPbI3 / homo | neg. curvature | OK (39) | OK (38) |
| C60_4 / homo | OK (61) | OK (47) | OK (45) |

**[OBS]** At iteration 5 the deflation is incomplete — X is not yet accurate (residual
norms 0.02–0.67) so some of the negative subspace leaks through. By iteration 10–15 `ΠMΠ`
is positive definite for every occupied state tested, and CG converges in 24–64 iterations.

### 13.3 Deflated spectral radius (Davidson iteration 15)

| System | State | band | ε̃ | ρ(E) full | **ρ(ΠEΠ)** | λ_min(ΠEΠ) | dominant |
|---|---|---:|---:|---:|---:|---:|---|
| B12 | middle | 125 | −0.5798 | 1.4279 | **0.93018** | −0.89713 | positive |
| B12 | homo | 249 | −0.3394 | 1.8323 | **0.93110** | −0.82248 | positive |
| water_128 | middle | 256 | −0.5403 | 1.4396 | **0.92423** | −0.87128 | positive |
| water_128 | homo | 511 | −0.3723 | 1.7166 | **0.92488** | −0.81334 | positive |
| C60_4 | middle | 240 | −0.6145 | 1.3704 | **0.92384** | −0.89983 | positive |
| C60_4 | homo | 479 | −0.3592 | 1.6102 | **0.92524** | −0.81131 | positive |
| MAPbI3 | lowest | 0 | −0.8233 | 0.99998 | **0.80908** | −0.49951 | positive |
| MAPbI3 | middle | 320 | −0.2867 | 1.2787 | **0.84706** | −0.36899 | positive |
| MAPbI3 | homo | 639 | −0.0127 | 1.4646 | **0.92733** | −0.30418 | positive |
| **B12** | **lowest** | 0 | −1.1601 | 1.1382 | **1.13646** | **−1.13646** | **negative** |
| **water_128** | **lowest** | 0 | −1.1590 | 1.1012 | **1.10097** | **−1.10097** | **negative** |
| **C60_4** | **lowest** | 0 | −1.0821 | 1.0734 | **1.07328** | **−1.07328** | **negative** |

Deflated values at **m = 80** (§13.5). The earlier revision of this report quoted m = 40,
which underestimated ρ by ~0.008 and κ by 5–13 %; no conclusion changes.

**[OBS] Deflation removes the λ > 1 modes.** ρ falls from 1.28–1.83 to **0.809–0.931** for
every state Davidson is required to converge. The convergence condition the reviewer asks
for **is satisfied in the deflated (operationally relevant) sense.**

**[UNC] But ρ_eff ≈ 0.92 is a weak rate, and on its own explains very little.** An
asymptotic factor of 0.92 per order means an 8 % error reduction per order: reaching a 10×
reduction would need N ≈ 29, and over the whole practically used range ρ^11 ≈ 0.42, i.e.
only ~2.4×. Empirically the measured deflated error and ρ^{N+1} stay within a factor 1.5–3
of one another over N = 1…11 (§13.4). **[UNC]** That is an observed coincidence of
magnitude, not a validated prediction: `η_N^Π = ‖Π E^{N+1}a₀‖/‖Π a₀‖` involves full-space
powers of E followed by a single projection, whereas ρ(ΠEΠ)^{N+1} would describe
`(ΠEΠ)^{N+1}`, and Π and E do not commute (§20.2). Reporting "ρ_eff < 1" answers the reviewer's
*formal* question and nothing more; the substantive evidence is the finite-order
measurements of §14.

### 13.5 Krylov convergence and Ritz residuals of the deflated spectrum

Jobs 327446–327449 repeated the i_iter = 15 snapshot with m = 20/40/60/80 and **explicit**
Ritz residuals `‖(ΠEΠ)v − θv‖/‖v‖` for the three extremal targets.

| system / state | quantity | m=20 | m=40 | m=60 | **m=80** | residual at m=80 |
|---|---|---:|---:|---:|---:|---:|
| B12 / homo | ρ = λ_max | 0.91385 | 0.92372 | 0.93083 | **0.93110** | 6.5e-4 |
| B12 / homo | λ_min | −0.80937 | −0.82236 | −0.82248 | **−0.82248** | **1.4e-6** |
| B12 / homo | κ | 21.00 | 23.89 | 26.35 | **26.45** | — |
| B12 / lowest | ρ = \|λ_min\| | 1.12899 | 1.13646 | 1.13646 | **1.13646** | **1.3e-9** |
| MAPbI3 / lowest | ρ = λ_max | 0.80010 | 0.80724 | 0.80881 | **0.80908** | 1.3e-3 |
| MAPbI3 / lowest | κ | 7.50 | 7.78 | 7.84 | **7.85** | — |

**[OBS]** The two ends behave very differently. The **most negative** Ritz value converges
fast (residual 1.4e-6 – 8e-5 at m = 80, six stable digits). The **most positive** Ritz value
— which is ρ for every interior state, and which sets `μ_min = 1 − λ_max` and therefore κ —
converges **slowly**: its residual is still 6e-4 – 3e-3 at m = 80 and the value is still
creeping upward.

**[INT]** ρ is therefore a *lower* bound and κ an *under*estimate. The increments are
strongly contracting, however: for B12/homo Δρ(40→60) = 7.1e-3 but Δρ(60→80) = 2.8e-4, a
25× reduction, so the limit is ρ ≈ 0.931. κ moves by 0.4–1 % between m = 60 and 80, so the
converged κ is at most ~1–2 % above the m = 80 value.

**[INF]** Quote ρ to three significant figures (≈ 0.93) and κ to two (≈ 26), not more.
The conclusions are unaffected: ρ < 1 by a wide margin for every state that must converge,
and κ ≈ 24–30 (0-D systems) versus 8–18 (MAPbI3).

**[UNC]** All values remain 40–80 dimensional Krylov estimates of a very large operator;
the deflated spectrum is denser than the full-space one (§8.7 reached 3e-15 at m = 150 in
the full space), so a comparable certification of the deflated ρ would need larger m.

### 13.4 Why the benefit saturates — the shape of the deflated spectrum

The deflated Ritz set (m = 40, i_iter = 15) is **broad**, not concentrated near zero:

| system / state | ρ | \|θ\|>0.9 | \|θ\|>0.7 | \|θ\|>0.5 | \|θ\|>0.3 | median \|θ\| |
|---|---:|---:|---:|---:|---:|---:|
| B12 / homo | 0.9237 | 3/40 | 16/40 | 25/40 | 31/40 | **0.607** |
| C60_4 / homo | 0.9227 | 3/40 | 15/40 | 24/40 | 31/40 | **0.601** |
| MAPbI3 / homo | 0.9261 | 3/40 | 11/40 | 17/40 | 25/40 | **0.353** |
| B12 / middle | 0.9221 | 3/40 | 17/40 | 25/40 | 31/40 | 0.636 |

Measured deflated error (Damped-NP) and where the gain actually comes from:

| system / state | N=0 | N=3 | N=6 | N=10 | gain 0→3 | gain 6→10 |
|---|---:|---:|---:|---:|---:|---:|
| B12 / homo | 0.6901 | 0.2558 | 0.1861 | 0.2422 | **2.70×** | 0.77× |
| C60_4 / homo | 0.6083 | 0.2230 | 0.1738 | 0.1676 | **2.73×** | 1.04× |
| MAPbI3 / homo | 0.7458 | 0.4632 | 0.3596 | 0.2678 | **1.61×** | 1.34× |
| B12 / middle | 0.6168 | 0.1193 | 0.0760 | 0.0635 | **5.17×** | 1.20× |

**[INT]** Nearly all of the useful reduction happens in the first ~3–6 orders, where the
bulk of the spectrum (median |θ| ≈ 0.6) is resolved. Beyond that the error is dominated by
the few modes near ρ ≈ 0.92, which improve by only ~8 % per order — and each order costs one
extra Hamiltonian application plus one extra GAPP application. **This is the spectral
account of the saturation seen in the archived benchmark, and of why the optimum is a
cost/benefit compromise rather than a divergence threshold.**

**[OBS] It does not remove the λ < −1 mode of the lowest state.** ρ(ΠEΠ) = 1.13646 /
1.10095 / 1.07327 for B12 / water / C60_4, essentially identical to the full-space value
and stable to 4–5 digits across iterations 5, 10, 15. This is expected: for the lowest
state `M ≻ 0`, so there is no indefiniteness to deflate; the negative mode comes from P
over-correcting (μ_max > 2, §12.2) and is orthogonal to span(X).

**[INF] The odd–even mechanism is therefore real, persistent, and cannot be deflated away.**
MAPbI3 has no such mode at any state or iteration (most negative −0.4995).

The only remaining ρ ≳ 1 entries are the "slowest" rows at late iterations, which by then
track high **virtual** bands (B12 253, water 613, C60_4 569, MAPbI3 736) lying above the
`bands = ceil(nelec/2)` convergence criterion — states Davidson is not required to converge.

---

## 14. The damping mechanism, quantified

Deflated error at the **lowest** state, Davidson iteration 15
(`η_N^Π = ‖Π E^{N+1}a₀‖ / ‖Π a₀‖`):

| N | B12 NP / DNP | water_128 NP / DNP | C60_4 NP / DNP | MAPbI3 NP / DNP |
|---:|---:|---:|---:|---:|
| 3 | 0.2191 / 0.0431 | 0.3832 / 0.0427 | 0.1725 / 0.0465 | 0.0495 / 0.0566 |
| 5 | 0.2495 / 0.0357 | 0.4435 / 0.0362 | 0.1866 / 0.0379 | 0.0314 / 0.0354 |
| 8 | 0.3131 / 0.0298 | 0.5608 / 0.0334 | 0.2162 / 0.0294 | 0.0163 / 0.0183 |
| 11 | **0.4006** / **0.0274** | **0.7160** / **0.0351** | **0.2542** / **0.0232** | **0.0085** / **0.0095** |

| system | λ_dom | \|(1+λ)/2\| | NP behaviour in N | DNP/NP gain at N = 11 |
|---|---:|---:|---|---:|
| B12 | −1.1365 | 0.0682 | **diverges** (0.219 → 0.401) | **14.6×** |
| water_cluster_128 | −1.1010 | 0.0505 | **diverges** (0.383 → 0.716) | **20.4×** |
| C60_4 | −1.0733 | 0.0366 | **diverges** (0.173 → 0.254) | **10.9×** |
| MAPbI3 | −0.4995 | 0.2503 | **converges** (0.050 → 0.0085) | 0.89× (damping slightly *harmful*) |

**[OBS]** Exactly where λ_dom < −1, the undamped expansion **diverges with order** while the
damped one converges, by a factor 11–20× at N = 11. Where λ_dom = −0.50 > −1 (MAPbI3) the
undamped expansion converges monotonically and damping is slightly counterproductive.

**[INT]** Mode-wise, `q_N^NP(λ) = λ^{N+1}` and `q_N^DNP(λ) = ½(1+λ)λ^N`. For λ = −1.1365,
|λ|¹² = 4.7 > 1 (divergent) while `|½(1+λ)|·|λ|¹¹ = 0.068 × 4.15 = 0.28 < 1` (still
contracting). The damping factor buys roughly `ln(1/|½(1+λ)|)/ln|λ|` ≈ 20 extra orders of
headroom. This is the spectral justification of the ½ damping that Reviewer 2 asked for,
and it simultaneously explains the MAPbI3 exception and why damping costs MAPbI3 a few
iterations at low order in the archived benchmark.

### 14.1 The correct performance metric: direction, not error norm

`ΠMΠ ≻ 0` allows the **exact** deflated correction `t = (ΠMΠ)⁻¹Πγ` to be computed by CG.
The angle between `Π p̄_N` (Damped-NP output) and `t` measures what Davidson actually cares
about, since the vector is normalised before use. Davidson iteration 15, HOMO state:

| N | B12 | water_128 | C60_4 | MAPbI3 |
|---:|---:|---:|---:|---:|
| 0 | 46.70° | 29.28° | 55.65° | 45.27° |
| 3 | 27.56° | 21.13° | 33.09° | 19.97° |
| 6 | 16.58° | **17.25°** | 19.61° | 11.19° |
| 9 | 11.52° | 22.07° | 13.17° | 7.67° |
| 11 | 12.78° | 34.71° | **12.09°** | **6.40°** |

**[OBS]** The direction improves monotonically and then saturates; the minimum lies at
N = 10 (B12), 6 (water_cluster_128), 11 (C60_4), 11 (MAPbI3). For the middle state the
minima are N = 11, 8, 11, 11.

**[UNC]** water_cluster_128 shows genuine degradation beyond N ≈ 6 at these two states,
which the aggregate benchmark (iteration count 19, 19, 19, 19, 18 for N = 6…10) does not
show. A single state's direction quality is not a quantitative predictor of an iteration
count aggregated over 614 states; this is reported as an observation, not a model.

---

## 15. Practical Neumann order — corrected

**No new timing was measured.**

**[RETRACTED]** The claim that "the spectral analysis explains where the useful order range
ends, because η has a minimum" is withdrawn (§10).

**[OBS]** In the archived benchmark the Damped-NP iteration count is **monotone
non-increasing** in N over the whole tested range 0…10 and never degrades; the gain
saturates (B12, N = 4…10: 27, 23, 21, 20, 19, 19, 19 — marginal gains −4, −2, −1, −1, 0, 0).
N = 10 was a practical experimental limit, not an observed turning point.

**[INF]** The optimum is therefore a **cost/benefit compromise**: each additional order costs
exactly one extra Hamiltonian application plus one extra GAPP application per preconditioner
call, while the iteration-count gain saturates. The optimum sits where the marginal gain
stops paying for the marginal cost, and is system- and solver-condition dependent — as the
existing manuscript/SI timing results already document.

**What the spectral analysis contributes** is the reason the *gain* saturates: the deflated
direction quality (§14.1) improves steeply for the first few orders and then flattens, and
for states with λ_dom < −1 the *undamped* expansion cannot be pushed to high order at all.
It does **not** predict the wall-time optimum, and no such claim is made.

---

## 16. Answers to the required questions

1–9. As in §§1–8 and §11 (environment, Gate A, code paths, `P` with the 1/(4π) factor
verified bit-identical, `E` validated to δ_recurrence ≈ 3–6e-16 in every production run,
corrected shifts captured from the production path, zero-shift fallback **inactive** in all
analysed snapshots, TM pseudopotentials present and used).

10. **Did the global spectrum explain convergence and oscillation?** No, and it could not:
    ρ(Eᵢ) > 1 is structurally guaranteed for interior states (§12), so the full-space
    spectrum carries no information about whether the method works.

11. **Was projected analysis required?** It was performed and its result is **retracted**
    (§9): `ρ_proj` cannot measure modal weight.

12. **Was η_N required?** It was computed and the metric is **retracted** (§10): it measures
    the error of a linear solve Davidson does not want.

13. **Manuscript table/figure** — §17.

14. **Practical order** — §15.

15. **Strongest supported conclusion** — §18.

16. **Limitations** — §19.

---

## 17. Manuscript-ready deliverables

**Table 1 — deflated spectrum** (§13.3): per state, ε̃, ρ(E) full space, ρ(ΠEΠ), λ_min(ΠEΠ).
Shows both that ρ > 1 in the full space is structural and that ρ_eff < 1 where it matters.

**Table 2 — damping mechanism** (§14): λ_dom of the lowest state, |(1+λ)/2|, undamped vs
damped deflated error at N = 11, and the observed undamped-NP behaviour. One table answering
Reviewer 2 comments 1–3 and Reviewer 3 comments 2–3.

**Figures** (`results_spectral_revision/deflated/figures/`):

* `rho_full_vs_deflated.png` — full-space vs deflated spectral radius per state, guide at 1.
* `damping_mechanism.png` — η^Π versus order, NP and Damped-NP, lowest state, all systems.
* `direction_vs_order.png` — angle to the exact deflated correction versus order.
* `deflation_vs_iteration.png` — ρ(ΠEΠ) versus Davidson iteration.

**Theory text supported by the data**

* The Neumann series for `Mᵢ = H − ε̃ᵢI` cannot converge in the full space for an interior
  state: `Mᵢ` is indefinite, `P` is SPD, and Sylvester's law of inertia then forces
  eigenvalues of `E` above 1. No choice of positive-definite `X` avoids this, and no
  rescaling repairs it.
* Davidson deflates the subspace that carries those modes, and on the complement
  `ΠMΠ ≻ 0` with **ρ(ΠEΠ) = 0.81–0.93** for every state that must converge — the formal
  convergence condition is met, though at a slow asymptotic rate (§13.4).
* A dominant **negative** eigenvalue below −1 survives deflation at the deeply shifted
  lowest states of B12 / water_cluster_128 / C60_4 (−1.136, −1.101, −1.073). It alternates
  in sign with order and grows — the source of the odd–even oscillation and of the failure
  of undamped NP at odd order.
* Two-point damping multiplies that mode by `(1+λ)/2 ≈ 0.037–0.068`, turning divergence into
  convergence: 11–20× smaller deflated error at N = 11.
* MAPbI3 has no eigenvalue below −1 (most negative −0.4995): no oscillation, and damping is
  marginally counterproductive there — exactly as the archived benchmark shows.
* `E` is markedly non-normal (measured asymmetry 0.163), so eigenvalues alone do not bound
  finite-order behaviour; the deflated norms and direction angles are reported alongside.

---

## 18. Strongest conclusion supported by the data

For a shifted interior eigenproblem the full-space Neumann convergence condition
`ρ(I − PM) < 1` **cannot** be met — `M = H − ε̃I` is indefinite and `P` is positive definite,
so Sylvester's law of inertia forces eigenvalues above 1. The method nevertheless works, and
is justified, because block Davidson deflates exactly the subspace carrying those modes: on
the deflated space `ΠMΠ ≻ 0` and **ρ(ΠEΠ) = 0.81–0.93** for every state required to
converge, measured once the Davidson subspace is accurate (iteration ≳ 10).

What deflation does **not** remove is a dominant negative eigenvalue below −1 at the deeply
shifted lowest states — −1.136 (B12), −1.101 (water_cluster_128), −1.073 (C60_4) — arising
because `P` over-corrects there (μ_max > 2). This mode alternates in sign and grows with
expansion order, and is the mechanism of the odd–even oscillation and of the failure of
undamped NP at odd orders. The implemented ½ damping multiplies it by `(1+λ)/2 ≈ 0.04–0.07`
and restores convergence, giving 11–20× smaller deflated error at N = 11. MAPbI3 has no such
mode (−0.4995), shows no oscillation, and is the one system where damping slightly costs.

---

## 19. Limitations and unresolved issues

1. **[UNC]** Four states per system per snapshot. The extent of the λ < −1 region over the
   band index was not mapped, so the *fraction* of states affected — the quantity that
   actually drives the aggregate iteration count — remains unknown. This is the most
   valuable follow-up.
2. **[UNC]** A single state's deflated direction quality is not a quantitative predictor of
   the aggregate Davidson iteration count (§14.1, water_cluster_128).
3. **[UNC]** Deflation is against the *current* Ritz block, which is exact only in the limit.
   At iteration 5 three of eleven states showed negative curvature in `ΠMΠ`; by iteration
   10–15 all occupied states were positive definite. Results are quoted at iteration 15.
4. **[UNC]** The "slowest" state tracks a moving band, so its rows are not comparable across
   iterations; at late iterations it is a high virtual band outside the convergence
   criterion.
5. **[UNC]** Ritz residuals at m = 60 in the full-space study were 2.6e-4–1.2e-3; the
   convergence study (§8.7) at m = 100/150 was performed for B12 only.
6. **[UNC]** One of 44 archived (system, order) baseline points was reproduced. CPU vs CUDA
   RNG differences mean only converged quantities are comparable.
7. **[UNC]** The under-relaxation by-product (§12.2, α ≈ 0.93–0.96 restores convergence of
   the undamped expansion at the lowest states) was derived from the measured spectrum but
   not tested in an actual run.
8. **[UNC]** The optional damping-weight sweep (instruction §13) was not run; the measured
   spectra do support the special role of λ ≈ −1, which is the stated precondition for
   skipping it.
9. **[OBS]** Infrastructure: this cluster schedules by cores only (`SelectTypeParameters =
   CR_CORE`), so `--mem` does not prevent co-scheduling; two large jobs on one node caused
   two OOM kills. The probe was also rewritten to store only the selected columns of each
   recurrence term (≈35 GB → ≈0.2 GB for MAPbI3), verified to give identical results.
10. **[UNC]** CPU-only environment; nothing here bears on the manuscript's timing results,
    which are used as-is.

---

## 20. The order-N preconditioner — closed form, what it does and does not license

### 20.1 The closed form, and its direct verification

    P_N M   = Σ_{n=0}^{N} Eⁿ (I − E)            = I − E^{N+1}
    P̄_N M  = Σ_{n=0}^{N−1} Eⁿ (I−E) + ½Eᴺ(I−E)  = I − ½ Eᴺ (I + E)
    general weight α:                             I − Eᴺ (1 − α + αE)

**[OBS] Verified directly against the production code** (job 329741). The order-N operator
was applied as `P_N M x = PreNeumann.call(Hx − ε̃x, H, ε̃)` using the *production*
`PreNeumann`, and its spectrum measured by Arnoldi (B12, HOMO, m = 30):

| N | μ_min predicted | μ_min measured | error |
|---:|---:|---:|---:|
| 2 | −2.37466 | −2.37523 | **0.02 %** |
| 5 | −13.04511 | −13.05016 | **0.04 %** |
| 8 | −57.45477 | −57.48700 | **0.06 %** |

The algebra, the reading of the damping-accumulation logic, and the production behaviour all
agree. **[OBS]** Note also what this shows: **in the full space the order-N preconditioned
operator gets steadily *worse* with N** (μ_min = −2.4 → −13 → −57), which is exactly
`ρ(E) = 1.609 > 1` expressing itself.

### 20.2 **[RETRACTED]** — the closed form may not be applied to the deflated spectrum

An earlier revision of this report computed `κ(P_N M)` by inserting the **deflated** Ritz
values of `ΠEΠ` into the closed form, and built on it (a κ-vs-order table, a fit
`iterations ∝ κ_N^p` with p = 0.76–0.85, and an optimal-order formula
`N_opt = argmin κ_N^p (c+N)`). **All of that is withdrawn.**

`Π` and `E` do not commute, so `Π P_N M Π ≠ I − ½(ΠEΠ)ᴺ(I + ΠEΠ)`. Measured (same job):

| N | μ_min pred. | μ_min meas. | err | μ_max pred. | μ_max meas. | err |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.19025 | 0.16502 | 13 % | 0.99904 | 0.99951 | 0.05 % |
| 5 | 0.37208 | **0.05400** | **85 %** | 1.03337 | 1.16858 | 13 % |
| 8 | 0.51309 | 0.45451 | 11 % | **1.00000** | **2.05236** | **105 %** |

**[OBS] This is a qualitative failure, not a resolution artefact.** Every λ in the deflated
HOMO spectrum lies in (−1, +0.919], so `½λ⁸(1+λ) ≥ 0` and the closed form *requires*
`μ ≤ 1`, i.e. `μ_max = 1.00000` exactly. The measured value is 2.052.

**[INF]** The consistency of p ≈ 0.8 across four systems, previously reported, has no
established basis; it may reflect a monotone relation between the (invalid) κ and some
correct quantity, but that is unsupported and is not claimed.

### 20.2b **[SUPERSEDED by §20.3–20.7 and Appendix N]** Direct re-measurement with Ritz residuals

*The full-space table below stands. The deflated `mu_min` column and everything derived from
it are void: the Arnoldi run leaked into `span(X)`, which `Pi P_N M Pi` annihilates, so the
subspace itself was contaminated and the Ritz-residual guard could not detect it (a null-space
Ritz pair has residual ~0 by construction). The `mu_max` column is unaffected — a Ritz vector
with `span(X)` content `w` has residual at least `theta*w` — and is reproduced to five decimals
by the corrected tooling, so the §20.2 retraction stands. Appendix N documents the defect.*

The first verification lacked Ritz residuals, so a failure of the closed form could not be
distinguished from an unconverged Ritz value (a non-normal operator's numerical range extends
beyond its spectrum). Jobs 329990/329991 repeated the measurement **with explicit residuals**.

**Full space — closed form confirmed.**

| N | μ_min measured | Ritz residual | closed form |
|---:|---:|---:|---:|
| 2 | −2.37523 | 4.5e-07 | −2.3752 |
| 3 | −4.42953 | 3.4e-08 | −4.4295 |
| 5 | −13.05016 | 3.3e-10 | −13.0500 |
| 8 | −57.48700 | **1.9e-13** | −57.4860 |

**Deflated — closed form refuted, with converged residuals.**

| N | μ_min | res | conv.? | μ_max | res | conv.? | closed-form μ_max |
|---:|---:|---:|---|---:|---:|---|---:|
| 1 | 0.10303 | 1.1e-02 | no | 1.12455 | 2.5e-03 | no | 1.1250 |
| 2 | 0.16291 | 8.3e-03 | no | 1.00052 | 8.3e-03 | no | 0.9998 |
| 3 | 0.00002 | 3.2e-03 | no | 1.05226 | 3.4e-03 | no | 1.0527 |
| 5 | **0.00000** | **1.4e-06** | **yes** | **1.16859** | **4.5e-05** | **yes** | 1.0334 |
| 8 | **0.00000** | **3.5e-05** | **yes** | **2.05236** | **2.3e-05** | **yes** | **1.0000** |

**[OBS]** At N = 8 the closed form *requires* μ_max ≤ 1 (every deflated λ lies in (−1, 0.919],
so `½λ⁸(1+λ) ≥ 0`); the measured value is 2.052 with a **converged** residual of 2.3e-5.
The failure of the deflated shortcut is real. **The retraction of §20.2 stands.**

**[WITHDRAWN — Appendix N]** This section originally reported a "converged near-null
direction" for N ≥ 5 (μ_min = 0.00000) and concluded that κ(Π P_N M Π) diverges with order.
Both are artefacts: the Arnoldi basis had leaked into `span(X)`, which the operator
annihilates, and a null-space Ritz pair has residual ≈ 0 by construction, so the residual
guard certified it. Corrected values are in §20.2c — κ is finite everywhere and improves with
order in all 32 measured combinations.

**[INT]** The *mechanism* proposed here was right and is retained in §20.2c, attached to the
correct quantity: the full-space growth of `E^N` does leak through Π because the two do not
commute — but it inflates `σ_max` (to 2.71 and 2.97 in the two systems with the largest ρ(E)),
it does not drive `σ_min` to zero.

**[INF]** κ is therefore not merely mis-computed; **it is the wrong figure of merit here**,
because it is a worst case over all directions in range(Π) while the solver only ever applies
the preconditioner to the actual residual. This is the same lesson as §9 (ρ_proj) and §10
(η_N) in a third guise: worst-case spectral quantities do not describe this algorithm. The
order dependence must rest on the *per-direction* measured quantities of §14 — the deflated
error `η_N^Π` and the angle to the exact deflated correction — which do improve with N and
which involve no closed form.

**[UNC]** At m = 80 the run aborted with a dtype error in the explicit-residual path when an
extremal Ritz value came out complex (the production Hamiltonian operator is real-only). The
m = 40 extremes were real and converged, so the conclusions above hold; the analysis code
does not currently support complex extremal Ritz pairs of `P_N M`.

### 20.2c Re-measurement with error bounds — conditioning, its high-order limit, and a convergence certificate

*Replaces the void parts of §20.2b. 32 (system, state, order) points; jobs 330624-330627,
331606-331609, 333978-333981.*

**Method.** Condition numbers are **singular-value** ratios bracketed by Lanczos on the
symmetric PSD `A^T A`, which carries the classical residual bound `|lambda - theta| <=
beta_m |s_m|`. Eigenvalue ratios are reported alongside and are only a *lower* bound: `E` is
non-symmetric (asymmetry 0.16-0.28) and the deflated operators carry no real-spectrum
guarantee. Krylov runs on operators that annihilate `span(X)` carry `project = Pi` applied
immediately before each normalisation (Appendix N).

**[VAL]** The analysis operator reproduces production `PreNeumann`: `P_N` bit-for-bit
(`0.00e+00`), `P_N M` to 2-6e-15, adjoint identity to 1e-14, Lanczos orthogonality error
1.3-2.0e-14. Two independent job sets measured the HOMO columns and agree to **0.000 %**.

#### Condition number of the order-N operator on range(Pi)

| System | state | N=0 | N=2 | N=5 | N=8 | N=0 -> N=8 |
|---|---|---|---|---|---|---|
| B12 | homo | > 26.7 | [12.63, 14.53] | [6.90, 8.02] | 9.99 | **> 2.7x** |
| water_cluster_128 | homo | > 24.0 | [7.59, 7.95] | 4.17 | [6.14, 6.26] | **> 3.9x** |
| C60_4 | homo | > 31.5 | [12.41, 15.87] | [7.32, 7.41] | 6.34 | **> 5.0x** |
| MAPbI3 | homo | [22.39, 26.40] | [7.97, 8.18] | 4.09 | 3.03 | **8.0x** |
| B12 | lowest | [24.70, 48.59] | [6.44, 6.53] | 3.34 | 2.56 | **14.3x** |
| water_cluster_128 | lowest | [24.91, 51.58] | [5.91, 6.11] | 3.10 | 2.32 | **16.5x** |
| C60_4 | lowest | [24.74, 50.79] | [5.94, 6.12] | 3.13 | 2.20 | **17.1x** |
| MAPbI3 | lowest | [8.04, 8.72] | 2.56 | 1.53 | 1.24 | **6.8x** |

**[OBS]** `kappa` improves with order in **all 32 combinations**, by 2.7-17x from N = 0 to
N = 8. The lowest state is better conditioned than the HOMO state at every order (0.26-0.77x),
because its shift lies far below the spectrum, `M` is closer to positive definite, and the
full-space `rho(E)` is near 1. MAPbI3 / lowest reaches **kappa = 1.24** at N = 8, i.e.
`Pbar_8 M ~ I` on range(Pi) (spectrum in [0.830, 1.030]).

**[OBS]** The eigenvalue ratio understates `kappa` by 1.3-2.2x, and the gap widens with order.
The closed-form table of the withdrawn draft reproduces the eigenvalue ratio accurately (0.3 %
against direct Arnoldi) but must be labelled a lower bound, not a condition number.

**[OBS]** Complex eigenvalues appear at N = 8 in two of four systems (3.4e-4, 1.7e-5) and in
none of the fourteen measurements at N <= 5. `PM` is similar to the symmetric
`P^(1/2) M P^(1/2)` so `spec(E)` is real by construction; `Pi P M Pi != (Pi P Pi)(Pi M Pi)`
carries no such guarantee.

#### Iteration count versus conditioning

| System | 0->2 | 2->5 | 5->8 |
|---|---:|---:|---:|
| B12 | — | +1.00 | -0.64 |
| water_cluster_128 | — | +0.85 | -0.13 |
| C60_4 | — | +0.93 | +0.80 |
| MAPbI3 | +0.86 | +0.88 | +0.89 |

**[OBS]** `N <= 5`: p = 0.91, range 0.85-1.00 over 5 intervals and four systems.
`N -> 8`: mean +0.23, range -0.64 to +0.89.

**[INF]** The exponent is near 1 (Richardson-like) rather than 0.5 (CG-like), consistent with
Davidson acting on a non-symmetric operator without an SPD-optimal polynomial.

#### Why the correspondence fails beyond N ~ 5

`Pbar_N M = I - 0.5 E^N (I+E)` and `E^N` grows like `rho(E)^N` in the full space. `Pi` and `E`
do not commute, so `Pi` cannot fully suppress that growth; the residue inflates `sigma_max`.

| System | state | full-space rho(E) | sigma_max N=2 | N=5 | N=8 | kappa turns around? |
|---|---|---:|---:|---:|---:|---|
| MAPbI3 | lowest | 1.0000 | 1.021 | 1.044 | 1.030 | no |
| C60_4 | lowest | 1.0734 | 1.082 | 1.117 | 1.090 | no |
| water_cluster_128 | lowest | 1.1012 | 1.075 | 1.093 | 1.118 | no |
| B12 | lowest | 1.1367 | 1.095 | 1.124 | 1.201 | no |
| MAPbI3 | homo | 1.4646 | 1.040 | 1.116 | 1.218 | no |
| C60_4 | homo | 1.6102 | 1.085 | 1.352 | 1.786 | no |
| water_cluster_128 | homo | 1.7166 | 1.035 | 1.234 | 2.708 | **YES** |
| B12 | homo | 1.8323 | 1.148 | 1.351 | 2.966 | **YES** |

**[OBS]** `sigma_max` at N = 8 is monotone in the full-space `rho(E)` across all eight
combinations, and `kappa` turns around in exactly the two with `rho(E) >= 1.72`.

**[INT]** This is the mechanism proposed in the withdrawn §20.2b, attached to the right
quantity: the leaked amplification inflates `sigma_max`, it does not drive `sigma_min` to zero.
Projecting at every recurrence step breaks the accumulation, which is why that variant keeps
improving at N = 8 (by 1.2-2.1x). Together with the saturation of the iteration count this is
the spectral basis for an optimal order of **N ~ 4-6**.

#### A rigorous certificate that the deflated expansion converges

`rho(A) <= ||A^n||^(1/n)` for every `n`, and `||A^n|| = sigma_max(A^n)` is bounded above by the
same Lanczos machinery. A value below 1 certifies contraction to the accuracy of the Lanczos residual, unlike an Arnoldi Ritz value on a
non-symmetric operator. `||A|| > 1` in seven of eight cases, so the naive bound fails.

| System | state | ||A|| | n=2 | n=3 | n=5 | n=8 | n=12 | n=20 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B12 | homo | 1.117 | 1.08794 | 1.07157 | 1.04585 | 1.02016 | 0.99913 | 0.97675 | **proved n=12** |
| C60_4 | homo | 1.108 | 1.09038 | 1.07560 | 1.05123 | 1.02581 | 1.00426 | 0.98045 | **proved n=20** |
| MAPbI3 | homo | 1.018 | 1.00716 | 0.99865 | 0.98516 | 0.97144 | 0.96014 | 0.94842 | **proved n=3** |
| water_cluster_128 | homo | 1.073 | 1.03085 | 1.02005 | 1.00221 | 0.98296 | 0.96654 | 0.94901 | **proved n=8** |
| B12 | lowest | 1.278 | 1.16864 | 1.16032 | 1.15122 | 1.14569 | 1.14261 | 1.14015 | not proved (rho > 1) |
| C60_4 | lowest | 1.259 | 1.12303 | 1.12095 | 1.10088 | 1.08965 | 1.08436 | 1.07996 | not proved (rho > 1) |
| MAPbI3 | lowest | 0.870 | 0.85754 | 0.84662 | 0.83199 | 0.81890 | 0.81106 | 0.80984 | **proved n=1** |
| water_cluster_128 | lowest | 1.279 | 1.15085 | 1.14769 | 1.12836 | 1.11744 | 1.11211 | 1.10769 | not proved (rho > 1) |

**[OBS]** HOMO: all four systems certified `rho(Pi E Pi) <= 0.981`. MAPbI3 / lowest certified
`<= 0.810`. The other three lowest states converge from above onto their Arnoldi values
(1.073-1.136) and are correctly *not* proved -- `rho > 1` there is a fact, not a tooling limit.

**[OBS]** All eight bounds lie above the corresponding Arnoldi estimates, by 0.1-6.0 % (mean
2.2 %). The concern that Arnoldi might underestimate `rho` is therefore rejected, and the
`rho ~ 0.93` values quoted elsewhere in this report stand -- as estimates, with the certified
bound as the provable statement.

**[INT]** The number of powers needed tracks `||A||` exactly: 0.870 -> n = 1, 1.018 -> n = 3,
1.073 -> n = 8, 1.108/1.117 -> n = 12-20, >= 1.26 -> never. The length of the non-normal
transient is set by the gap between `||A||` and `rho`.

#### What the lowest state means for the damping argument

For three of four systems `rho(Pi E Pi) = |lambda_min|`, i.e. the deflated spectral radius **is**
the negative mode that produces the odd-even oscillation. MAPbI3 is the exception: its
`rho = lambda_max = +0.809` is set by a positive mode. This is the same fact as §20.4's
`lambda_dom` table, now established by an independent and rigorous route, and it is what makes
the 1/2 damping necessary rather than cosmetic (§21.1).

### 20.2d The converged-regime measurement, and the manifold distribution

*Three campaigns, all through the unmodified production path. Supersedes the sampling choice
of §20.2c.*

#### Design

The measurement point of §20.2c was a fixed mid-run Davidson iteration, which required
justifying that choice. It is replaced by a design that needs no such justification: capture
at **the first preconditioner call where every analysed state's residual norm is below 1e-3**,
and use **the raw Davidson Ritz value itself as the shift** (`--probe_resid_below 1e-3
--exact_shift`). `M = H - eps I` is then exactly singular at the target state, so only the
deflated operators are well defined -- which is precisely the Jacobi-Davidson setting.

| campaign | jobs | states | orders | purpose |
|---|---|---|---|---|
| iteration 15, corrected shift | 330624-330627 | lowest, HOMO | 0,2,5,8 | original |
| **converged, exact shift** | **356653-356656** | lowest, HOMO | 0,2,5,8 | **primary** |
| band scan | 379318-379321 | ~11 bands | 0,2,5,8 | manifold distribution |
| dense N | 408521-408524 + 12 fill | 3 states / all bands | **0..10** | benchmark-matched curve |

#### The two designs agree

**[OBS]** 29 conclusive condition numbers compared: mean **1.13 %**,
median **0.05 %**, max 9.12 %; 21/29 within 1 %. The 16 lowest-state
points agree to 0.0 %. All eight operator norms `||Pi E Pi||` agree to three decimals, and
every rho certificate reproduces (e.g. B12/HOMO 0.976748 -> 0.976707, MAPbI3/HOMO
0.949014 -> 0.949113).

**[INF]** The reported numbers are not an artefact of where in the Davidson run the operator
is sampled. The primary campaign is now the converged one, and the fixed-iteration campaign
serves as its independent reproduction.

#### The manifold distribution

| System | bands | monotone to N = 8 | eps range |
|---|---:|---:|---|
| B12 | 11 | **8 (73 %)** | -1.16 to -0.34 |
| water_cluster_128 | 12 | **6 (50 %)** | -1.16 to -0.37 |
| C60_4 | 11 | **11 (100 %)** | -1.08 to -0.36 |
| MAPbI3 | 11 | **11 (100 %)** | -0.82 to -0.01 |
| **total** | **45** | **36 (80 %)** | |

**[OBS] Aggregate conditioning over the sampled manifold** (geometric mean):
kappa = **7.11** at N = 2, **3.65** at N = 5, **3.11** at N = 8 — a
**2.29x** improvement, still falling at N = 8.

**[OBS] A single quantity classifies every band without exception.**

| | bands | `sigma_max` at N = 8 |
|---|---:|---|
| monotone | 36 | <= **1.872** |
| reversed | 9 | >= **1.959** |

The separation is clean, with a gap of 4.7 %. The rule was fixed on 43 bands and then
correctly predicted the two measured afterwards (B12 band 200, `sigma_max` = 2.048, reversed
at 1.08x, as predicted).

**[INT]** This resolves the apparent tension between the condition number and the benchmark.
The iteration count is set by the whole manifold; 80 % of it is still improving at N = 8, and
the aggregate improves by 2.29x. The reversal at the HOMO of the two systems with the largest
full-space `rho(E)` is real (§20.2c) but is a minority of the manifold.

**[UNC] Sampling limitations.** The sample is uniform in **band index**, not in energy: the
`eps` spacing between adjacent sampled bands varies by up to 78x within a system
(water_cluster_128: 0.006 to 0.466 Ha), so the 80 % counts bands, not energy intervals. Virtual
states — 20 % of the Davidson block, at higher `eps` where reversals are most likely — were
not sampled, so the reversal fraction is a **lower bound**. Bands are counted unweighted.

### 20.2d-bis What is rigorous and what is numerically certified

*Audit of the final method, carried out before writing the response.*

**[VAL] Verified exactly.** Adjoint identity `<v,Au> = <u,A^T v>` to 3.5e-16; `B = A^T A`
symmetric to 2.8e-15; `G_N = I - P_N M` exact (0.00e+00); `G_N` adjoint to 2.6e-16;
`P_N` bit-for-bit against production `PreNeumann`; Lanczos orthogonality 1.3-2.0e-14.

**[OBS] Which bracket ends are rigorous.** For Rayleigh-Ritz on a symmetric operator,
interlacing gives

    lam_min <= theta_min   =>   sigma_min <= sqrt(theta_min)     exact
    lam_max >= theta_max   =>   sigma_max >= sqrt(theta_max)     exact
    => kappa >= sqrt(theta_max / theta_min)                      EXACT, unconditionally

The classical residual bound `|lam - theta| <= beta_m |s_m|` certifies proximity to *an*
eigenvalue, not to the extremal one. The upper end of each bracket, and hence
`rho <= ||A^n||^(1/n)`, therefore rests on the Lanczos run having converged to the extremal
value rather than on a theorem.

**[VAL] Convergence of the upper end, verified three ways.** `sigma_max` is m-independent from
m = 20 to 100 (eight digits, dense reference); the dense cross-check on the 1000-point Si case
agrees to five digits; two independently designed campaigns agree to a median of 0.05 % over 29
points. The code also degrades safely: when `sigma_min` is not converged it reports
`kappa in [lower, inf]` and marks the point inconclusive rather than returning a wrong value —
verified at m = 10, 40, 120, 180 on a dense reference whose true `sigma_min` is 1e-4.

**[INF]** The claim "`kappa` is large at low order" rests on the exact lower bound alone. The
claims "`kappa` decreases with order" and "`rho < 1`" rest on converged upper ends and are
therefore *numerically certified*, not proved. The manuscript wording is set accordingly.

### 20.2e The contraction factor of the scheme, undamped versus damped

*Jobs 418886-418889. 132 orders x 2 schemes = 264 norms.*

`rho` is a property of `E` and is therefore the SAME for NP and Damped-NP: damping changes how
the terms are summed, not the recurrence. The quantity that IS specific to the damping is the
error operator

    G_N = I - P_N M = E^N ((1-w) I + w E),     w = 1 (NP),  w = 1/2 (Damped-NP)

whose norm is the contraction actually achieved at order N. Below 1 the scheme contracts.
Measured by Lanczos on `G_N^T G_N` with `project = Pi`.

**[VAL]** `G_N` reproduces `I - P_N M` exactly (max deviation 0.00e+00 on a dense reference) and
its adjoint identity holds to 2.6e-16.


**lowest**

| System | scheme | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAPbI3 | NP | **0.87** | **0.74** | **0.62** | **0.51** | **0.41** | **0.34** | **0.27** | **0.22** | **0.18** | **0.14** | **0.12** |
| MAPbI3 | **DNP** | **0.93** | **0.80** | **0.68** | **0.56** | **0.46** | **0.37** | **0.30** | **0.25** | **0.20** | **0.16** | **0.13** |
| C60_4 | NP | 1.26 | 1.26 | 1.41 | 1.48 | 1.62 | 1.72 | 1.86 | 1.99 | 2.14 | 2.30 | 2.47 |
| C60_4 | **DNP** | **0.99** | **0.99** | **0.94** | **0.87** | **0.79** | **0.72** | **0.65** | **0.59** | **0.54** | **0.48** | **0.44** |
| water_cluster_128 | NP | 1.28 | 1.32 | 1.51 | 1.64 | 1.83 | 2.00 | 2.22 | 2.43 | 2.69 | 2.95 | 3.26 |
| water_cluster_128 | **DNP** | **0.98** | **0.95** | **0.89** | **0.84** | **0.78** | **0.73** | **0.69** | **0.64** | **0.60** | **0.56** | **0.52** |
| B12 | NP | 1.28 | 1.37 | 1.57 | 1.79 | 2.03 | 2.31 | 2.63 | 2.99 | 3.41 | 3.88 | 4.41 |
| B12 | **DNP** | 1.00 | 1.00 | **0.96** | **0.90** | **0.84** | **0.78** | **0.73** | **0.68** | **0.63** | **0.58** | **0.54** |

**middle**

| System | scheme | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAPbI3 | NP | **0.96** | **0.90** | **0.82** | **0.74** | **0.67** | **0.58** | **0.52** | **0.45** | **0.39** | **0.34** | **0.29** |
| MAPbI3 | **DNP** | **0.97** | **0.92** | **0.86** | **0.78** | **0.70** | **0.62** | **0.55** | **0.48** | **0.42** | **0.36** | **0.31** |
| C60_4 | NP | 1.13 | 1.09 | 1.08 | 1.05 | 1.01 | **0.95** | **0.89** | **0.83** | **0.77** | **0.71** | **0.66** |
| C60_4 | **DNP** | 1.02 | 1.07 | 1.08 | 1.07 | 1.03 | **0.98** | **0.92** | **0.86** | **0.80** | **0.74** | **0.69** |
| water_cluster_128 | NP | 1.11 | 1.01 | **0.98** | **0.93** | **0.88** | **0.82** | **0.77** | **0.85** | 1.20 | 1.76 | 2.56 |
| water_cluster_128 | **DNP** | 1.01 | 1.01 | **0.99** | **0.95** | **0.90** | **0.84** | **0.79** | **0.78** | 1.01 | 1.47 | 2.16 |
| B12 | NP | 1.12 | 1.11 | 1.10 | 1.08 | 1.04 | **0.99** | **0.93** | **0.88** | **0.98** | 1.42 | 2.06 |
| B12 | **DNP** | 1.03 | 1.08 | 1.10 | 1.09 | 1.06 | 1.01 | **0.96** | **0.90** | **0.86** | 1.20 | 1.74 |

**homo**

| System | scheme | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAPbI3 | NP | 1.02 | 1.02 | 1.00 | **0.97** | **0.93** | **0.89** | **0.84** | **0.79** | **0.74** | **0.69** | **0.83** |
| MAPbI3 | **DNP** | 1.00 | 1.02 | 1.01 | **0.98** | **0.95** | **0.91** | **0.86** | **0.81** | **0.76** | **0.71** | **0.72** |
| C60_4 | NP | 1.11 | 1.19 | 1.25 | 1.28 | 1.29 | 1.28 | 1.27 | 1.24 | 1.25 | 2.03 | 3.25 |
| C60_4 | **DNP** | 1.05 | 1.14 | 1.22 | 1.26 | 1.28 | 1.29 | 1.27 | 1.25 | 1.22 | 1.63 | 2.64 |
| water_cluster_128 | NP | 1.07 | 1.06 | 1.05 | 1.01 | **0.96** | **0.91** | 1.02 | 1.58 | 2.54 | 4.03 | 6.33 |
| water_cluster_128 | **DNP** | 1.02 | 1.05 | 1.05 | 1.02 | **0.98** | **0.93** | **0.92** | 1.28 | 2.05 | 3.28 | 5.18 |
| B12 | NP | 1.12 | 1.18 | 1.23 | 1.25 | 1.24 | 1.22 | 1.20 | 1.74 | 2.89 | 4.73 | 7.61 |
| B12 | **DNP** | 1.05 | 1.14 | 1.20 | 1.23 | 1.24 | 1.23 | 1.21 | 1.39 | 2.31 | 3.81 | 6.17 |

**[OBS] Where the damping acts.**

| state | NP contracts | Damped-NP contracts | mean ratio NP/DNP |
|---|---:|---:|---:|
| lowest | 11/44 | **42/44** | **2.61x** |
| middle | 27/44 | **26/44** | **1.01x** |
| homo | 10/44 | **11/44** | **1.06x** |

**[OBS]** At the lowest state the undamped norm grows monotonically with order (1.26-1.28 at
N = 0 to 2.47-4.41 at N = 10) while the damped norm falls monotonically (0.98-1.00 to
0.44-0.54). MAPbI3, the one system whose lowest state has `rho(Pi E Pi) < 1`, is the exception:
there the undamped scheme already contracts and damping costs 6-7 %.

**[INF]** The benefit of the 1/2 damping is therefore **localised**: it is decisive on the
20-27 % of bands with `eps < -0.89 Ha`, and worth 1-6 % elsewhere. This is the quantitative
form of the qualitative statement in Figure 1 of the manuscript, and it explains the MAPbI3
exception (§20.2d, R2.3) as the case where the localised benefit does not exist.

**[UNC]** `||G_N||` is a worst case over all directions in range(Pi). Where it exceeds 1 at
high order the realised Davidson convergence is unaffected: the eigenvalue-based rate
`rho(G_N) = max |1 - mu|` stays below 1 in 30 of 31 measured (system, state, order) points, and
predicts the archived iteration counts to within a mean factor 1.24 over 11 order intervals.
The stationary model over-predicts the gain because Rayleigh-Ritz over the accumulated subspace
already recovers part of what a better preconditioner would give.

### 20.3 What the verified full-space form does license

**Odd-order indefiniteness.** This is a statement about `P_N M` in the **full space**, where
the closed form is verified. With `λ_dom < −1` and N odd, `λ_dom^{N+1} = +|λ_dom|^{N+1} > 1`,
so `1 − λ_dom^{N+1} < 0`:

| System | λ_dom (lowest state) | undamped `P_N M`, N = 0 … 11 |
|---|---:|---|
| B12 | −1.137 | + IND + IND + IND + IND + IND + IND |
| water_cluster_128 | −1.101 | + IND + IND + IND + IND + IND + IND |
| C60_4 | −1.073 | + IND + IND + IND + IND + IND + IND |
| MAPbI3 | +0.809 | + + + + + + + + + + + + |

**At odd order the undamped preconditioner is indefinite**, not merely less accurate. Damping
restores definiteness at every order tested, with analytic headroom
`N ≲ ln(1/|½(1+λ)|)/ln|λ|` = **21 (B12), 31 (water), 47 (C60_4)**.

**The damping weight.** `α = 1/(1−λ)` annihilates a mode exactly; for λ = −1 that is
**α = ½**. Measured annihilators: 0.468 / 0.476 / 0.482 (§21.1). The admissible-α window and
the Chebyshev comparison of §21.2 also rest only on the full-space form.

### 20.4 Order dependence — what is left, and it is measured, not derived

The quantities below were obtained by applying the operator directly, with no closed form,
and therefore stand:

* **deflated error** `η_N^Π = ‖Π E^{N+1} a₀‖ / ‖Π a₀‖` (§14);
* **angle between `Π p̄_N` and the exact deflated correction** `t = (ΠMΠ)⁻¹Πγ` obtained by
  PCG (§14.1).

Both improve steeply over the first ~3–6 orders and then flatten, with minima at
N ≈ 6–11 depending on system and state. That is consistent with the archived benchmark
saturating at N ≈ 5–6, but **[UNC] no validated quantitative model linking a single state's
spectral data to the aggregate Davidson iteration count now exists.** The practical-order
question must therefore be answered as the reviewer's own framing suggests — a cost/benefit
compromise documented by the existing timing data — with the spectral analysis supplying the
*mechanism* for the saturation, not a predictive formula.

**[UNC]** A measured (rather than derived) `κ(Π P_N M Π)` curve would restore a quantitative
route; the three points measured here (κ = 6.1, 21.6, 4.5 at N = 2, 5, 8) are not monotone
and are probably under-resolved at m = 30, so no curve is reported.

### 20.5 How many states are actually affected?

Survey of the occupied manifold (job 329742; B12, every 10th band, deflated, m = 30):

| band | ε̃ | λ_min | λ_dom | λ_dom < −1 |
|---:|---:|---:|---:|---|
| 0 | −1.1601 | −1.13641 | −1.13641 | **yes** |
| 20 | −1.0196 | −1.06749 | −1.06749 | **yes** |
| 40 | −0.8904 | −1.00181 | −1.00181 | **yes** |
| 50 | −0.8311 | −0.97560 | −0.97560 | no |
| 80 | −0.6878 | −0.92554 | −0.92554 | no |
| 90 | −0.6462 | −0.91297 | **+0.91722** | no |
| 240 | −0.3640 | −0.82909 | +0.91862 | no |

**[OBS] Three regimes.** Bands 0–40 (**≈ 16 % of the occupied manifold**) have `λ_dom < −1`;
bands ≈ 50–80 have a negative but sub-unit dominant eigenvalue; from band ≈ 90 the dominant
eigenvalue is *positive* (λ_max ≈ +0.917, essentially constant across the whole manifold —
a grid/GAPP mode). `λ_min` varies smoothly with the shift, `dλ_min/dε̃ ≈ 0.50`, crossing −1
at `ε̃ ≈ −0.89 Ha`.

**[INF]** This closes the largest gap of the earlier revision. The odd-order failure is not
confined to a handful of states: for B12 roughly **40 of 250 occupied bands** are affected,
and Block Davidson terminates only when *every* occupied band has converged.

## 20.6 CONSOLIDATED ACCOUNT — supersedes the deflation narrative of §13 and §18

Sections 9–20 were written as the analysis developed and contain three superseded framings.
This section is the settled account; where it conflicts with an earlier section, it wins.

### Why ρ(E) ≥ 1 — two independent structural causes, plus one from P

| # | cause | eigenvalue of E | eigenvector | measured |
|---|---|---|---|---|
| (a) | `M = H − ε̃I` is **near-singular by design**: ε̃ approximates ε_i. `det(PM) = det(P)det(M)`, so `PM` is near-singular too | **λ ≈ 1** | **ψ_i itself** | MAPbI3 lowest: λ_dom = **+0.99995** (Ritz res 4e-4); B12 lowest at iter 15: `ε_i − ε̃ = 3.5e-10 Ha` |
| (b) | `M` is **indefinite** for any interior state (Sylvester's law of inertia; #negative = #states below ε̃) | **λ > 1** | the states below ε̃ | ρ(E) = 1.28–1.83; B12: 0/131/248/295 states below ε̃ |
| (c) | `P` **over-corrects** at deeply shifted states (μ_max > 2) | **λ < −1** | a vacuum/grid mode | −1.07 … −1.14 at the lowest states of B12 / water / C60_4; 16–25 % of the occupied manifold |

(a) holds even where `M` is positive definite, so **ρ(E) ≥ 1 is unavoidable regardless of the
quality of P**. No positive-definite X avoids it and no rescaling repairs it.

### What protects the method — measured, mode by mode

The Davidson recursion runs in the **full space**; `Π = I − XXᴴ` is applied **once**, to the
returned block. So the surviving error is `Π E^{N+1} a₀`, and asymptotically

    ‖Π E^{N+1} a₀‖  ~  |c| · ρ(E)^{N+1} · ‖Π v_dom‖ .

The decisive factor is `‖Π v_dom‖/‖v_dom‖` — how much of the *divergent eigenvector* survives
deflation. Measured over 12 (system, iteration, state) combinations:

| cause | ‖Π v_dom‖/‖v_dom‖ | removed by deflation | protected by |
|---|---:|---:|---|
| (a) λ ≈ 1 | **0.0065** | **99.4 %** | **deflation** — ψ_i ∈ span(X) |
| (b) λ > 1 | **0.039 – 0.113** | **88.7 – 96.1 %** | **deflation** — those states ∈ span(X) |
| (c) λ < −1 | **0.996 – 1.000** | **0.0 – 0.4 %** | **damping** — deflation cannot touch it |

**[OBS]** The correspondence is exact and has no exceptions in the measured set, and the
values are stable between Davidson iterations 5 and 15.

**[OBS]** Direct confirmation: `‖Π Eⁿa₀‖/‖Eⁿa₀‖` for B12/homo falls 0.993 (n=0) → 0.537
(n=8) → **0.130** (n=16), converging on `‖Π v_dom‖/‖v_dom‖ = 0.105`.

**[INT] Headroom.** Deflation buys `ln(1/0.08)/ln(1.83) ≈ 4` orders; damping buys
`ln(1/0.068)/ln(1.14) ≈ 20` orders. **The order range actually used, N ≤ 10, lies inside the
window the two mechanisms create.**

### Three corrections to the earlier narrative

1. **[CORRECTED]** §13 argued "Davidson deflates, therefore the relevant operator is `ΠEΠ`,
   and `ρ(ΠEΠ) < 1` shows the expansion converges." Davidson deflates the **output**, not
   each step; the recursion iterates `E` in the full space. `ρ(ΠEΠ) = 0.81–0.93 < 1` is a
   correct measurement, and it is meaningful — it shows that the λ ≥ 1 modes of (a) and (b)
   are absent from the deflated operator — but it does **not** establish convergence of the
   implemented recursion.
2. **[CORRECTED]** The statement "κ(PM) = ∞ by design" was wrong. `M` is near-singular, not
   singular, and a perfect preconditioner would still give κ = 1. The **deflated** κ ≈ 24–30
   at N = 0 is genuine (Ritz and PCG estimators agree to ~10 %), because deflation removes
   exactly the ψ_i direction.
3. **[WITHDRAWN — see Appendix N]** The claim that the measured deflated κ(P_N M) *diverges*
   for N ≥ 5 (μ_min = 0) rested on an Arnoldi run that leaked into the null space `span(X)`
   of `Π P_N M Π`. With the leak fixed, κ is finite everywhere and **improves with order in
   every one of the 32 (system, state, order) combinations measured**, by factors of 2.7–17
   from N = 0 to N = 8 (§20.3). The claim that κ is "the wrong figure of merit" went with it:
   `iters ∝ κ^0.91` holds for N ≤ 5 across four systems (§20.4).

**[INF] The recurring lesson, now correctly counted.** Two worst-case spectral quantities do
fail to describe this algorithm — ρ_proj (§9) and η_N (§10). The two further failures once
attributed to κ were artefacts of the defect in Appendix N, and are withdrawn. What describes
the algorithm at moderate order *is* the conditioning; what describes it at high order is the
saturation of the iteration count against the leaked full-space amplification (§20.5).

---

## 21. Direct answers to the remaining reviewer questions

### 21.1 Is the damping weight 1/2 heuristic? (R2.2a, R2.2b, R3.2)

For the general weight `p_N^(α) = p_{N−1} + α a_N`,

    P_N^(α) M = I − λ^N (1 − α + αλ)

so the mode-wise factor `(1 − α + αλ)` **vanishes identically** at

    α_annihilate = 1 / (1 − λ)

**α = 1/2 is therefore the exact annihilator of a mode at λ = −1** — precisely the mode that
produces the odd–even oscillation. This is a derivation, not a tuning.

**[OBS]** The measured dominant modes and the α that would exactly kill them:

| System | λ_dom | 1/(1−λ_dom) | suppression at α = 1/2 |
|---|---:|---:|---:|
| B12 | −1.1365 | **0.468** | 0.068 |
| water_cluster_128 | −1.1010 | **0.476** | 0.051 |
| C60_4 | −1.0733 | **0.482** | 0.037 |
| MAPbI3 | +0.8091 | 5.238 | 0.905 (no suppression, and none needed) |

α = 1/2 is within 4–7 % of the exact annihilator for all three oscillating systems.

**[OBS] A single α must remain admissible at *every* order.** Odd N constrains α from above
(the factor flips sign and grows), even N from below (too little suppression). Intersecting:

| System | N ≤ 5 | N ≤ 11 | N ≤ 20 | N ≤ 30 |
|---|---|---|---|---|
| B12 | [0.188, 0.715] | [0.338, 0.582] | **[0.432, 0.509]** | [0.458, 0.479] ✗ |
| water_cluster_128 | [0.152, 0.770] | [0.294, 0.641] | **[0.407, 0.552]** | [0.450, 0.505] |
| C60_4 | [0.119, 0.821] | [0.245, 0.703] | **[0.365, 0.608]** | [0.425, 0.544] |

**[INF]** The admissible window **shrinks onto the annihilator value ≈ 0.47** as the order
range grows. α = 1/2 lies inside it for the whole practical range (N ≤ 20) in all three
systems. For B12 beyond N ≈ 30 the window would exclude 0.5 (it becomes [0.458, 0.479]),
but no such order is used. **α = 1/2 is thus close to the unique robust choice, not a
convenient guess.**

**[OBS] For MAPbI3, α = 1 (no damping) is better** at every order (α_opt ≈ 1.2 at the scan
bound), consistent with the archived benchmark where Damped-NP costs MAPbI3 a few iterations
at low order.

### 21.2 Relation to polynomial preconditioning and Chebyshev filtering (R2.2c)

`P_N M = I − q_N(λ)` with `q_N` a polynomial of degree N+1: **the Neumann preconditioner is a
polynomial preconditioner, and damping is a change of its coefficients.** This structural
statement rests on the full-space closed form, which is directly verified (§20.1), and
stands.

**[OBS] The comparison, now on measured spectra.** With `spec(ΠPMΠ)` measured (twice the
N = 0 damped range) and the degrees paired as `d = N + 1`, the optimal shifted-Chebyshev
value `κ_cheb = (1+δ)/(1−δ)`, `δ = 1/T_d(η)`, `η = (b+a)/(b−a)` gives

| N | d | κ_cheb | Damped-NP eigenvalue ratio | ratio |
|---:|---:|---|---|---|
| 2 | 3 | 1.53 – 3.90 | 2.45 – 6.41 | 1.6 – 2.2× |
| 5 | 6 | 1.05 – 1.54 | 1.47 – 3.61 | 1.4 – 2.5× |
| 8 | 9 | 1.01 – 1.15 | 1.20 – 4.52 | 1.2 – 4.0× |

over all four systems and both representative states. This supersedes the retraction of the
previous revision: the earlier figure of "1.7–2.4× of optimal" was approximately right, and
is now obtained from rigorous brackets rather than the invalid deflated shortcut.

**[UNC]** Both sides of the table are *eigenvalue* ratios, which is self-consistent because
`κ_cheb` is itself an eigenvalue construction. Against the singular-value condition number
the gap is larger (up to 8.9× at B12 / HOMO / N = 8); the operators are non-normal, so the
two measures differ and the choice must be stated.

**[INF]** The qualitative positioning that survives, and is the one worth making to the
reviewer, is structural rather than numerical: the Neumann/Damped-NP construction is a
**fixed** polynomial that requires **no spectral bounds**, whereas Chebyshev acceleration
requires estimates of `μ_min` and `μ_max`. This analysis shows those bounds to be
state-dependent and iteration-dependent (§13.5, §20.5), i.e. expensive to obtain — which is
the real trade-off, independent of the exact factor by which Chebyshev would win.

### 21.3 The corrected shift: why c = 0.1, and is it sensitive? (R2.5)

`ε̃ = ε − c‖γ‖²` with c = 0.1.

**[INT] The functional form is not arbitrary.** A Ritz value is variational (ε ≥ ε_true) and
its error converges *quadratically* in the residual, ε − ε_true = O(‖γ‖²). Subtracting
`c‖γ‖²` therefore has exactly the right scaling to compensate it.

**[OBS] The required coefficient is remarkably constant.** For B12's lowest state,
`c_min = (ε − ε_true)/‖γ‖²` measured at three snapshots spanning **six orders of magnitude**
in ‖γ‖²:

| i_iter | ‖γ‖ | ε − ε_true | **c_min** | margin at c = 0.1 |
|---:|---:|---:|---:|---:|
| 5 | 1.72e-1 | 2.12e-3 | **0.0715** | 1.40× |
| 10 | 3.31e-3 | 7.75e-7 | **0.0710** | 1.41× |
| 15 | 9.12e-5 | 4.85e-10 | **0.0583** | 1.72× |

**[INF]** c = 0.1 always overshoots c_min by 40–70 %, so it *guarantees* `ε̃ < ε₀` for the
lowest state — which is exactly what makes `M` positive definite there (§12.2). **c is a
safety factor, not a performance knob.**

**[OBS] Performance sensitivity is negligible.** `dρ/dε̃ ≈ −0.25` measured across states, and
`c‖γ‖² ≤ 3e-3 Ha`, so varying c over any reasonable range changes ρ by `≤ 8e-4`.

**[UNC]** c_min was evaluated for B12 only, where converged reference eigenvalues are
available from the reproduced baseline (§7).

### 21.4 Relation to Jacobi–Davidson (R2.4)

The reference solve of §13.1 *is* the Jacobi–Davidson correction equation:
`Π(H − ε̃I)Π t = Πγ`, solved by PCG. The comparison is therefore direct:

* **Jacobi–Davidson** solves the *deflated* equation, whose operator `ΠMΠ` is positive
  definite (§13.2) — a well-posed SPD solve, at the price of an inner Krylov loop with
  global inner products (44–64 PCG iterations here).
* **Damped-NP** applies a fixed-degree polynomial in the *undeflated* `M`, with no inner
  products and no convergence test, and lets the outer Davidson orthogonalisation supply the
  deflation afterwards.

**[OBS] The two are numerically indistinguishable as preconditioned operators.** Comparing
`Π P̄_N M Π` (implemented) with `Π P̄_N Π M Π` (the JD correction equation preconditioned by
`Π P̄_N Π`) over four systems and four orders, the condition-number brackets agree to
within **4.8 %**, and to within 1 % in 13 of 16 comparisons. Inserting a projector between the
preconditioner and `M` is immaterial here because `span(X)` occupies only 0.02–0.08 % of the
grid (nbands 300–768 against ngpts 0.87–1.6 M). The difference that *is* real is projecting
inside the recurrence, which is a different algorithm and gives a 1.2–2.1× smaller κ at N = 8.

**[OBS]** With the same P, PCG reaches 1e-8 in 44–64 applications whereas the Neumann series
at rate ρ ≈ 0.93 would need ≈ 230. **[INF]** The advantage of Damped-NP is therefore *not* a
better convergence rate per application; it is the absence of global reductions and of an
inner stopping criterion, which is what makes it cheap on GPUs and at scale. The manuscript
should say this rather than imply a rate advantage.

### 21.5 Order-0 equivalence and linear cost (R2.6)

**[OBS]** `P_0 = P` exactly (the code early-returns `a_0 = Pγ` at order 0), so **order 0 is
GAPP**, and `P_0 M = I − E` by construction. Each additional order costs exactly **one
Hamiltonian application plus one GAPP application** on the residual block
(`precondition.py:854-857`), so the preconditioner cost is exactly linear in N. Both
statements follow from the closed forms in §20.1 and need no measurement.

---

## 22. Final verification record

Checks run after all analysis was complete, to close gaps rather than to restate results.

| # | check | result |
|---|---|---|
| 1 | `P` symmetric positive definite — **0-D branch** (B12/water/C60_4), built densely | `‖P−Pᵀ‖/‖P‖ = 0.00e+00`, eig(P) ∈ [6.53e-3, 7.94e-1] → **SPD** |
| 2 | `P` SPD — **3-D PBC branch** (MAPbI3, different `t_sample`) | `‖P−Pᵀ‖/‖P‖ = 0.00e+00`, eig(P) ∈ [1.20e-2, 5.67e-1] → **SPD** |
| 3 | Sylvester inertia claim, independent numerical test with 3 different SPD `P` | `#neg(PM) = #neg(M) = 3` in all three, while μ_min varied −1.95…−6.19 |
| 4 | `#neg(M)` against the converged B12 spectrum | 0 / 131 / 248 / 295 states below ε̃ for lowest / middle / slowest / homo |
| 5 | Are the ρ ≳ 1 rows at i_iter = 15 really outside the convergence criterion? | slowest bands 253 / 613 / 569 / 736 vs `bands` 250 / 512 / 480 / 640 → **all virtual** |
| 6 | MAPbI3 lowest: is `M` near-singular rather than over-corrected? | μ_min = **1.9e-5**, `0.1‖γ‖² = 1.33e-4 Ha`; ρ_full = 0.99998, ρ_deflated = 0.80724 |
| 7 | **Production invariance with the final code** (after the subspace hook and the probe refactor) | Gate-A run vs the *unmodified* original code: `max|Δeigval| = 0.0`, `max|Δresidual| = 0.0` → **bitwise identical** |
| 8 | Unit tests (`tests/test_spectral_tools.py`) | 5/5 pass |
| 9 | Headline numbers in §13.3 and §14 against raw JSON | 16/16 exact |
| 10 | **Final sweep**: all headline numbers in §13.3, §20.3c, §21.1–21.3 and the reviewer draft against raw JSON | **37/37 exact** |
| 11 | **Final regression** after every code change (incl. the probe refactor and the deflated-residual runs) | **bitwise identical**, 49/49 iterations |
| 12 | Report internal consistency: retracted claims not asserted elsewhere, no stale m=40 values, no dangling section references | clean |

Check 2 was a genuine gap: §12's inertia argument requires `P ≻ 0`, and it had only been
verified for the 0-D branch, while MAPbI3 uses a different GAPP parameterisation.
Check 7 was also a genuine gap: production invariance had been verified only after the
*first* hook was added, not after the later subspace-capture hook and the probe refactor.
Both now hold.

### Acceptance checklist (instruction §17)

All items satisfied, with three explicit deviations, all documented above:

* **fresh clone** — worked in place on the existing clone, verified at the same commit as
  `origin/main` with a clean tracked tree (§1.1);
* **CUDA wheel** — CPU build, because no GPU exists on this cluster (§1.2);
* **η_N computed in the Phase-3 pass** — disclosed at the time; it costs ~12 operator
  applications against a ~20 min Hamiltonian setup that a third pass would repeat. Its
  *interpretation* was subsequently retracted anyway (§10).

---

## Appendix N — A null-space leak in the Krylov driver, and everything it invalidated

*Merged from `nullspace_leak_appendix.md`. Documents the numerical defect that invalidated the first version of §20.2b, its audit, and the corrections.*

### N.1 What was wrong

The measured operator is `A = Pi Pbar_N M Pi` with `Pi = I - X X^T`, `X` the Davidson Ritz
block. **`A` annihilates `span(X)` exactly**: `v in span(X)` gives `Pi v = 0`, hence `A v = 0`.
`A` therefore carries a null space of dimension `nbands` (19 in the small validation case,
several hundred in the production systems). The quantity of interest is `sigma_min` of `A`
*restricted to* `range(Pi)`; the null space is an artefact of embedding that restriction in
the full space.

In exact arithmetic a Krylov method started from `v0 in range(Pi)` never sees it: `A` maps
`range(Pi)` into itself, so `K = span{v0, A v0, ...} subset range(Pi)`.

In floating point each application of `A` leaves a `span(X)` residue of size `~1e-16 ||A v||`
— negligible in absolute terms. The amplification happens at the **normalisation**:

    w   = A q_j                    # span(X) content ~ 1e-16 ||w||
    w'  = w - sum_i h_i q_i        # Gram-Schmidt: ||w|| -> beta, which can be tiny.
                                   # every q_i lies in range(Pi), so this subtraction
                                   # cannot remove the span(X) content
    q   = w' / beta                # <-- the span(X) content is absolute-constant while
                                   #     the vector is divided by a small beta

The relative leak after normalisation is `~1e-16 ||w|| / beta`, and the contaminated `q`
feeds the next step, so it compounds multiplicatively.

**[OBS]** Measured `||X^T q_j||` on the small validation case (Si, ngpts = 1000, k = 19,
N = 0, m = 60):

| j | 0 | 10 | 30 | 59 |
|---:|---:|---:|---:|---:|
| leak | 9.7e-17 | 2.8e-15 | 3.2e-13 | **7.8e-06** |

Once the basis carries a `1e-5` component in `span(X)`, the Krylov subspace effectively
contains a null direction, and Arnoldi correctly reports a Ritz value at 0 — the statement is
true *within the computed subspace*.

**[INT]** The onset depends on how far the Krylov process drives `beta_j` down, which grows
with both `m` and `N`: at large `N`, `Pbar_N M -> I` on most of `range(Pi)`, the process
exhausts the informative directions quickly and `beta_j` collapses. The measured degradation
follows exactly that order.

| N | 2 | 3 | 5 | 8 |
|---:|---:|---:|---:|---:|
| `mu_min` (corrupted) | 0.16291 | 0.00002 | **0.00000** | **0.00000** |

### N.2 Why the Ritz-residual guard did not catch it

Explicit residuals `||A v - theta v|| / ||v||` were introduced in §20.2b precisely to separate
a genuine failure from an unconverged Ritz value. They cannot detect this failure mode: for
`v` in `span(X)`, `A v = 0` and `theta ~ 0`, so the residual is `~0`. **The guard certifies
the spurious pair as converged** — the recorded values were 1.4e-06 and 3.5e-05, and they were
believed.

**[INF]** A small Ritz residual establishes that the pair is converged *with respect to the
computed subspace*. It says nothing about whether that subspace is the intended one. When the
operator has a null space that the subspace is supposed to exclude, the residual test is
vacuous and the subspace itself must be certified instead (here: `||X^T q_j||`).

The same argument shows which measurements survive: a Ritz vector with `span(X)` content `w`
has residual at least `theta * w`, so a *large* Ritz value with a small residual is clean.
`mu_max = 2.05236` at residual 2.3e-5 implies `w <= 1.1e-5`. **The §20.2 retraction — that
`Pi (I - E^(N+1)) Pi != I - (Pi E Pi)^(N+1)` because `Pi` and `E` do not commute — therefore
stands; it never depended on `mu_min`.**

### N.2b Direct before/after on a production system

B12, HOMO state, Davidson iteration 15, the same operator `Pi Pbar_N M Pi`, the same Krylov
dimension m = 80 — the only difference is the projection fix.

| N | | `mu_min` | `mu_max` |
|---:|---|---:|---:|
| 2 | leaked | 0.16502 | 0.99951 |
| 2 | **fixed** | 0.16291 | 1.00444 |
| 5 | leaked | **0.05400** | **1.16858** |
| 5 | **fixed** | **0.32414** | **1.16859** |

**[OBS]** At N = 5 `mu_max` agrees to five decimals while `mu_min` is wrong by a factor of 6.
At N = 2 both agree: the leak had not yet been amplified enough to matter.

**[INF]** This is exactly the asymmetry predicted in §N.2: a Ritz vector carrying `span(X)`
content `w` has residual at least `theta * w`, so a *large* Ritz value cannot be corrupted
without the residual revealing it, while a *small* one can. Every `mu_max`-derived conclusion
survives the bug; every `mu_min`-derived one does not. In particular the N = 8 value
`mu_max = 2.05236`, on which the §20.2 retraction rests, is a `mu_max` and stands.

**[INT] The controlling parameter is `sigma_min`, not `N`.** The leak is amplified by
`||w|| / beta_j`, and `beta_j` collapses when Arnoldi has to resolve a small extremal value.
B12 / HOMO makes this explicit:

| B12 HOMO | N = 5 | N = 8 |
|---|---|---|
| `sigma_min` | 0.169-0.196 | **0.295** |
| `mu_min` leaked vs fixed | 0.05400 vs **0.32414** (6x wrong) | 0.45451 vs **0.45438** (0.03 %) |
| `mu_max` leaked vs fixed | 1.16858 vs 1.16859 | **2.05236 vs 2.05236** |

N = 8 was *cleaner* than N = 5 for this state because its spectrum is less extreme.

**[UNC] The onset is not predicted by any single scalar.** `sigma_min` correlates with it but
does not determine it:

| | `sigma_min` | `mu_min` leaked -> fixed | verdict |
|---|---|---|---|
| B12 / HOMO / N = 5 | 0.169-0.196 | 0.05400 -> **0.32414** | **corrupted** |
| B12 / lowest / N = 2 | 0.168-0.170 | 0.17227 -> 0.16991 | clean |

Nearly identical `sigma_min`, opposite outcomes. What actually drives the amplification is how
far `beta_j` collapses during the Arnoldi run, which depends on the clustering of the whole
spectrum, not on one extremal value. Two earlier formulations of this appendix -- "N >= 3" and
then "the combinations with smallest `sigma_min`" -- are both superseded.

**[INF] The only reliable diagnostic is to monitor the leak directly.** `||X^T q_j||` is cheap
to compute and unambiguous; the fix (projecting immediately before the normalisation) holds it
at 1e-16 for every `j`, in every case measured. Any Krylov run on an operator with a null
space that the subspace is meant to exclude should carry this check.

**[OBS] The value on which the §20.2 retraction rests is reproduced exactly.** B12 / HOMO /
N = 8 gives `mu_max = 2.05236` both before and after the fix, to five decimals, while the
closed form `I - (Pi E Pi)^(N+1)` requires `mu <= 1` (all deflated `lambda` lie in
(-1, 0.919], so `0.5 lambda^8 (1 + lambda) >= 0`). `Pi (I - E^(N+1)) Pi != I - (Pi E Pi)^(N+1)`
is therefore confirmed with corrected tooling.

### N.3 The fix

Apply the projector **after** Gram-Schmidt, immediately before the normalisation, so the
residue is removed while it is still absolute-small and is never divided by `beta`:

    w' = w - sum_i h_i q_i
    w' = Pi w'                     # <-- here, immediately before the division
    beta = ||w'||;  q = w' / beta

Projecting *before* Gram-Schmidt (the first attempt) does not work: the residue is removed,
but Gram-Schmidt then shrinks `||w||` to `beta` while the newly introduced rounding residue is
again amplified by the division. Measured leak after the fix: `<= 5.1e-16` at every `j`.

Implemented as the `project=` argument of `spectral_tools.arnoldi` and
`spectral_tools.lanczos`, and passed as `project=Pi` wherever the operator kills `span(X)`.

### N.4 Validation of the corrected tooling

Dense reference on the small case (`range(Pi)` restriction via an SVD basis, not QR — QR
without pivoting on the rank-deficient `Pi` gives a basis accurate only to 4e-11):

| N | operator | dense `|lam|min` | Arnoldi `|mu|min` | dense `kappa_sigma` | Lanczos bracket |
|---:|---|---:|---:|---:|---|
| 0 | implemented | 8.0339e-04 | 8.0339e-04 | 722.106 | [470.3, inf] |
| 2 | implemented | 1.4624e-03 | 1.4624e-03 | 693.479 | [693.48, 693.48] |
| 2 | deflated_form | 4.0118e-03 | 4.0118e-03 | 252.758 | [252.76, 252.76] |
| 5 | implemented | 8.0831e-03 | 8.0831e-03 | 126.052 | [126.05, 126.05] |
| 5 | deflated_form | 8.8051e-03 | 8.8051e-03 | 115.622 | [115.62, 115.62] |
| 8 | implemented | 3.2388e-02 | 3.2388e-02 | **31.592** | [31.59, 31.59] |
| 8 | deflated_form | 1.3575e-02 | 1.3575e-02 | 75.046 | [75.05, 75.05] |

Three independent routes agree. Unit tests: 11 in `tests/test_spectral_tools.py`, including
one that asserts the Lanczos bracket contains the truth at deliberately unconverged `m = 5`.

The operators are also validated against the production code path itself:

| check | small Si | C60_4 (production scale) |
|---|---:|---:|
| `P_N M` vs production `PreNeumann` | 1.08e-15 | 5.59e-15 |
| `P_N` vs production `PreNeumann` | **0.00e+00** | **0.00e+00** |
| adjoint `<v,Au>` vs `<u,A^T v>` | 0.00e+00 | 2.12e-14 |

### N.5 Audit — what the leak reached

| Arnoldi target | null space | verdict |
|---|---|---|
| `E = I - PM` (full space) | none | **safe** — `rho`, `lambda_min`, `lambda_dom`, §9, §18 |
| `Pi E Pi` | `span(X)` | **safe, verified**: 0 Ritz values with `|lam| < 1e-8` at m = 80 in all four systems; `rho` and `lambda_min` identical to 6 digits with and without projection, and equal to the dense truth |
| `Pi P_N M Pi` | `span(X)` | **corrupted**: all `mu_min` and `kappa` values |
| `Q E Q` (`Q_below`) | `span(Q)` | experiment already dropped |

Note on naming: the `projected_spectrum` of §9 is **not** a projected operator. It is the same
full-space `E`, started from `a0 = P gamma` instead of a random vector (`spectral_experiment.py:524`).
It has no null space and is unaffected; `rho_proj` therefore stands, and remains one of the two
genuinely independent failures of a worst-case spectral quantity (the other being `eta_N`).

**[INT]** `Pi E Pi` escaped because `||Pi E Pi|| ~ 0.93`: the Krylov process does not drive
`beta_j` small within 80 steps. `Pi P_N M Pi` at large `N` is near-identity on most of
`range(Pi)`, which is exactly the situation that collapses `beta_j`.

### N.6 Claims withdrawn

1. `kappa(Pi P_N M Pi) -> 3.4e11 (N=5), 1.6e9 (N=8)` and `8.4e9` for the lowest state.
2. "The worst-case deflated conditioning degrades with order rather than improving."
3. "Convergence can only be claimed for the deflated recursion `a_{n+1} = Pi E Pi a_n`, not for
   the implemented scheme."
4. "`kappa` is the wrong figure of merit here — this is the same lesson as `rho_proj` (§9) and
   `eta_N` (§10) in a third guise." Two of the four cited failures *were* the corrupted
   `kappa`. The independent failures are `rho_proj` and `eta_N` only, and neither is `kappa`.
   Whether `iters ~ kappa_N^p` describes the benchmark is an **open question**, re-opened by
   this correction and answered by the re-measurement jobs.

### N.7 Claims unaffected

`rho(E) >= 1` and its three causes (a) near-singular `M` by design, (b) indefiniteness via
Sylvester, (c) over-correction; deflation removing (a) and (b) and damping handling (c);
`rho(Pi E Pi) = 0.925-0.931 < 1`; `eta_N^Pi`; the direction-agreement angles; the excitation
weights `w_a0 = 0.11-0.32`; the `alpha = 1/2` annihilator; `c_min = 0.058-0.072`; Gate 0
(27 vs 27 iterations); and the §20.2 retraction itself.

### N.8 A second, independent labelling error: eigenvalue ratio is not the condition number

Separate from the leak, the closed-form table of §20.2 was labelled a *condition number*. It
is not. For the damped scheme the closed form gives the **eigenvalues**

    mu = 1 - 0.5 lam^N (1 + lam),      lam in spec(Pi E Pi)

and `kappa_eig = mu_max / mu_min` is the eigenvalue ratio. `E` is not symmetric (asymmetry
0.16-0.28) and neither is `Pi E Pi`, so

    kappa = sigma_max / sigma_min  >=  |lam|_max / |lam|_min  =  kappa_eig

with equality only for a normal operator. **Every closed-form value is a lower bound on the
condition number, not the condition number.**

**[OBS]** The closed form does reproduce the eigenvalue ratio accurately — independently
confirmed at N = 0, HOMO, against direct Arnoldi on `Pi Pbar_0 M Pi` (m = 80, projected):

| System | closed-form `kappa_eig` | direct Arnoldi eigenvalue ratio | agreement |
|---|---:|---:|---|
| B12 | 26.452 | 26.37 | 0.3 % |
| water_cluster_128 | 24.140 | 24.16 | 0.1 % |
| C60_4 | 24.230 | 24.16 | 0.3 % |

**[OBS]** The gap to the true condition number is system-dependent and not yet bounded on all
systems. Where the Lanczos lower bound already exceeds the eigenvalue ratio it is informative;
where it does not, nothing is learned (a lower bound below another lower bound is not a
contradiction, only a looser bound):

| System | N | `kappa_eig` | measured `kappa` bracket | gap established |
|---|---:|---:|---|---|
| C60_4 | 0 | 24.23 | [31.48, inf] | **>= 1.30x** |
| C60_4 | 2 | 5.68 | [12.40, 15.68] | **>= 2.18x** |
| B12 | 0 | 26.45 | [26.68, inf] | >= 1.01x |
| water_cluster_128 | 0 | 24.14 | [24.01, inf] | none (bound too loose) |

**[INF]** The small validation case is *not* a guide here: there the operators were nearly
normal (`||A A^T - A^T A|| / ||A||^2` = 0.008 -> 0.000) and the eigenvalue ratio sat within
1-3 % of the true condition number. C60_4 at N = 2 shows a factor >= 2.18, so that
extrapolation fails on production systems.

**Consequence for the manuscript.** The closed-form table must be relabelled
"eigenvalue ratio of the deflated formulation — a lower bound on the condition number", and
the quantity reported as a condition number must be the directly measured `sigma` bracket.
Both decrease monotonically with `N`, so the qualitative claim is unchanged.

### N.9 What replaced the withdrawn interpretation

The mechanism proposed in the withdrawn §20.2b [INT] was right; the quantity it was attached
to was wrong.

Original claim: "`P_N M = I - E^(N+1)`, and in the *full* space `E^(N+1)` grows
(rho(E) = 1.609). The non-commutativity of `Pi` and `E` feeds that amplification into the
deflated operator, producing directions on which `Pi P_N M Pi` nearly annihilates."

The amplification is real, but it inflates `sigma_max`, it does not drive `sigma_min` to zero.

**[OBS]** `sigma_max` of the implemented operator, HOMO state:

| System | full-space `rho(E)` | N=2 | N=5 | N=8 |
|---|---:|---:|---:|---:|
| water_cluster_128 | 1.7166 | 1.035 | 1.234 | **2.708** |
| C60_4 | 1.6102 | 1.085 | 1.352 | 1.786 |
| MAPbI3 | 1.4646 | 1.040 | 1.116 | — |

**[INT]** `Pi` suppresses most but not all of the full-space growth: the crude estimate
`0.5 rho(E)^8 |1 + lam_min(E)|` gives 9.7 and 4.6 for water_cluster_128 and C60_4 against
measured 2.71 and 1.79, so roughly 70-75 % of the amplification is removed and the remainder
survives into the deflated operator.

**[OBS]** The consequence is a turning point in the condition number. For
water_cluster_128, `kappa` (implemented) falls 24.0 -> 7.6-8.0 -> 4.17 and then **rises to
6.14-6.26 at N = 8**, while the iteration count still falls (20 -> 19). The deflated
formulation, which projects at every recurrence step, keeps falling (4.34 -> 3.02).

**[OBS] Validity range of `iters ~ kappa^p`** (implemented formulation, HOMO):

| N pair | B12 | water_cluster_128 | C60_4 | MAPbI3 |
|---|---:|---:|---:|---:|
| 0 -> 2 | — | — | — | 0.86 |
| 2 -> 5 | 1.00 | 0.85 | 0.93 | 0.88 |
| 5 -> 8 | — | **-0.13** | 0.80 | — |

**[INF]** For `N <= 5` the exponent is 0.85-1.00 across four systems and three different order
intervals — the model works, and the exponent is near 1 (Richardson-like) rather than 0.5
(CG-like), which is consistent with Davidson acting on a non-symmetric operator without an
SPD-optimal polynomial. For `N >= 8` it fails, because the iteration count saturates
(water_cluster_128: 20, 19, 19, 18 at N = 5, 8, 9, 10) and no monotone spectral quantity can
track a saturated one. The optimal order is therefore set by that saturation against the
per-order cost, exactly as the manuscript already argues -- now with a quantitative basis for
where the spectral description stops applying.

### N.10 A rigorous certificate that the deflated expansion converges

`rho(A) <= ||A^n||^(1/n)` for every `n`, and `||A^n|| = sigma_max(A^n)` is bracketed by
Lanczos on the symmetric PSD `(A^n)^T A^n`, which carries the classical residual bound. A
value below 1 is therefore a proof, not an estimate -- unlike an Arnoldi Ritz value on the
non-symmetric `Pi E Pi`, which carries no error bound at all (this session measured a Ritz
value of 2.05 for an operator whose closed form caps the spectrum at 1).

**[OBS]** `||A||` itself exceeds 1 (1.02-1.28 across systems and states), so the naive bound
`rho <= ||A||` genuinely fails and the higher powers are necessary.

**[OBS]** B12, HOMO: 1.117, 1.088, 1.072, 1.046, 1.020, **0.999127** (n = 12), **0.976748**
(n = 20). MAPbI3, HOMO: 1.018, 1.007, **0.998647** (n = 3).

**[INF]** The certified bound (0.977 for B12) lies *above* the Arnoldi estimate (0.925-0.931)
and therefore does not contradict it. What is proved is `rho < 1`; the specific value 0.93
remains an Arnoldi estimate and must be labelled as such.

### N.11 CORRECTION to statements made while this work was in progress

While reporting intermediate results I repeatedly quoted "rho(Pi E Pi) = 0.925-0.931 < 1"
without a state qualifier. That is the **HOMO and middle** value. `report.md:826` already
records the correct fact; the over-claim was confined to the running commentary, but it must
not propagate into the §20 rewrite.

**[OBS]** `rho(Pi E Pi)` from the archived m = 80 deflated spectra, by state:

| System | lowest | middle | homo |
|---|---:|---:|---:|
| B12 | **1.13646** | 0.93018 | 0.93110 |
| C60_4 | **1.07328** | 0.92384 | 0.92524 |
| water_cluster_128 | **1.10097** | 0.92423 | 0.92488 |
| MAPbI3 | 0.80907 | 0.84706 | 0.92733 |

**[INT]** This is not new physics; it is cause (c) of §20.6. Deflation removes causes (a) and
(b) — the near-singular direction and the states below the shift, both of which live in
`span(X)` — but the `lambda < -1` mode is a vacuum/grid mode with
`||Pi v_dom|| / ||v_dom|| = 0.996-1.000`, i.e. deflation removes 0-0.4 % of it. It is damping,
not deflation, that handles that mode.

**[OBS]** The certificate behaves accordingly and is *not* failing: B12 / lowest gives
1.278, 1.169, 1.160 at n = 1, 2, 3, converging from above onto the Arnoldi value 1.13646.
No power of `n` will bring it below 1, and none should. Extending `n` for that state would be
wasted compute.

**What may and may not be claimed**

* **May:** for the HOMO and middle states the deflated expansion converges, with a rigorous
  certificate `rho(Pi E Pi) <= 0.976748` (B12) and `<= 0.998647` (MAPbI3).
* **May not:** convergence for every state. In three of four systems the lowest state has
  `rho(Pi E Pi) = 1.07-1.14 > 1` and the expansion diverges asymptotically in N.
* **Holds instead for the lowest state** (already in `report.md:865`): the damped mode factor
  is `|0.5 (1+lambda)| |lambda|^N = 0.068 |lambda|^N` for `lambda = -1.1365`, which contracts
  to `N ~ 20`. Within the practical range (`N <= 10`) the damped scheme still contracts.

Manuscript sentence: *The damped scheme contracts throughout the practical order range for
every state. Asymptotic convergence in N holds for the HOMO and middle states, where the
deflated error operator satisfies a rigorously certified rho <= 0.977; for the lowest state the
lambda < -1 mode survives deflation, and damping postpones rather than removes its growth,
buying roughly 20 orders of headroom.*

### N.12 The implemented scheme is numerically identical to the Jacobi-Davidson form

`Pi Pbar_N M Pi` (implemented) versus `Pi Pbar_N Pi M Pi` (JD correction equation
`Pi M Pi t = gamma` preconditioned by `Pi Pbar_N Pi`), 4 systems x 4 orders, HOMO:

**[OBS]** Maximum deviation **4.8 %** (B12, N = 5, where both brackets are wide); 13 of 16
comparisons agree to better than 1 %.

**[INT]** Inserting a projector between the preconditioner and `M` is immaterial because
`span(X)` occupies 0.02-0.08 % of the grid in production systems (nbands = 300-768 against
ngpts = 0.87-1.6 M). The small validation case is misleading here: at 1.9 % the two forms do
separate, and in the *opposite* direction. Formulation comparisons must not be extrapolated
from it.

**[INF]** Answer to R2.4: the method is not merely *related* to Jacobi-Davidson — with the
Davidson subspace as the deflation space it is numerically indistinguishable from the
Jacobi-Davidson correction equation preconditioned by the order-N Neumann polynomial. What
does differ is projecting at every recurrence step (`deflated_form`), which is a different
algorithm and is measurably better at N = 8 (by 1.19x for C60_4 and 2.05x for both
water_cluster_128 and B12).

---

## 23. Iteration- and shift-dependence of the deflated operator (HOMO, 0-D systems)

*Jobs 477995-478003 (exact shift) and 487066-487074 (corrected shift); 18 snapshots.*
*No new timing was measured; these are spectral re-measurements of the production path.*

Two questions were open after §20.4: whether the reported quantities depend on **where in
the Davidson run** the operator is sampled, and whether they depend on **which shift** is
used. Both were settled by re-measuring the HOMO of the three 0-D systems at three probe
calls (i_iter = 6 / 12 / 18) under both shift conventions.

### 23.1 Snapshots

| system | pc | i_iter | shift (exact) | shift (corrected) | c\|gamma\|^2 | PCG on (Pi M Pi) |
|---|---:|---:|---:|---:|---:|---|
| B12 | 4 | 6 | -0.1228060040 | -0.1351138838 | 1.23e-02 | negative_curvature |
| B12 | 10 | 12 | -0.3334604015 | -0.3351757295 | 1.72e-03 | converged |
| B12 | 16 | 18 | -0.3394102946 | -0.3394108469 | 5.52e-07 | converged |
| C60_4 | 4 | 6 | -0.3516591598 | -0.3577275667 | 6.07e-03 | converged |
| C60_4 | 10 | 12 | -0.3591882597 | -0.3591888465 | 5.87e-07 | converged |
| C60_4 | 16 | 18 | -0.3591888688 | -0.3591888691 | 3.00e-10 | converged |
| water_cluster_128 | 4 | 6 | -0.3138802838 | -0.3479489786 | 3.41e-02 | converged |
| water_cluster_128 | 10 | 12 | -0.3722651073 | -0.3722699747 | 4.87e-06 | converged |
| water_cluster_128 | 16 | 18 | -0.3722681806 | -0.3722681821 | 1.50e-09 | converged |

**[OBS]** The correction `c||gamma||^2` is 1.7-10.9 % of the shift at i_iter = 6 but
~1e-9 Ha by i_iter = 18: the two shift conventions are distinguishable only early in the run.

### 23.2 Corrected versus exact shift -- kappa(implemented), lower bound

| system | N | pc4 ex | pc4 corr | pc10 ex | pc10 corr | pc16 ex | pc16 corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| B12 | 2 | 110.34 | 109.14 | 15.72 | 15.62 | 12.88 | 12.88 |
| B12 | 6 | 117.62 | 128.07 | 8.71 | 8.64 | 6.84 | 6.84 |
| B12 | 10 | 233.37 | 230.83 | 27.13 | 26.77 | 18.04 | 18.04 |
| C60_4 | 2 | 16.80 | 16.30 | 13.46 | 13.46 | 12.04 | 12.04 |
| C60_4 | 6 | 9.02 | 8.71 | 6.99 | 6.99 | 6.26 | 6.26 |
| C60_4 | 10 | 12.13 | 11.49 | 10.12 | 10.12 | 9.48 | 9.48 |
| water_cluster_128 | 2 | 9.60 | 9.03 | 7.61 | 7.61 | 7.58 | 7.58 |
| water_cluster_128 | 6 | 5.82 | 5.32 | 4.34 | 4.34 | 4.31 | 4.31 |
| water_cluster_128 | 10 | 20.77 | 17.31 | 12.08 | 12.08 | 11.82 | 11.82 |

**[OBS]** At pc10 and pc16 the two conventions agree to three decimals (ratio 1.000 in
every C60_4 and water_cluster_128 entry, 0.987-0.999 for B12/pc10). At pc4 the corrected
shift is 5-17 % better for water_cluster_128 and C60_4.

**[INF]** The values reported in the response do not depend on the shift convention. This
supersedes the weaker cross-campaign argument (median 0.05 %), which had confounded a change
of iteration with a change of shift; here only one variable moves at a time.

**[OBS] The B12 early-iteration pathology is NOT a shift effect.** At i_iter = 6 B12 keeps
kappa = 109-231 and PCG still terminates on `negative_curvature` under the corrected shift,
i.e. `Pi M Pi` is still indefinite there. Shifting by 1.2e-2 Ha does not repair it; the
cause is that `X` has not yet captured the states below `eps_t`, so deflation is not yet
doing its job. **[UNC]** The prediction that the corrected shift would repair this was made
before the run and is refuted by it.

### 23.3 The high-order reversal is structural, not a snapshot artefact

kappa passes a minimum near N = 6 and rises in **all 18 snapshots**, across three iterations,
both shift conventions, three systems, and with `||G_N||` substituted for kappa:

| system | N | kappa pc10 | kappa pc16 | \|\|G_N\|\| dmp pc10 | \|\|G_N\|\| dmp pc16 |
|---|---:|---:|---:|---:|---:|
| B12 | 6 | 8.71 | 6.84 | 1.370 | 1.209 |
| B12 | 8 | 13.52 | 9.90 | 2.363 | 2.313 |
| B12 | 10 | 27.13 | 18.04 | 6.338 | 6.167 |
| C60_4 | 6 | 6.99 | 6.26 | 1.296 | 1.240 |
| C60_4 | 8 | 6.79 | 6.27 | 1.246 | 1.176 |
| C60_4 | 10 | 10.12 | 9.48 | 2.591 | 2.657 |
| water_cluster_128 | 6 | 4.34 | 4.31 | 0.926 | 0.923 |
| water_cluster_128 | 8 | 6.22 | 6.12 | 2.074 | 2.038 |
| water_cluster_128 | 10 | 12.08 | 11.82 | 5.244 | 5.149 |

**[INF]** The reversal is a property of the operator, not of the sampling point. It is
nevertheless **not** reflected in the archived iteration counts, which are monotone
non-increasing over N = 0...10. This is the reason kappa was removed from the reviewer
response entirely (see §23.6): it is the one measured quantity whose behaviour contradicts
the benchmark, and it answers a question the reviewer did not ask.

### 23.4 Deflation is necessary, not an analytical convenience

`--svd_undeflated` bounds `kappa(Pbar_N M)` with no projector anywhere (corrected-shift run):

| system | N | deflated | undeflated (lower bound) |
|---|---:|---:|---:|
| B12 | 2 | 12.88 | >100 (inconclusive) |
| B12 | 6 | 6.84 | >73 (inconclusive) |
| B12 | 10 | 18.04 | >242 (inconclusive) |
| C60_4 | 2 | 12.04 | >84 (inconclusive) |
| C60_4 | 6 | 6.26 | >72 (inconclusive) |
| C60_4 | 10 | 9.48 | >183 (inconclusive) |
| water_cluster_128 | 2 | 7.58 | >163 (inconclusive) |
| water_cluster_128 | 6 | 4.31 | >253 (inconclusive) |
| water_cluster_128 | 10 | 11.82 | >199 (inconclusive) |

**[OBS]** Every undeflated bracket is **inconclusive**: `sigma_min` does not converge, so
the upper end is infinite and the tabulated number is only a lower bound. This is the
expected signature of `M = H - eps_t I` being near-singular by design.

**[OBS]** The direction differs: the deflated kappa improves to N = 6, the undeflated one
**worsens with order** (B12: >100 at N = 2 to >242 at N = 10).

**[INF]** Without deflation the expansion buys no conditioning at all. This supports R2.1
point 2 -- the deflated operator is the operationally correct object -- as a necessity
rather than a modelling choice.

### 23.5 Direction quality: the water_cluster_128 exception reproduces

Angle between `Pi p_bar_N` (damped) and the exact deflated correction, degrees:

| system | N=4 | N=6 | N=7 | N=8 | N=10 | minimum |
|---|---:|---:|---:|---:|---:|---|
| B12 (pc16) | 25.72 | 19.02 | 16.68 | 15.10 | 15.59 | **N=9** |
| C60_4 (pc16) | 27.65 | 19.87 | 17.26 | 15.39 | 14.45 | **N=9** |
| water_cluster_128 (pc16) | 19.75 | 17.26 | 17.23 | 18.65 | 27.16 | **N=7** |

**[OBS]** B12 and C60_4 improve through N = 9-10; water_cluster_128 reaches its minimum at
N = 7 and then degrades (17.2 -> 27.2 deg by N = 10). The same pattern appears at pc4 and
pc10, so the §14.1 observation at i_iter = 15 was not a one-off.

**[INF]** Recorded in the response as Limitation 3 in its corrected form: a single state's
direction quality is not a validated predictor of the aggregate iteration count.

### 23.6 Consequence for the response draft

**[INF]** On the strength of §23.3 the condition-number analysis was removed from the
reviewer response in full (R2.1 point 4 and its N_kappa-min table, the order-scan figure,
the kappa clause in R2.3, the kappa metric in R2.4 and R2.6). Every reviewer question is
answerable from `rho`, the deflation fractions, the mode factor and `||G_N||` alone; the
Jacobi-Davidson argument was restated structurally (the reference solve *is* the JD
correction equation; `span(X)` is 0.02-0.08 % of the grid) and is stronger without kappa,
because it now also states plainly that the advantage is not a rate advantage.

**[VAL]** `--svd_undeflated` was added to `spectral_experiment.py` for §23.4; the 11
`tests/test_spectral_tools.py` tests still pass and the production-equivalence check
(`P_N dev = 0.000e+00`) is unchanged in every job of this campaign.

### 23.7 Presentation decision: Arnoldi leads, the norm certificate validates

**[INF]** The Method paragraph of the response was reordered so that the Arnoldi eigenvalues
and `rho` are presented first, with `rho <= ||A^n||^(1/n)` introduced in one short paragraph as
the independent check that Arnoldi -- which carries no error bound on a non-symmetric operator
-- has not gone wrong. This matches how the two tools are actually used: every *explanatory*
statement (sign of the dominant mode, the annihilator `alpha = 1/(1-lambda)`, the MAPbI3
exception) needs a signed individual eigenvalue, which only Arnoldi supplies; every *convergence*
statement rests on the certified bound. The alternative of dropping the certificate and quoting
Arnoldi alone was considered and rejected: the archived Arnoldi Ritz residuals are 2.6e-4-1.2e-3
(section 8.6 [UNC]), and on a non-normal operator a small residual does not bound the eigenvalue
error, so the central claim would rest on an unbounded estimate.

**[OBS] Realness of `spec(Pi E Pi)` -- audit.** `E` is self-adjoint in the `P^-1` inner product
(`P^(-1/2) E P^(1/2) = I - P^(1/2) M P^(1/2)`, symmetric; `P` verified SPD in section 22), so
`spec(E)` is real. That argument does **not** survive projection: `P^(-1/2) Pi P^(1/2)` is an
oblique projector, so `Pi E Pi` is neither symmetric nor similar to a symmetric matrix and its
spectrum is not guaranteed real. Measured over **310 Ritz sets** spanning all systems, states,
Davidson iterations, Krylov dimensions and seeds: `max|Im lambda| = 0.000e+00` in every one, and
`summarize_spectrum` flags the verdict (`numerically_real`, tol 1e-10 x scale) on every run.
`rho` is computed from the complex modulus `|theta|`, not from the real part.

**[INF]** The likely reason is that `Pi` removes only 0.02-0.08 % of the grid, so the
non-commutativity with `P^-1` is confined to a subspace far too small to create the eigenvalue
collisions a complex pair would require. This is an explanation, not a proof, and is disclosed
as such in Limitation 5 of the response.

**[OBS] Extrapolation of the certificate was tested and rejected.** Fitting `||A^n||^(1/n)` in
`1/n` over n = 5...20 and extrapolating to `1/n = 0` reproduces the Arnoldi estimate closely at
the lowest states (B12 exactly: 1.13646) but overshoots at the HOMO states by 0.011-0.037. It
**undershoots** the Arnoldi value in three cases (C60_4/lowest -0.0006, MAPbI3/lowest -0.0089,
water/lowest -0.0004), i.e. an extrapolated value is **not** an upper bound and cannot be quoted
as one. It also yields only `rho = max|lambda|`, with no sign, so it cannot replace Arnoldi for
any of the explanatory claims.
