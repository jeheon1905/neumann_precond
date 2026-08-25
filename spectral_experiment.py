"""Driver for the JCTC spectral-analysis revision.

The calculation workflow itself is *not* re-implemented here: this script
attaches a disabled-by-default probe to the production ``PreNeumann`` object
and then calls ``test.run_once`` unchanged, so the Hamiltonian, GAPP,
corrected shift and Neumann recurrence are exactly the production ones.

Usage (example)::

    python spectral_experiment.py \
        --filepath data/systems/B12.sdf --pbc 0 0 0 --supercell 1 1 1 \
        --spacing 0.2 --pp_type TM --filtering --outerorder 4 \
        --seed 2 --probe_call 6 --state_rule slowest \
        --krylov_dims 20 40 60 --seeds 0 1 \
        --outdir results_spectral_revision/global_spectrum/raw
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

from typing import Optional

import numpy as np
import torch

import spectral_tools as st


class _ProbeDone(Exception):
    """Raised to stop the Davidson run once the snapshot has been captured."""


class NeumannSpectralProbe:
    """One-shot probe attached to ``PreNeumann``.

    It captures, for one selected preconditioner call:
      * the residual block and its norms,
      * the raw Ritz values and the corrected shifts actually used,
      * the zero-shift fallback status,
      * the production Neumann recurrence terms ``a_0 ... a_ncapture``,
      * the exact code-level action of ``P`` and the Hamiltonian operator.
    """

    def __init__(self, target_call: int, n_capture: int = 4,
                 state_rules=None, virtual_factor: float = 1.2,
                 resid_below: float = 0.0):
        # ``resid_below`` > 0 replaces the fixed call index by a convergence
        # trigger: capture at the first call where EVERY selected state's
        # residual norm has fallen below the threshold.  This removes the
        # arbitrariness of picking an iteration and puts the measurement in the
        # regime where the shift is closest to the true eigenvalue.
        self.resid_below = float(resid_below)
        self.target_call = int(target_call)
        self.n_capture = int(n_capture)
        self.state_rules = list(state_rules or ["slowest"])
        self.virtual_factor = float(virtual_factor)
        self.armed = False
        self.captured = False
        self.data = {}
        self.terms = {}
        self.cols = []          # selected ACTIVE column indices
        self.rules = []         # rule name per entry of self.cols

    # -- hook 1: entry of PreNeumann.call -----------------------------
    def __call__(
        self,
        residue,
        H,
        raw_eigval,
        corrected_eigval,
        residue_norm,
        zero_shift_mask,
        apply_P,
        preconditioner,
        call_index,
    ):
        meta = getattr(preconditioner, "_davidson_meta", None)
        subspace = getattr(preconditioner, "_davidson_subspace", None)
        if self.captured:
            self.armed = False
            return
        rn = residue_norm.detach().reshape(-1)
        n_active = rn.numel()
        n_occupied = int(round(n_active / self.virtual_factor))
        if self.resid_below > 0.0:
            # convergence-triggered: wait until every selected state is below
            # the threshold; the bands are still active (Davidson locks at
            # diag_tol, which must be set tighter than this threshold).
            probe_cols = []
            for r in self.state_rules:
                if isinstance(r, str) and r.startswith("stride:"):
                    continue
                c = _select_column(rn, r, n_active, n_occupied)
                if c is not None:
                    probe_cols.append(c)
            if not probe_cols or float(rn[probe_cols].max().item()) >= self.resid_below:
                self.armed = False
                return
        elif call_index != self.target_call:
            self.armed = False
            return
        self.armed = True

        # Choose the columns HERE so that only those are ever copied; storing
        # whole blocks would cost n_active x ngpts x 8 bytes per recurrence term
        # (~30 GB for MAPbI3).
        rules = []
        for r in self.state_rules:
            if isinstance(r, str) and r.startswith("stride:"):
                k = int(r.split(":")[1])
                rules += [str(b) for b in range(0, n_occupied, k)]
            else:
                rules.append(r)
        sel, seen = [], set()
        for r in rules:
            c = _select_column(rn, r, n_active, n_occupied)
            c = max(0, min(int(c), n_active - 1))
            if c not in seen:
                seen.add(c)
                sel.append((r, c))
        self.rules = [r for r, _ in sel]
        self.cols = [c for _, c in sel]
        idx = torch.tensor(self.cols, device=residue.device)

        self.data = {
            "call_index": int(call_index),
            "davidson_meta": meta,
            "subspace": subspace,      # reference to U_ (= X_ at i_b == 1); not copied
            "residue_sel": residue.detach().index_select(1, idx).clone(),
            "residue_norm": rn.clone(),
            "raw_eigval": raw_eigval.detach().reshape(-1).clone(),
            "corrected_eigval": corrected_eigval.detach().reshape(-1).clone(),
            "zero_shift_mask": (
                None if zero_shift_mask is None
                else zero_shift_mask.detach().reshape(-1).clone()
            ),
            "n_active": n_active,
            "n_occupied": n_occupied,
            "H": H,
            "apply_P": apply_P,
            "preconditioner": preconditioner,
        }

    # -- hook 2: each recurrence term ---------------------------------
    def on_term(self, n, term, active_indices):
        if not self.armed:
            return
        if active_indices is not None:
            # a fixed integer order performs no column filtering; verify it
            ai = active_indices.tolist()
            if ai[: len(ai)] != list(range(len(ai))) or len(ai) != self.data["n_active"]:
                raise RuntimeError(
                    "active-column filtering occurred; column mapping is no longer "
                    "valid for the probe (use a fixed integer --outerorder)"
                )
        idx = torch.tensor(self.cols, device=term.device)
        self.terms[int(n)] = term.detach().index_select(1, idx).clone()
        if int(n) >= self.n_capture:
            self.captured = True
            raise _ProbeDone()


# ----------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _select_column(rn, rule: str, n_active: int, n_occupied: int) -> int:
    """Pick an active column from the residual norms."""
    if rule == "slowest":
        return int(torch.argmax(rn).item())
    if rule == "fastest":
        return int(torch.argmin(rn).item())
    if rule == "lowest":
        return 0
    if rule == "middle":
        return n_occupied // 2
    if rule == "homo":
        return int(n_occupied - 1)
    if rule == "lumo":
        return int(n_occupied)
    if rule == "highest":
        return n_active - 1
    return int(rule)


def select_state(probe_data, rule: str, n_occupied: Optional[int] = None) -> int:
    """Pick the active column to analyse."""
    rn = probe_data["residue_norm"]
    n_active = rn.numel()
    if rule == "slowest":
        return int(torch.argmax(rn).item())
    if rule == "fastest":
        return int(torch.argmin(rn).item())
    if rule == "lowest":
        return 0
    if rule == "middle":
        return (n_occupied or n_active) // 2
    if rule == "homo":
        return int(n_occupied - 1) if n_occupied else n_active // 2
    if rule == "lumo":
        return int(n_occupied) if n_occupied else n_active // 2
    if rule == "highest":
        return n_active - 1
    return int(rule)  # explicit active-column index


# ----------------------------------------------------------------------
def validate_operator(E, probe, pos, apply_P, verbose=True):
    """Phase-1 validation: apply_P equivalence, linearity, recurrence equivalence."""
    out = {}
    dev = probe.data["residue_sel"].device
    dtype = probe.data["residue_sel"].dtype
    ngpts = probe.data["residue_sel"].shape[0]

    # (a) apply_P vs the production expression gapp(x).mul_(INV_4PI)
    pre = probe.data["preconditioner"]
    g = torch.Generator(device="cpu").manual_seed(12345)
    x = torch.randn(ngpts, 1, generator=g, dtype=torch.float64).to(dev).to(dtype)
    p_new = apply_P(x)
    p_old = pre.gapp(x.clone()).mul_(0.25 / np.pi)
    out["delta_apply_P"] = float(
        (torch.linalg.vector_norm(p_new - p_old) / torch.linalg.vector_norm(p_old)).item()
    )

    # (b) linearity
    g2 = torch.Generator(device="cpu").manual_seed(54321)
    y = torch.randn(ngpts, 1, generator=g2, dtype=torch.float64).to(dev).to(dtype)
    alpha, beta = 1.7, -0.35
    lhs = E(alpha * x + beta * y)
    rhs = alpha * E(x) + beta * E(y)
    out["delta_linearity"] = float(
        (torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(rhs)).item()
    )

    # (c) recurrence equivalence: a_{n+1}^existing vs E(a_n)
    deltas = {}
    ns = sorted(k for k in probe.terms if k + 1 in probe.terms)
    for n in ns:
        a_n = probe.terms[n][:, pos : pos + 1].contiguous()
        a_np1 = probe.terms[n + 1][:, pos : pos + 1].contiguous()
        pred = E(a_n)
        deltas[str(n)] = float(
            (
                torch.linalg.vector_norm(a_np1 - pred)
                / torch.linalg.vector_norm(a_np1)
            ).item()
        )
    out["delta_recurrence"] = deltas
    if verbose:
        print("[validate] delta_apply_P   =", out["delta_apply_P"])
        print("[validate] delta_linearity =", out["delta_linearity"])
        print("[validate] delta_recurrence=", deltas)
    return out


def run_arnoldi_study(E, v0, dims, label, apply_for_residual=True, verbose=True,
                      project=None):
    """Arnoldi at several Krylov dimensions; returns a list of records.

    ``project`` must be supplied for operators with a null space (Pi E Pi kills
    span(X)); see the note in ``spectral_tools.arnoldi``.
    """
    records = []
    for m in dims:
        t0 = time.time()
        res = st.arnoldi(E, v0, m, project=project)
        theta, Y = st.ritz_pairs(res)
        summ = st.summarize_spectrum(theta)
        est = st.arnoldi_residual_estimate(res, Y)

        targets = {
            "largest_magnitude": summ["idx_largest_magnitude"],
            "most_negative": summ["idx_most_negative"],
            "most_positive": summ["idx_most_positive"],
        }
        tgt_out = {}
        for name, idx in targets.items():
            lam = complex(theta[idx])
            rec = {
                "eigenvalue_real": lam.real,
                "eigenvalue_imag": lam.imag,
                "magnitude": abs(lam),
                "damping_factor": st.damping_factor(lam),
                "arnoldi_residual_estimate": float(est[idx]),
            }
            if apply_for_residual:
                r, _ = st.explicit_ritz_residual(res, lam, Y[:, idx], E)
                rec["ritz_residual"] = r
            tgt_out[name] = rec

        records.append(
            {
                "label": label,
                "krylov_dim_requested": m,
                "krylov_dim_built": res.m,
                "breakdown": res.breakdown,
                "ortho_error": res.ortho_error,
                "summary": summ,
                "targets": tgt_out,
                "ritz_values_real": theta.real.tolist(),
                "ritz_values_imag": theta.imag.tolist(),
                "seconds": time.time() - t0,
            }
        )
        if verbose:
            print(
                f"[arnoldi:{label}] m={res.m:3d} rho={summ['rho']:.6f} "
                f"lam_min={summ['lambda_min_real']:.6f} lam_max={summ['lambda_max_real']:.6f} "
                f"real={summ['numerically_real']} ortho={res.ortho_error:.2e} "
                f"({time.time()-t0:.1f}s)"
            )
    return records


# ----------------------------------------------------------------------
def main():
    import test as test_mod  # production driver

    p = test_mod.build_argparser()
    # analysis-specific arguments
    p.add_argument("--probe_call", type=int, default=6,
                   help="PreNeumann call index to probe (0-based)")
    p.add_argument("--state_rules", type=str, nargs="+", default=None,
                   help="analyse several states in one snapshot, e.g. lowest middle homo slowest")
    p.add_argument("--state_rule", type=str, default="slowest",
                   help="slowest|lowest|middle|homo|<int active column index>")
    p.add_argument("--krylov_dims", type=int, nargs="+", default=[20, 40, 60])
    p.add_argument("--arnoldi_seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--outdir", type=str, default="results_spectral_revision/global_spectrum/raw")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--projected", action="store_true",
                   help="also run the residual-generated (a_0-started) Krylov projection")
    p.add_argument("--projected_dims", type=int, nargs="+", default=None,
                   help="Krylov dimensions for the projected spectrum (default: --krylov_dims)")
    p.add_argument("--skip_global", action="store_true",
                   help="skip the global spectrum (use when it has already been computed)")
    p.add_argument("--deflated", action="store_true",
                   help="deflated analysis: Pi = I - X X^H with X the Davidson subspace")
    p.add_argument("--deflated_dims", type=int, nargs="+", default=[20, 40],
                   help="Krylov dimensions for rho(Pi E Pi)")
    p.add_argument("--cg_tol", type=float, default=1e-10)
    p.add_argument("--cg_maxiter", type=int, default=300)
    p.add_argument("--deflated_recursion", type=int, default=0,
                   help="if >0: measure ||(Pi E Pi)^n a0|| -- does the DEFLATED recursion "
                        "converge, as rho(Pi E Pi) < 1 predicts?")
    p.add_argument("--minimal_projector", action="store_true",
                   help="also use Q_i = I - X_<i X_<i^H, removing ONLY the Ritz vectors whose "
                        "value lies below eps_t (the minimal projector making M pos. def.)")
    p.add_argument("--weights", type=int, default=0,
                   help="if >0: measure how strongly the residual excites the DIVERGENT modes "
                        "and how much of the dominant eigenvector survives deflation")
    p.add_argument("--verify_order_form", type=int, nargs="+", default=None,
                   help="directly measure the spectrum of P_N M (production PreNeumann) and "
                        "compare with the closed form I - E^(N+1) / I - 0.5 E^N (I+E)")
    p.add_argument("--verify_dim", type=int, default=30)
    p.add_argument("--svd_bounds", type=int, nargs="+", default=None,
                   help="orders N for RIGOROUS singular-value / condition-number bounds")
    p.add_argument("--svd_dim", type=int, default=80,
                   help="Lanczos dimension for the A^T A run")
    p.add_argument("--svd_undeflated", action="store_true",
                   help="additionally bound kappa(P_N M) with NO projector at all, on the "
                        "full space.  M is near-singular by design (eps_t approximates an "
                        "eigenvalue), so this is expected to be huge and to get WORSE as "
                        "Davidson converges -- it quantifies what deflation is worth")
    p.add_argument("--error_norm", type=int, nargs="+", default=None,
                   help="orders N at which to measure ||Pi G_N Pi|| rigorously, for BOTH "
                        "the undamped (w=1) and damped (w=0.5) schemes; G_N = I - P_N M is "
                        "the contraction factor actually achieved at order N")
    p.add_argument("--error_norm_dim", type=int, default=30)
    p.add_argument("--probe_resid_below", type=float, default=0.0,
                   help="capture at the first call where every selected state's residual "
                        "norm is below this value, instead of at a fixed --probe_call; "
                        "puts the measurement in the converged regime where the shift is "
                        "closest to the true eigenvalue")
    p.add_argument("--exact_shift", action="store_true",
                   help="use the raw Davidson Ritz value as the shift instead of the "
                        "corrected eps - c||gamma||^2; makes M exactly singular at the "
                        "target state, so only the DEFLATED operators are well defined")
    p.add_argument("--rho_bound", type=int, nargs="+", default=None,
                   help="powers n for a RIGOROUS upper bound rho(Pi E Pi) <= ||A^n||^(1/n); "
                        "Arnoldi Ritz values carry no error bound on a non-symmetric "
                        "operator, so they cannot certify rho < 1 on their own")
    p.add_argument("--rho_bound_dim", type=int, default=40)
    p.add_argument("--jd_form", action="store_true",
                   help="also bound kappa(Pi P_N Pi M Pi), the Jacobi-Davidson form, "
                        "in which the projector sits BETWEEN the preconditioner and M")
    p.add_argument("--svd_arnoldi_dim", type=int, default=0,
                   help="if > 0, also re-run Arnoldi on the implemented operator at this "
                        "dimension to test whether mu_min -> 0 was a convergence artifact")
    p.add_argument("--eta_max_order", type=int, default=0,
                   help="if >0, also report eta_N from the captured recurrence terms")
    p.add_argument("--dense", action="store_true",
                   help="build the explicit dense E (small systems only)")
    p.add_argument("--dense_max_ngpts", type=int, default=4000)

    args = p.parse_args()
    test_mod.args = args  # resolve_upf_files() reads a module-level `args`

    from gospel.ParallelHelper import ParallelHelper as PH
    from gospel.util import set_global_seed

    PH.init_from_env(args.use_cuda)
    set_global_seed(args.seed + PH.rank)
    torch.set_num_threads(os.cpu_count() if args.threads is None else args.threads)

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or (
        f"{os.path.splitext(os.path.basename(args.filepath))[0]}"
        f"_o{args.outerorder}_seed{args.seed}_call{args.probe_call}"
    )

    probe = NeumannSpectralProbe(
        target_call=args.probe_call,
        n_capture=min(int(args.outerorder), 4),
        state_rules=args.state_rules or [args.state_rule],
        virtual_factor=args.virtual_factor,
        resid_below=args.probe_resid_below,
    )

    # attach the probe by wrapping the production preconditioner builder
    orig_build = test_mod.build_preconditioner

    def build_and_attach(calc, a):
        orig_build(calc, a)
        pre = calc.eigensolver.preconditioner
        pre.spectral_probe = probe
        pre.term_probe = probe.on_term
        print(f"[probe] attached to {type(pre).__name__}, target_call={probe.target_call}")

    test_mod.build_preconditioner = build_and_attach

    t_start = time.time()
    try:
        test_mod.run_once(args)
        raise SystemExit("probe never fired: increase --diag_iter or lower --probe_call")
    except _ProbeDone:
        print(f"[probe] snapshot captured after {time.time()-t_start:.1f}s")

    d = probe.data
    ngpts = d["residue_sel"].shape[0]
    n_active = d["n_active"]
    n_occupied = d["n_occupied"]

    # (rule, active-column index, position inside the stored slices)
    rules = probe.rules
    cols = [(r, c, i) for i, (r, c) in enumerate(zip(probe.rules, probe.cols))]

    print(f"[snapshot] ngpts={ngpts} n_active={n_active} n_occupied={n_occupied}")
    print(f"[snapshot] davidson_meta i_iter={d['davidson_meta']['i_iter']} "
          f"i_b={d['davidson_meta']['i_b']} call={d['call_index']}")
    print(f"[snapshot] states: {[(r, c) for r, c, _ in cols]}")

    payload = {
        "tag": tag,
        "datetime": datetime.datetime.now().isoformat(),
        "commit": git_commit(),
        "host": platform.node(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in vars(args).items()},
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(PH.get_device()),
            "threads": torch.get_num_threads(),
        },
        "system": os.path.basename(args.filepath),
        "ngpts": int(ngpts),
        "n_active": int(n_active),
        "n_occupied": n_occupied,
        "preconditioner_call": int(d["call_index"]),
        "davidson_meta": d["davidson_meta"],
        "memory_estimate_bytes": st.memory_estimate_bytes(ngpts, max(args.krylov_dims)),
        "states": [],
    }

    band_list = (d["davidson_meta"] or {}).get("band_index")

    for rule, col, pos in cols:
        band_index = band_list[col] if band_list is not None else None
        raw_eps = float(d["raw_eigval"][col].item())
        # --exact_shift makes M = H - eps I exactly singular at this state, so
        # only the DEFLATED operators remain well defined; the full-space
        # Neumann series does not exist in that limit.
        shift = raw_eps if args.exact_shift else float(d["corrected_eigval"][col].item())
        gamma_norm = float(d["residue_norm"][col].item())
        zsf = (
            bool(d["zero_shift_mask"][col].item())
            if d["zero_shift_mask"] is not None else False
        )

        print(f"\n[state:{rule}] col={col} band={band_index} "
              f"raw_eps={raw_eps:.10f} shift_used={shift:.10f} "
              f"|gamma|={gamma_norm:.6e} zero_shift_fallback={zsf}"
              f"{'  [EXACT SHIFT]' if args.exact_shift else ''}")
        # How well does the Davidson block actually contain this state?  If it
        # does not, the lambda ~ 1 mode is not removed by Pi and sigma_min
        # collapses -- the same failure mode as Appendix N, from a different cause.
        _X = d.get("subspace")
        if _X is not None:
            _psi = d["subspace_target"][:, col:col+1] if "subspace_target" in d else None
            _v = d["residue_sel"][:, pos:pos+1]
            _Xc = _X.to(_v.dtype)
            leak_gamma = float(torch.linalg.vector_norm(
                _v - (_v - _Xc @ (_Xc.transpose(0, 1) @ _v))).item()
                / torch.linalg.vector_norm(_v).item())
            print(f"[state:{rule}] ||X^T gamma||/||gamma|| = {leak_gamma:.3e}  "
                  f"(residual is orthogonal to the Davidson block by construction)")

        E = st.ErrorPropagationOperator(d["H"], d["apply_P"], shift)

        entry = {
            "state_rule": rule,
            "active_index": int(col),
            "state_index": band_index,
            "residual_norm": gamma_norm,
            "raw_epsilon": raw_eps,
            "corrected_epsilon": shift,
            "zero_shift_fallback": zsf,
            "dtype": str(d["residue_sel"].dtype),
        }

        # ---- Phase 1 validation (per state: the shift differs) ----
        entry["validation"] = validate_operator(
            E, probe, pos, d["apply_P"], verbose=(rule == rules[0])
        )

        # ---- optional dense check (small systems only) ----
        if args.dense:
            if ngpts > args.dense_max_ngpts:
                entry["dense"] = {"skipped": f"ngpts={ngpts} > {args.dense_max_ngpts}"}
            else:
                I = torch.eye(ngpts, dtype=d["residue_sel"].dtype, device=d["residue_sel"].device)
                Edense = E(I)
                ev = np.linalg.eigvals(Edense.cpu().numpy())
                entry["dense"] = {
                    "eigenvalues_real": ev.real.tolist(),
                    "eigenvalues_imag": ev.imag.tolist(),
                    "rho": float(np.max(np.abs(ev))),
                    "lambda_min_real": float(np.min(ev.real)),
                    "lambda_max_real": float(np.max(ev.real)),
                    "asymmetry": float(
                        (Edense - Edense.T).abs().max().item() / Edense.abs().max().item()
                    ),
                }
                print("[dense] rho =", entry["dense"]["rho"],
                      " lam_min =", entry["dense"]["lambda_min_real"],
                      " asym =", entry["dense"]["asymmetry"])

        # ---- Phase 2: global spectrum ----
        if not args.skip_global:
            recs = []
            for sd in args.arnoldi_seeds:
                g = torch.Generator(device="cpu").manual_seed(int(sd))
                v0 = torch.randn(ngpts, 1, generator=g, dtype=torch.float64)
                v0 = v0.to(d["residue_sel"].device).to(d["residue_sel"].dtype)
                rr = run_arnoldi_study(E, v0, args.krylov_dims,
                                       label=f"{rule}/global_seed{sd}")
                for r in rr:
                    r["seed"] = int(sd)
                recs += rr
            entry["global_spectrum"] = recs

        # ---- Phase 3 (conditional): residual-generated Krylov projection ----
        if args.projected:
            a0 = probe.terms[0][:, pos : pos + 1].contiguous()
            dims = args.projected_dims or args.krylov_dims
            recs = run_arnoldi_study(E, a0, dims, label=f"{rule}/projected")
            for r in recs:
                r["seed"] = None
                r["start_vector"] = "a0 = P gamma"
            entry["projected_spectrum"] = recs
            entry["a0_norm"] = float(torch.linalg.vector_norm(a0).item())

        # ---- Phase 4 (conditional): eta_N directly from the recurrence ----
        if args.eta_max_order > 0:
            a = probe.terms[0][:, pos : pos + 1].contiguous()
            a0n = float(torch.linalg.vector_norm(a).item())
            a_norms = [a0n]
            eta_np, eta_dnp = [], []
            prev = a
            for n in range(1, args.eta_max_order + 2):
                cur = E(prev)
                a_norms.append(float(torch.linalg.vector_norm(cur).item()))
                eta_np.append(a_norms[-1] / a0n)
                eta_dnp.append(
                    float(torch.linalg.vector_norm(prev + cur).item()) / (2.0 * a0n)
                )
                prev = cur
            entry["eta"] = {
                "a_norms": a_norms,
                "eta_NP": eta_np,
                "eta_DNP": eta_dnp,
                "max_order": int(args.eta_max_order),
            }
            print(f"[eta:{rule}] N :  eta_NP        eta_DNP")
            for n in range(len(eta_np)):
                print(f"[eta:{rule}] {n:2d}: {eta_np[n]:.6e}  {eta_dnp[n]:.6e}")


        # ---- Deflated analysis --------------------------------------
        # Davidson removes the span(X) component of the preconditioned block
        # before using it, and the unstable directions of E live in span(X).
        # Everything below therefore measures the part that actually survives.
        if args.deflated:
            X = d.get("subspace")
            if X is None:
                entry["deflated"] = {"skipped": "no Davidson subspace captured"}
            else:
                X = X.to(d["residue_sel"].dtype)
                Pi = st.Deflator(X)
                dfl = {"n_subspace": int(X.shape[1]),
                       "X_orthonormality_error": Pi.orthonormality_error()}
                print(f"[deflate:{rule}] X shape={tuple(X.shape)} "
                      f"||X^H X - I||_max={dfl['X_orthonormality_error']:.2e}")

                gamma = d["residue_sel"][:, pos:pos+1].contiguous()
                dfl["gamma_norm"] = float(torch.linalg.vector_norm(gamma).item())
                dfl["Pi_gamma_norm"] = float(torch.linalg.vector_norm(Pi(gamma)).item())
                print(f"[deflate:{rule}] ||gamma||={dfl['gamma_norm']:.4e} "
                      f"||Pi gamma||={dfl['Pi_gamma_norm']:.4e} "
                      f"(residual is orthogonal to X by construction)")

                # (a) rho(Pi E Pi) -- via run_arnoldi_study so that EXPLICIT Ritz
                #     residuals ||Ev - theta v||/||v|| are computed for the extremal
                #     pairs (largest magnitude / most negative / most positive).
                #     kappa needs both ends, so all three targets are relevant.
                Edef = st.DeflatedOperator(E, Pi)
                g = torch.Generator(device="cpu").manual_seed(0)
                v0 = Pi(torch.randn(ngpts, 1, generator=g,
                                    dtype=torch.float64).to(X.device).to(X.dtype))
                recs = run_arnoldi_study(Edef, v0, args.deflated_dims,
                                         label=f"{rule}/deflated", project=Pi)
                for r in recs:
                    r["seed"] = 0
                    r["start_vector"] = "Pi * random"
                dfl["deflated_spectrum"] = recs

                # (b) exact deflated correction  (Pi M Pi) t = Pi gamma  by PCG
                shift_l = shift
                def applyA(v):
                    v = Pi(v)
                    return Pi(E.H @ v - shift_l * v)
                def applyMprec(v):
                    return Pi(d["apply_P"](Pi(v)))
                if args.cg_maxiter <= 0:
                    t_ref, hist, status = Pi(gamma), [], "skipped"
                else:
                    t_ref, hist, status = st.pcg(applyA, applyMprec, Pi(gamma),
                                                 tol=args.cg_tol, maxiter=args.cg_maxiter)
                dfl["cg_status"] = status
                dfl["cg_iters"] = len(hist)
                dfl["cg_final_relres"] = hist[-1] if hist else None
                print(f"[deflate:{rule}] PCG on (Pi M Pi)t = Pi*gamma : {status}, "
                      f"{len(hist)} iters, "
                      f"relres={hist[-1] if hist else float('nan'):.2e}")

                # (c) deflated eta and direction agreement, order by order
                a = probe.terms[0][:, pos:pos+1].contiguous()
                Pa0 = Pi(a)
                n0 = float(torch.linalg.vector_norm(Pa0).item())
                p_np = a.clone()                # p_N   (undamped, running sum)
                rows = []
                prev = a
                for N in range(0, args.eta_max_order + 1):
                    cur = E(prev)               # a_{N+1}
                    Pcur = Pi(cur)
                    # p_bar_N = p_{N-1} + 0.5 a_N ;  p_np currently holds p_N, prev holds a_N
                    p_dnp = (p_np - prev) + 0.5 * prev
                    rows.append({
                        "N": N,
                        "eta_defl_NP": float(torch.linalg.vector_norm(Pcur).item()) / n0,
                        "eta_defl_DNP": float(
                            torch.linalg.vector_norm(Pi(prev + cur)).item()) / (2.0 * n0),
                        "dir_NP": st.direction_agreement(Pi(p_np), t_ref),
                        "dir_DNP": st.direction_agreement(Pi(p_dnp), t_ref),
                    })
                    p_np = p_np + cur
                    prev = cur
                dfl["order_scan"] = rows
                print(f"[deflate:{rule}]  N |  eta^Pi_NP   eta^Pi_DNP |  angle_NP  angle_DNP (deg)")
                for r in rows:
                    print(f"[deflate:{rule}] {r['N']:2d} | {r['eta_defl_NP']:11.5f} "
                          f"{r['eta_defl_DNP']:11.5f} | {r['dir_NP']['angle_deg']:9.3f} "
                          f"{r['dir_DNP']['angle_deg']:10.3f}")
                entry["deflated"] = dfl


        # ---- Direct verification of the closed form for P_N M -------------
        # P_N M = I - E^(N+1)  (undamped) / I - 0.5 E^N (I+E)  (damped) is an identity
        # in the FULL space.  Pi and E do not commute, so applying it to the DEFLATED
        # spectrum is an approximation -- both are tested here against the production
        # PreNeumann.call used as the order-N operator.
        if args.verify_order_form:
            from precondition import create_preconditioner
            pre = d["preconditioner"]
            X = d.get("subspace")
            Pi = st.Deflator(X.to(d["residue_sel"].dtype)) if X is not None else None
            shift_l = shift
            ver = []
            for N in args.verify_order_form:
                # a second PreNeumann with correction_scale = 0 so that the shift we pass
                # IS the corrected shift (production applies the correction internally)
                pn = create_preconditioner("neumann", pre.grid, False, {
                    "fp": "DP", "order": int(N), "correction_scale": 0.0,
                    "no_shift_thr": pre.no_shift_thr, "error_cutoff": pre.error_cutoff,
                    "verbosityLevel": 1, "max_order": pre.max_order, "timing": False,
                    "averaged_sum": pre.averaged_sum, "weight": pre.weight})
                ev_t = torch.full((1,), shift_l, dtype=d["residue_sel"].dtype,
                                  device=d["residue_sel"].device)

                def apply_PNM(v, _pn=pn, _ev=ev_t):
                    if v.ndim == 1:
                        v = v.unsqueeze(1)
                    Mv = E.H @ v - shift_l * v
                    return _pn(Mv.contiguous(), E.H, _ev.clone())

                def apply_PNM_defl(v, _f=apply_PNM):
                    return Pi(_f(Pi(v)))

                rec = {"N": int(N)}
                # closed-form prediction from the E spectrum at the SAME Krylov dimension
                a_w = 1.0 if not pre.averaged_sum else float(pre.weight)
                g0 = torch.Generator(device="cpu").manual_seed(1)
                vE = torch.randn(ngpts, 1, generator=g0, dtype=torch.float64
                                 ).to(d["residue_sel"].device).to(d["residue_sel"].dtype)
                for space, op, start in [("full", E, vE),
                                       ("deflated", st.DeflatedOperator(E, Pi) if Pi else None,
                                        Pi(vE) if Pi else None)]:
                    if op is None:
                        continue
                    rr = st.arnoldi(op, start, args.verify_dim)
                    lam = st.ritz_pairs(rr)[0].real
                    mu_p = 1 - lam**int(N) * (1 - a_w + a_w*lam)
                    rec["pred_"+space] = {"mu_min": float(mu_p.min()), "mu_max": float(mu_p.max()),
                        "kappa": float(mu_p.max()/mu_p.min()) if mu_p.min() > 0 else None,
                        "rho_E": float(np.max(np.abs(lam)))}
                    print(f"[verify:{rule}] N={N:2d} {space:>8}  PREDICTED "
                          f"mu in [{mu_p.min():+.5f}, {mu_p.max():+.5f}]  "
                          f"kappa={rec['pred_'+space]['kappa'] if rec['pred_'+space]['kappa'] else float('nan'):.4f}"
                          f"   (from rho(E)={rec['pred_'+space]['rho_E']:.5f})")
                g = torch.Generator(device="cpu").manual_seed(1)
                v0 = torch.randn(ngpts, 1, generator=g, dtype=torch.float64
                                 ).to(d["residue_sel"].device).to(d["residue_sel"].dtype)
                for space, op, start in [("full", apply_PNM, v0),
                                       ("deflated", apply_PNM_defl, Pi(v0) if Pi else None)]:
                    if start is None:
                        continue
                    r = st.arnoldi(op, start, args.verify_dim)
                    th, Y = st.ritz_pairs(r)
                    mu = th.real
                    # EXPLICIT Ritz residuals for the two ends -- without them a
                    # non-normal operator can show an unconverged Ritz value outside
                    # the true spectrum, which would mimic a failure of the closed form.
                    i_lo, i_hi = int(np.argmin(mu)), int(np.argmax(mu))
                    res_lo, _ = st.explicit_ritz_residual(r, complex(th[i_lo]), Y[:, i_lo], op)
                    res_hi, _ = st.explicit_ritz_residual(r, complex(th[i_hi]), Y[:, i_hi], op)
                    rec[space] = {"m": r.m, "rho": float(np.max(np.abs(th))),
                                "mu_min": float(mu.min()), "mu_max": float(mu.max()),
                                "res_mu_min": res_lo, "res_mu_max": res_hi,
                                "ortho_error": r.ortho_error,
                                "kappa": float(mu.max()/mu.min()) if mu.min() > 0 else None,
                                "ritz_values_real": mu.tolist(),
                                "ritz_values_imag": th.imag.tolist()}
                    print(f"[verify:{rule}] N={N:2d} {space:>8}  MEASURED  "
                          f"mu in [{mu.min():+.5f}, {mu.max():+.5f}]  "
                          f"kappa={rec[space]['kappa'] if rec[space]['kappa'] else float('nan'):.4f}"
                          f"   res(min)={res_lo:.2e} res(max)={res_hi:.2e}")
                ver.append(rec)
            entry["order_form_verification"] = ver


        # ---- Rigorous condition-number bounds ------------------------------
        # Arnoldi on the NON-SYMMETRIC order-N operator carries no error bound, so a
        # computed mu_min ~ 0 cannot be told apart from an unconverged Ritz value.
        # A^T A is symmetric PSD, so Lanczos on it does carry the classical bound and
        # brackets sigma_min / sigma_max -- hence kappa -- rigorously.  P and H are
        # symmetric to machine precision, so (I - P M)^T = I - M P.
        if args.svd_bounds:
            from precondition import create_preconditioner
            pre = d["preconditioner"]
            X = d.get("subspace")
            if X is None:
                entry["svd_bounds"] = {"skipped": "no Davidson subspace"}
            else:
                dt = d["residue_sel"].dtype
                Pi = st.Deflator(X.to(dt))
                Et = st.AdjointErrorPropagationOperator(E)
                a_w = 1.0 if not pre.averaged_sum else float(pre.weight)
                F = st.DeflatedOperator(E, Pi)                 # Pi E Pi
                Ft = st.DeflatedOperator(Et, Pi)               # (Pi E Pi)^T = Pi E^T Pi
                g = torch.Generator(device="cpu").manual_seed(11)
                _raw = torch.randn(ngpts, 1, generator=g, dtype=torch.float64
                                   ).to(d["residue_sel"].device).to(dt)
                v0 = Pi(_raw)
                v0 = v0 / torch.linalg.vector_norm(v0)
                # same draw, NOT projected -- start vector for the undeflated run
                v0_full = _raw / torch.linalg.vector_norm(_raw)
                out = {"weight": a_w, "svd_dim": args.svd_dim, "checks": {}}

                # (0) the E-polynomial must reproduce the PRODUCTION order-N operator
                Nc = int(max(args.svd_bounds))
                pn = create_preconditioner("neumann", pre.grid, False, {
                    "fp": "DP", "order": Nc, "correction_scale": 0.0,
                    "no_shift_thr": pre.no_shift_thr, "error_cutoff": pre.error_cutoff,
                    "verbosityLevel": 1, "max_order": pre.max_order, "timing": False,
                    "averaged_sum": pre.averaged_sum, "weight": pre.weight})
                ev_t = torch.full((1,), shift, dtype=dt, device=d["residue_sel"].device)
                _mv = E.H @ v0 - shift * v0
                prod = pn(_mv.contiguous(), E.H, ev_t.clone())
                mine = st.neumann_operator(E, Nc, a_w)(v0)
                dev = float(torch.linalg.vector_norm(prod - mine).item()
                            / torch.linalg.vector_norm(prod).item())
                out["checks"]["poly_vs_production"] = dev
                # P_N alone (not P_N M): production applies pn() to the residual
                mineP = st.neumann_precond(E, E.apply_P, Nc, a_w)(v0)
                prodP = pn(v0.contiguous(), E.H, ev_t.clone())
                out["checks"]["precond_vs_production"] = float(
                    torch.linalg.vector_norm(prodP - mineP).item()
                    / torch.linalg.vector_norm(prodP).item())
                # (0b) adjoint sanity: <v, A u> == <u, A^T v>
                u_ = torch.randn(ngpts, 1, generator=g, dtype=torch.float64).to(dt)
                lhs = float((v0 * st.neumann_operator(E, Nc, a_w)(u_)).sum().item())
                rhs = float((u_ * st.neumann_operator(Et, Nc, a_w)(v0)).sum().item())
                out["checks"]["adjoint_rel_err"] = abs(lhs - rhs) / max(abs(lhs), 1e-300)
                print(f"[svd:{rule}] vs production PreNeumann: P_N M dev = {dev:.3e}  "
                      f"P_N dev = {out['checks']['precond_vs_production']:.3e}  "
                      f"adjoint check = {out['checks']['adjoint_rel_err']:.3e}", flush=True)

                rows = []
                for N in args.svd_bounds:
                    N = int(N)
                    # (a) IMPLEMENTED scheme: recursion in the full space, Pi applied once
                    A_i = st.neumann_operator(E, N, a_w)
                    At_i = st.neumann_operator(Et, N, a_w)
                    # Pi N(E) Pi has span(X) as a null space; without project=Pi the
                    # Lanczos run converges to that spurious zero (verified on a dense
                    # 1000-point case: theta_min collapses from 6.4e-5 to 2e-16).
                    imp = st.singular_value_bounds(
                        lambda v, _f=A_i: Pi(_f(Pi(v))),
                        lambda v, _f=At_i: Pi(_f(Pi(v))), v0, args.svd_dim, project=Pi)
                    # (b) DEFLATED formulation: a_{n+1} = Pi E Pi a_n
                    dfm = st.singular_value_bounds(st.neumann_operator(F, N, a_w),
                                                   st.neumann_operator(Ft, N, a_w),
                                                   v0, args.svd_dim, project=Pi)
                    rec = {"N": N, "implemented": imp, "deflated_form": dfm}
                    if args.svd_undeflated:
                        # No projector anywhere: the raw P_N M on the full space.
                        # M is near-singular by design, so sigma_min is expected to
                        # be at the level of the Ritz residual and kappa to be huge.
                        rec["undeflated"] = st.singular_value_bounds(
                            A_i, At_i, v0_full, args.svd_dim)
                    if args.jd_form:
                        # Jacobi-Davidson form: solve (Pi M Pi) t = gamma with the
                        # preconditioner Pi P_N Pi, so the preconditioned operator is
                        # Pi P_N Pi M Pi -- one extra projector compared with the
                        # implemented Pi P_N M Pi.
                        PN = st.neumann_precond(E, E.apply_P, N, a_w)
                        PNt = st.neumann_precond_adjoint(Et, E.apply_P, N, a_w)
                        aM = lambda v: E.H @ v - shift * v
                        rec["jd_form"] = st.singular_value_bounds(
                            lambda v: Pi(PN(Pi(aM(Pi(v))))),
                            lambda v: Pi(aM(Pi(PNt(Pi(v))))),
                            v0, args.svd_dim, project=Pi)
                    if args.svd_arnoldi_dim > 0:
                        r = st.arnoldi(lambda v, _f=A_i: Pi(_f(Pi(v))), v0,
                                       args.svd_arnoldi_dim, project=Pi)
                        th2, _Y = st.ritz_pairs(r)
                        rec["arnoldi_recheck"] = {
                            "m": r.m, "mu_min": float(th2.real.min()),
                            "mu_max": float(th2.real.max()),
                            "max_abs_imag": float(np.abs(th2.imag).max()),
                            "eig_ratio": float(np.abs(th2).max() / np.abs(th2).min()),
                            "ortho_error": r.ortho_error,
                            "ritz_values_real": th2.real.tolist(),
                            "ritz_values_imag": th2.imag.tolist()}
                    rows.append(rec)
                    _fam = [("implemented", imp), ("deflated_form", dfm)]
                    if "undeflated" in rec:
                        _fam.append(("undeflated", rec["undeflated"]))
                    if "jd_form" in rec:
                        _fam.append(("jd_form", rec["jd_form"]))
                    for lab, rr in _fam:
                        print(f"[svd:{rule}] N={N:2d} {lab:>14}  "
                              f"s_min in [{rr['sigma_min_lo']:.5f}, {rr['sigma_min_hi']:.5f}]  "
                              f"s_max in [{rr['sigma_max_lo']:.5f}, {rr['sigma_max_hi']:.5f}]  "
                              f"kappa in [{rr['kappa_lower']:.3f}, {rr['kappa_upper']:.3f}]  "
                              f"{'CONCLUSIVE' if rr['conclusive'] else 'inconclusive'}",
                              flush=True)
                    if "arnoldi_recheck" in rec:
                        a = rec["arnoldi_recheck"]
                        print(f"[svd:{rule}] N={N:2d}   arnoldi m={a['m']}: "
                              f"mu in [{a['mu_min']:+.5f}, {a['mu_max']:+.5f}]  "
                              f"max|Im|={a['max_abs_imag']:.2e}  "
                              f"eig_ratio={a['eig_ratio']:.4g}", flush=True)
                out["orders"] = rows
                entry["svd_bounds"] = out

        # ---- Contraction factor of the scheme, undamped vs damped ----------
        # rho is a property of E and is therefore the SAME for NP and Damped-NP:
        # damping changes how the terms are summed, not the recurrence.  What the
        # damping does change is the error operator G_N = I - P_N M, whose norm is
        # the contraction actually achieved at order N.  Measured for w = 1 and
        # w = 0.5 side by side.
        if args.error_norm:
            X = d.get("subspace")
            if X is None:
                entry["error_norm"] = {"skipped": "no Davidson subspace"}
            else:
                dte = d["residue_sel"].dtype
                Pie = st.Deflator(X.to(dte))
                Ete = st.AdjointErrorPropagationOperator(E)
                ge = torch.Generator(device="cpu").manual_seed(5)
                ve = Pie(torch.randn(ngpts, 1, generator=ge, dtype=torch.float64
                                     ).to(d["residue_sel"].device).to(dte))
                ve = ve / torch.linalg.vector_norm(ve)
                out_e = []
                for N in args.error_norm:
                    N = int(N)
                    rec = {"N": N}
                    for lab, w in (("undamped", 1.0), ("damped", 0.5)):
                        A = st.neumann_error_operator(E, N, w)
                        At = st.neumann_error_operator(Ete, N, w)
                        b = st.singular_value_bounds(
                            lambda v, _f=A: Pie(_f(Pie(v))),
                            lambda v, _f=At: Pie(_f(Pie(v))),
                            ve, args.error_norm_dim, project=Pie)
                        rec[lab] = {"norm_lo": b["sigma_max_lo"], "norm_hi": b["sigma_max_hi"],
                                    "contracts": bool(b["sigma_max_hi"] < 1.0), "m": b["m"]}
                    out_e.append(rec)
                    print(f"[err:{rule}] N={N:2d}  ||G_N||  undamped <= "
                          f"{rec['undamped']['norm_hi']:.4f} {'(contracts)' if rec['undamped']['contracts'] else '(NO)':<12}"
                          f"  damped <= {rec['damped']['norm_hi']:.4f} "
                          f"{'(contracts)' if rec['damped']['contracts'] else '(NO)'}", flush=True)
                entry["error_norm"] = {"orders": out_e, "m": args.error_norm_dim}

        # ---- Rigorous certificate that rho(Pi E Pi) < 1 --------------------
        # rho(A) <= ||A^n||^(1/n) for every n, and the sequence decreases to rho.
        # ||A^n|| = sigma_max(A^n) comes from the same Lanczos machinery, which
        # DOES carry an error bound, so a value below 1 is a proof rather than an
        # estimate.  Arnoldi on the non-symmetric Pi E Pi cannot do this: its Ritz
        # values lie in the numerical range, which is strictly larger than the
        # spectrum (measured: a Ritz value of 2.05 for an operator whose closed
        # form caps the spectrum at 1).
        if args.rho_bound:
            X = d.get("subspace")
            if X is None:
                entry["rho_bound"] = {"skipped": "no Davidson subspace"}
            else:
                dtq = d["residue_sel"].dtype
                Piq = st.Deflator(X.to(dtq))
                Fq = st.DeflatedOperator(E, Piq)
                Ftq = st.DeflatedOperator(st.AdjointErrorPropagationOperator(E), Piq)
                gq = torch.Generator(device="cpu").manual_seed(3)
                vq = Piq(torch.randn(ngpts, 1, generator=gq, dtype=torch.float64
                                     ).to(d["residue_sel"].device).to(dtq))
                vq = vq / torch.linalg.vector_norm(vq)

                def _pow(op, k):
                    def f(v, _op=op, _k=k):
                        for _ in range(_k):
                            v = _op(v)
                        return v
                    return f

                res_rb, proved = [], None
                for nn in args.rho_bound:
                    nn = int(nn)
                    b = st.singular_value_bounds(_pow(Fq, nn), _pow(Ftq, nn), vq,
                                                 args.rho_bound_dim, project=Piq)
                    bound = b["sigma_max_hi"] ** (1.0 / nn)
                    ok = bound < 1.0
                    if ok and proved is None:
                        proved = nn
                    res_rb.append({"n": nn, "norm_An_upper": b["sigma_max_hi"],
                                   "norm_An_lower": b["sigma_max_lo"],
                                   "rho_upper_bound": bound, "proves_rho_lt_1": bool(ok),
                                   "m": b["m"]})
                    print(f"[rho:{rule}] n={nn:3d}  ||A^n|| <= {b['sigma_max_hi']:.6e}  "
                          f"=> rho(Pi E Pi) <= {bound:.6f}  "
                          f"{'PROVES rho < 1' if ok else 'inconclusive'}", flush=True)
                entry["rho_bound"] = {"powers": res_rb, "m": args.rho_bound_dim,
                                      "smallest_n_proving_rho_lt_1": proved}



        # ---- Excitation of the divergent modes -----------------------------
        # rho(E) > 1 is structural, but the divergent eigendirections are the occupied
        # states BELOW eps_t, i.e. span(X), and the Davidson residual is orthogonal to X
        # by construction.  Two independent suppression factors decide whether the
        # divergence ever appears in  Pi E^n a0 :
        #   (i)  w   = ||(I-Pi) a0|| / ||a0||          how much P scatters gamma back into X
        #   (ii) s   = ||Pi v_dom|| / ||v_dom||        how much of the dominant eigenvector
        #                                              of the FULL-space E survives Pi
        if args.weights > 0:
            X = d.get("subspace")
            if X is None:
                entry["weights"] = {"skipped": "no Davidson subspace"}
            else:
                Pi = st.Deflator(X.to(d["residue_sel"].dtype))
                gamma = d["residue_sel"][:, pos:pos+1].contiguous()
                a0 = probe.terms[0][:, pos:pos+1].contiguous()
                nz = lambda v: float(torch.linalg.vector_norm(v).item())
                w0 = nz(a0 - Pi(a0)) / nz(a0)
                wg = nz(gamma - Pi(gamma)) / nz(gamma)
                out = {"w_gamma_in_X": wg, "w_a0_in_X": w0,
                       "gamma_norm": nz(gamma), "a0_norm": nz(a0)}
                print(f"[weights:{rule}] ||(I-Pi)gamma||/||gamma|| = {wg:.3e}   "
                      f"(residual is orthogonal to X by construction)")
                print(f"[weights:{rule}] ||(I-Pi)a0||/||a0||       = {w0:.3e}   "
                      f"<- how much P scatters it back into span(X)")

                # surviving fraction of E^n a0 as n grows
                seq = []
                v = a0
                for n in range(args.weights + 1):
                    nv = nz(v)
                    seq.append({"n": n, "norm": nv, "surviving_fraction": nz(Pi(v)) / nv,
                                "growth": nv / out["a0_norm"]})
                    v = E(v)
                out["E_power_scan"] = seq
                print(f"[weights:{rule}]  n | {'||E^n a0||/||a0||':>18} | {'||Pi E^n a0||/||E^n a0||':>26}")
                for r in seq:
                    print(f"[weights:{rule}] {r['n']:2d} | {r['growth']:18.4e} | "
                          f"{r['surviving_fraction']:26.4e}")

                # dominant eigenvector of the FULL-space E: how much survives Pi ?
                g = torch.Generator(device="cpu").manual_seed(7)
                v0 = torch.randn(ngpts, 1, generator=g, dtype=torch.float64
                                 ).to(d["residue_sel"].device).to(d["residue_sel"].dtype)
                r = st.arnoldi(E, v0, 40)
                th, Y = st.ritz_pairs(r)
                idx = int(np.argmax(np.abs(th)))
                res_d, vdom = st.explicit_ritz_residual(r, complex(th[idx]), Y[:, idx], E)
                vr = vdom.real if vdom.is_complex() else vdom
                s_dom = nz(Pi(vr)) / nz(vr)
                out["lambda_dom_full"] = float(th[idx].real)
                out["ritz_residual_dom"] = res_d
                out["surviving_fraction_vdom"] = s_dom
                print(f"[weights:{rule}] lambda_dom(full E) = {th[idx].real:+.5f} "
                      f"(Ritz res {res_d:.1e});  ||Pi v_dom||/||v_dom|| = {s_dom:.4e}")
                entry["weights"] = out


        # ---- Does the DEFLATED recursion converge?  and a minimal projector ----
        if args.deflated_recursion > 0 or args.minimal_projector:
            X = d.get("subspace")
            if X is None:
                entry["recursion"] = {"skipped": "no Davidson subspace"}
            else:
                Xd = X.to(d["residue_sel"].dtype)
                Pi = st.Deflator(Xd)
                nz = lambda v: float(torch.linalg.vector_norm(v).item())
                a0 = probe.terms[0][:, pos:pos+1].contiguous()
                out = {}

                projs = [("Pi_all", Pi, Xd.shape[1])]
                if args.minimal_projector:
                    # Q_i removes ONLY the Ritz vectors lying below eps_t: the minimal
                    # subspace whose removal makes M positive definite (Sylvester).
                    rv = d["raw_eigval"]
                    below = (rv < shift)
                    k = int(below.sum().item())
                    if k > 0:
                        Xq = Xd.index_select(1, torch.nonzero(below).flatten().to(Xd.device))
                        projs.append(("Q_below", st.Deflator(Xq), k))
                    else:
                        projs.append(("Q_below", None, 0))
                    print(f"[recur:{rule}] Ritz vectors below eps_t: {k} of {Xd.shape[1]}")

                for name, Q, k in projs:
                    rec = {"n_removed": k}
                    if Q is None:
                        rec["note"] = "no state below eps_t -> Q = I"
                        Q = st.Deflator(Xd[:, :0])
                    Eq = st.DeflatedOperator(E, Q)
                    # (i) spectrum of Q E Q
                    g = torch.Generator(device="cpu").manual_seed(0)
                    v0 = Q(torch.randn(ngpts, 1, generator=g, dtype=torch.float64
                                       ).to(Xd.device).to(Xd.dtype))
                    r = st.arnoldi(Eq, v0, 40)
                    th = st.ritz_pairs(r)[0].real
                    mu = 1 - th
                    rec["rho"] = float(np.max(np.abs(th)))
                    rec["lambda_min"] = float(th.min()); rec["lambda_max"] = float(th.max())
                    rec["mu_min"] = float(mu.min()); rec["mu_max"] = float(mu.max())
                    rec["kappa_N0"] = float(mu.max()/mu.min()) if mu.min() > 0 else None
                    print(f"[recur:{rule}] {name:>8} (removes {k:4d}): rho(QEQ)={rec['rho']:.5f}  "
                          f"lam in [{th.min():+.4f},{th.max():+.4f}]  "
                          f"kappa(N=0)={rec['kappa_N0'] if rec['kappa_N0'] else float('nan'):.2f}")
                    # (ii) the DEFLATED recursion  a_{n+1} = Q E Q a_n
                    if args.deflated_recursion > 0:
                        seq=[]; v = Q(a0); n0 = nz(v)
                        for n in range(args.deflated_recursion + 1):
                            seq.append({"n": n, "norm_ratio": nz(v)/n0})
                            v = Eq(v)
                        rec["deflated_recursion"] = seq
                        # for contrast, the UNdeflated one projected once at the end
                        seq2=[]; v = a0
                        for n in range(args.deflated_recursion + 1):
                            seq2.append({"n": n, "norm_ratio": nz(Q(v))/n0})
                            v = E(v)
                        rec["undeflated_recursion"] = seq2
                        print(f"[recur:{rule}] {name:>8}  n | {'||(QEQ)^n a0||':>15} | "
                              f"{'||Q E^n a0||':>13}   (deflated vs implemented)")
                        for i in range(0, len(seq), 2):
                            print(f"[recur:{rule}] {'':>8} {seq[i]['n']:2d} | "
                                  f"{seq[i]['norm_ratio']:15.4e} | {seq2[i]['norm_ratio']:13.4e}")
                    out[name] = rec
                entry["recursion"] = out

        entry["matvecs"] = E.nmatvec
        payload["states"].append(entry)

    payload["total_seconds"] = time.time() - t_start

    out = os.path.join(args.outdir, f"{tag}.json")
    st.save_json(out, payload)
    print(f"[done] wrote {out}  ({len(payload['states'])} states, {payload['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
