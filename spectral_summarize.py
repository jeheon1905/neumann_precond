"""Aggregate spectral_experiment.py JSON output into manuscript tables/figures."""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


FIELDS = [
    "system", "state_index", "active_index", "davidson_iteration",
    "preconditioner_call", "residual_norm", "raw_epsilon", "corrected_epsilon",
    "zero_shift_fallback", "target", "eigenvalue_real", "eigenvalue_imag",
    "magnitude", "damping_factor", "ritz_residual", "krylov_dim", "seed",
    "commit", "dtype", "device",
]


def rows_from_payload(payload: dict, section: str):
    snap = payload["snapshot"]
    meta = snap.get("davidson_meta") or {}
    base = {
        "system": snap["system"],
        "state_index": snap.get("state_index"),
        "active_index": snap["active_index"],
        "davidson_iteration": meta.get("i_iter"),
        "preconditioner_call": snap["preconditioner_call"],
        "residual_norm": snap["residual_norm"],
        "raw_epsilon": snap["raw_epsilon"],
        "corrected_epsilon": snap["corrected_epsilon"],
        "zero_shift_fallback": snap["zero_shift_fallback"],
        "commit": payload["commit"],
        "dtype": snap["dtype"],
        "device": payload["environment"]["device"],
    }
    out = []
    for rec in payload.get(section, []):
        for tname, t in rec["targets"].items():
            r = dict(base)
            r.update(
                target=tname,
                eigenvalue_real=t["eigenvalue_real"],
                eigenvalue_imag=t["eigenvalue_imag"],
                magnitude=t["magnitude"],
                damping_factor=t["damping_factor"],
                ritz_residual=t.get("ritz_residual", t["arnoldi_residual_estimate"]),
                krylov_dim=rec["krylov_dim_built"],
                seed=rec.get("seed"),
            )
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results_spectral_revision/global_spectrum/raw")
    ap.add_argument("--section", default="global_spectrum",
                    choices=["global_spectrum", "projected_spectrum"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    outdir = os.path.dirname(args.indir.rstrip("/"))
    out_csv = args.out or os.path.join(outdir, "summary.csv")
    figdir = args.figdir or os.path.join(outdir, "figures")

    files = sorted(glob.glob(os.path.join(args.indir, "*.json")))
    if not files:
        raise SystemExit(f"no JSON in {args.indir}")

    import csv

    rows, payloads = [], []
    for f in files:
        with open(f) as fh:
            p = json.load(fh)
        if args.section not in p:
            continue
        payloads.append((os.path.basename(f), p))
        rows += rows_from_payload(p, args.section)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out_csv} ({len(rows)} rows from {len(payloads)} runs)")

    # --- console table at the largest converged Krylov dimension ---
    print("\n| System | State | |gamma| | eps_tilde | lam_min | lam_max | rho(E) | max Ritz resid |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, p in payloads:
        recs = [r for r in p[args.section]]
        if not recs:
            continue
        mmax = max(r["krylov_dim_built"] for r in recs)
        sel = [r for r in recs if r["krylov_dim_built"] == mmax]
        s = p["snapshot"]
        lam_min = min(r["summary"]["lambda_min_real"] for r in sel)
        lam_max = max(r["summary"]["lambda_max_real"] for r in sel)
        rho = max(r["summary"]["rho"] for r in sel)
        rr = max(
            t.get("ritz_residual", t["arnoldi_residual_estimate"])
            for r in sel for t in r["targets"].values()
        )
        print(
            f"| {s['system']} | {s.get('state_index')} | {s['residual_norm']:.4e} | "
            f"{s['corrected_epsilon']:.6f} | {lam_min:.6f} | {lam_max:.6f} | "
            f"{rho:.6f} | {rr:.2e} |"
        )

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable:", e)
        return

    os.makedirs(figdir, exist_ok=True)

    # (1) Ritz spectrum per system at the largest Krylov dim
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, p) in enumerate(payloads):
        recs = p[args.section]
        mmax = max(r["krylov_dim_built"] for r in recs)
        rec = [r for r in recs if r["krylov_dim_built"] == mmax][0]
        th = np.array(rec["ritz_values_real"])
        ax.scatter(th, np.full_like(th, i), s=18, label=p["snapshot"]["system"])
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.axvline(-1.0, color="r", ls="--", lw=0.8)
    ax.set_xlabel(r"Re $\lambda$ of $E = I - P(H-\tilde\epsilon I)$")
    ax.set_yticks(range(len(payloads)))
    ax.set_yticklabels([p["snapshot"]["system"] for _, p in payloads], fontsize=8)
    ax.set_title(f"{args.section.replace('_',' ')}: Ritz values")
    fig.tight_layout()
    f1 = os.path.join(figdir, f"{args.section}_ritz.png")
    fig.savefig(f1, dpi=160)
    print("wrote", f1)

    # (2) Krylov-dimension convergence of the extremal quantities
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in payloads:
        recs = sorted(p[args.section], key=lambda r: r["krylov_dim_built"])
        lab = p["snapshot"]["system"]
        seen = set()
        for seed in {r.get("seed") for r in recs}:
            rs = [r for r in recs if r.get("seed") == seed]
            l = lab if lab not in seen else None
            seen.add(lab)
            axes[0].plot([r["krylov_dim_built"] for r in rs],
                         [r["summary"]["rho"] for r in rs], "o-", label=l)
            axes[1].plot([r["krylov_dim_built"] for r in rs],
                         [r["summary"]["lambda_min_real"] for r in rs], "o-", label=l)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("Krylov dimension $m$"); axes[0].set_ylabel(r"$\rho(E)$")
    axes[1].axhline(-1.0, color="r", ls="--", lw=0.8)
    axes[1].set_xlabel("Krylov dimension $m$"); axes[1].set_ylabel(r"$\lambda_{\min}$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    f2 = os.path.join(figdir, f"{args.section}_convergence.png")
    fig.savefig(f2, dpi=160)
    print("wrote", f2)


if __name__ == "__main__":
    main()
