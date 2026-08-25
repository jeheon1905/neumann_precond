"""Condition number of the order-N Neumann preconditioner, from RIGOROUS bounds.

Consumes ``results_spectral_revision/svd_bounds/*.json`` produced by
``spectral_experiment.py --svd_bounds``.  Three formulations are distinguished:

    implemented    Pi Pbar_N M Pi        what the production code applies
    deflated_form  I - Fbar_N, F=Pi E Pi  a_{n+1} = Pi E Pi a_n  (a variant)
    jd_form        Pi Pbar_N Pi M Pi     the Jacobi-Davidson correction equation

Each is bracketed by Lanczos on ``A^T A`` (symmetric PSD, so the classical
residual bound applies).  Eigenvalue ratios are NOT used: they are only a lower
bound on the condition number for a non-normal operator, and Arnoldi on the
non-symmetric operator carries no error bound at all.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

RAW = "results_spectral_revision/svd_bounds"
OUT = "results_spectral_revision/svd_bounds"
FIG = os.path.join(OUT, "figures")

SY = ["B12", "water_cluster_128", "C60_4", "MAPbI3"]
FAM = ["implemented", "deflated_form", "jd_form"]

# archived Damped-NP Davidson iteration counts (median run), orders 0..10
OBS = {
    "B12":               [158, 73, 42, 33, 27, 23, 21, 20, 19, 19, 19],
    "water_cluster_128": [127, 58, 34, 27, 22, 20, 19, 19, 19, 19, 18],
    "C60_4":             [123, 57, 33, 26, 21, 18, 17, 16, 16, 16, 16],
    "MAPbI3":            [80, 51, 31, 24, 20, 17, 15, 14, 13, 12, 12],
}


def load():
    """{(system, state, family): {N: record}} plus the arnoldi rechecks."""
    kap, arn, chk = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        p = json.load(open(f))
        s = p["system"].split(".")[0]
        for e in p.get("states", []):
            sb = e.get("svd_bounds")
            if not sb or "orders" in sb is None or "orders" not in sb:
                continue
            st = e["state_rule"]
            chk[(s, st)] = sb.get("checks", {})
            for r in sb["orders"]:
                for fam in FAM:
                    if fam in r:
                        kap.setdefault((s, st, fam), {})[r["N"]] = r[fam]
                if "arnoldi_recheck" in r:
                    arn.setdefault((s, st), {})[r["N"]] = r["arnoldi_recheck"]
    return kap, arn, chk


def fmt(rec):
    lo, hi = rec["kappa_lower"], rec["kappa_upper"]
    if not np.isfinite(hi):
        return f"> {lo:.1f}"
    return f"{lo:.2f}" if hi / max(lo, 1e-30) < 1.01 else f"[{lo:.2f}, {hi:.2f}]"


def main():
    os.makedirs(FIG, exist_ok=True)
    kap, arn, chk = load()
    if not kap:
        print(f"no results yet in {RAW}/")
        return

    print("### Validation against the production preconditioner\n")
    print("| System | state | P_N M dev | P_N dev | adjoint |")
    print("|---|---|---:|---:|---:|")
    for (s, st), c in sorted(chk.items()):
        print(f"| {s} | {st} | {c.get('poly_vs_production', float('nan')):.2e} | "
              f"{c.get('precond_vs_production', float('nan')):.2e} | "
              f"{c.get('adjoint_rel_err', float('nan')):.2e} |")

    orders = sorted({n for d in kap.values() for n in d})
    rows = []
    for st in sorted({k[1] for k in kap}):
        print(f"\n### kappa brackets — {st} state (Lanczos on A^T A, rigorous)\n")
        hdr = "| System | formulation | " + " | ".join(f"N={n}" for n in orders) + " |"
        print(hdr)
        print("|---|---|" + "---:|" * len(orders))
        for s in SY:
            for fam in FAM:
                d = kap.get((s, st, fam))
                if not d:
                    continue
                cells = [fmt(d[n]) if n in d else "—" for n in orders]
                print(f"| {s} | {fam} | " + " | ".join(cells) + " |")
                for n in orders:
                    if n in d:
                        rows.append({"system": s, "state": st, "formulation": fam, "N": n,
                                     **{k: d[n][k] for k in
                                        ("kappa_lower", "kappa_upper", "sigma_min_lo",
                                         "sigma_min_hi", "sigma_max_lo", "sigma_max_hi",
                                         "m", "conclusive")}})
    if rows:
        with open(os.path.join(OUT, "kappa_bounds.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            [w.writerow(r) for r in rows]
        print(f"\nwrote {OUT}/kappa_bounds.csv  ({len(rows)} rows)")

    if arn:
        print("\n### Arnoldi re-check with the projection fix "
              "(same m=80 as the corrupted run)\n")
        print("| System | state | N | mu_min | mu_max | max abs imag | eig ratio |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for (s, st), d in sorted(arn.items()):
            for n in sorted(d):
                a = d[n]
                print(f"| {s} | {st} | {n} | {a['mu_min']:+.5f} | {a['mu_max']:+.5f} | "
                      f"{a['max_abs_imag']:.2e} | {a['eig_ratio']:.4g} |")

    # ---- iterations vs kappa, using the IMPLEMENTED formulation ----
    print("\n### iterations ~ kappa_N^p  (implemented formulation, homo state)\n")
    print("| System | orders used | p | mean err | max err |")
    print("|---|---|---:|---:|---:|")
    for s in SY:
        d = kap.get((s, "homo", "implemented"))
        if not d:
            continue
        ns = [n for n in sorted(d) if d[n]["conclusive"] and n < len(OBS[s])]
        if len(ns) < 3:
            print(f"| {s} | only {len(ns)} conclusive order(s) — fit skipped | | | |")
            continue
        k = np.array([0.5 * (d[n]["kappa_lower"] + d[n]["kappa_upper"]) for n in ns])
        o = np.array([OBS[s][n] for n in ns], float)
        p_, c_ = np.polyfit(np.log(k), np.log(o), 1)
        err = np.abs(np.exp(c_) * k ** p_ - o) / o
        print(f"| {s} | {ns} | {p_:.2f} | {err.mean()*100:.1f} % | {err.max()*100:.1f} % |")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable:", exc)
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {"implemented": ("o-", 1.8), "deflated_form": ("s--", 1.0),
              "jd_form": ("^:", 1.0)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, st in zip(axes, ["homo", "lowest"]):
        for i, s in enumerate(SY):
            for fam in FAM:
                d = kap.get((s, st, fam))
                if not d:
                    continue
                ns = [n for n in sorted(d) if d[n]["conclusive"]]
                if not ns:
                    continue
                y = [0.5 * (d[n]["kappa_lower"] + d[n]["kappa_upper"]) for n in ns]
                m, lw = styles[fam]
                ax.plot(ns, y, m, color=colors[i % len(colors)], lw=lw, ms=5,
                        label=f"{s} / {fam}" if st == "homo" else None)
        ax.set_yscale("log")
        ax.set_xlabel("Neumann order $N$")
        ax.set_title(f"{st} state", fontsize=10)
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\kappa$  (rigorous, $\sigma_{max}/\sigma_{min}$)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Condition number of the order-$N$ preconditioned operator on "
                 "range($\\Pi$) — three formulations", fontsize=11)
    fig.tight_layout()
    f = os.path.join(FIG, "kappa_bounds_vs_order.png")
    fig.savefig(f, dpi=170)
    print("\nwrote", f)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Fallback: build the tables from the job logs
# ----------------------------------------------------------------------
# The JSON payload is written only when a job finishes, but every measurement is
# printed with flush=True as soon as it is made.  This parser recovers the table
# from a partially complete run.
import re

_RE_SVD = re.compile(
    r"^\[svd:(?P<state>\w+)\] N=\s*(?P<N>\d+)\s+(?P<fam>\w+)\s+"
    r"s_min in \[(?P<smnlo>[-\d.e+]+), (?P<smnhi>[-\d.e+]+)\]\s+"
    r"s_max in \[(?P<smxlo>[-\d.e+]+), (?P<smxhi>[-\d.e+]+)\]\s+"
    r"kappa in \[(?P<klo>[-\d.e+]+), (?P<khi>[-\d.e+inf]+)\]\s+(?P<verdict>\w+)")
_RE_ARN = re.compile(
    r"^\[svd:(?P<state>\w+)\] N=\s*(?P<N>\d+)\s+arnoldi m=(?P<m>\d+): "
    r"mu in \[(?P<lo>[-+\d.e]+), (?P<hi>[-+\d.e]+)\]\s+"
    r"max\|Im\|=(?P<im>[-\d.e+]+)\s+eig_ratio=(?P<er>[-\d.e+]+)")
_RE_RHO = re.compile(
    r"^\[rho:(?P<state>\w+)\] n=\s*(?P<n>\d+)\s+\|\|A\^n\|\| <= (?P<An>[-\d.e+]+)\s+"
    r"=> rho\(Pi E Pi\) <= (?P<rho>[-\d.e+]+)\s+(?P<verdict>.*)$")


def from_logs(pattern="results_spectral_revision/jobs/*.txt"):
    """{(system, state, family): {N: rec}}, {(system,state): {N: arnoldi}}, rho rows."""
    kap, arn, rho = {}, {}, {}
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)
        m = re.match(r"(?:svd|jd|rho|exact|band|dense|fill)_(?P<sys>.+?)(?:_g\d)?_\d+\.txt$", base)
        if not m:
            continue
        s = m.group("sys")
        for line in open(f, errors="replace"):
            g = _RE_SVD.match(line)
            if g:
                hi = float("inf") if "inf" in g["khi"] else float(g["khi"])
                kap.setdefault((s, g["state"], g["fam"]), {})[int(g["N"])] = {
                    "kappa_lower": float(g["klo"]), "kappa_upper": hi,
                    "sigma_min_lo": float(g["smnlo"]), "sigma_min_hi": float(g["smnhi"]),
                    "sigma_max_lo": float(g["smxlo"]), "sigma_max_hi": float(g["smxhi"]),
                    "conclusive": g["verdict"] == "CONCLUSIVE", "m": None}
                continue
            g = _RE_ARN.match(line)
            if g:
                arn.setdefault((s, g["state"]), {})[int(g["N"])] = {
                    "m": int(g["m"]), "mu_min": float(g["lo"]), "mu_max": float(g["hi"]),
                    "max_abs_imag": float(g["im"]), "eig_ratio": float(g["er"])}
                continue
            g = _RE_RHO.match(line)
            if g:
                rho.setdefault((s, g["state"]), {})[int(g["n"])] = {
                    "norm_An_upper": float(g["An"]), "rho_upper_bound": float(g["rho"]),
                    "proves": "PROVES" in g["verdict"]}
    return kap, arn, rho


def report_logs():
    kap, arn, rho = from_logs()
    orders = sorted({n for d in kap.values() for n in d})
    for st in sorted({k[1] for k in kap}):
        print(f"\n### kappa (from logs) — {st}\n")
        print("| System | formulation | " + " | ".join(f"N={n}" for n in orders) + " |")
        print("|---|---|" + "---:|" * len(orders))
        for s in SY:
            for fam in FAM:
                d = kap.get((s, st, fam))
                if d:
                    print(f"| {s} | {fam} | " +
                          " | ".join(fmt(d[n]) if n in d else "—" for n in orders) + " |")
    if arn:
        print(f"\n### Arnoldi re-check (projection fixed)\n")
        print("| System | state | N | mu_min | mu_max | max abs imag | eig ratio |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for (s, st), d in sorted(arn.items()):
            for n in sorted(d):
                a = d[n]
                print(f"| {s} | {st} | {n} | {a['mu_min']:+.5f} | {a['mu_max']:+.5f} | "
                      f"{a['max_abs_imag']:.1e} | {a['eig_ratio']:.4g} |")
    if rho:
        print(f"\n### Rigorous certificate  rho(Pi E Pi) <= ||A^n||^(1/n)\n")
        ns = sorted({n for d in rho.values() for n in d})
        print("| System | state | " + " | ".join(f"n={n}" for n in ns) + " | smallest n proving rho<1 |")
        print("|---|---|" + "---:|" * len(ns) + "---|")
        for (s, st), d in sorted(rho.items()):
            cells = [f"{d[n]['rho_upper_bound']:.5f}" if n in d else "—" for n in ns]
            pr = [n for n in sorted(d) if d[n]["proves"]]
            print(f"| {s} | {st} | " + " | ".join(cells) +
                  f" | {pr[0] if pr else '**none yet**'} |")


# ----------------------------------------------------------------------
# Converged-regime (exact-shift) cross-check
# ----------------------------------------------------------------------
def compare_exact_shift():
    """iteration-15 corrected-shift measurement vs converged exact-shift measurement.

    The second design removes the arbitrariness of choosing an iteration: it
    captures at the first call where every selected state's residual is below a
    threshold and uses the raw Davidson Ritz value as the shift, so that
    ``M = H - eps I`` is exactly singular at the target state. Agreement between
    the two is a robustness statement, not an improvement.
    """
    base, _, base_rho = from_logs("results_spectral_revision/jobs/svd_*.txt")
    _, _, rho15 = from_logs("results_spectral_revision/jobs/rho_*.txt")
    base_rho.update(rho15)
    ex, ex_arn, ex_rho = from_logs("results_spectral_revision/jobs/exact_*.txt")
    orders = sorted({n for d in ex.values() for n in d})

    for st in ["homo", "lowest"]:
        rows = [(s, n) for s in SY for n in sorted((ex.get((s, st, "implemented")) or {}))]
        if not rows:
            continue
        print(f"\n### kappa — {st}: iteration 15 (corrected shift) vs converged (exact shift)\n")
        print("| System | N | iteration 15 | converged, exact shift | agree? |")
        print("|---|---:|---|---|---|")
        for s, n in rows:
            a_ = (base.get((s, st, "implemented")) or {}).get(n)
            b_ = (ex.get((s, st, "implemented")) or {}).get(n)
            if not b_:
                continue
            if not a_:
                print(f"| {s} | {n} | — | {fmt(b_)} | — |")
                continue
            ka = 0.5 * (a_["kappa_lower"] + a_["kappa_upper"])
            kb = 0.5 * (b_["kappa_lower"] + b_["kappa_upper"])
            if ka > 1e30 or kb > 1e30:
                v = "both open"
            else:
                d = abs(kb / ka - 1) * 100
                v = f"{d:.1f} %"
            print(f"| {s} | {n} | {fmt(a_)} | {fmt(b_)} | {v} |")

        print(f"\n### monotonicity of kappa in N — {st}, converged regime\n")
        print("| System | " + " | ".join(f"N={n}" for n in orders) + " | monotone? |")
        print("|---|" + "---:|" * len(orders) + "---|")
        for s in SY:
            d = ex.get((s, st, "implemented")) or {}
            if len(d) < 2:
                continue
            ks = [0.5 * (d[n]["kappa_lower"] + d[n]["kappa_upper"]) if n in d else None
                  for n in orders]
            fin = [(n, k) for n, k in zip(orders, ks) if k is not None and k < 1e30]
            mono = all(b < a for (_, a), (_, b) in zip(fin, fin[1:])) if len(fin) > 1 else None
            print(f"| {s} | " + " | ".join(fmt(d[n]) if n in d else "—" for n in orders)
                  + f" | {'**yes**' if mono else '**NO — turns around**'} |")

    if ex_rho:
        print("\n### rho certificate — converged regime vs iteration 15\n")
        print("| System | state | ||A|| (it 15) | ||A|| (converged) | best bound (conv.) | proved at |")
        print("|---|---|---:|---:|---:|---|")
        for k in sorted(ex_rho, key=lambda x: (x[1], x[0])):
            d = ex_rho[k]
            pr = [n for n in sorted(d) if d[n]["proves"]]
            a15 = base_rho.get(k, {}).get(1, {}).get("norm_An_upper", float("nan"))
            print(f"| {k[0]} | {k[1]} | {a15:.3f} | {d[1]['norm_An_upper']:.3f} | "
                  f"{min(d[n]['rho_upper_bound'] for n in d):.6f} | "
                  f"{'n='+str(pr[0]) if pr else 'not proved (rho > 1)'} |")


def band_distribution(pattern="results_spectral_revision/jobs/band_*.txt"):
    """Distribution of kappa over the occupied manifold.

    The two representative states (lowest, HOMO) bracket the manifold but do not
    describe it: the observed Davidson iteration count falls monotonically to
    N = 10, whereas kappa at the HOMO turns around at N = 8 in the two systems
    with the largest full-space rho(E). If most bands are monotone, the aggregate
    conditioning still improves and the two pictures are consistent.
    """
    kap, _, _ = from_logs(pattern)
    orders = sorted({n for d in kap.values() for n in d})
    print("| System | band | " + " | ".join(f"N={n}" for n in orders)
          + " | N=5 -> N=8 | monotone? |")
    print("|---|---:|" + "---:|" * len(orders) + "---|---|")
    tally = {}
    for s in SY:
        bands = sorted({k[1] for k in kap if k[0] == s and k[2] == "implemented"},
                       key=lambda x: (x.isdigit() is False, int(x) if x.isdigit() else 0))
        for b in bands:
            d = kap.get((s, b, "implemented")) or {}
            ks = {n: 0.5 * (d[n]["kappa_lower"] + d[n]["kappa_upper"])
                  for n in orders if n in d and d[n]["conclusive"]}
            fin = [ks[n] for n in orders if n in ks and ks[n] < 1e30]
            mono = all(y < x for x, y in zip(fin, fin[1:])) if len(fin) > 1 else None
            r58 = (f"{ks[8]/ks[5]:.2f}x" if 5 in ks and 8 in ks else "—")
            if mono is not None:
                tally.setdefault(s, []).append(mono)
            print(f"| {s} | {b} | " + " | ".join(fmt(d[n]) if n in d else "—" for n in orders)
                  + f" | {r58} | {'yes' if mono else '**NO**'} |")
    if tally:
        print("\n| System | bands measured | monotone | fraction |")
        print("|---|---:|---:|---:|")
        allv = []
        for s, v in tally.items():
            allv += v
            print(f"| {s} | {len(v)} | {sum(v)} | {100*sum(v)/len(v):.0f} % |")
        print(f"| **total** | **{len(allv)}** | **{sum(allv)}** | **{100*sum(allv)/len(allv):.0f} %** |")
