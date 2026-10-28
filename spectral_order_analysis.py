"""Condition number of the order-N Neumann preconditioner, and the optimal order.

Uses the identities

    P_N M      = sum_{n=0}^{N} E^n (I - E) = I - E^{N+1}
    Pbar_N M   = I - (1/2) E^N (I + E)

so the eigenvalues of the order-N preconditioned operator follow in closed form from the
eigenvalues of E.  Everything here is computed from the already-saved deflated Ritz
spectra: no new electronic-structure calculation is performed.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

RAW = "results_spectral_revision/deflated_residuals"   # m up to 80, with explicit Ritz residuals
OUT = "results_spectral_revision/order_analysis"
FIG = os.path.join(OUT, "figures")

SY = ["B12", "water_cluster_128", "C60_4", "MAPbI3"]

# archived Damped-NP Davidson iteration counts (median run), orders 0..10
OBS = {
    "B12":               [158, 73, 42, 33, 27, 23, 21, 20, 19, 19, 19],
    "water_cluster_128": [127, 58, 34, 27, 22, 20, 19, 19, 19, 19, 18],
    "C60_4":             [123, 57, 33, 26, 21, 18, 17, 16, 16, 16, 16],
    "MAPbI3":            [80, 51, 31, 24, 20, 17, 15, 14, 13, 12, 12],
}
NMAX = 12


def load(state="homo", iiter=15):
    out = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        p = json.load(open(f))
        if p["davidson_meta"]["i_iter"] != iiter:
            continue
        s = p["system"].split(".")[0]
        for e in p["states"]:
            if e["state_rule"] != state:
                continue
            recs = e["deflated"]["deflated_spectrum"]
            r = max(recs, key=lambda x: x["krylov_dim_built"])   # largest m available
            out[s] = {"lam": np.array(r["ritz_values_real"]),
                      "m": r["krylov_dim_built"],
                      "res": max(t.get("ritz_residual", np.nan)
                                 for t in r["targets"].values()),
                      "eps": e["corrected_epsilon"], "band": e["state_index"]}
    return out


def eig_PN_M(lam, N, damped):
    """Eigenvalues of the order-N preconditioned operator."""
    return 1.0 - 0.5 * lam ** N * (1.0 + lam) if damped else 1.0 - lam ** (N + 1)


def kappa(lam, N, damped):
    v = eig_PN_M(lam, N, damped)
    if (v <= 0).any():
        return np.inf          # indefinite: condition number undefined
    return float(v.max() / v.min())


def main():
    os.makedirs(FIG, exist_ok=True)
    d = load()

    # ---------------- table + csv ----------------
    dlow0 = load(state="lowest")
    rows = []
    print("### kappa(P_N M), LOWEST state (where lambda_dom < -1), Davidson iteration 15\n")
    print("| System | λ_dom | N | κ undamped | κ damped | undamped definite? |")
    print("|---|---:|---:|---:|---:|---|")
    for s in SY:
        lam = dlow0[s]["lam"]
        ld = lam[np.argmax(np.abs(lam))]
        for N in [0, 1, 2, 3, 5, 8, 11]:
            kn, kd = kappa(lam, N, False), kappa(lam, N, True)
            rows.append({"system": s, "lambda_dom": ld, "N": N,
                         "kappa_NP": kn, "kappa_DNP": kd,
                         "NP_definite": np.isfinite(kn)})
            print(f"| {s} | {ld:+.4f} | {N} | "
                  f"{'indefinite' if not np.isfinite(kn) else f'{kn:.3f}'} | "
                  f"{kd:.3f} | {'yes' if np.isfinite(kn) else '**no**'} |")
    with open(os.path.join(OUT, "kappa_vs_order.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"\nwrote {OUT}/kappa_vs_order.csv")

    # ---------------- fit iters ~ kappa^p ----------------
    fits = {}
    print("\n### iterations ~ kappa_N^p, fitted against the archived benchmark\n")
    print("| System | p | mean error | max error |")
    print("|---|---:|---:|---:|")
    for s in SY:
        lam = d[s]["lam"]
        kd = np.array([kappa(lam, N, True) for N in range(11)])
        o = np.array(OBS[s], float)
        p_, c_ = np.polyfit(np.log(kd), np.log(o), 1)
        pred = np.exp(c_) * kd ** p_
        err = np.abs(pred - o) / o
        fits[s] = (p_, np.exp(c_), kd, o, pred)
        print(f"| {s} | {p_:.2f} | {err.mean()*100:.1f} % | {err.max()*100:.1f} % |")

    # ---------------- optimal order ----------------
    print("\n### N_opt = argmin  kappa_N^p (c + N)\n")
    cs = [1, 3, 5, 8, 12]
    print("| System | " + " | ".join(f"c={c}" for c in cs) + " |")
    print("|---|" + "---:|" * len(cs))
    for s in SY:
        p_, A, kd, o, pred = fits[s]
        cells = [int(np.argmin(pred * (c + np.arange(11)))) for c in cs]
        print(f"| {s} | " + " | ".join(f"N={c}" for c in cells) + " |")

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable:", exc)
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Fig A: kappa vs N, NP and DNP -- at the LOWEST state, which is where
    # lambda_dom < -1 and the undamped operator therefore turns indefinite.
    dlow = load(state="lowest")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    N = np.arange(NMAX)
    for i, s in enumerate(SY):
        lam = dlow[s]["lam"]
        ld = lam[np.argmax(np.abs(lam))]
        kn = np.array([kappa(lam, n, False) for n in N])
        kd = np.array([kappa(lam, n, True) for n in N])
        c = colors[i % len(colors)]
        lab = f"{s} ($\\lambda_{{dom}}$={ld:+.3f})"
        good = np.isfinite(kn)
        off = (i - 1.5) * 0.12                      # small offset so markers do not overlap
        axes[0].plot(N[good] + off, kn[good], "o-", color=c, label=lab, ms=5, lw=0.8)
        axes[0].plot(N[~good] + off, np.full((~good).sum(), 45), "x", color=c, ms=8, mew=2)
        axes[1].plot(N, kd, "s--", color=c, label=lab, ms=5)
    axes[0].set_title("undamped NP  —  × marks INDEFINITE $P_N M$", fontsize=10)
    axes[1].set_title("Damped-NP (weight 0.5)", fontsize=10)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Neumann order $N$")
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\kappa(P_N M)$  (deflated)")
    axes[0].set_ylim(0.9, 80)
    axes[0].text(5.5, 55, "indefinite", ha="center", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.suptitle("Condition number of the order-$N$ Neumann preconditioner "
                 "(lowest state, Davidson iteration 15)", fontsize=11)
    fig.tight_layout()
    f = os.path.join(FIG, "kappa_vs_order.png")
    fig.savefig(f, dpi=170)
    print("\nwrote", f)

    # Fig B: predicted vs observed iteration counts
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, s in enumerate(SY):
        p_, A, kd, o, pred = fits[s]
        c = colors[i % len(colors)]
        ax.plot(range(11), o, "o-", color=c, label=f"{s} (obs)")
        ax.plot(range(11), pred, "--", color=c, alpha=0.75,
                label=f"{s} ($\\kappa^{{{p_:.2f}}}$)")
    ax.set_yscale("log")
    ax.set_xlabel("Neumann order $N$")
    ax.set_ylabel("Davidson iterations (Damped-NP)")
    ax.set_title(r"$\kappa(P_N M)$ from the spectrum predicts the archived benchmark")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(FIG, "kappa_predicts_iterations.png")
    fig.savefig(f, dpi=170)
    print("wrote", f)

    # Fig C: work vs order
    fig, axes = plt.subplots(1, len(SY), figsize=(15, 3.6), sharex=True)
    for i, s in enumerate(SY):
        p_, A, kd, o, pred = fits[s]
        for c in [1, 3, 5, 8, 12]:
            w = pred * (c + np.arange(11))
            axes[i].plot(range(11), w / w.min(), "o-", ms=3, label=f"c={c}")
        axes[i].set_title(s, fontsize=9)
        axes[i].set_xlabel("$N$")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("relative work  $\\kappa_N^p (c+N)$")
    axes[0].legend(fontsize=7)
    fig.suptitle("Predicted total work versus Neumann order, for several per-iteration "
                 "overheads $c$", fontsize=10)
    fig.tight_layout()
    f = os.path.join(FIG, "work_vs_order.png")
    fig.savefig(f, dpi=170)
    print("wrote", f)


if __name__ == "__main__":
    main()
