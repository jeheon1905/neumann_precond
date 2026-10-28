"""Manuscript tables and figures for the deflated analysis."""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

RAW = "results_spectral_revision/deflated/raw"
OUT = "results_spectral_revision/deflated"
FIG = os.path.join(OUT, "figures")

SY = ["B12", "water_cluster_128", "C60_4", "MAPbI3"]
RULES = ["lowest", "middle", "homo", "slowest"]
ITERS = [5, 10, 15]


def load():
    D, meta = {}, {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        p = json.load(open(f))
        it = p["davidson_meta"]["i_iter"]
        s = p["system"].split(".")[0]
        meta[(s, it)] = p
        for e in p["states"]:
            if e.get("deflated", {}).get("deflated_spectrum"):
                D[(s, e["state_rule"], it)] = e
    return D, meta


def spec(e, m=40):
    r = [x for x in e["deflated"]["deflated_spectrum"] if x["krylov_dim_built"] == m]
    return r[0]["summary"] if r else None


def main():
    os.makedirs(FIG, exist_ok=True)
    D, meta = load()

    # ---------------- CSV ----------------
    fields = ["system", "davidson_iteration", "state_rule", "state_index",
              "corrected_epsilon", "residual_norm", "rho_deflated", "lambda_min_deflated",
              "damping_factor", "cg_status", "cg_iters", "n_subspace",
              "X_orthonormality_error", "krylov_dim", "commit"]
    rows = []
    for (s, r, it), e in sorted(D.items()):
        su = spec(e)
        d = e["deflated"]
        rows.append({
            "system": s, "davidson_iteration": it, "state_rule": r,
            "state_index": e["state_index"],
            "corrected_epsilon": e["corrected_epsilon"],
            "residual_norm": e["residual_norm"],
            "rho_deflated": su["rho"], "lambda_min_deflated": su["lambda_min_real"],
            "damping_factor": abs(0.5 * (1 + su["lambda_min_real"])),
            "cg_status": d["cg_status"], "cg_iters": d["cg_iters"],
            "n_subspace": d["n_subspace"],
            "X_orthonormality_error": d["X_orthonormality_error"],
            "krylov_dim": 40,
            "commit": meta[(s, it)]["commit"],
        })
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}/summary.csv ({len(rows)} rows)")

    # ---------------- Table: deflated spectrum at the converged iteration ------
    print("\n### Deflated spectrum at i_iter = 15 (X converged enough that Pi M Pi is SPD)\n")
    print("| System | State | band | ε̃ | ρ(ΠEΠ) | λ_min(ΠEΠ) | dominant | |(1+λ)/2| | ref. solve |")
    print("|---|---|---:|---:|---:|---:|---|---:|---|")
    for s in SY:
        for r in RULES:
            e = D.get((s, r, 15))
            if e is None:
                continue
            su = spec(e)
            dom = "**negative**" if abs(su["lambda_min_real"]) >= su["rho"] - 1e-9 else "positive"
            cg = "OK" if e["deflated"]["cg_status"] == "converged" else "neg.curv."
            print(f"| {s} | {r} | {e['state_index']} | {e['corrected_epsilon']:+.4f} | "
                  f"{su['rho']:.5f} | {su['lambda_min_real']:+.5f} | {dom} | "
                  f"{abs(0.5*(1+su['lambda_min_real'])):.4f} | {cg} |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable:", exc)
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Fig 1: full-space vs deflated rho, at i_iter = 15 -------------------
    full = {}
    for f in glob.glob("results_spectral_revision/state_dependence/raw/*.json"):
        p = json.load(open(f))
        for e in p["states"]:
            r = max(e["global_spectrum"], key=lambda x: x["krylov_dim_built"])
            full[(p["system"].split(".")[0], e["state_rule"])] = r["summary"]["rho"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels, fv, dv = [], [], []
    for s in SY:
        for r in ["lowest", "middle", "homo"]:
            e = D.get((s, r, 15))
            if e is None or (s, r) not in full:
                continue
            labels.append(f"{s}\n{r}")
            fv.append(full[(s, r)])
            dv.append(spec(e)["rho"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, fv, 0.4, label=r"full space  $\rho(E)$", color="#c44e52")
    ax.bar(x + 0.2, dv, 0.4, label=r"deflated  $\rho(\Pi E \Pi)$", color="#4c72b0")
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(r"spectral radius")
    ax.set_title(r"Deflation removes the modes that make $\rho>1$   (Davidson iteration 15)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(FIG, "rho_full_vs_deflated.png")
    fig.savefig(p1, dpi=170)
    print("\nwrote", p1)

    # --- Fig 2: the damping mechanism at the lowest state -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, s in enumerate(SY):
        e = D.get((s, "lowest", 15))
        if e is None:
            continue
        sc = e["deflated"]["order_scan"]
        N = [r["N"] for r in sc]
        lam = spec(e)["lambda_min_real"]
        c = colors[i % len(colors)]
        lab = f"{s}  ($\\lambda_{{dom}}$={lam:+.3f})"
        axes[0].semilogy(N, [r["eta_defl_NP"] for r in sc], "o-", color=c, label=lab)
        axes[1].semilogy(N, [r["eta_defl_DNP"] for r in sc], "s--", color=c, label=lab)
    for ax, t in zip(axes, ["undamped NP", "Damped-NP (weight 0.5)"]):
        ax.set_xlabel("expansion order $N$")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"deflated error  $\eta_N^{\Pi}$")
    axes[0].legend(fontsize=7)
    fig.suptitle("lowest state, Davidson iteration 15: damping is required exactly where "
                 r"$\lambda_{dom}<-1$", fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(FIG, "damping_mechanism.png")
    fig.savefig(p2, dpi=170)
    print("wrote", p2)

    # --- Fig 3: direction quality vs order ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for j, rule in enumerate(["middle", "homo"]):
        for i, s in enumerate(SY):
            e = D.get((s, rule, 15))
            if e is None:
                continue
            sc = e["deflated"]["order_scan"]
            axes[j].plot([r["N"] for r in sc],
                         [r["dir_DNP"]["angle_deg"] for r in sc],
                         "o-", color=colors[i % len(colors)], label=s)
        axes[j].set_xlabel("expansion order $N$")
        axes[j].set_title(f"{rule} state", fontsize=10)
        axes[j].grid(alpha=0.3)
    axes[0].set_ylabel("angle to exact deflated correction (deg)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Damped-NP: direction quality improves with order, then saturates",
                 fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(FIG, "direction_vs_order.png")
    fig.savefig(p3, dpi=170)
    print("wrote", p3)

    # --- Fig 4: iteration dependence of the deflation ------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for i, s in enumerate(SY):
        for k, rule in enumerate(["lowest", "homo"]):
            xs, ys = [], []
            for it in ITERS:
                e = D.get((s, rule, it))
                if e is None:
                    continue
                xs.append(it)
                ys.append(spec(e)["rho"])
            if xs:
                ax.plot(xs, ys, ["o-", "s--"][k], color=colors[i % len(colors)],
                        label=f"{s} / {rule}")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("Davidson iteration at which the snapshot is taken")
    ax.set_ylabel(r"$\rho(\Pi E \Pi)$")
    ax.set_title("Deflation becomes exact as the Davidson subspace converges")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(FIG, "deflation_vs_iteration.png")
    fig.savefig(p4, dpi=170)
    print("wrote", p4)


if __name__ == "__main__":
    main()
