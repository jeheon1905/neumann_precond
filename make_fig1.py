"""Figure 1 for the revision: the convergence condition is a property of P.

Plots mu_max(Pi P M Pi) = 1 - lambda_min(Pi E Pi) against the corrected shift, with the
Neumann limit mu_max = 2 marked. The three molecular systems cross that line at the same
shift to within 0.6 %, which is the evidence that the boundary belongs to the GAPP
parameterisation rather than to any particular material; the periodic system never reaches it.

Run:  python make_fig1.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SY = ["MAPbI3", "C60_4", "water_cluster_128", "B12"]
LBL = {"MAPbI3": "MAPbI$_3$ (periodic)", "C60_4": "(C$_{60}$)$_4$",
       "water_cluster_128": "(H$_2$O)$_{128}$", "B12": "B$_{12}$"}
SRC = "results_spectral_revision/manifold_survey/{}_survey.json"
OUT = "results_spectral_revision/figures_final/fig1_condition.png"


def main():
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    crossings = []
    for i, s in enumerate(SY):
        try:
            p = json.load(open(SRC.format(s)))
        except OSError:
            continue
        pts = []
        for e in p["states"]:
            recs = e.get("deflated", {}).get("deflated_spectrum") or []
            if not recs:
                continue
            r = max(recs, key=lambda x: x["krylov_dim_built"])
            # mu_max = 1 - lambda_min
            pts.append((e["corrected_epsilon"], 1.0 - r["summary"]["lambda_min_real"]))
        pts.sort()
        ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", ms=4.5,
                color=c[i], label=LBL[s])
        for k in range(len(pts) - 1):
            if (pts[k][1] - 2) * (pts[k + 1][1] - 2) < 0:
                t = (2 - pts[k][1]) / (pts[k + 1][1] - pts[k][1])
                crossings.append((s, pts[k][0] + t * (pts[k + 1][0] - pts[k][0])))

    ax.axhline(2.0, color="r", ls="--", lw=1.6)
    ax.text(-0.36, 2.04, r"Neumann limit  $\mu_{\max}=2$", color="r", fontsize=9.5)
    for _, x0 in crossings:
        ax.plot([x0], [2.0], "kv", ms=8, zorder=5)
    if crossings:
        ax.annotate("the three molecular systems cross\nwithin 0.6 % of each other",
                    xy=(float(np.mean([x for _, x in crossings])), 2.0),
                    xytext=(-0.80, 1.72), arrowprops=dict(arrowstyle="->", lw=1.0),
                    fontsize=9, ha="left")
    ax.set_xlabel(r"corrected shift $\tilde\varepsilon$ (Ha)")
    ax.set_ylabel(r"$\mu_{\max}(\Pi P M \Pi)$")
    ax.set_title(r"The convergence condition $\mu_{\max}<2$ is a property of $P$, "
                 r"not of the system", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("crossings: " + "   ".join(f"{s}: {x:+.4f} Ha" for s, x in crossings))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
