"""Aggregate the state-resolved spectral results into manuscript tables and figures."""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

RAW = "results_spectral_revision/state_dependence/raw"
OUT = "results_spectral_revision/state_dependence"
FIG = os.path.join(OUT, "figures")

SYS_ORDER = ["B12.sdf", "water_cluster_128.xyz", "C60_4.xyz", "MAPbI3.cif"]
SHORT = {"B12.sdf": "B12", "water_cluster_128.xyz": "water_cluster_128",
         "C60_4.xyz": "C60_4", "MAPbI3.cif": "MAPbI3"}
RULE_ORDER = ["lowest", "middle", "homo", "slowest"]

# undamped-NP Davidson iterations at odd orders, from the archived benchmark
NP_ODD = {"B12": "1000 (no conv.)", "water_cluster_128": "173-322",
          "C60_4": "68-223", "MAPbI3": "no oscillation"}

FIELDS = [
    "system", "state_rule", "state_index", "active_index", "davidson_iteration",
    "preconditioner_call", "residual_norm", "raw_epsilon", "corrected_epsilon",
    "zero_shift_fallback", "krylov_dim", "seed", "rho", "lambda_min", "lambda_max",
    "dominant_is_negative", "damping_factor_lambda_min", "ritz_residual_rho",
    "ritz_residual_lambda_min", "numerically_real", "commit", "dtype", "device",
]


def load():
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return sorted(out, key=lambda p: SYS_ORDER.index(p["system"]))


def best(entry):
    """Record at the largest Krylov dimension."""
    recs = entry["global_spectrum"]
    m = max(r["krylov_dim_built"] for r in recs)
    return [r for r in recs if r["krylov_dim_built"] == m][0]


def main():
    os.makedirs(FIG, exist_ok=True)
    payloads = load()

    rows = []
    for p in payloads:
        for e in p["states"]:
            for r in e["global_spectrum"]:
                su, t = r["summary"], r["targets"]
                lam_min = su["lambda_min_real"]
                rows.append({
                    "system": p["system"], "state_rule": e["state_rule"],
                    "state_index": e["state_index"], "active_index": e["active_index"],
                    "davidson_iteration": p["davidson_meta"]["i_iter"],
                    "preconditioner_call": p["preconditioner_call"],
                    "residual_norm": e["residual_norm"],
                    "raw_epsilon": e["raw_epsilon"],
                    "corrected_epsilon": e["corrected_epsilon"],
                    "zero_shift_fallback": e["zero_shift_fallback"],
                    "krylov_dim": r["krylov_dim_built"], "seed": r.get("seed"),
                    "rho": su["rho"], "lambda_min": lam_min,
                    "lambda_max": su["lambda_max_real"],
                    "dominant_is_negative": abs(lam_min) >= su["rho"] - 1e-9,
                    "damping_factor_lambda_min": t["most_negative"]["damping_factor"],
                    "ritz_residual_rho": t["largest_magnitude"].get("ritz_residual"),
                    "ritz_residual_lambda_min": t["most_negative"].get("ritz_residual"),
                    "numerically_real": su["numerically_real"],
                    "commit": p["commit"], "dtype": e["dtype"],
                    "device": p["environment"]["device"],
                })

    csv_path = os.path.join(OUT, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---------- Table 1: state-resolved spectrum ----------
    print("\n### Table 1 — state-resolved spectrum of E at the same snapshot "
          "(Davidson iteration 5, order 4, seed 2)\n")
    print("| System | State | band | ‖γ‖ | ε̃ | ρ(E) | λ_min | dominant | (1+λ_min)/2 |")
    print("|---|---|---:|---:|---:|---:|---:|---|---:|")
    for p in payloads:
        for rule in RULE_ORDER:
            es = [e for e in p["states"] if e["state_rule"] == rule]
            if not es:
                continue
            e = es[0]
            r = best(e)
            su, t = r["summary"], r["targets"]
            dom = "**negative**" if abs(su["lambda_min_real"]) >= su["rho"] - 1e-9 else "positive"
            print(f"| {SHORT[p['system']]} | {rule} | {e['state_index']} | "
                  f"{e['residual_norm']:.4e} | {e['corrected_epsilon']:+.6f} | "
                  f"{su['rho']:.6f} | {su['lambda_min_real']:+.6f} | {dom} | "
                  f"{t['most_negative']['damping_factor']:.4f} |")

    # ---------- Table 2: the mechanism ----------
    print("\n### Table 2 — dominant negative mode of the lowest state vs observed "
          "undamped-NP behaviour\n")
    print("| System | λ_dom (lowest state) | \\|λ_dom\\| | \\|(1+λ)/2\\| | "
          "½\\|1+λ\\|·\\|λ\\|¹⁰ | undamped NP, odd order |")
    print("|---|---:|---:|---:|---:|---|")
    mech = []
    for p in payloads:
        e = [x for x in p["states"] if x["state_rule"] == "lowest"][0]
        su = best(e)["summary"]
        lam = su["lambda_min_real"]
        dominant = abs(lam) >= su["rho"] - 1e-9
        damp = abs(0.5 * (1 + lam))
        name = SHORT[p["system"]]
        mech.append((name, lam, abs(lam), damp, damp * abs(lam) ** 10, dominant))
        note = "" if dominant else f" (subdominant; ρ = {su['rho']:.6f} > 0)"
        print(f"| {name} | {lam:+.6f}{note} | {abs(lam):.4f} | {damp:.4f} | "
              f"{damp*abs(lam)**10:.4f} | {NP_ODD[name]} |")

    # ---------- figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable:", exc)
        return

    # Figure A: rho and lambda_min vs corrected shift, per system
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, p in enumerate(payloads):
        eps, rho, lmin = [], [], []
        for e in sorted(p["states"], key=lambda x: x["corrected_epsilon"]):
            su = best(e)["summary"]
            eps.append(e["corrected_epsilon"])
            rho.append(su["rho"])
            lmin.append(su["lambda_min_real"])
        lab = SHORT[p["system"]]
        axes[0].plot(eps, rho, "o-", color=colors[i % len(colors)], label=lab)
        axes[1].plot(eps, lmin, "o-", color=colors[i % len(colors)], label=lab)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.9)
    axes[0].set_xlabel(r"corrected shift $\tilde\epsilon$ (Ha)")
    axes[0].set_ylabel(r"$\rho(E)$")
    axes[0].set_title(r"spectral radius vs shift")
    axes[1].axhline(-1.0, color="r", ls="--", lw=0.9)
    axes[1].set_xlabel(r"corrected shift $\tilde\epsilon$ (Ha)")
    axes[1].set_ylabel(r"$\lambda_{\min}$")
    axes[1].set_title(r"most negative eigenvalue vs shift")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    f = os.path.join(FIG, "spectrum_vs_shift.png")
    fig.savefig(f, dpi=170)
    print("\nwrote", f)

    # Figure B: eta_N for NP and Damped-NP, lowest state
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, p in enumerate(payloads):
        e = [x for x in p["states"] if x["state_rule"] == "lowest"][0]
        if "eta" not in e:
            continue
        eta = e["eta"]
        N = np.arange(len(eta["eta_NP"]))
        c = colors[i % len(colors)]
        axes[0].semilogy(N, eta["eta_NP"], "o-", color=c, label=SHORT[p["system"]])
        axes[1].semilogy(N, eta["eta_DNP"], "s--", color=c, label=SHORT[p["system"]])
    for ax, title in zip(axes, [r"undamped NP: $\eta_N=\|a_{N+1}\|/\|a_0\|$",
                                r"Damped-NP: $\eta_N=\|a_N+a_{N+1}\|/2\|a_0\|$"]):
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.set_xlabel("expansion order $N$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"$\eta_N$")
    axes[0].legend(fontsize=8)
    fig.suptitle("lowest state", fontsize=10)
    fig.tight_layout()
    f = os.path.join(FIG, "eta_lowest_state.png")
    fig.savefig(f, dpi=170)
    print("wrote", f)

    # Figure C: eta_N for the slowest state (the Phase-2/3 snapshot)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, p in enumerate(payloads):
        cand = [x for x in p["states"] if x["state_rule"] == "slowest"] or \
               [x for x in p["states"] if x["state_rule"] == "homo"]
        e = cand[0]
        if "eta" not in e:
            continue
        eta = e["eta"]
        N = np.arange(len(eta["eta_NP"]))
        c = colors[i % len(colors)]
        ax.semilogy(N, eta["eta_NP"], "o-", color=c, label=f"{SHORT[p['system']]} NP")
        ax.semilogy(N, eta["eta_DNP"], "s--", color=c, alpha=0.6,
                    label=f"{SHORT[p['system']]} DNP")
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xlabel("expansion order $N$")
    ax.set_ylabel(r"$\eta_N$")
    ax.set_title("slowest-converging state: transient decay then growth")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    f = os.path.join(FIG, "eta_slowest_state.png")
    fig.savefig(f, dpi=170)
    print("wrote", f)


if __name__ == "__main__":
    main()
