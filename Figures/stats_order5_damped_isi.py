"""Per-iteration mean/std of Diag. and Precond. time for Damped NP vs ISI.

Two views:
  (1) ALL iters
  (2) STEADY-STATE only (excludes 1st iter [no precond] and last iter
      [convergence/partial step]; both have empty Precond entries)
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "order5_np_damped_isi_per_iteration_time.csv"

SYSTEMS = ["$(H_2O)_{128}$", "$C_{60}$ tetramer", "$MAPbI_3$", "Vitamin $B_{12}$"]
SYS_PRETTY = {
    "$(H_2O)_{128}$":     "(H2O)128",
    "$C_{60}$ tetramer":  "C60 tetramer",
    "$MAPbI_3$":          "MAPbI3",
    "Vitamin $B_{12}$":   "Vitamin B12",
}
METHODS = ["Damped NP", "ISI"]
METHOD_LABEL = {"Damped NP": "Damped-NP", "ISI": "ISI"}


def load():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r["Method"] not in METHODS or r["System"] not in SYSTEMS:
                continue
            d = r["Diag. iter. time (s)"].strip()
            p = r["Preconditioning time (s)"].strip()
            rows.append({
                "method": r["Method"],
                "system": r["System"],
                "iter":   int(r["Diag. iter. index"]),
                "diag":   float(d) if d else None,
                "pre":    float(p) if p else None,
            })
    return rows


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0, 1
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)   # sample std
    return mu, math.sqrt(var), n


def summarize(rows, mode):
    """mode: 'all' or 'steady'."""
    out = {}
    for sys_key in SYSTEMS:
        out[sys_key] = {}
        for m in METHODS:
            sel = [r for r in rows if r["method"] == m and r["system"] == sys_key]
            if mode == "steady":
                sel = [r for r in sel if r["pre"] is not None]
            diag = [r["diag"] for r in sel]
            pre  = [r["pre"]  for r in sel]
            out[sys_key][m] = {
                "diag": mean_std(diag),
                "pre":  mean_std(pre),
                "n":    len(sel),
            }
    return out


def fmt(mu, sd):
    if math.isnan(mu):
        return "        n/a       "
    return f"{mu:7.4f} ± {sd:7.4f}"


def print_table(stats, title):
    print(f"\n=== {title} ===")
    header = f"{'System':<15} {'Method':<10} {'n':>3}  {'Diag (s)':>20}  {'Precond (s)':>20}"
    print(header)
    print("-" * len(header))
    for sys_key in SYSTEMS:
        for m in METHODS:
            s = stats[sys_key][m]
            mu_d, sd_d, _ = s["diag"]
            mu_p, sd_p, _ = s["pre"]
            print(f"{SYS_PRETTY[sys_key]:<15} {METHOD_LABEL[m]:<10} {s['n']:>3}  "
                  f"{fmt(mu_d, sd_d):>20}  {fmt(mu_p, sd_p):>20}")
        print()


def print_ratio(stats_steady):
    print("=== Std ratio  (ISI / Damped-NP)   [steady-state]  ===")
    header = f"{'System':<15} {'Diag std ratio':>18} {'Precond std ratio':>22}"
    print(header)
    print("-" * len(header))
    for sys_key in SYSTEMS:
        damped = stats_steady[sys_key]["Damped NP"]
        isi    = stats_steady[sys_key]["ISI"]
        rd = isi["diag"][1] / damped["diag"][1] if damped["diag"][1] > 0 else float("nan")
        rp = isi["pre"][1]  / damped["pre"][1]  if damped["pre"][1]  > 0 else float("nan")
        print(f"{SYS_PRETTY[sys_key]:<15} {rd:>18.2f} {rp:>22.2f}")
    print()


def write_csv(stats_steady, stats_all):
    out_path = HERE / "order5_np_damped_isi_per_iter_std.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "System", "Method", "Region", "n",
            "Diag mean (s)", "Diag std (s)",
            "Precond mean (s)", "Precond std (s)",
        ])
        for region, stats in (("steady", stats_steady), ("all", stats_all)):
            for sys_key in SYSTEMS:
                for m in METHODS:
                    s = stats[sys_key][m]
                    w.writerow([
                        SYS_PRETTY[sys_key], METHOD_LABEL[m], region, s["n"],
                        f"{s['diag'][0]:.6f}", f"{s['diag'][1]:.6f}",
                        f"{s['pre'][0]:.6f}",  f"{s['pre'][1]:.6f}",
                    ])
    print(f"Saved: {out_path}")


def main():
    rows = load()
    stats_all    = summarize(rows, "all")
    stats_steady = summarize(rows, "steady")
    print_table(stats_all,    "All iterations")
    print_table(stats_steady, "Steady-state (excl. 1st & last iter)")
    print_ratio(stats_steady)
    write_csv(stats_steady, stats_all)


if __name__ == "__main__":
    main()
