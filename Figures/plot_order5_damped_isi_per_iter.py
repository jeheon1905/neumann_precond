"""Plot per-iteration Diag. time and Precond. time for Damped (order=5) vs ISI.

4 subplots (a)-(d) for systems: H2O, C60, MAPbI3, Vitamin B12.
Single y-axis per subplot; single shared legend at the figure bottom.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "order5_np_damped_isi_per_iteration_time.csv"

SYSTEMS = [
    ("$(H_2O)_{128}$",     r"(a) $(\mathrm{H}_2\mathrm{O})_{128}$"),
    ("$C_{60}$ tetramer",  r"(b) $\mathrm{C}_{60}$ tetramer"),
    ("$MAPbI_3$",          r"(c) $\mathrm{MAPbI}_3$"),
    ("Vitamin $B_{12}$",   r"(d) Vitamin $\mathrm{B}_{12}$"),
]
METHODS = ["Damped NP", "ISI"]
METHOD_LABEL = {"Damped NP": "Damped-NP (order=5)", "ISI": "ISI"}
METHOD_COLOR = {"Damped NP": "tab:blue", "ISI": "tab:orange"}

TITLE_FS = 18
LABEL_FS = 16
TICK_FS  = 14
LEGEND_FS = 15


def load():
    data = {m: {s: {"iter": [], "diag": [], "pre": []} for s, _ in SYSTEMS} for m in METHODS}
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m, s = row["Method"], row["System"]
            if m not in METHODS or s not in data[m]:
                continue
            try:
                it = int(row["Diag. iter. index"])
            except ValueError:
                continue
            d = row["Diag. iter. time (s)"].strip()
            p = row["Preconditioning time (s)"].strip()
            data[m][s]["iter"].append(it)
            data[m][s]["diag"].append(float(d) if d else float("nan"))
            data[m][s]["pre"].append(float(p) if p else float("nan"))
    return data


def main():
    data = load()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes = axes.ravel()

    legend_handles, legend_labels = [], []

    for ax_idx, (ax, (sys_key, title)) in enumerate(zip(axes, SYSTEMS)):
        for m in METHODS:
            d = data[m][sys_key]
            color = METHOD_COLOR[m]
            label_base = METHOD_LABEL[m]
            (h_diag,) = ax.plot(
                d["iter"], d["diag"], color=color, linestyle="-",
                marker="o", markersize=4, linewidth=1.6,
                label=f"{label_base} - Diag. time",
            )
            (h_pre,) = ax.plot(
                d["iter"], d["pre"], color=color, linestyle="--",
                marker="s", markersize=4, linewidth=1.6,
                label=f"{label_base} - Precond. time",
            )
            if ax_idx == 0:
                legend_handles.extend([h_diag, h_pre])
                legend_labels.extend([h_diag.get_label(), h_pre.get_label()])

        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_xlabel("Diag. iter.", fontsize=LABEL_FS)
        ax.set_ylabel("Time (s)", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=2, fontsize=LEGEND_FS,
        bbox_to_anchor=(0.5, 0.0), frameon=True,
    )

    out = HERE / "order5_np_damped_isi_per_iteration_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
