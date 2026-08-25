#!/usr/bin/env python3
"""Extract per-iteration diagonalization / preconditioning times from run logs.

Produces the CSV consumed by ``Figures/plot_order5_damped_isi_per_iter.py``
(Figure S1) and ``Figures/stats_order5_damped_isi.py`` (Table S2).

The per-iteration numbers exist only as the individual lines

    [Time: Diag. Iter.]: 1.456300 s
    [Time: Preconditioning]: 0.988017 s

which ``Timer.stop`` prints only when ``verbosity >= 1``.  The aggregate
"Timer Summary" block carries totals and counts alone and cannot give a
standard deviation, so this script must be pointed at a run made with
``configs/paper/config.fixed.periter.yaml``.

Alignment follows the published CSV: with ``n`` ``Diag. Iter.`` lines there are
``n - 2`` ``Preconditioning`` lines, because the first outer iteration runs no
preconditioning step and the last is a convergence step.  Iteration indices are
1-based; the preconditioning column is left empty at the first and last index.

Usage
-----
    python extract_per_iteration_time.py \\
        --results_root results_paper/fixed.periter \\
        --order 5 \\
        --out Figures/order5_np_damped_isi_per_iteration_time.csv

Add ``--label-np`` to also pick up an ``avgsum=0`` sweep as a third method.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

DIAG_RE = re.compile(r"^\[Time: Diag\. Iter\.\]: ([\d.]+) s", re.M)
PRECOND_RE = re.compile(r"^\[Time: Preconditioning\]: ([\d.]+) s", re.M)

# Directory stem -> label used in the published CSV.
SYSTEM_LABELS = {
    "water_cluster_128": r"$(H_2O)_{128}$",
    "C60_4": r"$C_{60}$ tetramer",
    "MAPbI3": r"$MAPbI_3$",
    "B12": r"Vitamin $B_{12}$",
}
SYSTEM_ORDER = ["water_cluster_128", "C60_4", "MAPbI3", "B12"]


def series(log: Path) -> tuple[list[str], list[str]]:
    """Return the timings verbatim, as printed, so the CSV is byte-reproducible."""
    text = log.read_text(encoding="utf-8", errors="ignore")
    return DIAG_RE.findall(text), PRECOND_RE.findall(text)


def pick_log(root: Path, system: str, *, precond: str, order: Optional[int],
             avgsum: Optional[int]) -> Optional[Path]:
    """Locate the median (or first) run log for one system/method.

    repeat_test.py copies the representative run to ``history/.../median/`` and
    keeps the individual seeds under ``logs/.../run-N/``.  The paper quotes the
    median of three runs, so ``history/`` is searched first.
    """
    want = [f"prec={precond}"]
    if order is not None:
        want.append(f"outerorder={order}")
    if avgsum is not None:
        want.append(f"avgsum={avgsum}")

    cands: list[Path] = []
    for sub in ("history", "logs"):
        base = root / sub / system
        if not base.is_dir():
            continue
        hit = [p for p in base.rglob("stdout.log") if all(w in str(p) for w in want)]
        if not hit and avgsum is not None:
            # Runs made before the --averaged_sum CLI existed (commit 5614bf2)
            # have no `avgsum=` component; fall back to matching without it so
            # this tool also works on the archived layout.
            relaxed = [w for w in want if not w.startswith("avgsum=")]
            hit = [p for p in base.rglob("stdout.log")
                   if all(w in str(p) for w in relaxed) and "avgsum=" not in str(p)]
            if hit:
                print(f"  [note] {system}: no avgsum= in paths; matched without it")
        cands.extend(hit)

    if not cands:
        return None
    for p in cands:
        if p.parent.name == "median":
            return p
    print(f"  [note] {system}/{precond}: no median/ run found; using {sorted(cands)[0].parent.name}")
    return sorted(cands)[0]


def rows_for(method: str, system: str, log: Path) -> list[dict]:
    diag, prec = series(log)
    if not diag:
        return []
    if len(prec) not in (len(diag), len(diag) - 2):
        print(f"  [WARN] {system}/{method}: {len(diag)} Diag. Iter. lines but "
              f"{len(prec)} Preconditioning lines; padding by position")
    out = []
    for i, d in enumerate(diag, start=1):
        # preconditioning runs on iterations 2 .. n-1
        p = ""
        if 2 <= i <= len(diag) - 1:
            k = i - 2
            if k < len(prec):
                p = prec[k]
        out.append({
            "Method": method,
            "System": SYSTEM_LABELS.get(system, system),
            "Diag. iter. index": i,
            "Diag. iter. time (s)": d,
            "Preconditioning time (s)": p,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", required=True, type=Path,
                    help="results_root of a config.fixed.periter.yaml run")
    ap.add_argument("--order", type=int, default=5,
                    help="Neumann expansion order to extract (default: 5)")
    ap.add_argument("--out", type=Path,
                    default=Path("order5_np_damped_isi_per_iteration_time.csv"))
    ap.add_argument("--label-np", action="store_true",
                    help="also extract the avgsum=0 sweep as method 'NP'")
    args = ap.parse_args()

    methods = [("Damped NP", "neumann", args.order, 1),
               ("ISI", "shift-and-invert", None, None)]
    if args.label_np:
        methods.insert(0, ("NP", "neumann", args.order, 0))

    rows: list[dict] = []
    for system in SYSTEM_ORDER:
        for method, precond, order, avgsum in methods:
            log = pick_log(args.results_root, system,
                           precond=precond, order=order, avgsum=avgsum)
            if log is None:
                print(f"  [MISS] {system:<20} {method:<10} no log found")
                continue
            r = rows_for(method, system, log)
            rows.extend(r)
            n_p = sum(1 for x in r if x["Preconditioning time (s)"])
            print(f"  [ OK ] {system:<20} {method:<10} {len(r):>3} iters, "
                  f"{n_p:>3} precond   {log.relative_to(args.results_root)}")

    if not rows:
        raise SystemExit("no data extracted; check --results_root and --order")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
