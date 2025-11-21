"""
Make Plots Flex — Task1/Task2 (v5: path auto-discovery)

Goal
- Remove hardcoded internal folders. Accept only --root / --out and discover files.
- Keep Task1/Task2 behaviors identical where inputs exist.

How path discovery works
- "results_summary" files are found by tokens in the filename.
- Prefer matches under a directory containing "density_fixed_diagonalization".
- Break ties by shorter path length.
- Task2 median files are found anywhere under root where path contains
  "/{material}/precond_Neumann/" and filename contains "median".

Why minimal inline comments?
- Focus on the *why*: highlight non-obvious heuristics and decisions.
"""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# Settings (materials only; all paths are now discovered)
# =============================================================
MATERIALS = [
    "CaTiO3",
    "CsPbI3",
    "MAPbI3",
    "C60",
    "ATP",
    "GSH",
    "diamond",
    "Fe",
    "aspirin",
]
DEFAULT_CUT_VALUES = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]

# ---- Task1 (select which series to draw from the same primary file) ----
TASK1_MODE = "second"  # "cutoff" | "second" | "both"

# Token rules replacing hardcoded paths
TASK1_PRIMARY_TOKENS = {
    # Typical primary summary file: results_summary_Neumann_{material}.txt
    "cutoff": ["results_summary", "Neumann"],
    "second": ["results_summary", "Neumann"],
}
TASK1_REFERENCE_TOKENS: List[Dict[str, Sequence[str]]] = [
    {"label": "shift", "tokens": ["results_summary", "shift", "gapp", "pcg", "5"]},
    {"label": "gapp", "tokens": ["results_summary", "gapp"]},
]

# =============================================================
# Regex
# =============================================================
RE_SPACING = re.compile(r"spacing\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_SUPERCELL = re.compile(r"supercell\s*=\s*\[([^\]]+)\]")
RE_ITER = re.compile(r"(?:iteration|max_iterations)\s*=\s*(\d+)")
RE_TDIAG = re.compile(r"total_diagonal_time\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_ERRCUT = re.compile(r"error_cutoff\s*=\s*(-?\d+(?:\.\d+)?)")
RE_USING_ORDER = re.compile(
    r"using\s+order(?:\((?P<reason>[^)]+)\))?\s*=\s*(?P<order>\d+)", re.I
)
RE_PATH_SUPERCELL = re.compile(r"supercell_(\d+)_(\d+)_(\d+)")
RE_THR_IN_PATH = re.compile(
    r"(?:cutoff|thr|error_cutoff)[^-\d]*(-?\d+(?:\.\d+)?)", re.I
)
RE_PATH_SPACING = re.compile(r"spacing_([0-9]+(?:\.[0-9]+)?)")
RE_REASON_INLINE = re.compile(r"(?:reason|why)\s*[:=]\s*([^\n]+)", re.I)
RE_FILE_THR = re.compile(
    r"(?:using\s+cutoff|error_cutoff)\s*=\s*(-?\d+(?:\.\d+)?)", re.I
)

# =============================================================
# Small utils
# =============================================================


def _parse_supercell(s: str) -> Tuple[int, int, int]:
    return tuple(int(x.strip()) for x in s.split(","))  # type: ignore[return-value]


def safe_log10(v: Optional[float]) -> float:
    return float("nan") if v is None or v <= 0 else math.log10(v)


# --- reason normalization & colors ---


def _normalize_reason(s: Optional[str]) -> str:
    if not s:
        return "default"
    t = re.sub(r"\s+", " ", s.strip()).lower().replace(" ", "")
    if t == "low_order_cutoff":
        return "low_error_cutoff"
    if t == "alwayspre>now":
        return "always pre>now"
    if t == "pre<now_break":
        return "pre<now_break"
    if t == "high_error_cutoff":
        return "high_error_cutoff"
    if t == "low_error_cutoff":
        return "low_error_cutoff"
    if "pre<now" in t and "break" in t:
        return "pre<now_break"
    if "always" in t and "pre>now" in t:
        return "always pre>now"
    if "high" in t and "error" in t and "cutoff" in t:
        return "high_error_cutoff"
    if ("low" in t and ("error" in t or "order" in t)) and "cutoff" in t:
        return "low_error_cutoff"
    return "default"


REASON_COLORS: Dict[str, str] = {
    "pre<now_break": "orange",
    "high_error_cutoff": "red",
    "low_error_cutoff": "green",
    "always pre>now": "purple",
    "default": "blue",
}

# =============================================================
# Path discovery
# =============================================================


def _tokens_in(s: str, tokens: Sequence[str]) -> bool:
    s_low = s.lower()
    return all(tok.lower() in s_low for tok in tokens)


def discover_results_summary(
    root: Path, material: str, tokens: Sequence[str], *, verbose: bool = False
) -> Optional[Path]:
    """Find the best-matching results_summary file for a material using tokens.

    Heuristics: prefer files under a directory containing "density_fixed_diagonalization",
    then prefer shorter paths. This avoids editing code when folder nesting changes.
    """
    candidates: List[Tuple[int, int, Path]] = []
    for p in root.rglob(f"*{material}*.txt"):
        name = p.name
        if "results_summary" not in name.lower():
            continue
        if not _tokens_in(name, tokens):
            continue
        posix = p.as_posix().lower()
        penalty_dir = 0 if "density_fixed_diagonalization" in posix else 1
        candidates.append((penalty_dir, len(posix), p))
    if not candidates:
        if verbose:
            print(f"[discover] no match for material={material}, tokens={tokens}")
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    best = candidates[0][2]
    if verbose:
        print(f"[discover] {material} tokens={tokens} -> {best}")
    return best


def discover_references(
    root: Path,
    material: str,
    refs: Sequence[Dict[str, Sequence[str]]],
    *,
    verbose: bool = False,
) -> List[Dict[str, Optional[Path]]]:
    out: List[Dict[str, Optional[Path]]] = []
    for r in refs:
        p = discover_results_summary(root, material, r["tokens"], verbose=verbose)
        out.append({"label": r.get("label", "ref"), "path": p})
    return out


# -------------------------------------------------------------
# Summary parser — supports merged files with multiple error_cutoff lines
# -------------------------------------------------------------


def read_summary_lines(path: Path) -> List[Dict[str, Optional[float]]]:
    out: List[Dict[str, Optional[float]]] = []
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "spacing" not in line:
            continue
        m_sp = RE_SPACING.search(line)
        m_sc = RE_SUPERCELL.search(line)
        m_it = RE_ITER.search(line)
        m_td = RE_TDIAG.search(line)
        m_ct = RE_ERRCUT.search(line)
        if not (m_sp and m_sc):
            continue
        out.append(
            {
                "spacing": float(m_sp.group(1)) if m_sp else None,
                "supercell": _parse_supercell(m_sc.group(1)) if m_sc else None,
                "iteration": int(m_it.group(1)) if m_it else None,
                "tdiag": float(m_td.group(1)) if m_td else None,
                "cutoff": float(m_ct.group(1)) if m_ct else None,
            }
        )
    return out


def series_from_summary(
    path: Path, metric: str
) -> Tuple[List[float], List[float], Optional[Tuple[int, int, int]], Optional[float]]:
    rows = read_summary_lines(path)
    if not rows:
        return [], [], None, None
    by_thr: Dict[float, List[float]] = {}
    supercells = {r["supercell"] for r in rows if r.get("supercell") is not None}
    spacings = {r["spacing"] for r in rows if r.get("spacing") is not None}
    for r in rows:
        thr = r.get("cutoff")
        val = r.get(metric)
        if thr is None or val is None:
            continue
        by_thr.setdefault(thr, []).append(val)
    xs = sorted(by_thr.keys())
    ys = [safe_log10(float(np.median(by_thr[t]))) for t in xs]
    sc = next(iter(supercells)) if len(supercells) == 1 else None
    sp = next(iter(spacings)) if len(spacings) == 1 else None
    return xs, ys, sc, sp


def reference_value_from_summary(
    path: Optional[Path], metric: str
) -> Tuple[Optional[float], Optional[Tuple[int, int, int]], Optional[float]]:
    if not path or not path.exists():
        return None, None, None
    rows = read_summary_lines(path)
    if not rows:
        return None, None, None
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    v = safe_log10(float(np.median(vals))) if vals else None
    supercells = {r["supercell"] for r in rows if r.get("supercell") is not None}
    spacings = {r["spacing"] for r in rows if r.get("spacing") is not None}
    sc = next(iter(supercells)) if len(supercells) == 1 else None
    sp = next(iter(spacings)) if len(spacings) == 1 else None
    return v, sc, sp


# =============================================================
# Task1 — line + reference horizontals (paths discovered per material)
# =============================================================


def _title(
    material: str,
    metric: str,
    mode: str,
    sc: Optional[Tuple[int, int, int]],
    sp: Optional[float],
) -> str:
    metric_name = "iteration" if metric == "iteration" else "total diagonalization time"
    mode_txt = "second cutoff" if mode == "second" else "cutoff"
    sc_txt = str(sc) if sc else "mixed"
    sp_txt = f"{sp}" if sp is not None else "mixed"
    return f"{material} {metric_name} — {mode_txt} (supercell = {sc_txt}, spacing = {sp_txt})"


def plot_task1_line(
    root: Path,
    outdir: Path,
    metric: str,
    *,
    mode: str,
    primary_tokens: Sequence[str],
    reference_tokens: Sequence[Dict[str, Sequence[str]]],
    materials: Sequence[str],
    verbose: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for material in materials:
        p = discover_results_summary(root, material, primary_tokens, verbose=verbose)
        if not p:
            if verbose:
                print(f"[Task1:{mode}] {material} -> no primary file")
            continue
        xs, ys, sc, sp = series_from_summary(p, metric)
        if verbose:
            print(f"[Task1:{mode}] {material} -> {p} xs={xs} n={len(ys)}")
        if not xs:
            continue

        plt.figure(figsize=(10, 7))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("cutoff")
        if metric == "iteration":
            plt.ylim(0.5, 3.0)
            plt.yticks(np.arange(0.5, 3.01, 0.1))
            plt.ylabel("iteration (log10)")
        else:
            plt.ylim(0.5, 4.0)
            plt.yticks(np.arange(0.5, 4.02, 0.2))
            plt.ylabel("total diagonalization time (log10)")

        x_max = max(xs)
        refs = discover_references(root, material, reference_tokens, verbose=verbose)
        for ref in refs:
            label = ref.get("label", "ref")
            y, _, _ = reference_value_from_summary(ref.get("path"), metric)
            if y is None:
                if verbose:
                    print(f"  ref '{label}' no data")
                continue
            plt.axhline(y, linestyle="--", label=label)
            plt.text(x_max, y, f" {label}", va="bottom", fontsize=9)

        title = _title(material, metric, mode, sc, sp)
        plt.title(title)
        plt.tight_layout()
        out_mat = outdir / material
        out_mat.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            out_mat / f"{material}_{mode}_{metric}_line.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()


def plot_task1(
    root: Path,
    outdir: Path,
    metric: str,
    *,
    task1_mode: str,
    primary_tokens_map: Dict[str, Sequence[str]],
    reference_tokens: Sequence[Dict[str, Sequence[str]]],
    materials: Sequence[str],
    verbose: bool = False,
) -> None:
    modes = ["cutoff", "second"] if task1_mode == "both" else [task1_mode]
    for m in modes:
        sub = outdir / ("cutoff" if m == "cutoff" else "second_cutoff")
        plot_task1_line(
            root,
            sub,
            metric,
            mode=m,
            primary_tokens=primary_tokens_map[m],
            reference_tokens=reference_tokens,
            materials=materials,
            verbose=verbose,
        )


# =============================================================
# Task2 — Neumann only + reason coloring (guarantee per-cutoff plots)
# =============================================================


def _infer_spacing(text: str, path: Path) -> Optional[str]:
    ms = RE_SPACING.search(text)
    if ms:
        return ms.group(1)
    mp = RE_PATH_SPACING.search(path.as_posix())
    return mp.group(1) if mp else None


def _get_thr_from_file(path: Path) -> Optional[str]:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = RE_FILE_THR.search(txt)
    if m:
        return m.group(1)
    m2 = RE_THR_IN_PATH.search(path.as_posix())
    return m2.group(1) if m2 else None


def find_median_files(root: Path, *, material: str) -> List[Path]:
    """Find *median* files for a material anywhere under root.

    Why: avoid enforcing a fixed folder like density_fixed_diagonalization/{material}/precond_Neumann.
    Filter: filename contains "median" and not "results_summary"; path contains "/{material}/precond_Neumann/".
    """
    out: List[Path] = []
    mat_low = material.lower()
    for p in root.rglob("*median*.txt"):
        name_low = p.name.lower()
        if "results_summary" in name_low:
            continue
        posix_low = p.as_posix().lower()
        if f"/{mat_low}/precond_neumann/" not in posix_low:
            continue
        out.append(p)
    return out


def parse_orders_and_reasons(txt: str) -> Tuple[List[int], List[str]]:
    orders: List[int] = []
    reasons: List[str] = []
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        m = RE_USING_ORDER.search(line)
        if not m:
            continue
        orders.append(int(m.group("order")))
        r = m.group("reason") if "reason" in m.groupdict() else None
        if not r:
            mr = RE_REASON_INLINE.search(line)
            if not mr:
                for j in (i + 1, i - 1, i + 2, i - 2):
                    if 0 <= j < len(lines):
                        mr2 = RE_REASON_INLINE.search(lines[j])
                        if mr2:
                            r = mr2.group(1)
                            break
            else:
                r = mr.group(1)
        reasons.append(_normalize_reason(r))
    if len(reasons) < len(orders):
        reasons += ["default"] * (len(reasons) - len(orders))
    return orders, reasons


def _group_files_by_thr(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for f in files:
        thr = _get_thr_from_file(f)  # prefer in-file → fallback to path
        if thr is None:
            thr = "unknown"
        groups.setdefault(thr, []).append(f)
    return groups


def _legend_handles(reasons_present: Iterable[str]):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REASON_COLORS.get(r, "blue"),
            label=r,
            markersize=8,
        )
        for r in reasons_present
    ]


def plot_task2(root: Path, outdir: Path, *, verbose: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for material in MATERIALS:
        if verbose:
            print(f"[Task2] material={material}")
        files_all = find_median_files(root, material=material)
        if not files_all:
            if verbose:
                print("  (no files)")
            continue
        groups = _group_files_by_thr(files_all)
        if verbose:
            print("  groups:", {k: len(v) for k, v in groups.items()})
        for thr, files in sorted(groups.items(), key=lambda kv: kv[0]):
            chosen: Dict[str, Path] = {}
            for f in files:
                m_sc = RE_PATH_SUPERCELL.search(f.as_posix())
                if not m_sc:
                    continue
                sc_key = f"{m_sc.group(1)}_{m_sc.group(2)}_{m_sc.group(3)}"
                chosen.setdefault(sc_key, f)  # one per supercell within a cutoff
            for sc_key, f in sorted(chosen.items()):
                txt = f.read_text(encoding="utf-8", errors="ignore")
                orders, reasons = parse_orders_and_reasons(txt)
                if not orders:
                    if verbose:
                        print(f"  skip(no 'using order'): {f}")
                    continue
                spacing = _infer_spacing(txt, f) or "0.2"
                sx, sy, sz = sc_key.split("_")
                sc_str = f"[{sx}, {sy}, {sz}]"
                plt.figure(figsize=(12, 8))
                x = np.arange(1, len(orders) + 1)
                plt.plot(x, orders, color="0.85")
                unique: List[str] = []
                for r in reasons:
                    if r not in unique:
                        unique.append(r)
                for rk in unique:
                    idx = [i for i, r in enumerate(reasons) if r == rk]
                    if not idx:
                        continue
                    plt.scatter(
                        x[idx],
                        np.array(orders)[idx],
                        s=30,
                        color=REASON_COLORS.get(rk, "blue"),
                    )
                plt.ylim(0, 21)
                plt.yticks(np.arange(0, 21, 1))
                ticks = np.arange(5, len(orders) + 1, 5) if len(orders) >= 5 else x
                plt.xticks(ticks)
                plt.xlabel("diagonalization iteration")
                plt.ylabel("using order")
                plt.title(
                    f"{material} cutoff {thr} using order plot (supercell = {sc_str}, spacing = {spacing})"
                )
                handles = _legend_handles(unique)
                if handles:
                    plt.legend(handles=handles, title="reason", loc="upper right")
                save_dir = (
                    outdir
                    / material
                    / "cutoff"
                    / f"cutoff_{thr}"
                    / f"supercell_{sc_key}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = (
                    f"{material}_cutoff_{thr}_supercell_{sc_key}_using_order_plot.png"
                )
                path_out = save_dir / fname
                plt.tight_layout()
                plt.savefig(path_out, dpi=180, bbox_inches="tight")
                plt.close()
                if verbose:
                    print(f"  -> saved: {path_out}")


# =============================================================
# CLI
# =============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, required=True, help="root directory of results"
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="output directory for plots"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    # Task1 — iteration
    plot_task1(
        args.root,
        args.out / "task1_iteration",
        "iteration",
        task1_mode=TASK1_MODE,
        primary_tokens_map=TASK1_PRIMARY_TOKENS,
        reference_tokens=TASK1_REFERENCE_TOKENS,
        materials=MATERIALS,
        verbose=verbose,
    )
    # Task1 — total diagonalization time
    plot_task1(
        args.root,
        args.out / "task1_tdiag",
        "tdiag",
        task1_mode=TASK1_MODE,
        primary_tokens_map=TASK1_PRIMARY_TOKENS,
        reference_tokens=TASK1_REFERENCE_TOKENS,
        materials=MATERIALS,
        verbose=verbose,
    )

    # Task2 (Neumann only)
    plot_task2(args.root, args.out / "task2", verbose=verbose)


if __name__ == "__main__":
    main()
