#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Set

import yaml

"""
계산 전용 스크립트 (YAML configuration 버전)
- SCF를 모든 조합에 대해 항상 실행
- FIXED는 옵션에 따라 SCF 밀도 필요
- SCF/FIXED 로그/요약 기록

사용 예시:
nohup python repeat_test.py --config config.yaml > log 2>&1 &
"""

# ========== 유틸 ==========

_slug_re = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(s: str) -> str:
    return _slug_re.sub("-", s).strip("-")


def pair_to_str(xyz: Tuple[int, int, int]) -> str:
    return "x".join(map(str, xyz))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def tail_print(path: Path, n: int = 40) -> None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        print("[TAIL]", path)
        for ln in lines:
            sys.stdout.write(ln)
        if lines and not lines[-1].endswith("\n"):
            sys.stdout.write("\n")
    except Exception as e:
        print(f"[TAIL][ERR] {path}: {e}")


# ========== Configuration Loading ==========


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ========== 시스템 스캔 ==========


def _as_optional_int_seq(v):
    if isinstance(v, (list, tuple)):
        return tuple(None if x is None else int(x) for x in v)
    return (None if v is None else int(v),)


def _as_tuple3_seq(v, default=(1, 1, 1)):
    if v is None:
        v = default
    if (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and not isinstance(v[0], (list, tuple))
    ):
        return (tuple(int(a) for a in v),)
    return tuple(tuple(int(a) for a in t) for t in v)


def _as_number_seq(v, default: float):
    if v is None:
        v = default
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return (float(v),)


def _get_override_by_name(name: str, overrides: Dict) -> Dict:
    if name in overrides:
        return overrides[name]
    name_l = name.lower()
    for k, v in overrides.items():
        if k.lower() == name_l:
            return v
    stem = Path(name).stem.lower()
    for k, v in overrides.items():
        if Path(k).stem.lower() == stem:
            return v
    return {}


def _mk_system_entry(p: Path, defaults: Dict, overrides: Dict) -> Dict[str, Sequence]:
    suffix = p.suffix.lower()
    default_pbc = (0, 0, 0) if suffix in (".sdf", ".xyz") else (1, 1, 1)
    cfg = {**defaults, **_get_override_by_name(p.name, overrides)}
    return {
        "nbands": _as_optional_int_seq(cfg.get("nbands")),
        "supercell": _as_tuple3_seq(cfg.get("supercell", (1, 1, 1))),
        "pbc": _as_tuple3_seq(cfg.get("pbc", default_pbc), default_pbc),
        "spacing": _as_number_seq(cfg.get("spacing", 0.2), 0.2),
    }


def scan_systems(config: Dict) -> Dict[str, Dict[str, Sequence]]:
    sys_cfg = config["systems"]
    names = set(sys_cfg["selected"]) if sys_cfg["selected"] else None
    defaults = sys_cfg["defaults"]
    overrides = sys_cfg.get("overrides", {})
    out: Dict[str, Dict[str, Sequence]] = {}
    
    for root in sys_cfg["roots"]:
        rp = Path(root)
        if not rp.exists():
            continue
        for pat in sys_cfg["extensions"]:
            for p in rp.rglob(pat):
                if names and p.name not in names:
                    continue
                out[str(p)] = _mk_system_entry(p, defaults, overrides)
    return out


# ========== 설정 컨테이너 ==========


@dataclass
class FixedConfig:
    python_exe: str = sys.executable
    test_script: str = str(Path(__file__).with_name("test.py"))

    DENSITY_ROOT: Path = Path("result_scail_up") / "density"
    HISTORY_ROOT: Path = Path("result_scail_up") / "history"
    LOG_ROOT: Path = Path("result_scail_up") / "logs"

    mode: str = "scf-then-fixed"
    phase: str = "fixed"
    temperature: float = 0.01
    scf_print_energies: bool = False
    scf_energy_tol: float = 1e-6
    scf_density_tol: float = 1e-5
    scf_mixing: str = "density"
    pp_type: str = "TM"
    use_cuda: bool = False
    warmup_when_cuda: int = 1
    virtual_factor: float = 1.2

    diag_iter_scf: int = 1000
    diag_iter_fixed: int = 1000

    diag_tol_global: Optional[float] = None
    diag_tol_scf: Optional[float] = None
    diag_tol_fixed: Optional[float] = None
    diag_tol_global_is_set: bool = False
    diag_tol_scf_is_set: bool = False
    diag_tol_fixed_is_set: bool = False

    nblock: int = 2
    locking: bool = False
    fill_block: bool = False
    verbosity: int = 1
    seed: int = 0

    merge_neu_steps: int = 5

    threads_list: Sequence[int] = (1,)
    preconds: Sequence[str] = ("neumann", "shift-and-invert", "neu_ISI")
    inner_for_isi: Sequence[str] = ("neumann",)
    outerorder_list: Sequence[str] = ("dynamic",)
    innerorder_list: Sequence[str] = ("0", "1", "2")
    pcg_iter_by_inner: Dict[str, Sequence[int]] = field(
        default_factory=lambda: {"neumann": (2,)}
    )
    error_cutoff_list: Sequence[float] = tuple(round(-0.1 * k, 1) for k in range(1, 8))

    virtual_factor_list: Sequence[float] = field(default_factory=lambda: (1.2,))
    merge_neu_steps_list: Sequence[int] = field(default_factory=lambda: (5,))

    systems: Dict[str, Dict[str, Sequence]] = field(default_factory=dict)

    runs_per_combo: int = 3
    resume: bool = True
    dry_run: bool = False
    require_density_for_fixed: bool = True

    summary_fields: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_yaml_config(cls, config: Dict) -> "FixedConfig":
        """Create FixedConfig from loaded YAML configuration"""
        cfg = cls()
        
        # Experiment settings
        exp = config.get("experiment", {})
        results_root = Path(exp.get("results_root", "result_scail_up"))
        cfg.DENSITY_ROOT = results_root / "density"
        cfg.HISTORY_ROOT = results_root / "history"
        cfg.LOG_ROOT = results_root / "logs"
        
        # Calculation settings
        calc = config.get("calculation", {})
        cfg.mode = calc.get("mode", "scf-then-fixed")
        cfg.phase = calc.get("phase", "fixed")
        cfg.temperature = float(calc.get("temperature", 0.0))
        cfg.scf_print_energies = calc.get("scf_print_energies", False)
        cfg.scf_energy_tol = float(calc.get("scf_energy_tol", 1e-6))
        cfg.scf_density_tol = float(calc.get("scf_density_tol", 1e-5))
        cfg.scf_mixing = calc.get("scf_mixing", "density")
        cfg.pp_type = calc.get("pp_type", "TM")
        cfg.use_cuda = calc.get("use_cuda", False)
        cfg.warmup_when_cuda = int(calc.get("warmup_when_cuda", 1))
        cfg.virtual_factor = float(calc.get("virtual_factor", 1.2))
        
        # Diagonalization settings
        diag = calc.get("diagonalization", {})
        cfg.diag_iter_scf = int(diag.get("scf", {}).get("iter", 11))
        cfg.diag_iter_fixed = int(diag.get("fixed", {}).get("iter", 1000))
        
        scf_tol = diag.get("scf", {}).get("tol")
        fixed_tol = diag.get("fixed", {}).get("tol")
        if scf_tol is not None:
            cfg.diag_tol_scf = float(scf_tol)
            cfg.diag_tol_scf_is_set = True
        if fixed_tol is not None:
            cfg.diag_tol_fixed = float(fixed_tol)
            cfg.diag_tol_fixed_is_set = True
        
        # Davidson settings
        dav = calc.get("davidson", {})
        cfg.nblock = int(dav.get("nblock", 2))
        cfg.locking = bool(dav.get("locking", False))
        cfg.fill_block = bool(dav.get("fill_block", False))
        
        # Execution settings
        exec_cfg = calc.get("execution", {})
        cfg.runs_per_combo = int(exec_cfg.get("runs_per_combo", 3))
        cfg.resume = bool(exec_cfg.get("resume", True))
        cfg.dry_run = bool(exec_cfg.get("dry_run", False))
        cfg.require_density_for_fixed = bool(exec_cfg.get("require_density_for_fixed", True))
        cfg.verbosity = int(exec_cfg.get("verbosity", 1))
        cfg.seed = int(exec_cfg.get("seed", 0))
        
        # Sweep parameters
        sweep = config.get("sweep", {})
        if sweep.get("threads"):
            cfg.threads_list = tuple(int(x) for x in sweep["threads"])
        if sweep.get("preconds"):
            cfg.preconds = tuple(sweep["preconds"])
        if sweep.get("outerorder"):
            cfg.outerorder_list = tuple(sweep["outerorder"])
        if sweep.get("innerorder"):
            cfg.innerorder_list = tuple(sweep["innerorder"])
        if sweep.get("pcg_neumann"):
            cfg.pcg_iter_by_inner["neumann"] = tuple(int(x) for x in sweep["pcg_neumann"])
        if sweep.get("error_cutoff"):
            vals = sweep["error_cutoff"]
            cfg.error_cutoff_list = tuple(float(x) for x in vals)
        if sweep.get("virtual_factor"):
            cfg.virtual_factor_list = tuple(float(x) for x in sweep["virtual_factor"])
        if sweep.get("merge_iter"):
            cfg.merge_neu_steps_list = tuple(int(x) for x in sweep["merge_iter"])
        
        # Systems
        cfg.systems = scan_systems(config)
        
        # Apply spacing and nbands from sweep to systems
        if sweep.get("spacing"):
            vals = tuple(float(x) for x in sweep["spacing"])
            for k in list(cfg.systems.keys()):
                cfg.systems[k]["spacing"] = vals
        if sweep.get("nbands"):
            vals = tuple(
                None if (x is None or str(x).lower() == "none") else int(x)
                for x in sweep["nbands"]
            )
            for k in list(cfg.systems.keys()):
                cfg.systems[k]["nbands"] = vals
        
        # Summary fields
        cfg.summary_fields = config.get("summary_fields", {})
        
        return cfg


CFG: Optional[FixedConfig] = None
VARY_TOKENS: Set[str] = set()

# ========== 결과 로그에서 원하는 값 추출 ==========

_davidson_re = re.compile(r"^\s*davidson\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M)
_timer_row_re = re.compile(
    r"^(?P<label>[A-Za-z0-9 .()_@&\\/\\-]+?)\s*\|\s*(?P<total>[0-9]*\.?[0-9]+)\s*\|\s*(?P<count>\d+)\s*$",
    re.M,
)


def parse_davidson_seconds(log_path: Path) -> Optional[float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = _davidson_re.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_timer_metrics(log_path: Path) -> Dict[str, Dict[str, float]]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for m in _timer_row_re.finditer(text):
        label = m.group("label").strip()
        total = float(m.group("total"))
        count = int(m.group("count"))
        out[label] = {"total": total, "count": count}
    return out


def pick_metric(
    metrics: Dict[str, Dict[str, float]], candidates: List[str], attr: str
) -> Optional[float]:
    for key in candidates:
        if key in metrics:
            return metrics[key].get(attr)
    return None


# ========== 조합 & density 키 ==========


@dataclass
class Combo:
    sys_path: str
    spacing: float
    nbands: Optional[int]
    supercell: Tuple[int, int, int]
    pbc: Tuple[int, int, int]
    threads: int
    precond: str
    inner: Optional[str]
    outerorder: Optional[str]
    innerorder: Optional[str]
    pcg_iter: Optional[int]
    error_cutoff: Optional[float]
    virtual_factor: Optional[float]
    merge_neu_steps: Optional[int]


def generate_combos(cfg: FixedConfig) -> Iterator[Combo]:
    for sys_path, opts in cfg.systems.items():
        for spacing, nbands, scell, pbc in itertools.product(
            opts.get("spacing", (0.2,)),
            opts.get("nbands", (None,)),
            opts.get("supercell", ((1, 1, 1),)),
            opts.get("pbc", ((1, 1, 1),)),
        ):
            for threads in cfg.threads_list:
                vf_list = cfg.virtual_factor_list if nbands is None else (None,)
                for vf in vf_list:
                    for precond in cfg.preconds:
                        if precond == "neumann":
                            for outer in cfg.outerorder_list:
                                for ec in cfg.error_cutoff_list:
                                    yield Combo(
                                        sys_path,
                                        spacing,
                                        nbands,
                                        scell,
                                        pbc,
                                        threads,
                                        precond,
                                        None,
                                        outer,
                                        None,
                                        None,
                                        ec,
                                        vf,
                                        None,
                                    )
                        elif precond == "shift-and-invert":
                            for order in cfg.innerorder_list:
                                for pcg in cfg.pcg_iter_by_inner.get("neumann", ()):
                                    yield Combo(
                                        sys_path,
                                        spacing,
                                        nbands,
                                        scell,
                                        pbc,
                                        threads,
                                        precond,
                                        "neumann",
                                        None,
                                        order,
                                        pcg,
                                        None,
                                        vf,
                                        None,
                                    )
                        elif precond == "neu_ISI":
                            for outer in cfg.outerorder_list:
                                for order in cfg.innerorder_list:
                                    for pcg in cfg.pcg_iter_by_inner.get("neumann", ()):
                                        for ec in cfg.error_cutoff_list:
                                            for miter in cfg.merge_neu_steps_list:
                                                yield Combo(
                                                    sys_path,
                                                    spacing,
                                                    nbands,
                                                    scell,
                                                    pbc,
                                                    threads,
                                                    precond,
                                                    "neumann",
                                                    outer,
                                                    order,
                                                    pcg,
                                                    ec,
                                                    vf,
                                                    int(miter),
                                                )
                        else:
                            raise ValueError(f"Unknown precond {precond}")


def build_density_subpath(
    *,
    sys_path: str,
    spacing: float,
    supercell: Tuple[int, int, int],
    pbc: Tuple[int, int, int],
    nbands: Optional[int],
    virtual_factor: Optional[float],
) -> Path:
    name = Path(sys_path).stem
    parts: List[str] = [slugify(name), "phase=scf", f"pp={CFG.pp_type}"]
    eff_nbands = (
        (nbands * supercell[0] * supercell[1] * supercell[2])
        if nbands is not None
        else None
    )
    parts.append(f"scell={pair_to_str(supercell)}")
    parts.append(f"nbands={eff_nbands if eff_nbands is not None else 'auto'}")
    if nbands is None:
        vf = virtual_factor if virtual_factor is not None else CFG.virtual_factor
        parts.append(f"vf={vf}")
    parts.append(f"spacing={spacing}")
    return Path(parts[0]).joinpath(*parts[1:])


@dataclass
class RunResult:
    run_idx: int
    ret_history: Path
    log_path: Path
    davidson_s: Optional[float]


@dataclass
class RunPaths:
    base_subpath_scf: Path
    base_subpath_fixed: Path
    density_dir: Path
    history_scf_dir: Path
    history_dir: Path
    logs_scf_dir: Path
    logs_fixed_dir: Path

    def run_dir(self, run_idx: int) -> Path:
        return ensure_dir(self.logs_fixed_dir / f"run-{run_idx}")

    def run_history_path(self, run_idx: int) -> Path:
        return self.history_dir / f"run-{run_idx}" / "history.pt"

    def run_log_path(self, run_idx: int) -> Path:
        return self.logs_fixed_dir / f"run-{run_idx}" / "stdout.log"

    def hist_run_log_path(self, run_idx: int) -> Path:
        return self.history_dir / f"run-{run_idx}" / "stdout.log"

    def density_file(self) -> Path:
        return self.density_dir / "density.pt"

    def class_dir(self, label: str) -> Path:
        return ensure_dir(self.history_dir / label)


# ========== diag_tol 전달/미전달 해석 ==========


def _parse_optional_float_arg(s: Optional[str]) -> Tuple[bool, Optional[float]]:
    if s is None:
        return (False, None)
    if str(s).strip().lower() in ("none", ""):
        return (True, None)
    return (True, float(s))


def _resolve_diag_tol_for_phase(phase: str) -> Optional[float]:
    if phase == "scf":
        if CFG.diag_tol_scf_is_set:
            return CFG.diag_tol_scf
    else:
        if CFG.diag_tol_fixed_is_set:
            return CFG.diag_tol_fixed
    if CFG.diag_tol_global_is_set:
        return CFG.diag_tol_global
    return None


def build_combo_subpath(
    *,
    sys_path: str,
    threads: int,
    precond: str,
    inner: Optional[str],
    outerorder: Optional[str],
    innerorder: Optional[str],
    pcg_iter: Optional[int],
    error_cutoff: Optional[float],
    nbands: Optional[int],
    supercell: Tuple[int, int, int],
    pbc: Tuple[int, int, int],
    spacing: float,
    phase_token: str,
    virtual_factor: Optional[float],
    merge_neu_steps: Optional[int],
) -> Path:
    name = Path(sys_path).stem
    parts: List[str] = [slugify(name)]

    def add(text: str, cond: bool = True):
        if cond:
            parts.append(text)

    eff_nbands = (
        (nbands * supercell[0] * supercell[1] * supercell[2])
        if nbands is not None
        else None
    )
    add(f"phase={phase_token}")
    add(f"pp={CFG.pp_type}")
    add(f"cuda={int(CFG.use_cuda)}")
    add(f"thr={threads}")
    add(f"prec={precond}")
    add(f"inner={inner}", inner is not None)
    add(f"outerorder={outerorder}", outerorder is not None)
    add(f"innerorder={innerorder}", innerorder is not None)
    add(f"pcg={pcg_iter}", pcg_iter is not None)
    add(
        f"ec={error_cutoff}",
        (precond in ("neumann", "neu_ISI") and error_cutoff is not None),
    )
    add(f"scell={pair_to_str(supercell)}")
    add(f"pbc={pair_to_str(pbc)}")
    add(f"nbands={eff_nbands if eff_nbands is not None else 'auto'}")
    add(f"spacing={spacing}")
    add(f"vf={virtual_factor}", (nbands is None and virtual_factor is not None))
    add(
        f"merge_iter={merge_neu_steps}",
        (precond == "neu_ISI" and merge_neu_steps is not None),
    )
    add(
        f"diag_iter={CFG.diag_iter_scf if phase_token == 'scf' else CFG.diag_iter_fixed}"
    )
    tol_for_phase = _resolve_diag_tol_for_phase(phase_token)
    add(f"diag_tol={tol_for_phase}", tol_for_phase is not None)
    add(f"nblock={CFG.nblock}")
    add(f"lock={int(CFG.locking)}")
    add(f"fill={int(CFG.fill_block)}")
    return Path(parts[0]).joinpath(*parts[1:])


def prepare_paths(cfg: FixedConfig, combo: Combo) -> RunPaths:
    dens_sub = build_density_subpath(
        sys_path=combo.sys_path,
        spacing=combo.spacing,
        supercell=combo.supercell,
        pbc=combo.pbc,
        nbands=combo.nbands,
        virtual_factor=combo.virtual_factor,
    )
    sub_scf = build_combo_subpath(
        sys_path=combo.sys_path,
        threads=combo.threads,
        precond=combo.precond,
        inner=combo.inner,
        outerorder=combo.outerorder,
        innerorder=combo.innerorder,
        pcg_iter=combo.pcg_iter,
        error_cutoff=combo.error_cutoff,
        nbands=combo.nbands,
        supercell=combo.supercell,
        pbc=combo.pbc,
        spacing=combo.spacing,
        phase_token="scf",
        virtual_factor=combo.virtual_factor,
        merge_neu_steps=combo.merge_neu_steps,
    )
    sub_fixed = build_combo_subpath(
        sys_path=combo.sys_path,
        threads=combo.threads,
        precond=combo.precond,
        inner=combo.inner,
        outerorder=combo.outerorder,
        innerorder=combo.innerorder,
        pcg_iter=combo.pcg_iter,
        error_cutoff=combo.error_cutoff,
        nbands=combo.nbands,
        supercell=combo.supercell,
        pbc=combo.pbc,
        spacing=combo.spacing,
        phase_token="fixed",
        virtual_factor=combo.virtual_factor,
        merge_neu_steps=combo.merge_neu_steps,
    )
    return RunPaths(
        base_subpath_scf=sub_scf,
        base_subpath_fixed=sub_fixed,
        density_dir=ensure_dir(cfg.DENSITY_ROOT / dens_sub),
        history_scf_dir=ensure_dir(cfg.HISTORY_ROOT / sub_scf),
        history_dir=ensure_dir(cfg.HISTORY_ROOT / sub_fixed),
        logs_scf_dir=ensure_dir(cfg.LOG_ROOT / sub_scf),
        logs_fixed_dir=ensure_dir(cfg.LOG_ROOT / sub_fixed),
    )


# ========== 실제 실행 커맨드 및 실행 ==========


def build_cmd(
    cfg: FixedConfig,
    combo: Combo,
    paths: RunPaths,
    run_idx: int,
    *,
    phase: str,
    include_ret_history: bool = True,
) -> List[str]:
    warmup = cfg.warmup_when_cuda if cfg.use_cuda else 0
    diag_iter_for_phase = cfg.diag_iter_scf if phase == "scf" else cfg.diag_iter_fixed
    diag_tol_for_phase = _resolve_diag_tol_for_phase(phase)

    cmd: List[str] = [
        cfg.python_exe,
        "-u",
        cfg.test_script,
        "--filepath",
        combo.sys_path,
        "--spacing",
        str(combo.spacing),
        "--supercell",
        *map(str, combo.supercell),
        "--pbc",
        *map(str, combo.pbc),
        "--phase",
        phase,
        "--pp_type",
        cfg.pp_type,
        "--threads",
        str(combo.threads),
        "--warmup",
        str(warmup),
        "--diag_iter",
        str(diag_iter_for_phase),
        "--nblock",
        str(cfg.nblock),
        "--verbosity",
        str(cfg.verbosity),
        "--seed",
        str(cfg.seed + run_idx),
        "--temperature",
        str(cfg.temperature),
        "--scf_energy_tol",
        str(cfg.scf_energy_tol),
        "--scf_density_tol",
        str(cfg.scf_density_tol),
        "--scf_mixing",
        str(cfg.scf_mixing),
        "--density_filename",
        str(paths.density_file()),
    ]
    if diag_tol_for_phase is not None:
        cmd.extend(["--diag_tol", str(diag_tol_for_phase)])
    if cfg.use_cuda:
        cmd.append("--use_cuda")
    if cfg.scf_print_energies:
        cmd.append("--scf_print_energies")
    if combo.nbands is not None:
        eff = int(
            combo.nbands * combo.supercell[0] * combo.supercell[1] * combo.supercell[2]
        )
        cmd.extend(["--nbands", str(eff)])
    else:
        vf = (
            combo.virtual_factor
            if combo.virtual_factor is not None
            else cfg.virtual_factor
        )
        cmd.extend(["--virtual_factor", str(vf)])
    if cfg.locking:
        cmd.append("--locking")
    if cfg.fill_block:
        cmd.append("--fill_block")

    if combo.precond == "shift-and-invert":
        cmd.extend(["--precond", "shift-and-invert"])
        cmd.extend(["--inner", "neumann"])
        if combo.innerorder is not None:
            cmd.extend(["--innerorder", str(combo.innerorder)])
        if combo.pcg_iter is not None:
            cmd.extend(["--pcg_iter", str(combo.pcg_iter)])
    elif combo.precond == "neumann":
        cmd.extend(["--precond", "neumann"])
        if combo.outerorder is not None:
            cmd.extend(["--outerorder", str(combo.outerorder)])
        if combo.error_cutoff is not None:
            cmd.extend(["--error_cutoff", str(combo.error_cutoff)])
    elif combo.precond == "neu_ISI":
        cmd.extend(["--precond", "merge"])
        miter = (
            combo.merge_neu_steps
            if combo.merge_neu_steps is not None
            else cfg.merge_neu_steps
        )
        cmd.extend(["--merge_iter", str(miter)])
        if combo.outerorder is not None:
            cmd.extend(["--outerorder", str(combo.outerorder)])
        if combo.error_cutoff is not None:
            cmd.extend(["--error_cutoff", str(combo.error_cutoff)])
        cmd.extend(["--inner", "neumann"])
        if combo.innerorder is not None:
            cmd.extend(["--innerorder", str(combo.innerorder)])
        if combo.pcg_iter is not None:
            cmd.extend(["--pcg_iter", str(combo.pcg_iter)])
    else:
        raise ValueError(f"Unknown precond {combo.precond}")

    if phase == "fixed" and include_ret_history:
        cmd.extend(["--retHistory", str(paths.run_history_path(run_idx))])
    return [x for x in cmd if x]


def run_once(cmd: List[str], log_path: Path, threads: int) -> int:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
        }
    )
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def classify_runs_by_time(
    times: List[Tuple[int, Optional[float]]]
) -> Dict[str, int]:
    def key(t: Tuple[int, Optional[float]]):
        _, sec = t
        return float("inf") if sec is None else sec

    ordered = sorted(times, key=key)
    idxs = [t[0] for t in ordered]
    if not idxs:
        return {"fast": 1, "median": 1, "slow": 1}
    if len(idxs) == 1:
        return {"fast": idxs[0], "median": idxs[0], "slow": idxs[0]}
    if len(idxs) == 2:
        return {"fast": idxs[0], "median": idxs[1], "slow": idxs[1]}
    return {"fast": idxs[0], "median": idxs[1], "slow": idxs[2]}


def write_setting_summary(
    results_root: Path, combos: Sequence[Combo], systems: Dict[str, Dict[str, Sequence]]
):
    ensure_dir(results_root)
    payload = {
        "targets": list(systems.keys()),
        "vary_args": sorted(list(VARY_TOKENS)) if VARY_TOKENS else [],
        "fixed_args": {
            "phase_mode": CFG.mode,
            "pp_type": CFG.pp_type,
            "use_cuda": CFG.use_cuda,
            "virtual_factor_default": CFG.virtual_factor,
            "diag_iter_scf": CFG.diag_iter_scf,
            "diag_iter_fixed": CFG.diag_iter_fixed,
            "diag_tol_effective": {
                "global": CFG.diag_tol_global if CFG.diag_tol_global_is_set else None,
                "scf": _resolve_diag_tol_for_phase("scf"),
                "fixed": _resolve_diag_tol_for_phase("fixed"),
            },
            "nblock": CFG.nblock,
            "locking": CFG.locking,
            "fill_block": CFG.fill_block,
            "verbosity": CFG.verbosity,
        },
        "runs_per_combo": CFG.runs_per_combo,
    }
    (results_root / "setting_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _fmt_val(v):
    if isinstance(v, str):
        return f'"{v}"'
    return json.dumps(v, ensure_ascii=False)


def write_pretty_summary(dirpath: Path, row: Dict[str, object], filename: str) -> None:
    ensure_dir(dirpath)
    ordered_keys = [
        "material",
        "spacing",
        "preconditioner",
        "nbands_input",
        "nbands_eff",
        "virtual_factor",
        "supercell",
        "threads",
        "solver_type",
        "order",
        "innerorder",
        "innerprecond",
        "error_cutoff",
        "pcg_number",
        "merge_iter",
        "scf_iterations",
    ] + [
        k
        for k in row.keys()
        if k
        not in {
            "material",
            "spacing",
            "preconditioner",
            "nbands_input",
            "nbands_eff",
            "virtual_factor",
            "supercell",
            "threads",
            "solver_type",
            "order",
            "innerorder",
            "innerprecond",
            "error_cutoff",
            "pcg_number",
            "merge_iter",
            "scf_iterations",
        }
    ]
    line = ", ".join(
        [f"{k} = {_fmt_val(row.get(k))}" for k in ordered_keys if k in row]
    )
    out_path = dirpath / filename
    if out_path.exists():
        out_path.write_text(
            out_path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8"
        )
    else:
        out_path.write_text(line + "\n", encoding="utf-8")


def write_scf_only_summary(combo: Combo, scf_log: Path) -> None:
    metrics = parse_timer_metrics(scf_log)
    
    # Get candidates from config
    fields = CFG.summary_fields
    scf_iter_cnt = pick_metric(
        metrics,
        fields.get("diag_iter_count", {}).get("candidates", ["SCF iter."]),
        "count"
    )
    dav_total = pick_metric(
        metrics,
        fields.get("davidson_total", {}).get("candidates", ["davidson"]),
        "total"
    )

    material = Path(combo.sys_path).stem
    base_row: Dict[str, object] = {
        "material": material,
        "spacing": combo.spacing,
        "preconditioner": combo.precond,
        "nbands_input": combo.nbands if combo.nbands is not None else None,
        "nbands_eff": (
            (
                combo.nbands
                * combo.supercell[0]
                * combo.supercell[1]
                * combo.supercell[2]
            )
            if combo.nbands is not None
            else "auto"
        ),
        "virtual_factor": (combo.virtual_factor if combo.nbands is None else None),
        "supercell": list(combo.supercell),
        "threads": combo.threads,
        "solver_type": "SCF-only",
    }

    if combo.precond == "neumann":
        base_row.update(
            {
                "order": combo.outerorder,
                "innerorder": None,
                "innerprecond": None,
                "error_cutoff": combo.error_cutoff,
                "pcg_number": None,
                "merge_iter": None,
            }
        )
    elif combo.precond == "shift-and-invert":
        base_row.update(
            {
                "order": None,
                "innerorder": combo.innerorder,
                "innerprecond": "neumann",
                "error_cutoff": None,
                "pcg_number": combo.pcg_iter,
                "merge_iter": None,
            }
        )
    elif combo.precond == "neu_ISI":
        base_row.update(
            {
                "order": combo.outerorder,
                "innerorder": combo.innerorder,
                "innerprecond": "neumann",
                "error_cutoff": combo.error_cutoff,
                "pcg_number": combo.pcg_iter,
                "merge_iter": combo.merge_neu_steps,
            }
        )
    else:
        base_row.update(
            {
                "order": None,
                "innerorder": None,
                "innerprecond": None,
                "error_cutoff": None,
                "pcg_number": None,
                "merge_iter": None,
            }
        )

    row = {
        **base_row,
        "scf_iterations": scf_iter_cnt,
        "davidson_total": dav_total,
        "scf_iter_count": scf_iter_cnt,
    }
    write_pretty_summary(
        CFG.DENSITY_ROOT.parent, row, filename="calculation_summary_scf.txt"
    )


def find_label_log(runpaths: RunPaths, label: str, idx: Optional[int]) -> Optional[Path]:
    cand: List[Path] = [
        runpaths.history_dir / label / "stdout.log",
        runpaths.logs_fixed_dir / label / "stdout.log",
    ]
    if idx is not None:
        cand += [
            runpaths.logs_fixed_dir / f"run-{idx}" / "stdout.log",
            runpaths.history_dir / f"run-{idx}" / "stdout.log",
        ]
    for p in cand:
        if p.exists():
            return p
    return None


def write_fixed_summary(runpaths: RunPaths, combo: Combo, labels: Dict[str, int]) -> None:
    median_idx = labels.get("median")
    log_path = find_label_log(runpaths, "median", median_idx)
    metrics: Dict[str, Dict[str, float]] = {}
    if log_path is not None:
        metrics = parse_timer_metrics(log_path)
    
    material = Path(combo.sys_path).stem
    base_row: Dict[str, object] = {
        "material": material,
        "spacing": combo.spacing,
        "preconditioner": combo.precond,
        "nbands_input": combo.nbands if combo.nbands is not None else None,
        "nbands_eff": (
            (
                combo.nbands
                * combo.supercell[0]
                * combo.supercell[1]
                * combo.supercell[2]
            )
            if combo.nbands is not None
            else "auto"
        ),
        "virtual_factor": (combo.virtual_factor if combo.nbands is None else None),
        "supercell": list(combo.supercell),
        "threads": combo.threads,
        "solver_type": (
            "ISI"
            if combo.precond == "shift-and-invert"
            else ("merge" if combo.precond == "neu_ISI" else combo.precond)
        ),
        "order": (
            combo.outerorder if combo.precond in ("neumann", "neu_ISI") else None
        ),
        "innerorder": (
            combo.innerorder
            if combo.precond in ("shift-and-invert", "neu_ISI")
            else None
        ),
        "innerprecond": (
            "neumann"
            if combo.precond == "neu_ISI"
            else (combo.inner if combo.precond == "shift-and-invert" else None)
        ),
        "error_cutoff": (
            combo.error_cutoff if combo.precond in ("neumann", "neu_ISI") else None
        ),
        "pcg_number": (
            combo.pcg_iter if combo.precond in ("shift-and-invert", "neu_ISI") else None
        ),
        "merge_iter": (combo.merge_neu_steps if combo.precond == "neu_ISI" else None),
    }
    
    row = dict(base_row)
    fields = CFG.summary_fields
    row["davidson_total"] = pick_metric(
        metrics,
        fields.get("davidson_total", {}).get("candidates", ["davidson"]),
        "total"
    )
    row["diag_iter_count"] = pick_metric(
        metrics,
        fields.get("diag_iter_count", {}).get("candidates", ["Diag. Iter."]),
        "count"
    )
    row["preconditioning_total"] = pick_metric(
        metrics,
        fields.get("preconditioning_total", {}).get("candidates", ["Preconditioning"]),
        "total"
    )

    if row["davidson_total"] is None:
        try:
            ranking = json.loads(
                (runpaths.history_dir / "run_ranking.json").read_text(encoding="utf-8")
            )
            m_idx = labels.get("median")
            if isinstance(ranking, dict) and "order" in ranking and m_idx is not None:
                for idx, sec in ranking["order"]:
                    if idx == m_idx and isinstance(sec, (int, float)):
                        row["davidson_total"] = float(sec)
                        break
        except Exception:
            pass
    
    write_pretty_summary(
        CFG.DENSITY_ROOT.parent, row, filename="calculation_summary_fixed.txt"
    )


# ========== 메인 ==========


def main():
    global CFG, VARY_TOKENS
    
    parser = argparse.ArgumentParser(description="DFT calculation sweep with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scf", "fixed", "scf-then-fixed"],
        default=None,
        help="Override mode from config"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode (don't execute calculations)"
    )
    parser.add_argument(
        "--runs_per_combo",
        type=int,
        default=None,
        help="Override runs per combo from config"
    )
    parser.add_argument(
        "--diag_tol",
        type=str,
        default=None,
        help="(legacy) Global diag_tol; 'none' means don't pass option"
    )
    parser.add_argument(
        "--diag_tol_scf",
        type=str,
        default=None,
        help="SCF-specific diag_tol; 'none' means don't pass option"
    )
    parser.add_argument(
        "--diag_tol_fixed",
        type=str,
        default=None,
        help="Fixed-specific diag_tol; 'none' means don't pass option"
    )
    parser.add_argument(
        "--diag_iter",
        type=int,
        default=None,
        help="(legacy) Global diag_iter"
    )
    parser.add_argument(
        "--diag_iter_scf",
        type=int,
        default=None,
        help="SCF-specific diag_iter"
    )
    parser.add_argument(
        "--diag_iter_fixed",
        type=int,
        default=None,
        help="Fixed-specific diag_iter"
    )

    args = parser.parse_args()

    # Load configuration from YAML
    try:
        config = load_config(args.config)
        print(f"[INFO] Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {args.config}")
        print("[INFO] Please create a config.yaml file or specify path with --config")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    # Create configuration object
    CFG = FixedConfig.from_yaml_config(config)

    # Apply command-line overrides
    if args.mode is not None:
        CFG.mode = args.mode
    if args.dry_run:
        CFG.dry_run = True
    if args.runs_per_combo is not None:
        CFG.runs_per_combo = int(args.runs_per_combo)

    # Handle diag_tol overrides
    g_set, g_val = _parse_optional_float_arg(args.diag_tol)
    s_set, s_val = _parse_optional_float_arg(args.diag_tol_scf)
    f_set, f_val = _parse_optional_float_arg(args.diag_tol_fixed)
    
    if g_set:
        CFG.diag_tol_global_is_set = True
        CFG.diag_tol_global = g_val
    if s_set:
        CFG.diag_tol_scf_is_set = True
        CFG.diag_tol_scf = s_val
    if f_set:
        CFG.diag_tol_fixed_is_set = True
        CFG.diag_tol_fixed = f_val

    # Handle diag_iter overrides
    if args.diag_iter is not None:
        CFG.diag_iter_scf = int(args.diag_iter)
        CFG.diag_iter_fixed = int(args.diag_iter)
    if args.diag_iter_scf is not None:
        CFG.diag_iter_scf = int(args.diag_iter_scf)
    if args.diag_iter_fixed is not None:
        CFG.diag_iter_fixed = int(args.diag_iter_fixed)

    # Check if systems were loaded
    if not CFG.systems:
        print("[ERROR] No systems found. Check your configuration file.")
        sys.exit(1)

    print(f"[INFO] Found {len(CFG.systems)} system(s) to calculate")
    print(f"[INFO] Mode: {CFG.mode}")
    print(f"[INFO] Runs per combo: {CFG.runs_per_combo}")

    # Generate combinations
    combos = list(generate_combos(CFG))
    if not combos:
        print("[ERROR] No combos to run – check sweep parameters.")
        sys.exit(1)

    print(f"[INFO] Generated {len(combos)} calculation combination(s)")

    # Identify varying parameters
    keys = {
        "phase",
        "pp",
        "cuda",
        "thr",
        "prec",
        "inner",
        "outerorder",
        "innerorder",
        "pcg",
        "ec",
        "scell",
        "pbc",
        "nbands",
        "spacing",
        "vf",
        "merge_iter",
        "diag_iter",
        "diag_tol_scf",
        "diag_tol_fixed",
        "nblock",
        "lock",
        "fill",
    }
    values: Dict[str, set] = {k: set() for k in keys}
    
    for c in combos:
        def put(k: str, v):
            if v is None:
                return
            values[k].add(v)

        put("phase", CFG.mode)
        put("pp", CFG.pp_type)
        put("cuda", int(CFG.use_cuda))
        put("thr", c.threads)
        put("prec", c.precond)
        put("inner", c.inner)
        put("outerorder", c.outerorder)
        put("innerorder", c.innerorder)
        put("pcg", c.pcg_iter)
        put("ec", c.error_cutoff if c.precond in ("neumann", "neu_ISI") else None)
        put("scell", c.supercell)
        put("pbc", c.pbc)
        put("nbands", c.nbands if c.nbands is not None else "auto")
        put("spacing", c.spacing)
        put("vf", c.virtual_factor if c.nbands is None else None)
        put("merge_iter", c.merge_neu_steps if c.precond == "neu_ISI" else None)
        put("diag_iter", CFG.diag_iter_scf)
        put("diag_tol_scf", _resolve_diag_tol_for_phase("scf"))
        put("diag_tol_fixed", _resolve_diag_tol_for_phase("fixed"))
        put("nblock", CFG.nblock)
        put("lock", int(CFG.locking))
        put("fill", int(CFG.fill_block))
    
    VARY_TOKENS = {k for k, s in values.items() if len(s) > 1}
    
    results_root = CFG.DENSITY_ROOT.parent
    write_setting_summary(results_root, combos, CFG.systems)

    # === Execute calculations ===
    for c_idx, combo in enumerate(combos, 1):
        paths = prepare_paths(CFG, combo)
        dens_file = paths.density_file()

        # --- SCF: Always execute ---
        scf_rc = 0
        if CFG.mode in ("scf", "scf-then-fixed"):
            scf_cmd = build_cmd(
                CFG, combo, paths, run_idx=0, phase="scf", include_ret_history=False
            )
            scf_log = paths.logs_scf_dir / "scf.log"
            print(f"\n[SCF] ({c_idx}/{len(combos)}) combo={paths.base_subpath_scf}")
            print("CMD:", " ".join(scf_cmd))
            
            if not CFG.dry_run:
                scf_rc = run_once(scf_cmd, scf_log, combo.threads)
                if scf_rc != 0:
                    print(f"[ERR][SCF] Return code {scf_rc} – see: {scf_log}")
                    tail_print(scf_log, 60)
            else:
                ensure_dir(scf_log.parent)
                scf_log.write_text("[dry_run] scf\n", encoding="utf-8")
            
            write_scf_only_summary(combo, scf_log)

        if CFG.mode == "scf":
            continue

        # FIXED requires SCF density if configured
        if CFG.require_density_for_fixed and (scf_rc != 0 or not dens_file.exists()):
            print(
                f"[SKIP][FIXED] Missing/failed SCF density for combo={paths.base_subpath_fixed} – skip fixed runs."
            )
            continue

        # --- FIXED (multi-run) ---
        results: List[RunResult] = []
        for run_idx in range(1, CFG.runs_per_combo + 1):
            cmd = build_cmd(
                CFG, combo, paths, run_idx, phase="fixed", include_ret_history=True
            )
            print(
                f"\n[RUN] ({c_idx}/{len(combos)}) combo={paths.base_subpath_fixed} run={run_idx}"
            )
            print("CMD:", " ".join(cmd))
            
            if CFG.dry_run:
                results.append(
                    RunResult(
                        run_idx,
                        paths.run_history_path(run_idx),
                        paths.run_log_path(run_idx),
                        None,
                    )
                )
                continue
            
            ensure_dir(paths.run_history_path(run_idx).parent)
            rc = run_once(cmd, paths.run_log_path(run_idx), combo.threads)
            
            if rc != 0:
                print(f"[ERR] Return code {rc} – see: {paths.run_log_path(run_idx)}")
                tail_print(paths.run_log_path(run_idx), 60)
            
            dtime = parse_davidson_seconds(paths.run_log_path(run_idx))
            print(f"  → davidson(s) = {dtime}")
            
            results.append(
                RunResult(
                    run_idx,
                    paths.run_history_path(run_idx),
                    paths.run_log_path(run_idx),
                    dtime,
                )
            )

        # Classify runs by time
        labels = classify_runs_by_time([(r.run_idx, r.davidson_s) for r in results])
        order = sorted(
            [(r.run_idx, r.davidson_s) for r in results],
            key=lambda t: float("inf") if t[1] is None else t[1],
        )
        ranking = {"order": order, "labels": labels}
        
        (paths.history_dir / "run_ranking.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )
        
        with open(paths.history_dir / "run_ranking.txt", "w", encoding="utf-8") as f:
            for rank, (idx, sec) in enumerate(order, 1):
                tag = [k for k, v in labels.items() if v == idx]
                f.write(
                    f"{rank}) run-{idx}: davidson={sec}  label={tag[0] if tag else '-'}\n"
                )

        # Move files to labeled directories
        for label, idx in labels.items():
            dst_dir = paths.class_dir(label)
            src_h = paths.run_history_path(idx)
            src_l = paths.run_log_path(idx)
            
            if src_h.exists():
                ensure_dir(dst_dir)
                shutil.move(str(src_h), str(dst_dir / "history.pt"))
            else:
                print(f"[WARN] history not found for label={label}: {src_h}")
            
            if src_l.exists():
                ensure_dir(dst_dir)
                shutil.move(str(src_l), str(dst_dir / "stdout.log"))
            else:
                print(f"[WARN] log not found for label={label}: {src_l}")
        
        # Clean up temporary run directories
        for d in paths.logs_fixed_dir.glob("run-*"):
            shutil.rmtree(d, ignore_errors=True)
        for d in paths.history_dir.glob("run-*"):
            shutil.rmtree(d, ignore_errors=True)

        # Write summary
        (paths.history_dir / "summary.json").write_text(
            json.dumps(
                {
                    "path": str(paths.history_dir),
                    "vary_tokens": sorted(list(VARY_TOKENS)) if VARY_TOKENS else [],
                    "runs": [
                        {"run": r.run_idx, "davidson_seconds": r.davidson_s}
                        for r in results
                    ],
                    "labels": labels,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        write_fixed_summary(paths, combo, labels)

    print("\n[INFO] All calculations completed!")
    print(f"[INFO] Results saved to: {results_root}")


if __name__ == "__main__":
    main()
