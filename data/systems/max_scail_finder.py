#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

"""
메모리 한계(supercell / 분자수) 탐색 스크립트.

- 위치: data/systems/mem_limit_scan.py (추천)
- test.py 경로: ../../test.py (이 스크립트 기준)

동작 개요
---------
1) CIF 파일들에 대해:
   - 미리 지정한 supercell 후보 리스트를 순서대로 돌면서
   - ../../test.py 를 spacing=0.2, phase=fixed, precond=neumann, outerorder=10 등으로 실행
   - 성공 / 실패(OOM 포함)를 기록하고,
   - 마지막으로 성공한 supercell / 처음 실패한 supercell 을 정리.

2) C60 / water_cluster (xyz 파일)에 대해:
   - base xyz 파일을 읽어서, (nx, ny, nz) 3D grid로 복제한 새로운 xyz 파일을 생성
   - 전체 분자수 = nx * ny * nz
   - 각 grid에 대해 test.py 실행 → 성공 / 실패를 확인하고
   - 분자수 기준으로 최대 성공 / 최초 실패를 정리.

3) 결과 요약:
   - mem_limit_summary.txt 에 사람이 읽기 쉬운 한국어 문장으로 정리
   - mem_limit_summary.json 에 기계가 쓰기 좋은 포맷으로도 저장
"""

# ---------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent  # 전체 파일의 절대 경로를 불러온다 .resolve
PROJECT_ROOT = HERE.parents[2]  # ../../  
TEST_SCRIPT = PROJECT_ROOT / "test.py"   # test.py 의 경로 완성

# C60 / H2O 클러스터 생성 스크립트 경로
C60_GEN_SCRIPT = HERE / "C60_cluster.py"
WATER_GEN_SCRIPT = HERE / "make_water_cluster.py"

# C60 생성 기본값
C60_GEN_DEFAULTS = dict(
    spacing=12.0,      # C60 분자 사이 거리
    wall_gap=5.0,      # 벽과 분자 간격
    seed=42,           # 랜덤 회전 seed (build_c60_cluster 에서 사용)
)

# H2O 생성 기본값
WATER_GEN_DEFAULTS = dict(
    spacing=3.0,       # 물 분자 격자 간격
    rotate=True,
    translate=0.4,
    seed=42,
)

PYTHON_EXE = sys.executable # 파이썬 경로 확인
# ---------------------------------------------------------------------
# test.py 를 돌릴 테스트 설정 값  --> 계산 설정 변화가 있다면 수정할 것 
# ---------------------------------------------------------------------

TEST_DEFAULTS = dict(
    spacing=0.2,
    phase="fixed",
    pp_type="TM",
    threads=1,
    precond="neumann",
    outerorder=10,
    nblock=2,
    diag_iter=1000,          
    temperature=0.00,
    scf_energy_tol=1e-6,
    virtual_factor=1.2,
    use_cuda=False,
    verbosity=1,
    warmup=1,
)

# ---------------------------------------------------------------------
# 스캔 대상 설정
# ---------------------------------------------------------------------

@dataclass
class CrystalTarget:    # cif 파일의 실험할 데이터를 만든다.
    filename : str
    pbc : Tuple[int, int, int]
    supercells : List[Tuple[int, int, int]]

@dataclass
class ClusterTarget:    # C60, H2O cluser 의 실험할 데이터를 만든다.    
    filename : str
    label : str          # "C60" / "H2O" 등
    pbc : Tuple[int, int, int]
    grids : List[Tuple[int, int, int]]  # (nx, ny, nz)   # x, y, z 방향으로의 분자 수 설정


# --- CIF 대상들: 필요에 따라 supercell 확인할 물질들은 아래와 같은 형식으로 작성 ---
CRYSTAL_TARGETS : List[CrystalTarget] = [
    CrystalTarget(
        filename = "CsPbI3.cif",
        pbc = (1, 1, 1),
        supercells = [
            (3, 3, 3),
            (4, 3, 3),
            (4, 4, 3),
            (4, 4, 4),
        ],  # L40S 기준 (3, 3, 2) 가 max
    ),
    CrystalTarget(
        filename = "MAPbI3.cif",
        pbc = (1, 1, 1),
        supercells = [
            (2, 2, 2),
            (3, 2, 2),
            (3, 3, 2),
            (3, 3, 3),
        ], # L40S 기준 [2, 2, 1] 가 max  
    ),
    CrystalTarget(
        filename = "Si_diamond.cif",
        pbc = (1, 1, 1),
        supercells = [
            (4, 3, 3),
            (4, 4, 3),
            (4, 4, 4),
            (5, 4, 4),
        ], # L40S 기준 [3, 3, 3] 가 max  
    ),
    CrystalTarget(
        filename = "MgO.cif",
        pbc = (1, 1, 1),
        supercells = [
            (4, 4, 3),
            (4, 4, 4),
            (5, 4, 4),
            (5, 5, 4),
        ], # L40S 기준 [4, 3, 3] 가 max  
    ),
]

# --- C60 / H2O 클러스터 대상들 ---
CLUSTER_TARGETS: List[ClusterTarget] = [
    # C60_tetramer.xyz: 주석에 따르면 C60 2개 → 4개로 스케일 업한 파일이라
    # 여기서는 "base_molecules = 4" 로 둠.
    ClusterTarget(
        filename = "C60.xyz",
        label = "C60",
        pbc = (0, 0, 0),
        grids = [ 
            (2, 1, 1), # 2개 (dimer)
            (3, 1, 1), # 3개 (trimer) 
            (2, 2, 1), # 4개 (tetramer)
            (2, 3, 1), # 6개 (hexamer)
            (2, 2, 2), # 8개 (octamer)
        ], # L40S 기준 2개가 max
    ),
    # water_cluster.xyz: 주석에 따르면 water 64개 기준
    ClusterTarget(
        filename="H2O",
        label="H2O",
        pbc=(0, 0, 0),
        grids=[
            (1, 1, 1),
            (2, 1, 1),
            (2, 2, 1),
            (2, 2, 2),
            (3, 2, 2),
        ],
    ),
]

# ---------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_test_once(
    filepath: Path,
    supercell: Tuple[int, int, int],
    pbc: Tuple[int, int, int],
    log_dir: Path,
) -> Tuple[bool, bool, Path]:
    """
    test.py를 한 번 실행해서:
    - success: 종료코드 0 이고, log에서 OOM 패턴 없음
    - oom:     log에서 OOM 관련 문자열 발견
    를 판단한다.

    반환값: (success, is_oom, log_path)
    """
    ensure_dir(log_dir)
    log_path = log_dir / f"{filepath.stem}_scell_{supercell[0]}x{supercell[1]}x{supercell[2]}.log"

    # make_water_cluster.py 위치 (mem_limit_scan.py 가 있는 data/systems 폴더 기준)
    WATER_GEN_SCRIPT = HERE / "make_water_cluster.py"  # 위치 다르면 여기만 바꿔줘

    WATER_GEN_DEFAULTS = dict(
        spacing=3.0,    # 물 클러스터 기본 간격
        rotate=True,    # 항상 회전 켜기
        translate=0.4,  # 약간 흔들어주기
        seed=42,
    )
    cmd = [
        PYTHON_EXE,
        str(TEST_SCRIPT),
        "--filepath",
        str(filepath),
        "--spacing",
        str(TEST_DEFAULTS["spacing"]),
        "--supercell",
        str(supercell[0]),
        str(supercell[1]),
        str(supercell[2]),
        "--pbc",
        str(pbc[0]),
        str(pbc[1]),
        str(pbc[2]),
        "--phase",
        TEST_DEFAULTS["phase"],
        "--pp_type",
        TEST_DEFAULTS["pp_type"],
        "--threads",
        str(TEST_DEFAULTS["threads"]),
        "--precond",
        TEST_DEFAULTS["precond"],
        "--outerorder",
        str(TEST_DEFAULTS["outerorder"]),
        "--diag_iter",
        str(TEST_DEFAULTS["diag_iter"]),
        "--nblock",
        str(TEST_DEFAULTS["nblock"]),
        "--temperature",
        str(TEST_DEFAULTS["temperature"]),
        "--scf_energy_tol",
        str(TEST_DEFAULTS["scf_energy_tol"]),
        "--virtual_factor",
        str(TEST_DEFAULTS["virtual_factor"]),
        "--verbosity",
        str(TEST_DEFAULTS["verbosity"]),
    ]

    if TEST_DEFAULTS["use_cuda"]:
        cmd.append("--use_cuda")
        cmd.extend(["--warmup", str(TEST_DEFAULTS["warmup"])])
    else:
        cmd.extend(["--warmup", "0"])

    env = os.environ.copy()
    # 스레드 1 고정
    env.update(
        {
            "OMP_NUM_THREADS": str(TEST_DEFAULTS["threads"]),
            "MKL_NUM_THREADS": str(TEST_DEFAULTS["threads"]),
            "NUMEXPR_NUM_THREADS": str(TEST_DEFAULTS["threads"]),
        }
    )

    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        rc = proc.wait()

    # 로그를 읽어서 OOM 패턴을 찾아본다.
    is_oom = False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
        oom_patterns = [
            "memoryerror",
            "out of memory",
            "cuda error: out of memory",
            "cannot allocate memory",
            "std::bad_alloc".lower(),
            "bad_alloc",
            "oom-kill",
            "killed process",
        ]
        if any(pat in text for pat in oom_patterns):
            is_oom = True
    except Exception:
        pass

    success = (rc == 0) and not is_oom
    return success, is_oom, log_path


# ---------------------------------------------------------------------
# XYZ 유틸 (C60 / H2O 클러스터용)
# ---------------------------------------------------------------------

def read_xyz(path: Path) -> Tuple[str, List[Tuple[str, float, float, float]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ 파일 형식 이상: {path}")
    try:
        natoms = int(lines[0].strip())
    except Exception as e:
        raise ValueError(f"첫 줄에서 atom 수를 읽지 못함: {path}") from e
    comment = lines[1].strip()
    atom_lines = lines[2 : 2 + natoms]
    atoms = []
    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((sym, x, y, z))
    if len(atoms) != natoms:
        # 그냥 경고만 하고 실제 읽은 개수로 진행
        print(f"[WARN] {path}: 예상 atom 수 {natoms}, 실제 읽은 수 {len(atoms)}")
    return comment, atoms


def write_xyz(path: Path, atoms: List[Tuple[str, float, float, float]], comment: str = "") -> None:
    lines = []
    lines.append(str(len(atoms)))
    lines.append(comment)
    for sym, x, y, z in atoms:
        lines.append(f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replicate_xyz_grid(
    atoms: List[Tuple[str, float, float, float]],
    nx: int,
    ny: int,
    nz: int,
    margin: float = 5.0,
) -> List[Tuple[str, float, float, float]]:
    xs = [a[1] for a in atoms]
    ys = [a[2] for a in atoms]
    zs = [a[3] for a in atoms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    dx = (max_x - min_x) + margin
    dy = (max_y - min_y) + margin
    dz = (max_z - min_z) + margin

    new_atoms: List[Tuple[str, float, float, float]] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift_x = ix * dx
                shift_y = iy * dy
                shift_z = iz * dz
                for sym, x, y, z in atoms:
                    new_atoms.append((sym, x + shift_x, y + shift_y, z + shift_z))
    return new_atoms


# ---------------------------------------------------------------------
# CIF 대상 스캔
# ---------------------------------------------------------------------

def scan_crystals(summary: Dict) -> None:
    log_root = ensure_dir(HERE / "mem_logs_crystal")

    for target in CRYSTAL_TARGETS:
        cif_path = HERE / target.filename
        if not cif_path.exists():
            print(f"[SKIP][CIF] {target.filename} (파일 없음)")
            continue

        print(f"\n[CIF] {target.filename} 에 대해 supercell 메모리 한계 탐색 시작")
        max_success: Optional[Tuple[int, int, int]] = None
        first_fail: Optional[Tuple[int, int, int]] = None
        first_fail_oom: Optional[bool] = None

        for sc in target.supercells:
            print(f"  - supercell = {sc[0]}x{sc[1]}x{sc[2]} 실행 중...")
            success, is_oom, log_path = run_test_once(
                filepath=cif_path,
                supercell=sc,
                pbc=target.pbc,
                log_dir=log_root,
            )
            if success:
                print(f"    → 성공 (OOM 아님)  log={log_path.name}")
                max_success = sc
                continue

            # 실패
            print(
                f"    → 실패 (OOM={is_oom})  log={log_path.name}  → 이후 supercell 후보는 스킵"
            )
            first_fail = sc
            first_fail_oom = is_oom
            break

        material = cif_path.stem
        summary.setdefault("crystals", {})[material] = {
            "filename": target.filename,
            "max_success_supercell": list(max_success) if max_success else None,
            "first_fail_supercell": list(first_fail) if first_fail else None,
            "first_fail_is_oom": bool(first_fail_oom) if first_fail is not None else None,
        }


# ---------------------------------------------------------------------
# C60 / H2O 클러스터 스캔
# ---------------------------------------------------------------------
def scan_clusters(summary: Dict) -> None:
    log_root = ensure_dir(HERE / "mem_logs_cluster")
    gen_root = ensure_dir(HERE / "mem_generated_xyz")

    for target in CLUSTER_TARGETS:
        print(f"\n[CLUSTER] {target.label} 클러스터 메모리 한계 탐색 시작")

        max_success_mols: Optional[int] = None
        max_success_grid: Optional[Tuple[int, int, int]] = None
        first_fail_mols: Optional[int] = None
        first_fail_grid: Optional[Tuple[int, int, int]] = None
        first_fail_oom: Optional[bool] = None

        for nx, ny, nz in target.grids:
            total_mol = nx * ny * nz   # ★ 여기서부터는 항상 nx*ny*nz
            print(
                f"  - grid = {nx}x{ny}x{nz} (분자수 = {total_mol}) 에 대해 클러스터 생성 및 test.py 실행 중..."
            )

            # -----------------------
            # 1) 클러스터 xyz 생성
            # -----------------------
            out_xyz = gen_root / f"{target.label}_nx{nx}_ny{ny}_nz{nz}.xyz"

            if target.label.upper().startswith("C60"):
                # C60: build_c60_cluster_compact.py 사용
                cmd = [
                    PYTHON_EXE,
                    str(C60_GEN_SCRIPT),
                    "--nx", str(nx),
                    "--ny", str(ny),
                    "--nz", str(nz),
                    "--spacing", str(C60_GEN_DEFAULTS["spacing"]),
                    "--wall_gap", str(C60_GEN_DEFAULTS["wall_gap"]),
                    "--output", str(out_xyz),
                ]
                # 랜덤 회전 seed 는 스크립트 내부에서 42로 고정되어 있으니 여기선 추가 인자 필요 없음
                proc = subprocess.run(cmd, cwd=str(HERE))
            else:
                # H2O: make_water_cluster.py 사용
                cmd = [
                    PYTHON_EXE,
                    str(WATER_GEN_SCRIPT),
                    "--nx", str(nx),
                    "--ny", str(ny),
                    "--nz", str(nz),
                    "--spacing", str(WATER_GEN_DEFAULTS["spacing"]),
                    "--translate", str(WATER_GEN_DEFAULTS["translate"]),
                    "--output", str(out_xyz),
                ]
                if WATER_GEN_DEFAULTS["rotate"]:
                    cmd.append("--rotate")
                if WATER_GEN_DEFAULTS["seed"] is not None:
                    cmd.extend(["--seed", str(WATER_GEN_DEFAULTS["seed"])])

                proc = subprocess.run(cmd, cwd=str(HERE))

            if proc.returncode != 0:
                print(
                    f"    → 클러스터 생성 실패 (returncode={proc.returncode})  → 이후 grid 후보는 스킵"
                )
                first_fail_mols = total_mol
                first_fail_grid = (nx, ny, nz)
                first_fail_oom = False
                break

            # -----------------------
            # 2) test.py 실행
            # -----------------------
            success, is_oom, log_path = run_test_once(
                filepath=out_xyz,
                supercell=(1, 1, 1),
                pbc=target.pbc,
                log_dir=log_root,
            )

            if success:
                print(
                    f"    → 성공 (OOM 아님)  분자수={total_mol}, log={log_path.name}"
                )
                max_success_mols = total_mol
                max_success_grid = (nx, ny, nz)
                continue

            print(
                f"    → 실패 (OOM={is_oom})  분자수={total_mol}, log={log_path.name}  → 이후 grid 후보는 스킵"
            )
            first_fail_mols = total_mol
            first_fail_grid = (nx, ny, nz)
            first_fail_oom = is_oom
            break

        # 요약 기록
        summary.setdefault("clusters", {})[target.label] = {
            "filename": target.filename,
            "max_success_molecules": max_success_mols,
            "max_success_grid": list(max_success_grid) if max_success_grid else None,
            "first_fail_molecules": first_fail_mols,
            "first_fail_grid": list(first_fail_grid) if first_fail_grid else None,
            "first_fail_is_oom": bool(first_fail_oom)
            if first_fail_mols is not None
            else None,
        }


# ---------------------------------------------------------------------
# 요약 파일 출력
# ---------------------------------------------------------------------

def write_human_readable_summary(summary: Dict, path: Path) -> None:
    lines: List[str] = []

    lines.append("=== 메모리 한계 스캔 요약 ===\n")

    # CIF
    crystals = summary.get("crystals", {})
    if crystals:
        lines.append("[결정 (CIF)]")
        for material, info in crystals.items():
            max_sc = info.get("max_success_supercell")
            fail_sc = info.get("first_fail_supercell")
            if max_sc is None:
                lines.append(f"- {material} 에 대해 성공한 supercell 이 없습니다.")
            else:
                sc_str = f"{max_sc[0]}x{max_sc[1]}x{max_sc[2]}"
                line = f"- {material} 의 최대 supercell 스케일은 {sc_str} 입니다."
                if fail_sc is not None:
                    fail_str = f"{fail_sc[0]}x{fail_sc[1]}x{fail_sc[2]}"
                    line += f" (최초로 실패한 supercell = {fail_str})"
                lines.append(line)
        lines.append("")

    # CLUSTER
    clusters = summary.get("clusters", {})
    if clusters:
        lines.append("[클러스터 (C60 / H2O 등)]")
        for label, info in clusters.items():
            max_mol = info.get("max_success_molecules")
            fail_mol = info.get("first_fail_molecules")
            if max_mol is None:
                lines.append(f"- {label} 에 대해 성공한 분자 수 설정이 없습니다.")
            else:
                if label.lower().startswith("c60"):
                    lines.append(f"- C60 의 최대 분자수는 {max_mol} 개 입니다.")
                elif label.lower().startswith("h2o") or label.lower() == "h2o":
                    lines.append(f"- H2O 의 최대 물 분자 수는 {max_mol} 개 입니다.")
                else:
                    lines.append(f"- {label} 의 최대 분자 수는 {max_mol} 개 입니다.")
                if fail_mol is not None:
                    lines.append(
                        f"  (최초로 실패한 분자 수 설정 ≈ {fail_mol} 개)"
                    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"[INFO] test.py = {TEST_SCRIPT}")
    if not TEST_SCRIPT.exists():
        print("[ERROR] test.py 경로를 찾지 못했습니다. PROJECT_ROOT / test.py 위치를 확인하세요.")
        sys.exit(1)

    summary: Dict = {}

    # 1) CIF (결정) supercell 스캔
    scan_crystals(summary)

    # 2) C60 / H2O 클러스터 분자 수 스캔
    scan_clusters(summary)

    # 3) 요약 저장
    txt_path = HERE / "mem_limit_summary.txt"
    json_path = HERE / "mem_limit_summary.json"
    write_human_readable_summary(summary, txt_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[DONE] 요약 텍스트: {txt_path}")
    print(f"[DONE] 요약 JSON:   {json_path}")


if __name__ == "__main__":
    main()

