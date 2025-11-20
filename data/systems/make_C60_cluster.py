# File: tools/build_c60_cluster_compact.py
"""
C60 클러스터(비주기) 컴팩트 배치 도우미.
- 1D/2D/3D 배치 지원(line/grid/lattice).
- 직육면체/정육면체 셀 선택, 최소 벽 여유(wall_gap) 보장.
- 메모리 ~ (Lx*Ly*Lz)/(Δx^3); Δx는 솔버 그리드 간격.

주의(왜 이 설계인가): 비주기 실공간 솔버에서 박스 부피가 메모리를 지배하므로,
스팬+여유만큼의 타이트한 셀을 설정해 낭비를 줄인다.

example
python make_C60_cluster.py \
    --nx 2 --ny 2 --nz 2 \
    --spacing 12.0 \
    --wall_gap 5.0 \
    --output C60_2x2x2.xyz
"""
# 맨 아래 main block 에서 설정

from __future__ import annotations
from typing import Iterable, Tuple, List, Optional
from math import ceil, sqrt
import random
from ase import Atoms
from ase.io import read, write
import argparse

C60_DIAMETER = 7.1  # Å 문헌값, 참고
C60_RADIUS = C60_DIAMETER / 2.0  # Å (참고)


def _random_rotate(mol: Atoms, enable: bool, seed: Optional[int] = None) -> Atoms:
    m = mol.copy()
    if not enable:
        return m
    rng = random.Random(seed)
    # why: 배향 무작위화로 특정 접촉면 편향 방지
    for ang, ax in zip(
        (rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)), ("x", "y", "z")
    ):
        m.rotate(ang, v=ax, center="COM", rotate_cell=False)
    return m


def _place_on_positions(
    mol: Atoms, offsets: Iterable[Tuple[float, float, float]]
) -> Atoms:
    out = None
    for off in offsets:
        m = mol.copy()
        m.translate(off)
        out = m if out is None else out + m
    return out


def _line_offsets(n: int, spacing: float) -> List[Tuple[float, float, float]]:
    return [(i * spacing, 0.0, 0.0) for i in range(n)]


def _grid_offsets(
    nx: int, ny: int, sx: float, sy: float
) -> List[Tuple[float, float, float]]:
    return [(ix * sx, iy * sy, 0.0) for iy in range(ny) for ix in range(nx)]


def _lattice_offsets(
    nx: int, ny: int, nz: int, sx: float, sy: float, sz: float
) -> List[Tuple[float, float, float]]:
    return [
        (ix * sx, iy * sy, iz * sz)
        for iz in range(nz)
        for iy in range(ny)
        for ix in range(nx)
    ]


def _compact_cell_from_positions(
    centers: List[Tuple[float, float, float]], wall_gap: float, rectangular: bool
) -> Tuple[float, float, float]:
    if not centers:
        L = C60_DIAMETER + 2 * wall_gap
        return (L, L, L)
    xs, ys, zs = zip(*centers)
    span_x = (max(xs) - min(xs)) + C60_DIAMETER
    span_y = (max(ys) - min(ys)) + C60_DIAMETER
    span_z = (max(zs) - min(zs)) + C60_DIAMETER
    if rectangular:
        return (span_x + 2 * wall_gap, span_y + 2 * wall_gap, span_z + 2 * wall_gap)
    Lcube = max(span_x, span_y, span_z) + 2 * wall_gap
    return (Lcube, Lcube, Lcube)


def build_c60_cluster(
    input_xyz: str = "C60_1.xyz",
    n_molecules: int = 2,
    spacing: float = 8.5,
    wall_gap: float = 4.0,
    rectangular: bool = True,
    random_orientation: bool = False,
    output_xyz: Optional[str] = None,
    # 신규: 배치/형상
    layout: str = "line",  # "line" | "grid" | "lattice"
    grid_shape: Optional[Tuple[int, int]] = None,  # 예: (2,2)
    lattice_shape: Optional[Tuple[int, int, int]] = None,  # 예: (2,2,2)
    spacings_xyz: Optional[Tuple[float, float, float]] = None,  # (sx, sy, sz)
    seed: Optional[int] = None,
) -> str:
    base: Atoms = read(input_xyz)
    base.set_pbc([False, False, False])
    base.translate(-base.get_center_of_mass())
    base = _random_rotate(base, random_orientation, seed=seed)

    if spacings_xyz is None:
        sx = sy = sz = spacing
    else:
        sx, sy, sz = spacings_xyz

    # 배치 결정
    if layout == "line":
        offsets = _line_offsets(n_molecules, spacing=sx)
    elif layout == "grid":
        if grid_shape is None:
            nx = max(1, int(round(sqrt(n_molecules))))
            ny = ceil(n_molecules / nx)
        else:
            nx, ny = grid_shape
        if nx * ny != n_molecules:
            raise ValueError(f"grid_shape({nx}x{ny}) != n_molecules({n_molecules})")
        offsets = _grid_offsets(nx, ny, sx, sy)
    elif layout == "lattice":
        if lattice_shape is None:
            raise ValueError("lattice_shape를 지정하세요 (nx, ny, nz).")
        nx, ny, nz = lattice_shape
        if nx * ny * nz != n_molecules:
            raise ValueError(
                f"lattice_shape({nx}x{ny}x{nz}) != n_molecules({n_molecules})"
            )
        offsets = _lattice_offsets(nx, ny, nz, sx, sy, sz)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    cluster = _place_on_positions(base, offsets)

    mins = cluster.get_positions().min(axis=0)
    cluster.translate(
        [
            max(0.0, wall_gap - mins[0]),
            max(0.0, wall_gap - mins[1]),
            max(0.0, wall_gap - mins[2]),
        ]
    )

    Lx, Ly, Lz = _compact_cell_from_positions(offsets, wall_gap, rectangular)
    cluster.set_cell([Lx, Ly, Lz])
    cluster.set_pbc([False, False, False])

    if output_xyz is None:
        tag = "rect" if rectangular else "cube"
        lay = {"line": "1d", "grid": "2d", "lattice": "3d"}[layout]
        output_xyz = f"C60_{lay}_{tag}_n{n_molecules}_s{sx:.1f}-{sy:.1f}-{sz:.1f}_gap{wall_gap:.1f}.xyz"
    write(output_xyz, cluster, format="extxyz")
    return output_xyz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True, help="x axis molecule number")
    parser.add_argument("--ny", type=int, required=True, help="y axis molecule number")
    parser.add_argument("--nz", type=int, required=True, help="z axis molecule number")
    parser.add_argument(
        "--spacing", type=float, default=12.0, help="distance of molecule to molecule"
    )
    parser.add_argument(
        "--wall_gap", default=5.0, type=float, help="distance moelcule to wall"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="file name of C60 clster cif file"
    )
    args = parser.parse_args()
    #    # 1) 원래 1D (레퍼런스)
    #    print(build_c60_cluster(
    #        input_xyz="C60.xyz",
    #        n_molecules=2,
    #        spacing=10.0,
    #        wall_gap=4.0,
    #        rectangular=True,
    #        random_orientation=False
    #    ))

    # 2) 2D: 4개 평면(2×2) — 질문 케이스
    #    spacing≈8.5~10.5 권장, wall_gap=4.0 기본
    #    print(build_c60_cluster(
    #        input_xyz="C60.xyz",
    #        n_molecules=4,                      # C60의 개수
    #        layout="grid",                      # 2차원으로 배치
    #        grid_shape=(2, 2),                  # 각 축당 2개씩 배치
    #        spacings_xyz=(10.0, 10.0, 0.0),     # 분자 사이의 간격
    #        wall_gap=4.0,                       # 벽과 원자의 간격
    #        rectangular=True,
    #        random_orientation=False
    #    ))
    #
    # 3) 3D: 2×2×2 예시
    out = build_c60_cluster(
        input_xyz="C60.xyz",
        n_molecules=args.nx * args.ny * args.nz,  # C60의 개수
        layout="lattice",  # 3차원으로 배치
        lattice_shape=(args.nx, args.ny, args.nz),  # 각 축당 2개씩 배치
        spacings_xyz=(args.spacing, args.spacing, args.spacing),  # 분자 사이의 간격
        wall_gap=args.wall_gap,  # 벽과 원자사이의 간격
        rectangular=True,
        random_orientation=True,
        seed=42,
        output_xyz=args.output,
    )
    print(out)
