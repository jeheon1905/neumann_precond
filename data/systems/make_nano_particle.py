"""ASE를 사용해 간단한 금속 나노입자 클러스터를 생성합니다.

사용 예시
---------
python make_nano_particle.py --material Pt --natoms 90 --output pt_90.xyz
python make_nano_particle.py --material Au --natoms 147
"""
from __future__ import annotations

import argparse
from math import sqrt
from typing import Iterable, Tuple

from ase.cluster import wulff_construction
from ase.data import atomic_numbers, covalent_radii
from ase.io import write


# 기본 표면 지수와 표면 에너지를 정의합니다.
# (100), (110), (111) 면을 같은 표면 에너지로 두어 대칭적인 Wulff 구조를 만듭니다.
DEFAULT_SURFACES: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
)
DEFAULT_SURFACE_ENERGIES = (1.0, 1.0, 1.0)


def guess_lattice_constant(symbol: str) -> float:
    """공유 반지름을 이용해 fcc 격자의 격자 상수를 추정합니다.

    밀집된 fcc 격자에서 ``a = 2 * sqrt(2) * r`` 관계를 사용합니다.
    """

    if symbol not in atomic_numbers:    # 올바른 원소 기호를 입력할 것 
        raise ValueError(f"Unknown element symbol: {symbol}")

    # ASE가 제공하는 공유 반지름 표에서 값을 가져옵니다.
    radius = covalent_radii[atomic_numbers[symbol]]
    return 2 * sqrt(2) * radius


def build_cluster(symbol: str, natoms: int, vacuum: float) -> "ase.Atoms":
    """주어진 원자 수에 맞춰 Wulff 나노입자를 생성합니다.

    Parameters
    ----------
    symbol:
        클러스터에 사용할 화학 원소 기호 (예: ``"Pt"``, ``"Au"``).
    natoms:
        목표 원자 수. Wulff 생성기는 이보다 크거나 같은 원자 수를 배치합니다.
    vacuum:
        주기적 경계 조건으로 인한 상호 작용을 막기 위한 진공(Å).
    """

    if natoms <= 0:
        raise ValueError("Number of atoms must be positive.")

    # 1) 격자 상수를 추정해 Wulff 생성기에 전달합니다.
    lattice_constant = guess_lattice_constant(symbol)

    # 2) ASE의 Wulff 구성 함수를 이용해 fcc 나노입자를 생성합니다. 
    cluster = wulff_construction(
        symbol,
        DEFAULT_SURFACES,
        DEFAULT_SURFACE_ENERGIES,
        natoms,
        latticeconstant=lattice_constant,
        structure="fcc",
    )
    # 3) 시뮬레이션 박스 중심에 배치하고 진공을 더합니다.
    cluster.center(vacuum=vacuum)
    return cluster


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASE로 금속 나노입자 클러스터를 생성하는 CLI입니다.",
    )
    parser.add_argument(
        "--material",
        required=True,
        help="Chemical symbol of the element (e.g., Au, Pt, Ni).",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        required=True,
        help="Target number of atoms in the nanoparticle.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="출력 파일 이름(지정하지 않으면 {material}_{natoms}.xyz).",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=3.0,
        help="Vacuum padding (Å) to add around the cluster.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    # 1) CLI 인자를 파싱합니다.
    args = parse_args(argv)

    # 2) Wulff 구조로 클러스터를 만들고 진공을 추가합니다.
    cluster = build_cluster(args.material, args.natoms, args.vacuum)
    actual_natoms = len(cluster)

    # 3) 출력 파일 이름 결정
    #    --output을 지정하지 않은 경우에만 {material}_{실제원자수}.xyz 사용
    output_path = args.output or f"{args.material}_{actual_natoms}.xyz"

    # 4) ASE의 write 함수를 사용해 결과를 저장합니다.
    write(output_path, cluster)
    print(
        f"{actual_natoms}개 원자로 이루어진 {args.material} 클러스터를 "
        f"'{output_path}' 파일로 저장했습니다 (진공 {args.vacuum} Å)."
    )


if __name__ == "__main__":
    main()

