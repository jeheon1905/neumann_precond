"""
내부에 존재하는 cif, sdf, xyz 파일의 구조를 보는 코드

예시
python show_structure.py molecule.sdf
python show_structure.py structure.cif
"""

import argparse
from ase.io import read
from ase.visualize import view

# 커맨드라인 인자 파서
parser = argparse.ArgumentParser()
parser.add_argument("input", help="구조 파일 (xyz, sdf, cif 등)")
args = parser.parse_args()

# args.input 을 사용해서 Atoms 만들기
atoms = read(args.input)   # molecule.xyz, molecule.sdf, molecule.cif 등등
view(atoms)                # GUI로 띄우기

