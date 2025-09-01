import numpy as np
from ase.build import nanotube, make_supercell
from ase.visualize import view
from ase.io import write


def main(n, m, vacuum=3.0, supercell=[1, 1, 1], prnt=True, symbol="C", bond=None):
    """
    Generate a nanotube structure with specified parameters.

    Parameters:
        n, m (int): Chiral indices of the nanotube.
        vacuum (float): Vacuum padding around the nanotube.
        supercell (list): Supercell size.
        prnt (bool): Whether to print nanotube details.
        symbol (str): Atomic symbol of the nanotube material.
        bond (float): Bond length override.

    Returns:
        cnt (ASE Atoms object): Generated nanotube.
    """
    cnt = nanotube(n, m, bond=bond, symbol=symbol)
    prim = np.diag(supercell)
    cnt = make_supercell(cnt, prim)
    cnt.center(vacuum=vacuum, axis=np.arange(3)[~cnt.get_pbc()])
    cnt.center()

    if prnt:
        if (2 * n + m) % 3 == 0 or (n - m) % 3 == 0:
            print(f"Nanotube ({symbol}) ({n}, {m}) is metallic")
        else:
            print(f"Nanotube ({symbol}) ({n}, {m}) is non-metallic")
        print(len(cnt), cnt)

    return cnt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nm", type=int, nargs=2, required=True, help="Make nanotube (n, m)"
    )
    parser.add_argument(
        "--vacuum", type=float, default=3.0, help="Vacuum size (angstroms)"
    )
    parser.add_argument("--save", type=str, help="Save nanotube to CIF file")
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=[1, 1, 1], help="Supercell dimensions"
    )
    parser.add_argument(
        "--symbol", type=str, default="C", help="Element symbol (default: C)"
    )
    parser.add_argument(
        "--bond", type=float, default=1.42, help="Override bond length (angstroms)"
    )
    parser.add_argument("--view", action="store_true", help="Visualize nanotube")
    args = parser.parse_args()

    n, m = args.nm
    cnt = main(n, m, args.vacuum, args.supercell, symbol=args.symbol, bond=args.bond)

    if args.view:
        view(cnt)

    if args.save:
        write(args.save, cnt)
        print(f"Saved to {args.save}")
