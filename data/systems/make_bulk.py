import numpy as np
from ase.build import bulk, make_supercell
from ase.visualize import view
from ase.io import write


def main(
    symbol="Cu",
    crystalstructure="fcc",
    a=None,
    c=None,
    covera=None,
    orthorhombic=False,
    cubic=False,
    supercell=[1, 1, 1],
    prnt=True,
):
    """
    Generate a bulk crystal structure with specified parameters.

    Parameters:
        symbol (str): Atomic symbol of the element (e.g., 'Cu').
        crystalstructure (str): Crystal structure (e.g., 'fcc', 'bcc', 'hcp').
        a (float): Lattice constant a.
        c (float): Lattice constant c (for hcp or tetragonal if needed).
        covera (float): c/a ratio (for hcp if needed).
        orthorhombic (bool): Whether to force the cell to be orthorhombic.
        cubic (bool): Whether to force the cell to be cubic.
        supercell (list): Supercell size factors.
        prnt (bool): Whether to print information about the structure.

    Returns:
        atoms (ASE Atoms object): Generated bulk crystal structure.
    """

    # Build the primitive bulk structure
    atoms = bulk(
        name=symbol,
        crystalstructure=crystalstructure,
        a=a,
        c=c,
        covera=covera,
        orthorhombic=orthorhombic,
        cubic=cubic,
    )

    # Apply supercell scaling
    prim = np.diag(supercell)
    atoms = make_supercell(atoms, prim)

    if prnt:
        print(f"Bulk structure: {symbol}, {crystalstructure}")
        print(f"Atoms count: {len(atoms)}")
        print(atoms)

    return atoms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbol", type=str, default="Cu", help="Atomic symbol (default: Cu)"
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="fcc",
        help="Crystal structure (fcc, bcc, hcp, etc.)",
    )
    parser.add_argument("--a", type=float, default=None, help="Lattice constant a")
    parser.add_argument(
        "--c", type=float, default=None, help="Lattice constant c (if needed)"
    )
    parser.add_argument(
        "--covera", type=float, default=None, help="c/a ratio (if needed)"
    )
    parser.add_argument(
        "--orthorhombic", action="store_true", help="Force orthorhombic cell"
    )
    parser.add_argument("--cubic", action="store_true", help="Force cubic cell")
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="Supercell size (default: 1 1 1)",
    )
    parser.add_argument("--save", type=str, help="Save to CIF file")
    parser.add_argument("--view", action="store_true", help="Visualize structure")

    args = parser.parse_args()

    atoms = main(
        symbol=args.symbol,
        crystalstructure=args.structure,
        a=args.a,
        c=args.c,
        covera=args.covera,
        orthorhombic=args.orthorhombic,
        cubic=args.cubic,
        supercell=args.supercell,
    )

    if args.view:
        view(atoms)

    if args.save:
        write(args.save, atoms)
        print(f"Saved structure to {args.save}")
