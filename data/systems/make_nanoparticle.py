#!/usr/bin/env python3
"""
Utility script for generating FCC nanoparticle structures using ASE's Wulff construction.

This script creates isolated metallic nanoparticles, centers them with vacuum,
and exports the final structure to an output file.
"""

import argparse
from math import sqrt
from typing import Tuple

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.cluster.wulff import wulff_construction
from ase.io import write


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

DEFAULT_SURFACES = ((1, 0, 0), (1, 1, 0), (1, 1, 1))
DEFAULT_SURFACE_ENERGIES = (1.0, 1.0, 1.0)


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

def guess_lattice_constant(symbol: str) -> float:
    """
    Return an approximate FCC lattice constant for the given element.

    The estimation uses: a = 2 * sqrt(2) * r
      where r is the covalent radius.

    This is only a rough approximation and not intended for accurate
    material modeling. It is sufficient for generating test nanoparticles.
    """
    if symbol not in atomic_numbers:
        raise ValueError(f"Unknown element symbol: {symbol}")

    radius = covalent_radii[atomic_numbers[symbol]]
    return 2 * sqrt(2) * radius


def build_cluster(symbol: str, natoms: int, vacuum: float) -> Atoms:
    """
    Generate an FCC nanoparticle using Wulff construction.

    Parameters
    ----------
    symbol : str
        Atomic symbol (e.g., "Pt", "Au").
    natoms : int
        Target number of atoms. The final number may differ depending
        on Wulff geometry.
    vacuum : float
        Amount of vacuum padding (Å) added around the cluster.

    Returns
    -------
    Atoms
        ASE Atoms object containing the nanoparticle.
    """
    if natoms <= 0:
        raise ValueError("The number of atoms must be a positive integer.")

    lattice_constant = guess_lattice_constant(symbol)

    # Build the initial Wulff cluster
    cluster = wulff_construction(
        symbol,
        surfaces=DEFAULT_SURFACES,
        energies=DEFAULT_SURFACE_ENERGIES,
        size=natoms,
        structure="fcc",
        latticeconstant=lattice_constant,
        rounding="closest",
    )

    # Center the cluster within a box with vacuum padding
    cluster.center(vacuum=vacuum)

    return cluster


# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate FCC nanoparticles via Wulff construction."
    )

    parser.add_argument(
        "--material",
        type=str,
        required=True,
        help="Element symbol for the nanoparticle (e.g., Pt, Au, Cu).",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        required=True,
        help="Target number of atoms in the nanoparticle.",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=5.0,
        help="Vacuum thickness (Å) added around the particle. "
             "For real-space DFT with isolated boundaries, 4–6 Å is typical.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path. Default: {MATERIAL}_NP_{NATOMS}.xyz",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cluster = build_cluster(args.material, args.natoms, args.vacuum)
    real_natoms = len(cluster)

    # Default output filename
    output_path = (
        args.output
        if args.output
        else f"{args.material}_NP_{real_natoms}.xyz"
    )

    write(output_path, cluster)

    print(f"Nanoparticle generated:")
    print(f"  Material     : {args.material}")
    print(f"  Requested N  : {args.natoms}")
    print(f"  Actual N     : {real_natoms}")
    print(f"  Vacuum (Å)   : {args.vacuum}")
    print(f"  Saved to     : {output_path}")


if __name__ == "__main__":
    main()

