#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate water cluster on a rectangular lattice and return as ASE Atoms object.
Recommended parameters for realistic water clusters:
  - spacing: 2.8-3.0 Å (liquid water O-O distance ~2.8 Å)
  - translate: 0.3-0.5 Å (10-15% of spacing for natural disorder)
  - rotate: always recommended for physical realism
Examples:
  # Standard configuration (most realistic liquid-like cluster)
  python make_water_cluster.py --nx 3 --ny 3 --nz 3 --spacing 3.0 --translate 0.4 --rotate --output water.xyz
"""
import argparse
import numpy as np
from ase import Atoms
from ase.io import write
def water_monomer(origin, rotate=False, translate=0.0, seed=None):
    """
    Return H2O coordinates around origin (Å).
    Args:
        origin: (x, y, z) tuple for the oxygen position
        rotate: If True, apply random rotation to the molecule
        translate: Maximum random displacement from origin in Angstroms (default: 0.0)
        seed: Random seed for reproducibility (only used if rotate=True or translate>0)
    """
    ox, oy, oz = origin
    # Rough gas-phase water: OH~0.958 Å, angle~104.5°
    coords = np.array([
        [0.0, 0.0, 0.0],           # O
        [0.757, 0.0, 0.586],       # H1
        [-0.757, 0.0, 0.586],      # H2
    ])
    if rotate or translate > 0:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
    if rotate:
        # Random rotation using Euler angles
        alpha = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(0, 2 * np.pi)
        gamma = rng.uniform(0, 2 * np.pi)
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])
        # Apply rotation
        R = Rz @ Ry @ Rx
        coords = coords @ R.T
    # Random translation
    if translate > 0:
        displacement = rng.uniform(-translate, translate, size=3)
        ox += displacement[0]
        oy += displacement[1]
        oz += displacement[2]
    # Translate to origin
    coords[:, 0] += ox
    coords[:, 1] += oy
    coords[:, 2] += oz
    return [
        ("O", tuple(coords[0])),
        ("H", tuple(coords[1])),
        ("H", tuple(coords[2])),
    ]
def build_water_cluster(nx: int, ny: int, nz: int, spacing: float = 3.0,
                        rotate: bool = False, translate: float = 0.0, seed: int = None) -> Atoms:
    """
    Pack water molecules on a rectangular lattice with specified dimensions.
    Args:
        nx: Number of water molecules along x-axis
        ny: Number of water molecules along y-axis
        nz: Number of water molecules along z-axis
        spacing: Lattice spacing in Angstroms (default: 3.0)
        rotate: If True, randomly rotate each water molecule (default: False)
        translate: Maximum random displacement from grid points in Angstroms (default: 0.0)
        seed: Random seed for reproducibility when rotate=True or translate>0 (default: None)
    Returns:
        ASE Atoms object containing the water cluster
    """
    atoms_list = []
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = (ix * spacing, iy * spacing, iz * spacing)
                # Each molecule gets a different random rotation and translation
                atoms_list.extend(water_monomer(origin, rotate=rotate, translate=translate))
    # Convert to ASE Atoms object
    symbols = [sym for sym, _ in atoms_list]
    positions = [xyz for _, xyz in atoms_list]
    return Atoms(symbols=symbols, positions=positions)
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate water cluster on rectangular lattice"
    )
    p.add_argument(
        "--nx",
        type=int,
        default=2,
        help="Number of water molecules along x-axis (default: 2)"
    )
    p.add_argument(
        "--ny",
        type=int,
        default=2,
        help="Number of water molecules along y-axis (default: 2)"
    )
    p.add_argument(
        "--nz",
        type=int,
        default=2,
        help="Number of water molecules along z-axis (default: 2)"
    )
    p.add_argument(
        "--spacing",
        type=float,
        default=3.0,
        help="Lattice spacing in Angstroms (default: 3.0)"
    )
    p.add_argument(
        "--rotate",
        action="store_true",
        help="Randomly rotate each water molecule for more natural arrangement"
    )
    p.add_argument(
        "--translate",
        type=float,
        default=0.0,
        help="Maximum random displacement from grid points in Angstroms (default: 0.0, e.g., 0.3)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible rotations/translations"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (e.g., cluster.xyz, cluster.traj, cluster.png)"
    )
    p.add_argument(
        "--view",
        action="store_true",
        help="Open ASE GUI viewer after generation"
    )
    return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    # Generate water cluster
    atoms = build_water_cluster(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        spacing=args.spacing,
        rotate=args.rotate,
        translate=args.translate,
        seed=args.seed
    )
    n_total = args.nx * args.ny * args.nz
    print(f"Generated (H2O)_{n_total} cluster")
    print(f"  Grid dimensions: {args.nx} × {args.ny} × {args.nz}")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Lattice spacing: {args.spacing} Å")
    if args.rotate:
        print(f"  Random rotation: enabled")
    if args.translate > 0:
        print(f"  Random translation: ±{args.translate} Å")
    if args.seed is not None and (args.rotate or args.translate > 0):
        print(f"  Random seed: {args.seed}")
    print(f"  Composition: {atoms.get_chemical_formula()}")
    # Save to file if requested
    if args.output:
        write(args.output, atoms)
        print(f"  Saved to: {args.output}")
    # View if requested
    if args.view:
        from ase.visualize import view
        view(atoms)
