"""
test_lattice.py — Visual test of the SquareLattice2D patch/super-patch ordering.

Prints, for each site, its (row, col) coordinates, its natural index, and its
MPS index, for a 4×4 lattice with 2×2 patches (nbx=nby=2).

Also prints the nat_to_mps and mps_to_nat arrays in grid form so the
patch structure is easy to verify visually.

Run from gen_mps/:
    python test_lattice.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symm_basis.lattice import SquareLattice2D, build_patch_ordering

def main():
    Lx, Ly = 4, 4
    nbx, nby = 2, 2

    # ── build directly via the helper function ─────────────────────────────────
    nat_to_mps, mps_to_nat = build_patch_ordering(Lx, Ly, nbx, nby)

    print("=" * 60)
    print(f"Lattice: {Lx}×{Ly},  patch size: {nbx}×{nby}")
    print(f"Total sites N = {Lx * Ly}")
    print("=" * 60)

    # ── per-site table ─────────────────────────────────────────────────────────
    print(f"\n{'row':>4}  {'col':>4}  {'nat_idx':>8}  {'mps_idx':>8}")
    print("-" * 32)
    for row in range(Ly):
        for col in range(Lx):
            nat_idx = col + Lx * row
            mps_idx = int(nat_to_mps[nat_idx].item())
            print(f"{row:>4}  {col:>4}  {nat_idx:>8}  {mps_idx:>8}")

    # ── nat_to_mps as a 2D grid (natural layout) ──────────────────────────────
    # Each cell shows: nat_idx → mps_idx
    print("\nnat_to_mps shown as Ly×Lx grid (cell = mps index of that natural site):")
    print("  rows = row index (0 = top), cols = col index (0 = left)")
    header = "      " + "  ".join(f"col{c}" for c in range(Lx))
    print(header)
    for row in range(Ly):
        row_str = f"row{row} "
        for col in range(Lx):
            nat_idx = col + Lx * row
            mps_idx = int(nat_to_mps[nat_idx].item())
            row_str += f"  {mps_idx:3d}"
        print(row_str)

    # ── mps_to_nat as a 1D sequence ───────────────────────────────────────────
    print("\nmps_to_nat: mps index → (row, col) in natural coordinates")
    print(f"  {'mps_idx':>8}  {'nat_idx':>8}  {'(row, col)':>12}")
    print("-" * 36)
    for mps_idx in range(Lx * Ly):
        nat_idx = int(mps_to_nat[mps_idx].item())
        row = nat_idx // Lx
        col = nat_idx % Lx
        print(f"  {mps_idx:>8}  {nat_idx:>8}  ({row}, {col})")

    # ── verify inverse consistency ─────────────────────────────────────────────
    import torch
    recovered = nat_to_mps[mps_to_nat]
    identity  = torch.arange(Lx * Ly, dtype=torch.long)
    assert (recovered == identity).all(), "nat_to_mps[mps_to_nat] != identity!"
    recovered2 = mps_to_nat[nat_to_mps]
    assert (recovered2 == identity).all(), "mps_to_nat[nat_to_mps] != identity!"
    print("\n✓  nat_to_mps and mps_to_nat are consistent inverses of each other.")

    # ── also test via SquareLattice2D ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SquareLattice2D object (same parameters):")
    lattice = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=nbx, nby=nby)
    print(f"  N = {lattice.N}")
    assert (lattice.nat_to_mps == nat_to_mps).all()
    assert (lattice.mps_to_nat == mps_to_nat).all()
    print("  nat_to_mps / mps_to_nat match build_patch_ordering() directly.")
    nn_bonds = lattice.nn_bonds()
    print(f"  NN bond table shape: {tuple(nn_bonds.shape)}  (expect [{2*Lx*Ly}, 2])")
    print("=" * 60)

    # ── all_bonds in MPS indices ───────────────────────────────────────────────
    print("\nall_bonds (MPS indices):  [bond_idx]  i_mps -- j_mps  "
          "  (i_nat=(row,col)) -- (j_nat=(row,col))")
    print("-" * 70)
    all_bonds = lattice.all_bonds()  # [nb, 2] in MPS order
    for b in range(all_bonds.shape[0]):
        i_mps = int(all_bonds[b, 0].item())
        j_mps = int(all_bonds[b, 1].item())
        i_nat = int(lattice.mps_to_nat[i_mps].item())
        j_nat = int(lattice.mps_to_nat[j_mps].item())
        i_row, i_col = i_nat // Lx, i_nat % Lx
        j_row, j_col = j_nat // Lx, j_nat % Lx
        print(f"  [{b:3d}]  {i_mps:3d} -- {j_mps:3d}"
              f"    ({i_row},{i_col}) -- ({j_row},{j_col})")
    print(f"\nTotal bonds: {all_bonds.shape[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
