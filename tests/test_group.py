"""
test_group.py — Visual test of all individual symmetry generator maps and the
full SymmetryGroup for a 4×4 lattice with 2×2 patches.

For each generator (Tx, Ty, Rot, Mrrx, Mrry, Dia1, Dia2) we print every
non-identity action as a Ly×Lx grid showing where each site maps to,
in both natural coordinates and MPS coordinates.

Then we build a full SymmetryGroup with all generators active and print
the complete mapping table and phase table (transtable).

Run from gen_mps/:
    python test_group.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from symm_basis.lattice import SquareLattice2D, build_patch_ordering
from symm_basis.symmetry_group import (
    SymmetryGroup,
    _make_tx_maps, _make_ty_maps, _make_rot_maps,
    _make_mrrx_maps, _make_mrry_maps, _make_dia1_maps, _make_dia2_maps,
)

# ── parameters ────────────────────────────────────────────────────────────────
Lx, Ly = 4, 4
nbx, nby = 2, 2
N = Lx * Ly
nRot = 4

lattice = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=nbx, nby=nby)
nat_to_mps = lattice.nat_to_mps   # [N]
mps_to_nat = lattice.mps_to_nat   # [N]


# ── helpers ───────────────────────────────────────────────────────────────────

def nat_to_rc(nat_idx: int):
    """Natural index → (row, col)."""
    return nat_idx // Lx, nat_idx % Lx


def print_perm_grid(perm_nat: torch.Tensor, label: str):
    """
    Print a site permutation (in natural coordinates) as two Ly×Lx grids:
      • natural grid : cell shows destination natural index  →  (row, col)
      • MPS grid     : same permutation expressed in MPS indices
    """
    print(f"\n  {label}")
    print(f"  {'─'*56}")

    # natural grid
    print("  Natural coords  (cell = destination site's (row,col)):")
    header = "        " + "  ".join(f"c{c}" for c in range(Lx))
    print(header)
    for r in range(Ly):
        row_str = f"  row{r} "
        for c in range(Lx):
            src_nat = c + Lx * r
            dst_nat = int(perm_nat[src_nat].item())
            dr, dc = nat_to_rc(dst_nat)
            row_str += f" ({dr},{dc})"
        print(row_str)

    # MPS grid: convert perm_nat to a permutation on MPS indices
    # perm_mps[i_mps] = j_mps  where  j_mps = nat_to_mps[ perm_nat[ mps_to_nat[i_mps] ] ]
    perm_mps = nat_to_mps[perm_nat[mps_to_nat]]
    print("  MPS indices  (cell = destination MPS index):")
    print(header)
    for r in range(Ly):
        row_str = f"  row{r} "
        for c in range(Lx):
            src_nat = c + Lx * r
            src_mps = int(nat_to_mps[src_nat].item())
            dst_mps = int(perm_mps[src_mps].item())
            row_str += f"  {dst_mps:2d}"
        print(row_str)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Individual generator maps
# ══════════════════════════════════════════════════════════════════════════════

print_section(f"Lattice: {Lx}×{Ly},  patch: {nbx}×{nby}")
print(f"\n  MPS index grid (nat_to_mps shown at each (row,col)):")
header = "        " + "  ".join(f"c{c}" for c in range(Lx))
print(header)
for r in range(Ly):
    row_str = f"  row{r} "
    for c in range(Lx):
        row_str += f"  {int(nat_to_mps[c + Lx*r]):2d}"
    print(row_str)

# ── Tx ────────────────────────────────────────────────────────────────────────
print_section("Translation X  (Tx):  (row, col) → (row, col+t) mod Lx")
tx_maps = _make_tx_maps(Lx, Ly)
for t in range(1, Lx):   # skip t=0 (identity)
    print_perm_grid(tx_maps[t], f"Tx t={t}")

# ── Ty ────────────────────────────────────────────────────────────────────────
print_section("Translation Y  (Ty):  (row, col) → (row+t, col) mod Ly")
ty_maps = _make_ty_maps(Lx, Ly)
for t in range(1, Ly):
    print_perm_grid(ty_maps[t], f"Ty t={t}")

# ── Rot (C4) ──────────────────────────────────────────────────────────────────
print_section("C4 Rotation  (Rot):  (row,col) → (col, Ly-1-row)  [×r]")
rot_maps = _make_rot_maps(Lx, Ly, nRot)
for r in range(1, nRot):
    print_perm_grid(rot_maps[r], f"Rot r={r}  ({r}×90°)")

# ── Mrrx ─────────────────────────────────────────────────────────────────────
print_section("Mirror X  (Mrrx):  (row, col) → (row, Lx-1-col)")
px_maps = _make_mrrx_maps(Lx, Ly)
print_perm_grid(px_maps[1], "Mrrx")

# ── Mrry ─────────────────────────────────────────────────────────────────────
print_section("Mirror Y  (Mrry):  (row, col) → (Ly-1-row, col)")
py_maps = _make_mrry_maps(Lx, Ly)
print_perm_grid(py_maps[1], "Mrry")

# ── Dia1 ─────────────────────────────────────────────────────────────────────
print_section("Main-diagonal Mirror  (Dia1):  (row, col) → (col, row)")
d1_maps = _make_dia1_maps(Lx, Ly)
print_perm_grid(d1_maps[1], "Dia1")

# ── Dia2 ─────────────────────────────────────────────────────────────────────
print_section("Anti-diagonal Mirror  (Dia2):  (row, col) → (Ly-1-col, Lx-1-row)")
d2_maps = _make_dia2_maps(Lx, Ly)
print_perm_grid(d2_maps[1], "Dia2")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Full SymmetryGroup: all generators active, momentum k=(1,1), all eigenvalues -1
# ══════════════════════════════════════════════════════════════════════════════

print_section("Full SymmetryGroup  (all generators, kx=ky=1, px=py=sgm1=sgm2=-1, rr=1)")

group = SymmetryGroup(
    Lx=Lx, Ly=Ly,
    nat_to_mps=nat_to_mps,
    mps_to_nat=mps_to_nat,
    kx=2, ky=2,
    px=1, py=1,
    rr=0, nRot=nRot,
    sgm1=1, sgm2=1,
    use_Tx=True, use_Ty=True, use_Rot=True,
    use_Mrrx=True, use_Mrry=True, use_Dia1=True, use_Dia2=True,
    #use_Mrrx=False, use_Mrry=False, use_Dia1=False, use_Dia2=False,
)

print(f"\n  Total unique group elements (after dedup): {group.ntrans}")
print(f"\n  {'α':>4}  {'(tx,ty,rot,mx,my,d1,d2)':>26}"
      f"  {'χ.real':>9}  {'χ.imag':>9}")
print(f"  {'─'*62}")
for α in range(group.ntrans):
    tx, ty, rot, mx, my, d1, d2 = group.transtep[α].tolist()
    χ = group.transtable[α]
    print(f"  {α:>4}  ({int(tx)},{int(ty)},{int(rot)},{int(mx)},{int(my)},{int(d1)},{int(d2)})"
          f"  {χ.real:>+9.5f}  {χ.imag:>+9.5f}")

print(f"\n  mapping[α] shown as MPS-index permutation grids:")
for α in range(group.ntrans):
    tx, ty, rot, mx, my, d1, d2 = group.transtep[α].tolist()
    label = (f"α={α:2d}  step=({int(tx)},{int(ty)},{int(rot)},"
             f"{int(mx)},{int(my)},{int(d1)},{int(d2)})"
             f"  χ={group.transtable[α].real:+.4f}{group.transtable[α].imag:+.4f}j")

    # mapping[α] is already in MPS order: mapping[α, i_mps] = j_mps
    perm_mps = group.mapping[α]     # [N]
    # convert back to natural for the natural grid
    perm_nat = mps_to_nat[perm_mps[nat_to_mps]]  # perm in natural coords

    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print("  MPS grid  (cell = destination MPS index):")
    hdr = "        " + "  ".join(f"c{c}" for c in range(Lx))
    print(hdr)
    for r in range(Ly):
        row_str = f"  row{r} "
        for c in range(Lx):
            src_nat = c + Lx * r
            src_mps = int(nat_to_mps[src_nat].item())
            dst_mps = int(perm_mps[src_mps].item())
            row_str += f"  {dst_mps:2d}"
        print(row_str)
    print("  Natural grid  (cell = destination (row,col)):")
    print(hdr)
    for r in range(Ly):
        row_str = f"  row{r} "
        for c in range(Lx):
            src_nat = c + Lx * r
            dst_nat = int(perm_nat[src_nat].item())
            dr, dc = nat_to_rc(dst_nat)
            row_str += f" ({dr},{dc})"
        print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

print_section("Sanity checks")

# Every mapping[α] must be a valid permutation of [0..N-1]
for α in range(group.ntrans):
    vals = group.mapping[α]
    assert vals.sort()[0].tolist() == list(range(N)), \
        f"mapping[{α}] is not a valid permutation!"
print(f"  ✓  All {group.ntrans} mappings are valid permutations of [0..{N-1}].")

# Identity element must be present (permutation = arange(N))
identity = torch.arange(N, dtype=torch.long)
has_identity = any((group.mapping[α] == identity).all().item()
                   for α in range(group.ntrans))
assert has_identity, "Identity element missing from group!"
print("  ✓  Identity element present.")

# Sum of characters must be non-negative real (orthogonality for non-trivial irreps
# may give 0; for the trivial irrep kx=ky=0, px=py=1 etc. it equals ntrans)
χ_sum = group.transtable.sum()
print(f"  Σ χ(g) = {χ_sum.real:+.6f} {χ_sum.imag:+.6f}j")

print("\n  Done.")
