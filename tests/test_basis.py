"""
test_basis.py — Generate and print the complete symmetric basis for a 4×4
2D Heisenberg system with user-specified quantum numbers.

The quantum numbers that label a basis sector are:
  kx, ky   : momentum (integers in [0, Lx) and [0, Ly))
  px, py   : mirror-x / mirror-y eigenvalue (±1, or irrelevant if not active)
  rr       : rotation quantum number (integer in [0, nRot))
  sgm1,sgm2: diagonal mirror eigenvalues (±1, or irrelevant if not active)
  zz       : spin-inversion eigenvalue (+1 or -1, 0 to disable)
  sec      : Sz sector: nup = (N + 2*sec) / 2  (sec=0 → half filling)

Edit the QUANTUM NUMBERS block below to explore different sectors.

Run from gen_mps/:
    python test_basis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from symm_basis.lattice import SquareLattice2D
from symm_basis.symmetry_group import SymmetryGroup
from symm_basis.symm_basis import SymmBasis

# ══════════════════════════════════════════════════════════════════════════════
# LATTICE
# ══════════════════════════════════════════════════════════════════════════════
Lx, Ly   = 4, 4
nbx, nby = 2, 2
N        = Lx * Ly

lattice     = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=nbx, nby=nby)
nat_to_mps  = lattice.nat_to_mps
mps_to_nat  = lattice.mps_to_nat

# ══════════════════════════════════════════════════════════════════════════════
# QUANTUM NUMBERS  ← edit here
# ══════════════════════════════════════════════════════════════════════════════
kx   = 0    # x-momentum quantum number (integer in [0, Lx))
ky   = 0    # y-momentum quantum number (integer in [0, Ly))
px   = 1    # mirror-x eigenvalue  (+1 or -1)
py   = 1    # mirror-y eigenvalue  (+1 or -1)
rr   = 0    # rotation quantum number (integer in [0, nRot))
nRot = 4    # rotation group order
sgm1 = 1    # main-diagonal mirror eigenvalue  (+1 or -1)
sgm2 = 1    # anti-diagonal mirror eigenvalue  (+1 or -1)
zz   = 1    # spin-inversion eigenvalue (+1, -1, or 0 to disable)
sec  = 0    # Sz sector: nup = (N + 2*sec) / 2

# Active symmetry generators
use_Tx   = True
use_Ty   = True
use_Rot  = True
use_Mrrx = True
use_Mrry = True
use_Dia1 = True
use_Dia2 = True

# ══════════════════════════════════════════════════════════════════════════════
# BUILD GROUP AND BASIS
# ══════════════════════════════════════════════════════════════════════════════
nup = (N + 2 * sec) // 2
assert nup * 2 == N + 2 * sec, "sec must give an integer nup"

print("=" * 65)
print(f"  4×4 lattice,  patch {nbx}×{nby},  N={N},  nup={nup}  (sec={sec})")
print(f"  kx={kx} ky={ky}  px={px} py={py}  rr={rr}/{nRot}  "
      f"sgm1={sgm1} sgm2={sgm2}  zz={zz}")
active = [name for name, flag in [
    ('Tx',use_Tx),('Ty',use_Ty),('Rot',use_Rot),
    ('Mrrx',use_Mrrx),('Mrry',use_Mrry),('Dia1',use_Dia1),('Dia2',use_Dia2)
] if flag]
print(f"  Active generators: {', '.join(active)}")
print("=" * 65)

group = SymmetryGroup(
    Lx=Lx, Ly=Ly,
    nat_to_mps=nat_to_mps, mps_to_nat=mps_to_nat,
    kx=kx, ky=ky,
    px=px, py=py,
    rr=rr, nRot=nRot,
    sgm1=sgm1, sgm2=sgm2,
    use_Tx=use_Tx, use_Ty=use_Ty, use_Rot=use_Rot,
    use_Mrrx=use_Mrrx, use_Mrry=use_Mrry,
    use_Dia1=use_Dia1, use_Dia2=use_Dia2,
)
print(f"\n  Spatial group size (after dedup): {group.ntrans}")
print(f"  Extended orbit size (with zz):    "
      f"{'disabled' if zz == 0 else group.ntrans * 2}")

basis = SymmBasis(group=group, N=N, nup=nup, zz=zz)

# ── enumerate ──────────────────────────────────────────────────────────────────
print(f"\n  Enumerating basis (iterating all 2^{N} = {2**N} configs)...")
repr_ints, norms = basis.enumerate_basis()
D = repr_ints.shape[0]
print(f"  Basis size D = {D}")

# Reference: number of Sz-sector states before symmetry projection
from math import comb
n_sz = comb(N, nup)
print(f"  Sz-sector size (before projection): C({N},{nup}) = {n_sz}")
print(f"  Compression ratio: {n_sz}/{D} = {n_sz/D:.2f}×")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT BASIS STATES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'idx':>5}  {'repr_int':>12}  {'norm N_R':>10}"
      f"  {'config (MPS order)':>{N+2}}  config (row,col grid)")
print(f"  {'─'*80}")

for i in range(D):
    r_int  = int(repr_ints[i].item())
    norm   = float(norms[i].item())
    config = basis.decode_repr(r_int)           # [N] in MPS order

    # MPS-order bit string
    mps_str = ''.join(str(int(config[j].item())) for j in range(N))

    # Natural-order grid: show spins at each (row,col)
    config_nat = config[mps_to_nat]             # [N] in natural order
    rows = []
    for r in range(Ly):
        row_str = ''.join(
            ('↑' if config_nat[c + Lx * r].item() == 1 else '↓')
            for c in range(Lx)
        )
        rows.append(row_str)
    grid_str = '|'.join(rows)                   # one row per bar-separated block

    print(f"  {i:>5}  {r_int:>12}  {norm:>10.4f}  {mps_str:>{N+2}}  {grid_str}")

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Verification:")

# (1) Every printed representative must be its own representative
print(f"  (1) Checking each state is its own representative...", end=' ')
configs_all = torch.stack([basis.decode_repr(int(repr_ints[i])) for i in range(D)])
repr_c, _, norms_check = basis.find_representative(configs_all)
all_self = (configs_all == repr_c).all(dim=1).all()
norms_ok  = (torch.abs(norms_check - norms) < 1e-8).all()
assert all_self, "Some states are not their own representative!"
assert norms_ok,  "Norm mismatch between enumeration and find_representative!"
print("OK")

# (2) No two representatives are the same
print(f"  (2) Checking all representatives are distinct...", end=' ')
unique_reprs = torch.unique(repr_ints)
assert unique_reprs.shape[0] == D, "Duplicate representatives found!"
print("OK")

# (3) Check norm sum: Σ N_R should equal n_sz / |G'|
# (Burnside's lemma: D = (1/|G'|) Σ_{σ∈G'} |Fix(σ)|, weighted by characters)
print(f"  (3) Sum of norms Σ N_R = {norms.sum().item():.4f}")
print(f"      Expected ~ n_sz / ntrans_ext = "
      f"{n_sz} / {basis.ntrans_ext} = {n_sz / basis.ntrans_ext:.4f}")

print(f"\n  Done.")
