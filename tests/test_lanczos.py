"""
test_lanczos.py — Run the Lanczos eigensolver on the 4×4 Heisenberg model
in the kx=ky=0, all-symmetric sector (same quantum numbers as test_basis.py).

Known benchmark (J1=1 nearest-neighbour antiferromagnet, 4×4 PBC):
  Ground-state energy per site:  E_0 / N ≈ −0.70178
  Total E_0 ≈ -11.2284832084   (sector-dependent; the true ground state lives in
  the (kx,ky)=(0,0), px=py=sgm1=sgm2=1 sector for the 4×4 PBC system)

Run from gen_mps/:
    python test_lanczos.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from symm_basis.lattice import SquareLattice2D
from symm_basis.symmetry_group import SymmetryGroup
from symm_basis.symm_basis import SymmBasis
from symm_basis.hamiltonian import HeisenbergHamiltonian
from symm_basis.lanczos import lanczos

# ══════════════════════════════════════════════════════════════════════════════
# LATTICE
# ══════════════════════════════════════════════════════════════════════════════
Lx, Ly   = 4, 4
nbx, nby = 2, 2
N        = Lx * Ly

lattice    = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=nbx, nby=nby)
nat_to_mps = lattice.nat_to_mps
mps_to_nat = lattice.mps_to_nat

# ══════════════════════════════════════════════════════════════════════════════
# QUANTUM NUMBERS  (trivial sector — contains the true ground state)
# ══════════════════════════════════════════════════════════════════════════════
kx   = 0
ky   = 0
px   = 1
py   = 1
rr   = 0
nRot = 4
sgm1 = 1
sgm2 = 1
zz   = 1
sec  = 0    # half filling: nup = N//2

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

group = SymmetryGroup(
    Lx=Lx, Ly=Ly,
    nat_to_mps=nat_to_mps, mps_to_nat=mps_to_nat,
    kx=kx, ky=ky,
    px=px, py=py,
    rr=rr, nRot=nRot,
    sgm1=sgm1, sgm2=sgm2,
    use_Tx=use_Tx, use_Ty=use_Ty, use_Rot=use_Rot,
    use_Mrrx=use_Mrrx, use_Mrry=use_Mrry, use_Dia1=use_Dia1, use_Dia2=use_Dia2,
)

basis = SymmBasis(group=group, N=N, nup=nup, zz=zz)

print("=" * 65)
print(f"  4×4 Heisenberg model,  J1=1  (antiferromagnet)")
print(f"  Sector: kx={kx} ky={ky}  px={px} py={py}  rr={rr}/{nRot}  "
      f"sgm1={sgm1} sgm2={sgm2}  zz={zz}")
print("=" * 65)

print(f"\n  Enumerating basis ...")
repr_ints, norms = basis.enumerate_basis()
D = repr_ints.shape[0]
print(f"  Basis dimension D = {D}")

# ══════════════════════════════════════════════════════════════════════════════
# HAMILTONIAN  (nearest-neighbour J1 = 1)
# ══════════════════════════════════════════════════════════════════════════════
# all_bonds returns bonds in MPS-site order; use only NN bonds (J1 = 1)
nn_bonds = lattice.nn_bonds()   # [nb, 2] in MPS indices
J1 = 1.0
J_per_bond = torch.full((nn_bonds.shape[0],), J1, dtype=torch.float64)

ham = HeisenbergHamiltonian(bonds=nn_bonds, J_per_bond=J_per_bond, basis=basis)

print(f"  Number of NN bonds: {nn_bonds.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# LANCZOS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Running Lanczos (serial, maxlan=300, n_eig=3, tol=1e-10) ...\n")

result = lanczos(
    hamiltonian=ham,
    repr_ints=repr_ints,
    norms=norms,
    maxlan=300,
    n_eig=3,
    tol=1e-10,
    dist=None,
    verbose=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print(f"  Lanczos finished:")
print(f"    converged : {result.converged}")
print(f"    steps     : {result.n_steps}")
print(f"    E_0       : {result.eigenvalues[0].item():.10f}")
print(f"    E_0 / N   : {result.eigenvalues[0].item() / N:.10f}")
if result.eigenvalues.shape[0] > 1:
    print(f"    E_1       : {result.eigenvalues[1].item():.10f}")
if result.eigenvalues.shape[0] > 2:
    print(f"    E_2       : {result.eigenvalues[2].item():.10f}")
print()

# ── Verify ground state is normalised ────────────────────────────────────────
gs_norm = torch.sqrt((result.eigenvector.conj() * result.eigenvector).real.sum()).item()
print(f"  ||ψ_0||   : {gs_norm:.10f}  (should be 1.0)")

# ── Verify energy via <ψ|H|ψ> directly ──────────────────────────────────────
w = ham.matrix_vector_product(result.eigenvector, repr_ints, norms, dist=None)
energy_direct = (result.eigenvector.conj() * w).real.sum().item()
print(f"  <ψ|H|ψ>  : {energy_direct:.10f}  (should match E_0)")

# ── Benchmark reference ───────────────────────────────────────────────────────
# For 4×4 PBC isotropic Heisenberg (J=1), the exact ground-state energy is
# E_0 ≈ -11.2284832...  (E_0/N ≈ -0.7018)
# This lives in the (kx,ky)=(0,0), all-symmetric sector computed here.
E_ref = -11.2284832
print()
print(f"  Reference E_0 ≈ {E_ref}  (4×4 PBC, J=1, exact)")
print(f"  Difference    : {abs(energy_direct - E_ref):.2e}")
print("=" * 65)
