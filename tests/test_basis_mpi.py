"""
test_basis_mpi.py — Multi-process generation of the complete symmetric basis
for a 4×4 2D system using torch.distributed (gloo backend).

The 2^N configuration space is partitioned across ranks in a round-robin
pattern.  Each rank processes its chunk independently; rank 0 gathers and
broadcasts the final sorted basis.

Usage (4 processes via torchrun):
    torchrun --nproc_per_node=4 test_basis_mpi.py

torchrun sets the RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT environment
variables that torch.distributed/gloo requires.  Plain mpirun does not set
these, so torchrun is the correct launcher here.

The quantum numbers and active generators are identical to test_basis.py so
the two scripts can be used to cross-check each other.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import timedelta
import torch
import torch.distributed as dist

from symm_basis.lattice import SquareLattice2D
from symm_basis.symmetry_group import SymmetryGroup
from symm_basis.symm_basis import SymmBasis

# ══════════════════════════════════════════════════════════════════════════════
# LATTICE
# ══════════════════════════════════════════════════════════════════════════════
Lx, Ly   = 4, 4
nbx, nby = 2, 2
N        = Lx * Ly

# ══════════════════════════════════════════════════════════════════════════════
# QUANTUM NUMBERS  ← edit here (must match test_basis.py for cross-check)
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
sec  = 0

use_Tx   = True
use_Ty   = True
use_Rot  = True
use_Mrrx = True
use_Mrry = True
use_Dia1 = True
use_Dia2 = True


def main():
    # ── MPI initialisation ────────────────────────────────────────────────────
    dist.init_process_group("gloo", timeout=timedelta(seconds=1800000))
    psize  = dist.get_world_size()
    myrank = dist.get_rank()

    # Each rank announces itself in order (useful for debugging)
    for i in range(psize):
        if myrank == i:
            print(f"[rank {myrank}/{psize}] process started", flush=True)
        dist.barrier()

    # ── Build lattice, group, basis (identical on all ranks) ──────────────────
    lattice    = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=nbx, nby=nby)
    nat_to_mps = lattice.nat_to_mps
    mps_to_nat = lattice.mps_to_nat
    nup        = (N + 2 * sec) // 2

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

    basis = SymmBasis(group=group, N=N, nup=nup, zz=zz)

    if myrank == 0:
        active = [name for name, flag in [
            ('Tx',use_Tx),('Ty',use_Ty),('Rot',use_Rot),
            ('Mrrx',use_Mrrx),('Mrry',use_Mrry),('Dia1',use_Dia1),('Dia2',use_Dia2)
        ] if flag]
        print("=" * 65)
        print(f"  4×4 lattice,  patch {nbx}×{nby},  N={N},  nup={nup}  (sec={sec})")
        print(f"  kx={kx} ky={ky}  px={px} py={py}  rr={rr}/{nRot}  "
              f"sgm1={sgm1} sgm2={sgm2}  zz={zz}")
        print(f"  Active generators: {', '.join(active)}")
        print(f"  MPI ranks: {psize}")
        print(f"  Spatial group size: {group.ntrans}")
        print(f"  Extended orbit size (with zz): {basis.ntrans_ext}")
        print("=" * 65)
        print(f"\n  Enumerating basis across {psize} MPI ranks ...", flush=True)

    dist.barrier()

    # ── MPI-parallel enumeration ───────────────────────────────────────────────
    repr_ints, norms = basis.enumerate_basis_mpi(dist)
    D = repr_ints.shape[0]

    # ── Print results from rank 0 ──────────────────────────────────────────────
    if myrank == 0:
        from math import comb
        n_sz = comb(N, nup)

        print(f"  Basis size D = {D}")
        print(f"  Sz-sector size (before projection): C({N},{nup}) = {n_sz}")
        print(f"  Compression ratio: {n_sz}/{D} = {n_sz/D:.2f}×")

        print(f"\n  {'idx':>5}  {'repr_int':>12}  {'norm N_R':>10}"
              f"  {'config (MPS order)':>{N+2}}  config (row,col grid)")
        print(f"  {'─'*80}")

        for i in range(D):
            r_int  = int(repr_ints[i].item())
            norm   = float(norms[i].item())
            config = basis.decode_repr(r_int)           # [N] in MPS order

            mps_str = ''.join(str(int(config[j].item())) for j in range(N))

            config_nat = config[mps_to_nat]
            rows = []
            for r in range(Ly):
                row_str = ''.join(
                    ('↑' if config_nat[c + Lx * r].item() == 1 else '↓')
                    for c in range(Lx)
                )
                rows.append(row_str)
            grid_str = '|'.join(rows)

            print(f"  {i:>5}  {r_int:>12}  {norm:>10.4f}  {mps_str:>{N+2}}  {grid_str}")

        # ── Verification ──────────────────────────────────────────────────────
        print(f"\n  Verification (on rank 0):")

        # (1) Every representative must be its own representative
        print(f"  (1) Checking each state is its own representative ...", end=' ')
        configs_all = torch.stack([basis.decode_repr(int(repr_ints[i])) for i in range(D)])
        repr_c, _, norms_check = basis.find_representative(configs_all)
        assert (configs_all == repr_c).all(dim=1).all(), \
            "Some states are not their own representative!"
        assert (torch.abs(norms_check - norms) < 1e-8).all(), \
            "Norm mismatch between enumeration and find_representative!"
        print("OK")

        # (2) No duplicates
        print(f"  (2) Checking all representatives are distinct ...", end=' ')
        assert torch.unique(repr_ints).shape[0] == D, "Duplicate representatives!"
        print("OK")

        # (3) Norm sum
        print(f"  (3) Sum of norms Σ N_R = {norms.sum().item():.4f}")
        print(f"      Expected ~ n_sz / ntrans_ext = "
              f"{n_sz} / {basis.ntrans_ext} = {n_sz / basis.ntrans_ext:.4f}")

        print(f"\n  Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
