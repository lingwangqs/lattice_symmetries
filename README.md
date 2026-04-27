# lattice_symmetries

A Python library for group-theoretic symmetric basis construction for quantum spin-1/2 systems, supporting exact diagonalization (ED) across arbitrary symmetry sectors and — coming soon — symmetric-basis-adapted neural-network variational optimization.

## Overview

For a lattice Hamiltonian with a spatial symmetry group G, every energy eigenstate belongs to an irreducible representation (irrep) labeled by a set of quantum numbers (momentum, mirror parity, rotation, spin-inversion eigenvalue, ...).  Working in a single irrep drastically reduces the Hilbert space dimension compared to the full computational basis, making ED tractable for much larger systems.

This library constructs the **symmetric basis** for spin-1/2 systems from first principles:

1. Build the full symmetry group G by composing elementary generators (translations, C4 rotation, mirrors).
2. For each generator combination, compute the complex character χ(g) of the target irrep.
3. Identify **representative configurations**: for each orbit {g|R⟩ : g ∈ G}, the lexicographically smallest configuration.
4. Compute the **basis norm** N_R = |Σ_{g: g|R⟩=|R⟩} χ(g)|; states with N_R = 0 are excluded.
5. Express the Hamiltonian as a sparse matrix in the representative basis and diagonalize with Lanczos.

## Supported symmetries

| Generator | Description | Quantum number |
|-----------|-------------|----------------|
| Tx | Translation in x | kx ∈ [0, Lx) |
| Ty | Translation in y | ky ∈ [0, Ly) |
| Rot (C4) | 90° rotation (square lattice) | rr ∈ {0,1,2,3} |
| Mrrx | Mirror reflection about vertical axis | px ∈ {+1, −1} |
| Mrry | Mirror reflection about horizontal axis | py ∈ {+1, −1} |
| Dia1 | Main-diagonal mirror (row↔col) | sgm1 ∈ {+1, −1} |
| Dia2 | Anti-diagonal mirror | sgm2 ∈ {+1, −1} |
| Spin-inversion | Flip all spins | zz ∈ {+1, −1, 0} |
| Sz conservation | Total magnetization | sec (nup = N/2 + sec) |

Each generator can be toggled independently.  For 1D chains, only Tx (and optionally Mrrx and spin-inversion) are relevant.

## Directory structure

```
lattice_symmetries/
├── symm_basis/             # Main library package
│   ├── __init__.py         # Public API re-exports
│   ├── lattice.py          # Lattice geometry: Chain1D, SquareLattice2D
│   ├── symmetry_group.py   # SymmetryGroup: permutation tables and characters
│   ├── symm_basis.py       # SymmBasis: representative finding, norm, enumeration
│   ├── hamiltonian.py      # HeisenbergHamiltonian: matrix elements in symmetric basis
│   ├── lanczos.py          # Lanczos eigensolver (serial and MPI-distributed)
│   └── io.py               # Disk I/O for caching the enumerated basis
└── tests/
    ├── test_lattice.py     # Visualize 2D patch/super-patch site ordering
    ├── test_group.py       # Print all generator maps and the full group table
    ├── test_basis.py       # Enumerate and print the symmetric basis
    ├── test_basis_mpi.py   # MPI-parallel basis enumeration (torchrun)
    └── test_lanczos.py     # Full ED demo: 4×4 Heisenberg ground state
```

## Module descriptions

### `lattice.py` — Lattice geometry

Provides the bond table and the **MPS site ordering** for use with tensor-network variational methods.

- **`Chain1D(L)`** — 1D periodic chain.  Builds nearest-neighbor (`nn_bonds`) and next-nearest-neighbor (`nnn_bonds`) bond tables.
- **`SquareLattice2D(Lx, Ly, nbx, nby)`** — 2D square lattice with periodic boundary conditions.  The site ordering uses a **patch/super-patch** blocking: the Lx×Ly lattice is divided into (Lx/nbx)×(Ly/nby) super-patches, each of size nbx×nby, so that spatially nearby sites are close in the MPS chain.  Builds NN and diagonal bond tables.

```
SquareLattice2D attributes:
  .nat_to_mps   LongTensor [N]   natural index → MPS index
  .mps_to_nat   LongTensor [N]   MPS index → natural index
  .nn_bonds()   LongTensor [2N, 2]  nearest-neighbor bonds in MPS order
```

### `symmetry_group.py` — Symmetry group

**`SymmetryGroup`** composes the active generators and stores two precomputed tables:

```
.mapping     LongTensor [ntrans, N]   site permutation for each group element
.transtable  ComplexTensor [ntrans]   character χ(g_α) for each group element
.ntrans      int                      number of distinct group elements
```

The character is:

```
χ(g) = exp(-i(kx·tx·2π/Lx + ky·ty·2π/Ly + rr·rot·2π/nRot))
       × px^mx × py^my × sgm1^d1 × sgm2^d2
```

Duplicate permutations (from redundant generator combinations) are automatically removed.  Incompatible quantum numbers raise a `ValueError` with a diagnostic message.

### `symm_basis.py` — Symmetric basis

**`SymmBasis`** wraps a `SymmetryGroup` and provides:

| Method | Description |
|--------|-------------|
| `find_representative(configs)` | Map each config to its canonical (lex-min) representative and compute N_R |
| `is_representative(config)` | Check if a config is its own representative |
| `norm(config)` | Return N_R for a single config |
| `enumerate_basis()` | Return all valid representatives and norms (serial, N ≤ ~28) |
| `enumerate_basis_mpi(dist)` | Same, distributed across MPI ranks (N up to ~40) |
| `log_amplitude(config, amplitude_fn)` | Evaluate log\|ψ(R; χ)\| from an external amplitude model |
| `decode_repr(int)` | Integer → binary config tensor |
| `encode_config(config)` | Binary config tensor → integer |

Spin inversion (zz ≠ 0) doubles the orbit: both g|R⟩ and g(σ_inv|R⟩) are considered.

### `hamiltonian.py` — Heisenberg Hamiltonian

**`HeisenbergHamiltonian`** implements H = Σ J_b S_i·S_j in the symmetric basis:

| Method | Description |
|--------|-------------|
| `connected_elements(config, norm)` | All (R', H_{R'R}) off-diagonal matrix elements |
| `diagonal_element(config)` | Sz·Sz diagonal contribution |
| `matrix_vector_product(v, repr_ints, norms)` | Sparse H·v for Lanczos/ED (serial or MPI) |
| `local_energy(config, norm, log_amp, amplitude_fn)` | Local energy E_loc for VMC |

### `lanczos.py` — Eigensolver

**`lanczos(...)`** runs the Lanczos algorithm with full re-orthogonalization (Gram-Schmidt) against all previous vectors, preventing ghost states from floating-point loss of orthogonality.  The tridiagonal system is solved at each step with LAPACK `dstev` (via scipy).

```python
result = lanczos(hamiltonian, repr_ints, norms, maxlan=300, n_eig=3, tol=1e-10)
result.eigenvalues   # FloatTensor [n_eig] — lowest Ritz values
result.eigenvector   # ComplexTensor [D]   — ground-state vector
result.converged     # bool
result.n_steps       # int
```

MPI-distributed H·v is supported by passing a `dist` (torch.distributed) handle.

### `io.py` — Basis caching

Enumerating the basis for a large system is expensive and should be done once.  `save_basis` / `load_basis` cache `repr_ints` and `norms` to binary files named by all quantum numbers.

```python
from symm_basis.io import save_basis, load_basis
save_basis(repr_ints, norms, params, basis_dir="./basis")
repr_ints, norms = load_basis(params, basis_dir="./basis")
```

## Installation and dependencies

```bash
pip install torch numpy scipy
```

No compiled extensions are required.  PyTorch is used for all tensor operations; scipy provides the LAPACK `dstev` tridiagonal eigensolver.

## Example: ground state of the 4×4 Heisenberg model

The 2D spin-1/2 Heisenberg antiferromagnet on a 4×4 square lattice with periodic boundaries has a well-known exact ground-state energy E₀ ≈ −11.2285 (J=1).  The ground state lives in the fully symmetric sector: zero momentum (kx=ky=0), all mirror parities +1, zero rotation quantum number (rr=0, C4), and positive spin-inversion eigenvalue (zz=+1).

The following reproduces this result using `test_lanczos.py`.

### Step 1: build the lattice

```python
import torch
from symm_basis.lattice import SquareLattice2D

Lx, Ly = 4, 4
lattice = SquareLattice2D(Lx=Lx, Ly=Ly, nbx=2, nby=2)
nat_to_mps = lattice.nat_to_mps
mps_to_nat = lattice.mps_to_nat
```

### Step 2: build the symmetry group

```python
from symm_basis.symmetry_group import SymmetryGroup

group = SymmetryGroup(
    Lx=Lx, Ly=Ly,
    nat_to_mps=nat_to_mps, mps_to_nat=mps_to_nat,
    # Momentum sector
    kx=0, ky=0,
    # Mirror parities
    px=1, py=1,
    # C4 rotation quantum number
    rr=0, nRot=4,
    # Diagonal mirror parities
    sgm1=1, sgm2=1,
    # Active generators
    use_Tx=True, use_Ty=True, use_Rot=True,
    use_Mrrx=True, use_Mrry=True, use_Dia1=True, use_Dia2=True,
)
print(f"Group order: {group.ntrans}")   # number of distinct group elements
```

### Step 3: build the symmetric basis

```python
from symm_basis.symm_basis import SymmBasis

N   = Lx * Ly      # 16 sites
nup = N // 2       # half filling (Sz = 0 sector)
zz  = 1            # positive spin-inversion eigenvalue

basis = SymmBasis(group=group, N=N, nup=nup, zz=zz)

repr_ints, norms = basis.enumerate_basis()
D = repr_ints.shape[0]
print(f"Basis dimension D = {D}")
# Expected: D = 232 for this sector of the 4×4 lattice
```

### Step 4: build the Hamiltonian

```python
from symm_basis.hamiltonian import HeisenbergHamiltonian

nn_bonds   = lattice.nn_bonds()                                      # [2N, 2]
J_per_bond = torch.full((nn_bonds.shape[0],), 1.0, dtype=torch.float64)

ham = HeisenbergHamiltonian(bonds=nn_bonds, J_per_bond=J_per_bond, basis=basis)
```

### Step 5: run Lanczos

```python
from symm_basis.lanczos import lanczos

result = lanczos(
    hamiltonian=ham,
    repr_ints=repr_ints,
    norms=norms,
    maxlan=300,
    n_eig=3,
    tol=1e-10,
    dist=None,      # serial; pass a torch.distributed handle for MPI
    verbose=True,
)

print(f"E_0       = {result.eigenvalues[0].item():.10f}")
print(f"E_0 / N   = {result.eigenvalues[0].item() / N:.10f}")
print(f"converged = {result.converged}  (steps = {result.n_steps})")
```

Expected output:

```
E_0       = -11.2284832084
E_0 / N   = -0.7017802005
converged = True  (steps ≈ 60–80)
```

The exact benchmark value is E₀/N ≈ −0.70178 (J=1, 4×4 PBC antiferromagnet).

Run the complete script directly:

```bash
cd lattice_symmetries/tests
python3 test_lanczos.py
```

## Workflow for exact diagonalization

```
Chain1D or SquareLattice2D
         │
         ▼
    SymmetryGroup          ← specify quantum numbers (kx, ky, px, ...)
         │
         ▼
      SymmBasis             ← enumerate_basis() → repr_ints, norms
         │
         ▼
  HeisenbergHamiltonian     ← bond table + J couplings
         │
         ▼
       lanczos()            ← eigenvalues, eigenvector
```

To scan all symmetry sectors, loop over the quantum numbers and call `SymmetryGroup` / `SymmBasis.enumerate_basis()` / `lanczos()` for each sector.

## Coming soon: symmetric neural-network optimization

The `SymmBasis.log_amplitude()` and `HeisenbergHamiltonian.local_energy()` methods provide the VMC interface.  An external **amplitude model** (neural network, MPS, or any callable) maps configurations to log-amplitudes:

```python
def amplitude_fn(configs):   # configs: LongTensor [B, N]
    ...
    return log_amps           # FloatTensor [B]
```

The symmetric-basis-adapted VMC optimizer — which uses this interface to minimize the variational energy while respecting all lattice symmetries — will be released in a future update.

## Tests

Run each test from the `tests/` directory:

```bash
cd lattice_symmetries/tests

python3 test_lattice.py    # print patch ordering for 4×4 lattice
python3 test_group.py      # print all generator maps and the group table
python3 test_basis.py      # enumerate and print the symmetric basis
python3 test_lanczos.py    # full ED of 4×4 Heisenberg model
torchrun --nproc_per_node=4 test_basis_mpi.py   # MPI-parallel basis enumeration
```
