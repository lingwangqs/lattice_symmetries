"""
symm_basis — Symmetric basis library for quantum spin systems.

Provides group-theoretic (momentum / symmetry-sector) basis construction
for use with variational Monte Carlo and exact diagonalization.  The
amplitude of a basis state is evaluated by an external callable (neural
network, MPS, etc.), making the library agnostic about the ansatz.

Typical workflow
----------------
1. Build a lattice (geometry + MPS site ordering).
2. Build a SymmetryGroup (spatial generators + quantum numbers).
3. Build a SymmBasis (adds spin-inversion, representative logic, norm).
4. Build a Hamiltonian (bond couplings + matrix elements in symmetric basis).
5. Use SymmBasis.log_amplitude() + Hamiltonian.local_energy() in a VMC loop,
   or Hamiltonian.matrix_vector_product() in a Lanczos solver.

Public API
----------
  lattice       : Chain1D, SquareLattice2D
  symmetry_group: SymmetryGroup
  symm_basis    : SymmBasis
  hamiltonian   : HeisenbergHamiltonian
  io            : save_basis, load_basis
"""

from .lattice import Chain1D, SquareLattice2D
from .symmetry_group import SymmetryGroup
from .symm_basis import SymmBasis
from .hamiltonian import HeisenbergHamiltonian
from .io import save_basis, load_basis
from .lanczos import lanczos, LanczosResult

__all__ = [
    "Chain1D",
    "SquareLattice2D",
    "SymmetryGroup",
    "SymmBasis",
    "HeisenbergHamiltonian",
    "save_basis",
    "load_basis",
    "lanczos",
    "LanczosResult",
]
