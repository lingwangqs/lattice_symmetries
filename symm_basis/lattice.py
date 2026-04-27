"""
lattice.py — Lattice geometry: site coordinates, bond tables, and MPS site orderings.

Design
------
A Lattice object provides two things:
  (a) A bond table: the list of (i, j) site-index pairs that appear in the
      Hamiltonian.  These indices are always in *MPS order* (see below).
  (b) A site ordering: a bijection between natural (row, col) coordinates and
      the 1D index used by the MPS.

For a 1D chain the two orderings coincide.  For a 2D lattice the MPS ordering
is a patch/super-patch blocking that keeps spatially nearby sites close in the
MPS chain, reducing entanglement across cuts.

MPS ordering for 2D (patch/super-patch)
-----------------------------------------
The Lx×Ly square lattice is divided into (Lx//nbx)×(Ly//nby) super-patches,
each containing nbx×nby sites.  Within every super-patch, sites are indexed in
column-major order (column varies fastest).  Super-patches are also indexed in
column-major order.

  natural index : i_nat = col + Lx * row    (row ∈ [0,Ly), col ∈ [0,Lx))
  MPS index     : determined by build_patch_ordering()

The site ordering is stored as two tensors:
  nat_to_mps[i_nat] = i_mps   (natural → MPS)
  mps_to_nat[i_mps] = i_nat   (MPS     → natural)

The SymmetryGroup applies spatial symmetries in *natural* coordinates and then
converts back to MPS order using these tensors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _identity_ordering(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return trivial (identity) nat_to_mps / mps_to_nat for a 1D chain."""
    idx = torch.arange(N, dtype=torch.long)
    return idx.clone(), idx.clone()


def build_patch_ordering(
    Lx: int, Ly: int, nbx: int, nby: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the patch/super-patch MPS site ordering for an Lx×Ly square lattice.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.  Must satisfy Lx % nbx == 0 and Ly % nby == 0.
        Currently requires Lx == Ly and nbx == nby (square patches).
    nbx, nby : int
        Patch size along x and y.  Each super-patch contains nbx×nby sites.

    Returns
    -------
    nat_to_mps : LongTensor of shape [Lx*Ly]
        nat_to_mps[i_nat] = i_mps.
    mps_to_nat : LongTensor of shape [Lx*Ly]
        mps_to_nat[i_mps] = i_nat  (inverse permutation).
    """
    assert Lx % nbx == 0 and Ly % nby == 0, \
        "Patch size must divide lattice size exactly."
    assert Lx == Ly and nbx == nby, \
        "Currently only square lattices with square patches are supported."

    N = Lx * Ly
    NB = Lx // nbx   # number of super-patches per direction
    nb = nbx          # sites per patch per direction

    # within-patch index (column-major): col + row*nb, for col,row ∈ [0, nb)
    row_patch = torch.arange(nb, dtype=torch.long).reshape(1, nb)  # [1, nb]
    col_patch = torch.arange(nb, dtype=torch.long).reshape(nb, 1)  # [nb, 1]
    patch_local = col_patch + nb * row_patch                        # [nb, nb]
    # Tile across all NB×NB super-patches and reshape to [NB, nb, NB, nb]:
    # axes mean [super_row, within_row, super_col, within_col]
    patch_local = torch.tile(patch_local, (NB, NB)).reshape(NB, nb, NB, nb)

    # super-patch offset: each super-patch contributes nb² consecutive MPS sites
    Row_super = torch.arange(NB, dtype=torch.long).reshape(1, NB)  # [1, NB]
    Col_super = torch.arange(NB, dtype=torch.long).reshape(NB, 1)  # [NB, 1]
    # super-patch index in column-major order: Row + Col*NB, scaled by patch size²
    patch_offset = (Row_super + NB * Col_super) * (nb * nb)         # [NB, NB]
    # Broadcast to [NB, nb, NB, nb] and permute to [NB, nb, NB, nb] → [super_row, within_row, super_col, within_col]
    patch_offset = torch.tile(patch_offset.reshape(NB, NB, 1, 1), (1, 1, nb, nb))
    patch_offset = patch_offset.permute(0, 2, 1, 3)                 # [NB, nb, NB, nb]

    # MPS index of each (super_row, within_row, super_col, within_col)
    mps_index = (patch_local + patch_offset).reshape(N)             # [N]

    # The natural index of the same site is col + Lx * row, where
    # col = super_col * nb + within_col, row = super_row * nb + within_row.
    # The flat layout of the loop above visits sites in the order:
    # (super_row=0..NB, within_row=0..nb, super_col=0..NB, within_col=0..nb)
    # which corresponds to natural index:
    nat_col = (torch.arange(NB, dtype=torch.long).reshape(NB, 1, 1, 1) * nb
               + torch.arange(nb, dtype=torch.long).reshape(1, 1, 1, nb))   # [NB,1,1,nb]
    nat_row = (torch.arange(NB, dtype=torch.long).reshape(1, 1, NB, 1) * nb
               + torch.arange(nb, dtype=torch.long).reshape(1, nb, 1, 1))   # [1,nb,NB,1]
    nat_index = (nat_col + Lx * nat_row).reshape(N)                 # [N]

    # Build the permutation arrays
    nat_to_mps = torch.empty(N, dtype=torch.long)
    nat_to_mps[nat_index] = mps_index                               # nat → mps
    mps_to_nat = torch.argsort(nat_to_mps)                         # mps → nat (inverse)

    return nat_to_mps, mps_to_nat


# ──────────────────────────────────────────────────────────────────────────────
# 1D chain
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chain1D:
    """
    1D chain with L sites and periodic boundary conditions.

    Site ordering: the natural order 0, 1, ..., L-1 is also the MPS order.

    Attributes
    ----------
    L : int
        Number of sites.
    J2_coupling : bool
        If True, include next-nearest-neighbor bonds (distance-2) in addition
        to nearest-neighbor bonds.  Controlled by whether J2 ≠ 0 in the
        Hamiltonian, but the bond table is built here.
    """
    L: int
    J2_coupling: bool = False

    # Site ordering: trivial for 1D.
    nat_to_mps: torch.Tensor = field(init=False, repr=False)
    mps_to_nat: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nat_to_mps, self.mps_to_nat = _identity_ordering(self.L)

    @property
    def N(self) -> int:
        """Total number of sites."""
        return self.L

    def nn_bonds(self) -> torch.Tensor:
        """
        Nearest-neighbor bond table in MPS order.

        Returns
        -------
        bonds : LongTensor of shape [L, 2]
            bonds[b] = (i, j) with periodic boundary: j = (i+1) % L.
        """
        i = torch.arange(self.L, dtype=torch.long)
        j = (i + 1) % self.L
        return torch.stack([i, j], dim=1)

    def nnn_bonds(self) -> torch.Tensor:
        """
        Next-nearest-neighbor bond table (distance 2, periodic).

        Returns
        -------
        bonds : LongTensor of shape [L, 2]
        """
        i = torch.arange(self.L, dtype=torch.long)
        j = (i + 2) % self.L
        return torch.stack([i, j], dim=1)

    def all_bonds(self) -> torch.Tensor:
        """
        Concatenation of NN (and optionally NNN) bond tables.

        Returns
        -------
        bonds : LongTensor of shape [nb, 2]
            nb = L if J2_coupling is False, 2L otherwise.
        """
        if self.J2_coupling:
            return torch.cat([self.nn_bonds(), self.nnn_bonds()], dim=0)
        return self.nn_bonds()


# ──────────────────────────────────────────────────────────────────────────────
# 2D square lattice
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SquareLattice2D:
    """
    Square Lx×Ly lattice with periodic boundary conditions and an MPS-friendly
    patch/super-patch site ordering.

    Natural coordinates: site (row, col) has natural index col + Lx * row.
    MPS ordering       : determined by nbx×nby patch blocking (see lattice module
                         docstring).

    Attributes
    ----------
    Lx, Ly : int
        Lattice dimensions.  Both must equal the same value (square lattice).
    nbx, nby : int
        Patch size.  Must divide Lx and Ly respectively (and currently nbx==nby).
    include_diagonal : bool
        If True, add diagonal (J2) bonds in addition to NN bonds.
    """
    Lx: int
    Ly: int
    nbx: int = 1
    nby: int = 1
    include_diagonal: bool = True

    nat_to_mps: torch.Tensor = field(init=False, repr=False)
    mps_to_nat: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.Lx == self.Ly, "Only square lattices (Lx == Ly) are supported."
        assert self.nbx == self.nby, "Only square patches (nbx == nby) are supported."
        if self.nbx > 1:
            self.nat_to_mps, self.mps_to_nat = build_patch_ordering(
                self.Lx, self.Ly, self.nbx, self.nby
            )
        else:
            self.nat_to_mps, self.mps_to_nat = _identity_ordering(self.N)

    @property
    def N(self) -> int:
        """Total number of sites."""
        return self.Lx * self.Ly

    def _nat_index(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Convert (row, col) tensors to natural indices."""
        return col + self.Lx * row

    def _mps_bonds_from_nat(
        self,
        i_nat: torch.Tensor,
        j_nat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a list of bonds specified in natural coordinates to MPS-order
        indices and stack into a [nb, 2] bond table.
        """
        i_mps = self.nat_to_mps[i_nat]
        j_mps = self.nat_to_mps[j_nat]
        return torch.stack([i_mps, j_mps], dim=1)

    def nn_bonds(self) -> torch.Tensor:
        """
        Nearest-neighbor bond table in MPS order, periodic boundaries.

        Returns
        -------
        bonds : LongTensor of shape [2*N, 2]
            First N bonds: horizontal (x-direction).
            Next  N bonds: vertical   (y-direction).
        """
        row = torch.arange(self.Ly, dtype=torch.long).reshape(self.Ly, 1)
        col = torch.arange(self.Lx, dtype=torch.long).reshape(1, self.Lx)

        # Horizontal bonds: (row, col) — (row, col+1)
        i_nat_h = self._nat_index(row, col).reshape(-1)
        j_nat_h = self._nat_index(row, (col + 1) % self.Lx).reshape(-1)

        # Vertical bonds: (row, col) — (row+1, col)
        i_nat_v = self._nat_index(row, col).reshape(-1)
        j_nat_v = self._nat_index((row + 1) % self.Ly, col).reshape(-1)

        h_bonds = self._mps_bonds_from_nat(i_nat_h, j_nat_h)
        v_bonds = self._mps_bonds_from_nat(i_nat_v, j_nat_v)
        return torch.cat([h_bonds, v_bonds], dim=0)

    def diagonal_bonds(self) -> torch.Tensor:
        """
        Diagonal bond table (next-nearest-neighbor, two diagonals) in MPS order.

        Returns
        -------
        bonds : LongTensor of shape [2*N, 2]
            First N bonds : (+1,+1) diagonal.
            Next  N bonds : (-1,+1) diagonal (anti-diagonal).
        """
        row = torch.arange(self.Ly, dtype=torch.long).reshape(self.Ly, 1)
        col = torch.arange(self.Lx, dtype=torch.long).reshape(1, self.Lx)

        # (+row+1, +col+1) diagonal
        i_nat_d1 = self._nat_index(row, col).reshape(-1)
        j_nat_d1 = self._nat_index((row + 1) % self.Ly, (col + 1) % self.Lx).reshape(-1)

        # (+row+1, -col+1) anti-diagonal
        i_nat_d2 = self._nat_index(row, col).reshape(-1)
        j_nat_d2 = self._nat_index((row + 1) % self.Ly, (col - 1) % self.Lx).reshape(-1)

        d1_bonds = self._mps_bonds_from_nat(i_nat_d1, j_nat_d1)
        d2_bonds = self._mps_bonds_from_nat(i_nat_d2, j_nat_d2)
        return torch.cat([d1_bonds, d2_bonds], dim=0)

    def all_bonds(self) -> torch.Tensor:
        """
        Concatenation of NN (and optionally diagonal) bond tables in MPS order.

        Returns
        -------
        bonds : LongTensor of shape [nb, 2]
        """
        if self.include_diagonal:
            return torch.cat([self.nn_bonds(), self.diagonal_bonds()], dim=0)
        return self.nn_bonds()
