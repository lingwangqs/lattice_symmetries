"""
symmetry_group.py — Spatial symmetry group of a spin lattice.

Physics background
------------------
For a lattice Hamiltonian with spatial symmetry group G = {g_α}, a symmetric
basis state is built by projecting a reference configuration |R⟩ onto the
irreducible representation labelled by a set of quantum numbers (k, p, ...):

    |R; k, p, ...⟩ ∝ Σ_α  χ*(g_α) · g_α |R⟩

where χ(g_α) is the character of the irrep evaluated at g_α.

This module represents the group G as:
  • mapping[α, i]    : site-index permutation for group element α (in MPS order).
  • transtable[α]    : complex phase χ(g_α) = e^{-i φ_α} × (product of discrete
                       eigenvalues for mirror/diagonal generators).

The full set of group elements is built by composing the elementary generators
in a fixed order:
    Tx → Ty → Rot → Mrrx → Mrry → Dia1 → Dia2

Only the generators flagged as active contribute to the composition loop.
Duplicate permutations (which arise when generators are redundant for the given
lattice size) are automatically removed.

Notes
-----
• All permutations are stored in MPS site order.  The Lattice object provides
  nat_to_mps / mps_to_nat for conversion.
• For 1D systems Ty, Rot, Mrry, Dia1, Dia2 are irrelevant — set them to False.
• Rotation (C4) is only valid on square lattices with kx == ky.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Individual generator maps (in natural coordinates)
# ──────────────────────────────────────────────────────────────────────────────

def _make_tx_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Translation-x permutation maps.

    Returns tx_maps[t, i_nat] = j_nat, where j_nat is the natural index of
    the site obtained by translating site i_nat by t steps in the x-direction.

    Shape: [Lx, N]   (t = 0, 1, ..., Lx-1)
    """
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    tx_maps = torch.zeros(Lx, Ly * Lx, dtype=torch.long)
    for t in range(Lx):
        new_col = (col + t) % Lx
        tx_maps[t] = (new_col + Lx * row).reshape(-1)
    return tx_maps


def _make_ty_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Translation-y permutation maps.

    Returns ty_maps[t, i_nat] = j_nat.  Shape: [Ly, N].
    """
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    ty_maps = torch.zeros(Ly, Ly * Lx, dtype=torch.long)
    for t in range(Ly):
        new_row = (row + t) % Ly
        ty_maps[t] = (col + Lx * new_row).reshape(-1)
    return ty_maps


def _make_rot_maps(Lx: int, Ly: int, nRot: int) -> torch.Tensor:
    """
    C_{nRot} rotation permutation maps (currently nRot=4, i.e., C4 symmetry).

    The elementary C4 rotation sends (row, col) → (col, Ly-1-row).
    rot_maps[r] = r applications of the elementary rotation.

    Shape: [nRot, N].  Requires Lx == Ly.
    """
    assert Lx == Ly, "Rotation symmetry requires a square lattice (Lx == Ly)."
    assert nRot == 4, "Only C4 rotation (nRot=4) is implemented."
    N = Lx * Ly
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    rot_maps = torch.zeros(nRot, N, dtype=torch.long)
    # r=0: identity
    rot_maps[0] = (col + Lx * row).reshape(-1)
    # r=1: elementary C4 rotation  (row, col) → (col, Ly-1-row)
    rot_maps[1] = (row + Lx * ((Ly - 1 - col) % Ly)).reshape(-1)
    # r=2,3: compose r=1 with itself
    for r in range(2, nRot):
        rot_maps[r] = rot_maps[1][rot_maps[r - 1]]
    return rot_maps


def _make_mrrx_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Mirror-x (reflection about vertical axis) permutation maps.

    (row, col) → (row, Lx-1-col).  Returns shape [2, N]: [identity, mirror].
    """
    N = Lx * Ly
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    maps = torch.zeros(2, N, dtype=torch.long)
    maps[0] = (col + Lx * row).reshape(-1)                         # identity
    maps[1] = ((Lx - 1 - col) + Lx * row).reshape(-1)             # mirror x
    return maps


def _make_mrry_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Mirror-y (reflection about horizontal axis) permutation maps.

    (row, col) → (Ly-1-row, col).  Returns shape [2, N].
    """
    N = Lx * Ly
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    maps = torch.zeros(2, N, dtype=torch.long)
    maps[0] = (col + Lx * row).reshape(-1)
    maps[1] = (col + Lx * ((Ly - 1 - row) % Ly)).reshape(-1)
    return maps


def _make_dia1_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Main-diagonal mirror (σ_d1): (row, col) → (col, row).

    Requires Lx == Ly.  Returns shape [2, N].
    """
    assert Lx == Ly
    N = Lx * Ly
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    maps = torch.zeros(2, N, dtype=torch.long)
    maps[0] = (col + Lx * row).reshape(-1)
    maps[1] = (row + Lx * col).reshape(-1)                         # swap row ↔ col
    return maps


def _make_dia2_maps(Lx: int, Ly: int) -> torch.Tensor:
    """
    Anti-diagonal mirror (σ_d2): (row, col) → (Ly-1-col, Lx-1-row).

    Requires Lx == Ly.  Returns shape [2, N].
    """
    assert Lx == Ly
    N = Lx * Ly
    row = torch.arange(Ly, dtype=torch.long).reshape(Ly, 1)
    col = torch.arange(Lx, dtype=torch.long).reshape(1, Lx)
    maps = torch.zeros(2, N, dtype=torch.long)
    maps[0] = (col + Lx * row).reshape(-1)
    maps[1] = ((Ly - 1 - row) + Lx * (Lx - 1 - col)).reshape(-1)
    return maps


# ──────────────────────────────────────────────────────────────────────────────
# SymmetryGroup
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SymmetryGroup:
    """
    Spatial symmetry group of an Lx×Ly square lattice.

    The group is generated by composing the active elementary generators and is
    stored as two precomputed tables (both in MPS site order):

      mapping[α, i]  : LongTensor [ntrans, N]
          For each distinct group element α, the site permutation in MPS order.
          mapping[α, i_mps] = j_mps means that g_α maps MPS site i to MPS site j.

      transtable[α]  : ComplexTensor [ntrans]
          The complex character χ(g_α) for each group element:
              χ(g_α) = exp(-i (kx·tx·2π/Lx + ky·ty·2π/Ly + rr·rot·2π/nRot))
                       × px^mx × py^my × sgm1^d1 × sgm2^d2
          where (tx, ty, rot, mx, my, d1, d2) are the generator step parameters
          and (kx, ky, px, py, sgm1, sgm2, rr) are the target quantum numbers.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    nat_to_mps, mps_to_nat : LongTensor [N]
        Site ordering from a Lattice object.  Pass torch.arange(N) for 1D.
    kx, ky : int
        Momentum quantum numbers (integers in [0, Lx) and [0, Ly) respectively).
        The momentum eigenvalue is exp(i kx 2π/Lx) for x-translation.
    px : int, optional (±1)
        Mirror-x eigenvalue.  Ignored if use_Mrrx is False.
    py : int, optional (±1)
        Mirror-y eigenvalue.  Ignored if use_Mrry is False.
    rr : int
        Rotation quantum number (integer in [0, nRot)).
    nRot : int
        Order of the rotation group (currently only 4 supported).
    sgm1 : int, optional (±1)
        Eigenvalue of the main-diagonal mirror σ_d1.
    sgm2 : int, optional (±1)
        Eigenvalue of the anti-diagonal mirror σ_d2.
    use_Tx, use_Ty, use_Rot, use_Mrrx, use_Mrry, use_Dia1, use_Dia2 : bool
        Flags activating each generator.
    """

    # Lattice dimensions
    Lx: int
    Ly: int

    # Site ordering (from a Lattice object)
    nat_to_mps: torch.Tensor
    mps_to_nat: torch.Tensor

    # Quantum numbers (irrep labels)
    kx: int = 0
    ky: int = 0
    px: int = 1    # mirror-x eigenvalue (±1)
    py: int = 1    # mirror-y eigenvalue (±1)
    rr: int = 0    # rotation quantum number
    nRot: int = 1  # rotation group order (1 = no rotation)
    sgm1: int = 1  # main-diagonal mirror eigenvalue (±1)
    sgm2: int = 1  # anti-diagonal mirror eigenvalue (±1)

    # Active generators
    use_Tx: bool = True
    use_Ty: bool = False
    use_Rot: bool = False
    use_Mrrx: bool = False
    use_Mrry: bool = False
    use_Dia1: bool = False
    use_Dia2: bool = False

    def __post_init__(self) -> None:
        self._build()

    # ── public attributes (set by _build) ─────────────────────────────────────
    # mapping   : LongTensor    [ntrans, N]
    # transtable: ComplexTensor [ntrans]
    # ntrans    : int
    # transtep  : LongTensor    [ntrans, 7]  (tx, ty, rot, mx, my, d1, d2)

    # ── internal construction ──────────────────────────────────────────────────

    def _build(self) -> None:
        """Compose all active generators and populate mapping / transtable."""
        N = self.Lx * self.Ly

        # --- Build individual generator maps (in natural coordinates) ----------
        tx_maps  = _make_tx_maps(self.Lx, self.Ly)  if self.use_Tx   else None
        ty_maps  = _make_ty_maps(self.Lx, self.Ly)  if self.use_Ty   else None
        rot_maps = _make_rot_maps(self.Lx, self.Ly, self.nRot) if self.use_Rot  else None
        px_maps  = _make_mrrx_maps(self.Lx, self.Ly) if self.use_Mrrx else None
        py_maps  = _make_mrry_maps(self.Lx, self.Ly) if self.use_Mrry else None
        d1_maps  = _make_dia1_maps(self.Lx, self.Ly) if self.use_Dia1 else None
        d2_maps  = _make_dia2_maps(self.Lx, self.Ly) if self.use_Dia2 else None

        # Identity maps for inactive generators (index 0 = identity in each map)
        id_nat = torch.arange(N, dtype=torch.long)

        # --- Determine loop ranges for each generator -------------------------
        Lx_range  = range(self.Lx) if self.use_Tx   else range(1)
        Ly_range  = range(self.Ly) if self.use_Ty   else range(1)
        rot_range = range(self.nRot) if self.use_Rot else range(1)
        mx_range  = range(2) if self.use_Mrrx else range(1)
        my_range  = range(2) if self.use_Mrry else range(1)
        d1_range  = range(2) if self.use_Dia1 else range(1)
        d2_range  = range(2) if self.use_Dia2 else range(1)

        # Upper bound on number of group elements before deduplication
        ntrans_max = (self.Lx if self.use_Tx else 1) \
                   * (self.Ly if self.use_Ty else 1) \
                   * (self.nRot if self.use_Rot else 1) \
                   * (2 if self.use_Mrrx else 1) \
                   * (2 if self.use_Mrry else 1) \
                   * (2 if self.use_Dia1 else 1) \
                   * (2 if self.use_Dia2 else 1)

        mapping_all  = torch.zeros(ntrans_max, N, dtype=torch.long)
        transstep_all = []
        count = 0

        # --- Compose generators -----------------------------------------------
        # Composition order: Tx → Ty → Rot → Mrrx → Mrry → Dia1 → Dia2.
        # Each step applies the next generator to the accumulated permutation.
        # "Applying generator g to permutation p" means: new_p[i] = g[p[i]]
        # (i.e., first apply p, then apply g — left-to-right composition).
        for tx in Lx_range:
            for ty in Ly_range:
                for rot in rot_range:
                    for mx in mx_range:
                        for my in my_range:
                            for d1 in d1_range:
                                for d2 in d2_range:
                                    # Start from the identity in natural coords
                                    perm = id_nat.clone()
                                    if self.use_Tx:
                                        perm = tx_maps[tx][perm]
                                    if self.use_Ty:
                                        perm = ty_maps[ty][perm]
                                    if self.use_Rot:
                                        perm = rot_maps[rot][perm]
                                    if self.use_Mrrx:
                                        perm = px_maps[mx][perm]
                                    if self.use_Mrry:
                                        perm = py_maps[my][perm]
                                    if self.use_Dia1:
                                        perm = d1_maps[d1][perm]
                                    if self.use_Dia2:
                                        perm = d2_maps[d2][perm]

                                    # Convert natural-coordinate permutation to
                                    # MPS-coordinate permutation:
                                    #   i_mps → nat → apply perm → back to mps
                                    perm_mps = self.nat_to_mps[perm[self.mps_to_nat]]
                                    mapping_all[count] = perm_mps

                                    transstep_all.append(torch.tensor(
                                        [tx, ty, rot, mx, my, d1, d2],
                                        dtype=torch.long,
                                    ))
                                    count += 1

        mapping_all  = mapping_all[:count]
        transstep_all = torch.stack(transstep_all, dim=0)

        # --- Remove duplicate permutations ------------------------------------
        # Two different combinations of generator parameters can produce the
        # same site permutation (e.g., translation by Lx is the identity).
        # We keep only one representative per unique permutation.
        mapping_unique, inverse_idx, _ = torch.unique(
            mapping_all,
            sorted=False,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        # Pick the first occurrence of each unique permutation
        #sort_idx  = torch.argsort(inverse_idx, stable=True)
        _, sort_idx = torch.sort(inverse_idx, descending = False, stable = True)
        ntrans_u  = mapping_unique.shape[0]
        stride    = count // ntrans_u if count % ntrans_u == 0 else 1
        if stride > 1:
            unique_first = sort_idx[::stride]

            # --- Validate quantum numbers ----------------------------------------
            # sort_idx groups all transstep entries that produce the same permutation
            # into consecutive blocks of size `stride`.  For each such block we
            # compute the phase of every member and perform two checks:
            #
            #       Consistency: all phases in the block must be equal.
            #       If they differ, the symmetry generators are incompatible with
            #       the chosen quantum numbers (e.g., C4 rotation combined with
            #       momentum kx ≠ ky), and the irrep labelling is ill-defined.
            #
            # This check loop over the unique permutations (ntrans_u blocks).
            for u in range(ntrans_u):
                block = sort_idx[u * stride : (u + 1) * stride]  # indices into transstep_all
                phases = [
                    self.phase_of(*transstep_all[idx].tolist())
                    for idx in block
                ]
                # Check: all phases in the block must agree
                ref = phases[0]
                for p in phases[1:]:
                    if abs(p - ref) > 1e-10:
                        # Build a human-readable list of the conflicting steps
                        steps = [transstep_all[idx].tolist() for idx in block]
                        raise ValueError(
                            f"Inconsistent quantum numbers: the permutation at "
                            f"block {u} is produced by {stride} different generator "
                            f"combinations, but their phases are not all equal.\n"
                            f"  steps  : {steps}\n"
                            f"  phases : {[f'{ph:.6f}' for ph in phases]}\n"
                            f"Hint: check that kx==ky when C4 rotation is active, "
                            f"or that the momentum sector is compatible with the "
                            f"active discrete symmetries."
                        )
        else:
            unique_first = sort_idx

        self.transtep  = transstep_all[unique_first]  # [ntrans, 7]
        self.mapping   = mapping_unique                # [ntrans, N]
        self.ntrans    = ntrans_u

        # --- Build phase table ------------------------------------------------
        self.transtable = self._build_transtable()

    def _build_transtable(self) -> torch.Tensor:
        """
        Compute the complex character χ(g_α) for each group element α.

        χ(g_α) = exp(-i(kx·tx·2π/Lx + ky·ty·2π/Ly + rr·rot·2π/nRot))
                 × px^mx × py^my × sgm1^d1 × sgm2^d2

        The exponential factor comes from the translation / rotation eigenvalue;
        the power-of-eigenvalue factors come from discrete Z2 symmetries.

        Returns
        -------
        transtable : ComplexTensor of shape [ntrans]
        """
        transtable = torch.zeros(self.ntrans, dtype=torch.complex128)
        for α in range(self.ntrans):
            tx, ty, rot, mx, my, d1, d2 = self.transtep[α].tolist()
            transtable[α] = self.phase_of(tx, ty, rot, mx, my, d1, d2)
        return transtable

    def phase_of(
        self,
        tx: int, ty: int, rot: int,
        mx: int, my: int, d1: int, d2: int,
    ) -> complex:
        """
        Return the complex character χ(g) for the group element specified by
        its generator step parameters.

        χ(g) = exp(-i (kx·tx·2π/Lx + ky·ty·2π/Ly + rr·rot·2π/nRot))
               × px^mx × py^my × sgm1^d1 × sgm2^d2

        Parameters
        ----------
        tx, ty : int
            Translation steps in x and y (integers in [0, Lx) and [0, Ly)).
        rot : int
            Rotation step (integer in [0, nRot)).
        mx, my : int
            Mirror-x / mirror-y action flag (0 = identity, 1 = mirror applied).
        d1, d2 : int
            Main-diagonal / anti-diagonal mirror flag (0 or 1).

        Returns
        -------
        complex : the character χ(g).
        """
        # Continuous phase: translation and rotation eigenvalues
        kkx    = float(self.kx) * 2.0 * np.pi / float(self.Lx)
        kky    = float(self.ky) * 2.0 * np.pi / float(self.Ly)
        kk_rot = float(self.rr) * 2.0 * np.pi / float(max(self.nRot, 1))
        phase  = kkx * tx + kky * ty + kk_rot * rot
        χ_cont = np.cos(phase) - 1j * np.sin(phase)

        # Discrete phase: Z2 eigenvalue (±1) for each active mirror generator
        sign_count = 0
        if self.use_Mrrx and self.px  == -1 and mx == 1:
            sign_count += 1
        if self.use_Mrry and self.py  == -1 and my == 1:
            sign_count += 1
        if self.use_Dia1 and self.sgm1 == -1 and d1 == 1:
            sign_count += 1
        if self.use_Dia2 and self.sgm2 == -1 and d2 == 1:
            sign_count += 1

        return χ_cont * ((-1) ** sign_count)
