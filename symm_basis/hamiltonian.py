"""
hamiltonian.py — Heisenberg Hamiltonian in the symmetric basis.

Physics background
------------------
The Heisenberg Hamiltonian on a bond (i, j) is:

    H_{ij} = J S_i · S_j = J [ (1/2)(S_i⁺ S_j⁻ + S_i⁻ S_j⁺) + Sz_i Sz_j ]
           = J [ (1/2) flip_{ij} + (1/4)(2·config[i]−1)(2·config[j]−1) ]

where flip_{ij} exchanges the spins at sites i and j (non-zero only when
config[i] ≠ config[j]).

Matrix elements in the symmetric basis
----------------------------------------
For two representatives |R⟩ and |R'⟩ connected by a single bond flip:

    ⟨R'; χ|H_{ij}|R; χ⟩ = χ(g_{α*}) · √(N_R' / N_R) · (J/2)

where g_{α*} is the (unique) group element that maps the flipped config to
its representative |R'⟩, and χ(g_{α*}) is the associated character.

This module computes:
  1. connected_elements  : all (R', weight) pairs for a given R.
  2. diagonal_element    : the Sz·Sz diagonal contribution.
  3. local_energy        : the full local energy E_loc = Σ_{R'} H_{R'R} ψ(R')/ψ(R)
                           needed for VMC gradient estimation.
  4. matrix_vector_product : H·v in the symmetric basis for Lanczos/ED.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import torch
from .symm_basis import SymmBasis


class HeisenbergHamiltonian:
    """
    Heisenberg Hamiltonian H = Σ_{bonds} J_b S_i · S_j.

    Supports multiple bond types (e.g., J1 and J2) distinguished by their
    coupling constants, which are stored per bond in the bond table.

    Parameters
    ----------
    bonds : LongTensor of shape [nb, 2]
        Table of bonds (i, j) in MPS site order.  Each row is one bond.
    J_per_bond : FloatTensor or list of shape [nb]
        Coupling constant for each bond.  For a uniform J1 model pass a
        tensor of all J1 values; for J1-J2, alternate J1 and J2 as needed.
    basis : SymmBasis
        The symmetric basis object (provides find_representative).
    """

    def __init__(
        self,
        bonds: torch.Tensor,
        J_per_bond: torch.Tensor,
        basis: SymmBasis,
    ) -> None:
        self.bonds     = bonds.long()                      # [nb, 2]
        self.J_per_bond = J_per_bond.to(torch.double)     # [nb]
        self.basis     = basis
        self.nb        = bonds.shape[0]
        self.N         = basis.N

        # Precompute off-diagonal (J/2) and diagonal (J/4) matrix elements per bond
        self.helem_off = (0.5 * J_per_bond).to(torch.complex128)  # [nb]
        self.helem_dia = (0.25 * J_per_bond).to(torch.double)     # [nb]

    # ── Matrix elements in the symmetric basis ─────────────────────────────────

    def connected_elements(
        self,
        config: torch.Tensor,
        config_norm: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find all representatives |R'⟩ connected to |R⟩ by one bond flip,
        and compute the off-diagonal matrix elements H_{R'R}.

        Steps:
          1. For each bond (i,j), create the spin-flipped config (if config[i] ≠ config[j]).
          2. Find the representative of each flipped config and its norm.
          3. Remove zero-norm states (not in the valid sector).
          4. De-duplicate: if multiple bonds lead to the same |R'⟩, sum their weights.
          5. Exclude |R'⟩ = |R⟩ (diagonal contribution handled separately). doesn't contribute to connectivity

        Parameters
        ----------
        config : LongTensor [N]
            Input representative configuration.
        config_norm : float
            N_R for the input configuration.

        Returns
        -------
        conn_reprs : LongTensor [nconn, N]
            The connected representatives.
        weights : ComplexTensor [nconn]
            Off-diagonal matrix elements H_{R'R} = χ(g_{α*}) √(N_R'/N_R) J/2.
        """
        config = config.reshape(self.N)
        N_R    = float(config_norm)

        # --- Step 1: build all bond-flipped configs ---------------------------
        # configs_flipped[b] = config with spins at bond b swapped
        configs_flipped = config.unsqueeze(0).repeat(self.nb, 1)  # [nb, N]
        sites_i = self.bonds[:, 0]  # [nb]
        sites_j = self.bonds[:, 1]  # [nb]
        spins_i = config[sites_i]   # [nb]
        spins_j = config[sites_j]   # [nb]
        configs_flipped[torch.arange(self.nb), sites_i] = spins_j
        configs_flipped[torch.arange(self.nb), sites_j] = spins_i

        # Mask: only bonds where the two spins differ contribute off-diagonally
        flip_mask = (spins_i != spins_j)  # [nb]

        if not flip_mask.any():
            empty = torch.zeros(0, self.N, dtype=torch.long)
            return empty, torch.zeros(0, dtype=torch.complex128)

        active_configs = configs_flipped[flip_mask]                    # [nflip, N]
        active_helem   = self.helem_off[flip_mask]                     # [nflip]

        # --- Step 2: find representatives ------------------------------------
        conn_reprs, itrans, conn_norms = self.basis.find_representative(active_configs)
        # conn_reprs: [nflip, N], itrans: [nflip], conn_norms: [nflip]

        # --- Step 3: remove zero-norm states ---------------------------------
        valid = conn_norms > 1e-10
        if not valid.any():
            empty = torch.zeros(0, self.N, dtype=torch.long)
            return empty, torch.zeros(0, dtype=torch.complex128)

        conn_reprs  = conn_reprs[valid]    # [nvalid, N]
        itrans      = itrans[valid]        # [nvalid]
        conn_norms  = conn_norms[valid]    # [nvalid]
        active_helem = active_helem[valid] # [nvalid]

        # --- Step 4: compute weights -----------------------------------------
        # H_{R'R} = χ(g_{α*}) · √(N_R' / N_R) · (J/2)
        phases  = self.basis.transtable_ext[itrans]         # [nvalid] complex
        sq_norm = torch.sqrt(conn_norms / N_R).to(torch.complex128)  # [nvalid]
        weights = phases * sq_norm * active_helem           # [nvalid] complex

        # --- Step 5: de-duplicate by representative --------------------------
        # Encode each representative as an integer for grouping
        repr_ints = (conn_reprs * self.basis.bitmap[None, :self.N]).sum(dim=1)  # [nvalid]
        unique_repr_ints, inverse = torch.unique(
            repr_ints, sorted=False, return_inverse=True
        )
        n_unique = unique_repr_ints.shape[0]

        # Sum weights for the same representative
        weights_merged = torch.zeros(n_unique, dtype=torch.complex128)
        weights_merged.scatter_add_(0, inverse, weights)

        # Recover one representative config per unique integer
        # (pick the first occurrence for each unique representative)
        first_occurrence = torch.zeros(n_unique, dtype=torch.long)
        first_occurrence.scatter_(0, inverse, torch.arange(inverse.shape[0]))
        conn_reprs_merged = conn_reprs[first_occurrence]  # [n_unique, N]

        # --- Step 6: exclude self (|R'⟩ = |R⟩) ------------------------------
        #Note by Ling, One should not rule out R'=R although this is a rare case
        is_self = (conn_reprs_merged == config.unsqueeze(0)).all(dim=1)
        keep    = ~is_self
        return conn_reprs_merged[keep], weights_merged[keep]

    def diagonal_element(self, config: torch.Tensor) -> float:
        """
        Compute the diagonal matrix element ⟨R; χ|H|R; χ⟩.

        This is the Sz·Sz contribution summed over all bonds:
            Σ_{(i,j)} J_b · (1/4) · (2·σ_i − 1) · (2·σ_j − 1)
          = Σ_{(i,j)} J_b · (1/4) · (+1 if parallel, −1 if antiparallel)

        Parameters
        ----------
        config : LongTensor [N]

        Returns
        -------
        float
        """
        config  = config.reshape(self.N)
        spins_i = config[self.bonds[:, 0]].double()   # [nb], values in {0, 1}
        spins_j = config[self.bonds[:, 1]].double()   # [nb]
        # (2σ−1)(2σ'−1) / 4 = +1/4 if parallel, −1/4 if antiparallel
        sz_sz   = (2.0 * spins_i - 1.0) * (2.0 * spins_j - 1.0) * 0.25
        return float((self.J_per_bond * sz_sz).sum().item())

    # ── Local energy for VMC ──────────────────────────────────────────────────

    def local_energy(
        self,
        config: torch.Tensor,
        config_norm: float,
        log_amp_config: torch.Tensor,
        amplitude_fn: Callable[[torch.Tensor], torch.Tensor],
        machine_pow: float = 1.0,
        amp_clip: float = 8.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the local energy E_loc(R) for variational Monte Carlo.

        E_loc(R) = Σ_{R'} H_{R'R} · exp(log ψ(R') − log ψ(R))  +  H_{RR}

        where the sum runs over all connected representatives |R'⟩ ≠ |R⟩.

        Parameters
        ----------
        config : LongTensor [N]
            Current representative configuration.
        config_norm : float
            N_R for the current configuration.
        log_amp_config : Tensor, scalar
            log|ψ(R; χ)| — the model log-amplitude at the current config.
            (Precomputed externally; avoids redundant model calls.)
        amplitude_fn : callable
            Maps LongTensor [B, N] → Tensor [B] of log-amplitudes.
        machine_pow : float
            Smoothing exponent for log_amplitude (passed to SymmBasis.log_amplitude).
        amp_clip : float
            Clip the real part of (log ψ(R') − log ψ(R)) to [−clip, +clip]
            for numerical stability during early training.

        Returns
        -------
        e_real : scalar Tensor  (real part of E_loc)
        e_imag : scalar Tensor  (imaginary part of E_loc)
        """
        # Off-diagonal contributions
        conn_reprs, weights = self.connected_elements(config, config_norm)
        # weights: [nconn] complex, H_{R'R}

        e_off = torch.tensor(0.0 + 0.0j, dtype=torch.complex128)
        if conn_reprs.shape[0] > 0:
            # Evaluate amplitude model on all connected representatives
            log_amps_prime = torch.stack([
                self.basis.log_amplitude(conn_reprs[k], amplitude_fn, machine_pow)
                for k in range(conn_reprs.shape[0])
            ])  # [nconn]

            # Amplitude ratio: exp(log ψ(R') − log ψ(R)), with real-part clipping
            log_ratio = log_amps_prime - log_amp_config
            log_ratio_clipped = log_ratio.clone()
            log_ratio_clipped = torch.clamp(log_ratio_clipped, -amp_clip, amp_clip)

            e_off = (weights * torch.exp(log_ratio_clipped.to(torch.complex128))).sum()

        # Diagonal contribution
        e_dia = self.diagonal_element(config)

        e_total = e_off + e_dia
        return e_total.real, e_total.imag

    # ── Matrix-vector product for Lanczos / ED ────────────────────────────────

    def matrix_vector_product(
        self,
        v: torch.Tensor,
        repr_ints: torch.Tensor,
        norms: torch.Tensor,
        dist=None,
    ) -> torch.Tensor:
        """
        Compute w = H · v in the symmetric basis.

        For each basis state |R_i⟩:
            w[i] = Σ_j H_{ij} v[j]   (sparse matrix-vector product)

        The basis is indexed by repr_ints (integer encodings of representatives).
        Binary search (torch.searchsorted) maps a connected representative to
        its position in the sorted repr_ints array.

        Parameters
        ----------
        v : ComplexTensor [D]
            Input vector in the symmetric basis.
        repr_ints : LongTensor [D]
            Sorted integer encodings of all representatives (from enumerate_basis).
        norms : DoubleTensor [D]
            Basis norms N_R for each representative.
        dist : torch.distributed handle, optional
            If provided, distribute rows across MPI ranks and all-reduce the result.

        Returns
        -------
        w : ComplexTensor [D]
        """
        D = v.shape[0]
        w = torch.zeros(D, dtype=torch.complex128)

        # Determine which rows this rank processes
        if dist is not None:
            psize  = dist.get_world_size()
            myrank = dist.get_rank()
            chunk_sizes = [(D // psize) + (1 if i < D % psize else 0) for i in range(psize)]
            start = sum(chunk_sizes[:myrank])
            end   = start + chunk_sizes[myrank]
        else:
            start, end = 0, D

        for idx in range(start, end):
            config = self.basis.decode_repr(int(repr_ints[idx].item()))
            N_R    = float(norms[idx].item())

            # Off-diagonal: connected representatives
            conn_reprs, weights = self.connected_elements(config, N_R)
            if conn_reprs.shape[0] > 0:
                conn_ints = (conn_reprs * self.basis.bitmap[None, :self.N]).sum(dim=1)
                positions = torch.searchsorted(repr_ints, conn_ints)
                for k in range(positions.shape[0]):
                    pos = positions[k].item()
                    if pos < D and repr_ints[pos].item() == conn_ints[k].item():
                        w[pos] += v[idx] * weights[k]

            # Diagonal
            w[idx] += v[idx] * self.diagonal_element(config)

        # MPI all-reduce
        if dist is not None:
            dist.barrier()
            dist.all_reduce(torch.view_as_real(w), op=dist.ReduceOp.SUM)

        return w
