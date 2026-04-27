"""
symm_basis.py — Symmetric basis for spin-1/2 systems.

Physics background
------------------
Given a spatial SymmetryGroup G and, optionally, the spin-inversion Z2
symmetry, the symmetric basis state labelled by a representative |R⟩ is:

    |R; χ⟩ = (1/√N_R) Σ_{α ∈ G'} χ*(g_α) · g_α|R⟩

where G' = G ∪ {Z₂ g : g ∈ G} (G extended by spin inversion if zz ≠ 0),
χ(g_α) is the group character, and N_R is the norm:

    N_R = |Σ_{α : g_α|R⟩ = |R⟩} χ(g_α)|

A representative |R⟩ is the lexicographically smallest configuration
in the full orbit {g_α|R⟩ : g_α ∈ G'}.  Configurations with N_R = 0
belong to a zero-norm subspace and are excluded from the basis.

Spin inversion
--------------
Spin inversion flips all spins: σ_inv |0110...⟩ = |1001...⟩.
If use_spin_inv is True, the orbit is doubled: for each spatial group
element α, both g_α|config⟩ and g_α(σ_inv|config⟩) are considered.
The eigenvalue of spin inversion is zz ∈ {+1, -1}, encoded in
transtable_inv[α] = zz × transtable[α].

Amplitude model interface
-------------------------
The variational wavefunction amplitude ψ(config) is provided by an external
callable (e.g., a quimb MPS or a neural network):

    amplitude_fn(configs) → torch.Tensor of shape [B]

where configs is a LongTensor of shape [B, N] and the output values are
the log-amplitudes log|ψ(config)|.  The amplitude of the symmetric state is:

    log|ψ(R; χ)| ≈ (1/p) log Σ_{orbit} exp(p · amplitude_fn(g_α|R⟩))

which is a smooth (differentiable) approximation to the maximum over the orbit
controlled by the parameter machine_pow (p).  As p → ∞ this becomes
max_α amplitude_fn(g_α|R⟩); at p = 1 it is a true log-sum-exp.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import torch
from .symmetry_group import SymmetryGroup


# ──────────────────────────────────────────────────────────────────────────────
# Bit-encoding helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_bitmap(max_bits: int = 64) -> torch.Tensor:
    """
    Powers of 2: bitmap[k] = 2^k.

    Used to encode a binary spin configuration as a (block of) integer(s) for
    lexicographic comparison.  torch.long holds up to 63 bits safely (signed).
    """
    return torch.pow(2, torch.arange(max_bits, dtype=torch.long))


def _configs_to_block_ints(
    configs: torch.Tensor,
    blksize: int,
    nblks: int,
    bitmap: torch.Tensor,
) -> torch.Tensor:
    """
    Encode a batch of binary spin configurations as block integers.

    Parameters
    ----------
    configs : LongTensor [B, ntrans_ext, N]
        Binary spin configurations (values 0 or 1).
    blksize : int
        Number of bits per block (≤ 62 to avoid signed overflow).
    nblks : int
        Number of blocks (nblks * blksize == N).
    bitmap : LongTensor [64]
        Powers of 2.

    Returns
    -------
    block_ints : LongTensor [B, ntrans_ext, nblks]
        block_ints[b, α, k] encodes bits [k*blksize .. (k+1)*blksize - 1]
        of configs[b, α, :] as a single integer.
    """
    B, ntrans_ext, N = configs.shape
    configs_blocked = configs.reshape(B, ntrans_ext, nblks, blksize)
    # Inner-product with powers of 2 along the blksize axis
    block_ints = (configs_blocked * bitmap[None, None, None, :blksize]).sum(dim=-1)
    return block_ints


# ──────────────────────────────────────────────────────────────────────────────
# SymmBasis
# ──────────────────────────────────────────────────────────────────────────────

class SymmBasis:
    """
    Symmetric basis for an N-site spin-1/2 system.

    Wraps a SymmetryGroup with the spin-inversion extension and provides:
      • find_representative : map any config to its canonical representative.
      • norm                : compute the basis norm N_R for each config.
      • is_representative   : check if a config is canonical.
      • log_amplitude       : evaluate log|ψ(R; χ)| from an external amplitude.
      • enumerate_basis     : iterate over all valid representatives (for ED).

    Parameters
    ----------
    group : SymmetryGroup
        Spatial symmetry group (already built, stores mapping and transtable).
    N : int
        Number of sites.
    nup : int
        Number of up-spins (Sz conservation sector).  Configs with sum ≠ nup
        are never representatives.
    zz : int
        Spin-inversion eigenvalue (+1 or -1).  Set to 0 to disable spin
        inversion entirely (e.g., when the Hamiltonian breaks this symmetry).
    """

    def __init__(
        self,
        group: SymmetryGroup,
        N: int,
        nup: int,
        zz: int,
    ) -> None:
        self.group = group
        self.N     = N
        self.nup   = nup
        self.zz    = zz

        # Determine block structure for large-N lexicographic comparison.
        # torch.long is 64-bit signed; we use at most 62 bits per block.
        if N <= 62:
            self.nblks   = 1
            self.blksize = N
        elif N <= 124:
            self.nblks   = 2
            self.blksize = N // 2
        else:
            raise NotImplementedError(
                f"N={N} > 124 sites: extend the block structure."
            )
        assert self.blksize * self.nblks == N, \
            "N must be evenly divisible for block-integer encoding."

        self.bitmap = _build_bitmap(64)

        # Extended transtable: [2 * ntrans]
        # First ntrans entries: spatial group only.
        # Next  ntrans entries: spatial group composed with spin inversion,
        #                        weighted by eigenvalue zz.
        ntrans = group.ntrans
        if zz != 0:
            transtable_inv = float(zz) * group.transtable
            self.transtable_ext = torch.cat(
                [group.transtable, transtable_inv], dim=0
            )
            self.ntrans_ext = 2 * ntrans
        else:
            self.transtable_ext = group.transtable
            self.ntrans_ext     = ntrans

    # ── Core: representative and norm ─────────────────────────────────────────

    def find_representative(
        self,
        configs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the canonical representative of each configuration.

        A representative is defined as the lexicographically smallest
        configuration in the full orbit {g_α|config⟩, g_α(σ_inv|config⟩)}.

        The norm N_R is computed simultaneously as:
            N_R = |Σ_{α : g_α(config) = repr} χ_ext(g_α)|

        where χ_ext is the extended character (including spin inversion phase).
        This quantity is real and non-negative; N_R = 0 means the config maps
        to a zero-norm state in the chosen symmetry sector.

        Parameters
        ----------
        configs : LongTensor of shape [B, N] or [N]
            Binary spin configurations (values in {0, 1}).

        Returns
        -------
        repr_configs : LongTensor [B, N]
            The representative configuration for each input.
        itrans : LongTensor [B]
            Index into transtable_ext identifying which group element maps
            config to its representative.
        norms : FloatTensor [B]
            The basis norm N_R (real, non-negative).
        """
        configs = configs.reshape(-1, self.N)
        B = configs.shape[0]
        ntrans = self.group.ntrans

        # Apply all spatial group elements: shape [B, ntrans, N]
        # configs[:, mapping[α, :]] = g_α applied to each config in the batch.
        orbit_spatial = configs[:, self.group.mapping]  # [B, ntrans, N]

        if self.ntrans_ext > ntrans:
            # Spin-inverted configs: flip all spins, then apply spatial group
            orbit_inv = (1 - configs)[:, self.group.mapping]  # [B, ntrans, N]
            orbit = torch.cat([orbit_spatial, orbit_inv], dim=1)  # [B, 2*ntrans, N]
        else:
            orbit = orbit_spatial

        # --- Lexicographic minimum via block integers -------------------------
        # Encode each orbit config as nblks integers (most significant block
        # has the highest index).
        block_ints = _configs_to_block_ints(
            orbit, self.blksize, self.nblks, self.bitmap
        )  # [B, ntrans_ext, nblks]

        # Iteratively narrow down candidates: compare from most-significant
        # block to least-significant, keeping only the minimum at each step.
        candidates = torch.ones(B, self.ntrans_ext, dtype=torch.bool)
        for blk in range(self.nblks - 1, -1, -1):
            # Add a large penalty to non-candidates so they never win the min
            penalized = block_ints[:, :, blk] + (~candidates) * self.bitmap[self.blksize]
            val_min, itrans = torch.min(penalized, dim=1)          # [B]
            # Update candidates: those matching the minimum in this block
            candidates = candidates & (block_ints[:, :, blk] == val_min[:, None])

        # itrans: [B] — index of the representative in the extended orbit
        #repr_configs = orbit[torch.arange(B), itrans, :]
        repr_configs = torch.gather(
            orbit,
            dim=1,
            index=itrans[:, None, None].expand(B, 1, self.N),
        ).squeeze(1)  # [B, N]

        # --- Norm computation ------------------------------------------------
        # N_R = |Σ_{α : orbit[α] = repr} χ_ext[α]|
        # 'candidates' now flags exactly those α where orbit[α] == repr.
        norms_complex = (candidates.to(torch.complex128)
                         * self.transtable_ext[None, :]).sum(dim=1)  # [B] complex

        # Divide by χ_ext[itrans] to cancel the phase of the representative
        # transform, leaving a real non-negative value (up to numerical noise).
        norms_complex = norms_complex / self.transtable_ext[itrans]
        norms = norms_complex.real.abs().to(torch.double)

        return repr_configs, itrans, norms

    def is_representative(self, config: torch.Tensor) -> bool:
        """
        Return True if config is already its own representative.

        Parameters
        ----------
        config : LongTensor of shape [N]
        """
        repr_c, _, norm = self.find_representative(config.unsqueeze(0))
        return bool(
            torch.all(repr_c.squeeze(0) == config).item() and norm.item() > 1e-10
        )

    def norm(self, config: torch.Tensor) -> float:
        """
        Return N_R for a single configuration.

        Parameters
        ----------
        config : LongTensor of shape [N]

        Returns
        -------
        float : the norm (0.0 if config is not in the valid sector).
        """
        _, _, n = self.find_representative(config.unsqueeze(0))
        return float(n.item())

    # ── Amplitude of the symmetric state ──────────────────────────────────────

    def log_amplitude(
        self,
        config: torch.Tensor,
        amplitude_fn: Callable[[torch.Tensor], torch.Tensor],
        machine_pow: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute log|ψ(R; χ)| for a single representative configuration.

        The model returns log-amplitudes for each orbit config,
        and we use a smooth log-sum-exp approximation:

            log|ψ(R; χ)| ≈ (1/p) log Σ_{unique g_α R} exp(p · amp(g_α R))

        where amp(c) = amplitude_fn(c) is the model's log-amplitude output and
        p = machine_pow controls the sharpness.  This formula avoids explicit
        normalization and is numerically stable.

        Parameters
        ----------
        config : LongTensor of shape [N]
            A representative configuration (assumed valid, i.e. N_R > 0).
        amplitude_fn : callable
            Maps a batch of configs LongTensor [B, N] → FloatTensor [B].
            Should return log|ψ(config)| for each config.
            For a quimb MPS this would be the MPS overlap log-amplitude.
        machine_pow : float
            Smoothing exponent p.  p=2
            gives a sharper approximation to the maximum.

        Returns
        -------
        log_amp : scalar Tensor
            log|ψ(R; χ)| (real-valued).
        """
        config = config.reshape(self.N)
        ntrans = self.group.ntrans

        # Build the orbit: spatial transforms of config
        orbit_spatial = config[self.group.mapping]  # [ntrans, N]
        if self.ntrans_ext > ntrans:
            orbit_inv = (1 - config)[self.group.mapping]   # [ntrans, N]
            orbit = torch.cat([orbit_spatial, orbit_inv], dim=0)  # [2*ntrans, N]
        else:
            orbit = orbit_spatial

        # De-duplicate orbit configs (many group elements may give the same config)
        orbit_unique = torch.unique(orbit, sorted=False, return_inverse = False, return_counts = False, dim=0)  # [nuniq, N]
        orbit_unique = orbit_unique.long()

        # Evaluate the amplitude model on the unique orbit configs
        log_amps = amplitude_fn(orbit_unique)  # [nuniq]

        # log-sum-exp with exponent p:
        # (1/p) log Σ exp(p · amp) = max_amp + (1/p) log Σ exp(p·(amp - max_amp))
        log_amp = (
            torch.logsumexp(machine_pow * log_amps, dim=0) / machine_pow
        )

        return log_amp

    # ── Basis enumeration (exact diagonalization) ──────────────────────────────

    def enumerate_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enumerate all valid representatives in the fixed-Sz sector.

        Iterates over all 2^N binary configurations, checks the Sz sector
        (sum == nup), finds the representative, and keeps only those configs
        that are their own representative with N_R > 0.

        The representative is encoded as an integer:
            repr_int = Σ_i config[i] · 2^i

        This is suitable for small systems (N ≤ ~12).  For larger systems
        use enumerate_basis_mpi().

        Returns
        -------
        repr_ints : LongTensor [D]
            Integer encoding of each representative.  Sorted in ascending order.
        norms : FloatTensor [D]
            N_R value for each representative.
        """
        repr_list  = []
        norm_list  = []
        chunk_size = 32  # process configs in batches for efficiency

        total = 2 ** self.N
        for i in range(0, total, chunk_size):
            # Decode integers i .. i+chunk-1 into binary configs
            end = min(i + chunk_size, total)
            indices = torch.arange(i, end, dtype=torch.long)
            # configs[b, k] = (indices[b] >> k) & 1
            spnconf = (indices[:, None] // self.bitmap[None, :self.N]) % 2  # [chunk, N]

            # Apply Sz sector filter
            sz_filter = spnconf.sum(dim=1) == self.nup
            if not sz_filter.any():
                continue
            spnconf = spnconf[sz_filter]
            indices = indices[sz_filter]

            # Find representatives
            repr_c, _, norms = self.find_representative(spnconf)

            # Keep only those that are their own representative with N_R > 0
            is_repr = (spnconf == repr_c).all(dim=1) & (norms > 1e-10)
            indices = indices[is_repr]
            # Recompute with correct indexing:
            valid_mask = sz_filter.clone()
            valid_mask[sz_filter] = is_repr
            valid_indices = torch.arange(i, end, dtype=torch.long)[valid_mask]
            #double check indices and valid_indices coinside
            assert (indices == valid_indices).all()
            repr_list.append(valid_indices)
            norm_list.append(norms[is_repr])

        if len(repr_list) == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.double)

        repr_ints = torch.cat(repr_list, dim=0)
        norms_out = torch.cat(norm_list, dim=0)

        # Sort by integer value (ascending) for reproducibility
        order     = torch.argsort(repr_ints)
        return repr_ints[order], norms_out[order]

    def enumerate_basis_mpi(self, dist) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MPI-parallel basis enumeration for larger systems (N up to ~40).

        The 2^N configuration space is partitioned across MPI ranks.  Each
        rank processes its chunk and sends valid representatives to rank 0,
        which gathers and broadcasts the final sorted basis.

        Parameters
        ----------
        dist : torch.distributed
            An initialised torch.distributed process group handle.

        Returns
        -------
        repr_ints : LongTensor [D]   (same on all ranks after broadcast)
        norms     : DoubleTensor [D] (same on all ranks after broadcast)
        """
        psize  = dist.get_world_size()
        myrank = dist.get_rank()
        chunk  = 32
        total  = 2 ** self.N

        assert total % (chunk * psize) == 0, (
            f"2^N = {total} must be divisible by chunk*psize = {chunk * psize}."
        )

        repr_list = []
        norm_list = []

        for i in range(total // chunk // psize):
            for j in range(chunk):
                # Interleave ranks in a round-robin pattern for load balance
                k = (i * psize + ((myrank + (i % psize)) % psize)) * chunk + j
                spnconf = (k // self.bitmap[:self.N]) % 2  # [N]
                if spnconf.sum().item() != self.nup:
                    continue
                repr_c, _, norm = self.find_representative(spnconf.unsqueeze(0))
                if (spnconf == repr_c.squeeze(0)).all() and norm.item() > 1e-10:
                    repr_list.append(k)
                    norm_list.append(norm)

        # Gather on rank 0
        gathered_reprs = [None] * psize if myrank == 0 else None
        gathered_norms = [None] * psize if myrank == 0 else None
        repr_tensor = torch.tensor(repr_list, dtype=torch.long)
        norm_tensor = torch.cat(norm_list, dim=0) if norm_list else torch.zeros(0)
        dist.gather_object(repr_tensor, gathered_reprs, dst=0)
        dist.gather_object(norm_tensor, gathered_norms, dst=0)

        nrepr = torch.zeros(1, dtype=torch.long)
        if myrank == 0:
            repr_tensor = torch.cat(gathered_reprs, dim=0)
            norm_tensor = torch.cat(gathered_norms, dim=0)
            order       = torch.argsort(repr_tensor)
            repr_tensor = repr_tensor[order]
            norm_tensor = norm_tensor[order]
            nrepr[0]    = repr_tensor.shape[0]

        dist.barrier()
        dist.broadcast(nrepr, src=0)

        if myrank > 0:
            repr_tensor = torch.zeros(nrepr[0].item(), dtype=torch.long)
            norm_tensor = torch.zeros(nrepr[0].item(), dtype=torch.double)

        dist.broadcast(repr_tensor, src=0)
        dist.broadcast(norm_tensor, src=0)

        return repr_tensor, norm_tensor

    # ── Decode repr integer → configuration ───────────────────────────────────

    def decode_repr(self, repr_int: int) -> torch.Tensor:
        """
        Convert an integer-encoded representative to a binary config tensor.

        Parameters
        ----------
        repr_int : int or scalar LongTensor

        Returns
        -------
        config : LongTensor [N]
        """
        return (torch.tensor(repr_int, dtype=torch.long) // self.bitmap[:self.N]) % 2

    def encode_config(self, config: torch.Tensor) -> torch.Tensor:
        """
        Encode a binary configuration as a scalar integer.

        Parameters
        ----------
        config : LongTensor [N]

        Returns
        -------
        repr_int : scalar LongTensor
        """
        return (config * self.bitmap[:self.N]).sum()
