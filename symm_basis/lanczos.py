"""
lanczos.py — Lanczos eigensolver for the symmetric Hamiltonian.

Algorithm
---------
Standard Lanczos tridiagonalization with full re-orthogonalization (Gram-Schmidt
against all previous Lanczos vectors).  Re-orthogonalization is essential for
correctness when running many steps: without it, the Ritz values converge to
the same eigenvalue (ghost states) due to floating-point loss of orthogonality.

At each step m the tridiagonal matrix T_m is diagonalized with LAPACK dstev
(scipy).  Convergence is declared when the lowest n_eig Ritz values change
by less than `tol` between consecutive steps.

After convergence, the ground-state vector is reconstructed from the Lanczos
basis vectors and the first eigenvector of T_m.

MPI parallelism
---------------
The H·v product (the expensive step) is distributed across ranks via
`hamiltonian.matrix_vector_product(..., dist=dist)`.  The Lanczos basis
vectors are stored only on rank 0 to avoid replicating O(D × maxlan) memory.
At the start of each step, rank 0 broadcasts the current Lanczos vector to all
ranks so every rank can participate in the H·v multiplication.

Serial usage (dist=None) is fully supported: all logic runs on one process.

Public API
----------
    result = lanczos(hamiltonian, repr_ints, norms, maxlan, n_eig, tol, dist)

    result.eigenvalues   : FloatTensor  [n_eig]  — lowest eigenvalues
    result.eigenvector   : ComplexTensor [D]      — ground-state vector (k=0)
    result.converged     : bool
    result.n_steps       : int                   — steps taken
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from scipy.linalg.lapack import dstev

from .hamiltonian import HeisenbergHamiltonian


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LanczosResult:
    eigenvalues: torch.Tensor   # [n_eig] lowest Ritz values
    eigenvector: torch.Tensor   # [D] ground-state vector (normalised)
    converged: bool
    n_steps: int


# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────

def lanczos(
    hamiltonian: HeisenbergHamiltonian,
    repr_ints: torch.Tensor,
    norms: torch.Tensor,
    maxlan: int = 200,
    n_eig: int = 3,
    tol: float = 3e-8,
    dist=None,
    verbose: bool = True,
) -> LanczosResult:
    """
    Run the Lanczos algorithm to find the lowest eigenvalues of H in the
    symmetric basis.

    Parameters
    ----------
    hamiltonian : HeisenbergHamiltonian
        Provides the H·v product via matrix_vector_product().
    repr_ints : LongTensor [D]
        Sorted integer encodings of all basis representatives.
    norms : DoubleTensor [D]
        Basis norms N_R for each representative.
    maxlan : int
        Maximum number of Lanczos iterations.
    n_eig : int
        Number of lowest eigenvalues to track for convergence.
    tol : float
        Convergence threshold: stop when the sum of absolute changes in the
        lowest n_eig Ritz values is less than tol between consecutive steps.
    dist : torch.distributed handle, optional
        If provided, the H·v product is distributed across MPI ranks.
        Pass None for a serial run.
    verbose : bool
        Print Ritz values at each convergence check (from rank 0 only).

    Returns
    -------
    LanczosResult
        .eigenvalues : FloatTensor [n_eig]
        .eigenvector : ComplexTensor [D]  (ground state, all ranks on exit)
        .converged   : bool
        .n_steps     : int
    """
    D = repr_ints.shape[0]

    # Determine rank / world size (fall back to serial if dist is None)
    if dist is not None:
        myrank = dist.get_rank()
        psize  = dist.get_world_size()
    else:
        myrank = 0
        psize  = 1

    def _is_rank0():
        return myrank == 0

    def _matvec(v: torch.Tensor) -> torch.Tensor:
        """H·v, distributed if dist is not None."""
        return hamiltonian.matrix_vector_product(v, repr_ints, norms, dist)

    def _broadcast_vec(v: torch.Tensor) -> torch.Tensor:
        """Broadcast a complex vector from rank 0 to all ranks."""
        if dist is not None:
            dist.broadcast(torch.view_as_real(v), src=0)
        return v

    def _barrier():
        if dist is not None:
            dist.barrier()

    # ── Allocate tridiagonal coefficient arrays ────────────────────────────────
    # alpha[m] = diagonal element at step m        (Ritz coefficient)
    # beta[m]  = off-diagonal element at step m+1  (beta[0] unused)
    alpha = torch.zeros(maxlan, dtype=torch.float64)
    beta  = torch.zeros(maxlan, dtype=torch.float64)

    # ── Step 0: random initial vector (rank 0), broadcast to all ranks ─────────
    v_cur = torch.zeros(D, dtype=torch.complex128)
    if _is_rank0():
        v_cur = torch.rand(D, dtype=torch.complex128)
        v_cur /= torch.sqrt((v_cur.conj() * v_cur).real.sum())
    v_cur = _broadcast_vec(v_cur)

    # H·v_0
    w = _matvec(v_cur)

    # alpha_0 = <v_0|H|v_0>
    if _is_rank0():
        alpha[0] = (v_cur.conj() * w).real.sum().item()
        # w_1 = H|v_0> - alpha_0 |v_0>
        w = w - alpha[0] * v_cur
        beta[1] = torch.sqrt((w.conj() * w).real.sum()).item()
        v_next = w / beta[1]
    else:
        v_next = torch.zeros(D, dtype=torch.complex128)

    # Lanczos basis: stored only on rank 0
    if _is_rank0():
        vec_list = [v_cur.clone(), v_next.clone()]

    # ── Convergence tracking ───────────────────────────────────────────────────
    ritz_prev  = torch.zeros(n_eig, dtype=torch.float64)
    converged  = torch.zeros(1, dtype=torch.bool)
    z_final    = None   # eigenvectors of final tridiagonal matrix

    # ── Main Lanczos loop ──────────────────────────────────────────────────────
    n_steps = 1   # steps completed so far
    for m in range(2, maxlan):

        # Broadcast current vector (v_{m-1}) from rank 0 to all ranks
        if _is_rank0():
            v_cur = vec_list[m - 1].clone()
        else:
            v_cur = torch.zeros(D, dtype=torch.complex128)
        v_cur = _broadcast_vec(v_cur)

        # H·v_{m-1} (distributed)
        w = _matvec(v_cur)

        if _is_rank0():
            # alpha_{m-1} = <v_{m-1}|H|v_{m-1}>
            alpha[m - 1] = (vec_list[m - 1].conj() * w).real.sum().item()

            # Three-term recurrence: w_m = H|v_{m-1}> - alpha_{m-1}|v_{m-1}> - beta_{m-1}|v_{m-2}>
            w = w - alpha[m - 1] * vec_list[m - 1] - beta[m - 1] * vec_list[m - 2]

            # beta_m = ||w_m||
            beta[m] = torch.sqrt((w.conj() * w).real.sum()).item()
            v_next = w / beta[m]

            # Full re-orthogonalization (Gram-Schmidt against all previous vectors).
            # This prevents ghost states caused by floating-point loss of orthogonality.
            for i in range(len(vec_list)):
                overlap = (vec_list[i].conj() * v_next).sum()
                v_next  = v_next - overlap * vec_list[i]
            # Re-normalise after Gram-Schmidt
            v_next = v_next / torch.sqrt((v_next.conj() * v_next).real.sum())

            vec_list.append(v_next.clone())

        n_steps = m

        # ── Convergence check: solve the m×m tridiagonal system ────────────────
        if m > n_eig and _is_rank0():
            d_np = alpha[:m].numpy().astype(np.float64)
            e_np = beta[1:m].numpy().astype(np.float64)
            # dstev: symmetric tridiagonal eigensolver (LAPACK)
            # vals: all m eigenvalues (ascending), z: eigenvectors [m, m]
            vals, z, info = dstev(d_np, e_np)

            ritz_cur = torch.from_numpy(vals[:n_eig].copy())
            delta    = torch.abs(ritz_cur - ritz_prev).sum().item()

            if verbose:
                ev_str = "  ".join(f"{v:.8f}" for v in vals[:n_eig])
                print(f"  [lanczos step {m:4d}]  Ritz[:{n_eig}] = {ev_str}  Δ={delta:.2e}",
                      flush=True)

            if delta < tol:
                converged[0] = True
                z_final = z.copy()
                ritz_prev = ritz_cur
            else:
                ritz_prev = ritz_cur

        # Broadcast convergence flag to all ranks so they can exit together
        if dist is not None:
            dist.broadcast(converged, src=0)

        if converged[0].item():
            _barrier()
            break

    # ── Reconstruct ground-state eigenvector ──────────────────────────────────
    # The ground state in the full basis is:
    #     |ψ_0⟩ = Σ_{k=0}^{m-1}  z[k, 0] · |v_k⟩
    # where z[:, 0] is the first column of the tridiagonal eigenvector matrix.
    gs_vec = torch.zeros(D, dtype=torch.complex128)
    if _is_rank0():
        if z_final is None:
            # Convergence was not reached; use the last available tridiagonal solution
            d_np = alpha[:n_steps].numpy().astype(np.float64)
            e_np = beta[1:n_steps].numpy().astype(np.float64)
            vals, z_final, info = dstev(d_np, e_np)
            ritz_prev = torch.from_numpy(vals[:n_eig].copy())
            if verbose:
                print(f"  [lanczos] WARNING: not converged after {n_steps} steps.")

        # Stack stored Lanczos vectors (drop the last one: it was appended after
        # the final H·v and was not included in the tridiagonal system)
        vec_stack = torch.stack(vec_list[:-1], dim=0)  # [n_steps, D]
        z0        = torch.tensor(z_final[:, 0], dtype=torch.complex128)  # [n_steps]
        gs_vec    = (z0.unsqueeze(1) * vec_stack).sum(dim=0)             # [D]

        if verbose:
            # Final energy check: <ψ_0|H|ψ_0> (single rank computation)
            print(f"  [lanczos] converged={bool(converged[0].item())}  "
                  f"steps={n_steps}  E_0={ritz_prev[0].item():.10f}", flush=True)

    # Broadcast ground-state vector to all ranks
    gs_vec = _broadcast_vec(gs_vec)

    # Final energy verification via one more H·v
    w = _matvec(gs_vec)
    if _is_rank0() and verbose:
        energy_check = (gs_vec.conj() * w).real.sum().item()
        print(f"  [lanczos] energy check <ψ|H|ψ> = {energy_check:.10f}", flush=True)

    _barrier()

    return LanczosResult(
        eigenvalues = ritz_prev,
        eigenvector = gs_vec,
        converged   = bool(converged[0].item()),
        n_steps     = n_steps,
    )
