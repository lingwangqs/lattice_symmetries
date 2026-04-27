"""
io.py — Disk I/O for precomputed symmetric bases.

Building the full basis by enumerating all 2^N configurations is expensive and
should be done once and cached.  These functions save / load the two essential
arrays: repr_ints (integer-encoded representatives) and norms (N_R values).

File naming convention
----------------------
Files are written to a directory (default: "./basis/") and named by the
physical parameters of the basis so that different parameter sets coexist:

    {prefix}reprs-{Lx}-{Ly}-{sec}-{kx}-{ky}-{px}-{py}-{zz}-{sgm1}-{sgm2}-{rr}.bin
    {prefix}norms-{Lx}-{Ly}-{sec}-{kx}-{ky}-{px}-{py}-{zz}-{sgm1}-{sgm2}-{rr}.bin
    {prefix}info- ...  .txt   (human-readable header with basis size)

Usage example
-------------
    from symm_basis.io import save_basis, load_basis

    # After building basis:
    save_basis(repr_ints, norms, params, basis_dir="./basis")

    # Later, in the VMC loop:
    repr_ints, norms = load_basis(params, basis_dir="./basis")
"""

from __future__ import annotations
import os
from array import array as _array
from typing import Dict, Any, Tuple
import torch


# ──────────────────────────────────────────────────────────────────────────────
# File-name helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_tag(params: Dict[str, Any]) -> str:
    """
    Build a canonical filename tag from the physical parameter dictionary.

    Required keys: Lx, Ly, sec, kx, ky, px, py, zz, sgm1, sgm2, rr.
    """
    keys = ["Lx", "Ly", "sec", "kx", "ky", "px", "py", "zz", "sgm1", "sgm2", "rr"]
    return "-".join(str(params[k]) for k in keys)


def _filenames(params: Dict[str, Any], basis_dir: str = "./basis"):
    """Return (info_path, reprs_path, norms_path) for the given parameters."""
    os.makedirs(basis_dir, exist_ok=True)
    tag = _make_tag(params)
    info  = os.path.join(basis_dir, f"info-{tag}.txt")
    reprs = os.path.join(basis_dir, f"reprs-{tag}.bin")
    norms = os.path.join(basis_dir, f"norms-{tag}.bin")
    return info, reprs, norms


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_basis(
    repr_ints: torch.Tensor,
    norms: torch.Tensor,
    params: Dict[str, Any],
    basis_dir: str = "./basis",
) -> None:
    """
    Write a precomputed basis to disk.

    Parameters
    ----------
    repr_ints : LongTensor [D]
        Sorted integer encodings of all representatives.
    norms : DoubleTensor [D]
        Corresponding N_R values.
    params : dict
        Physical parameters (must include keys: Lx, Ly, sec, kx, ky, px, py,
        zz, sgm1, sgm2, rr).
    basis_dir : str
        Directory to write files into (created if absent).
    """
    info_path, reprs_path, norms_path = _filenames(params, basis_dir)
    D = repr_ints.shape[0]

    # Human-readable info file
    with open(info_path, "w") as f:
        f.write(f"basis_size={D}\n")
        for k, v in params.items():
            f.write(f"{k}={v}\n")

    # Binary: repr_ints as signed 64-bit integers
    reprs_np = repr_ints.numpy()
    with open(reprs_path, "wb") as f:
        reprs_np.tofile(f)

    # Binary: norms as float64
    norms_np = norms.double().numpy()
    with open(norms_path, "wb") as f:
        norms_np.tofile(f)


def load_basis(
    params: Dict[str, Any],
    basis_dir: str = "./basis",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a precomputed basis from disk.

    Parameters
    ----------
    params : dict
        Physical parameters used when saving (identifies the file).
    basis_dir : str
        Directory to read from.

    Returns
    -------
    repr_ints : LongTensor [D]
    norms : DoubleTensor [D]

    Raises
    ------
    FileNotFoundError if the expected files do not exist.
    """
    info_path, reprs_path, norms_path = _filenames(params, basis_dir)

    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"Basis info file not found: {info_path}\n"
            "Run enumerate_basis() and save_basis() first."
        )

    # Read basis size from info file
    D = None
    with open(info_path, "r") as f:
        for line in f:
            if line.startswith("basis_size="):
                D = int(line.split("=")[1].strip())
                break
    assert D is not None, f"Could not parse basis_size from {info_path}"

    # Read repr_ints (signed 64-bit integers)
    raw = _array("q")
    with open(reprs_path, "rb") as f:
        raw.fromfile(f, D)
    repr_ints = torch.tensor(raw.tolist(), dtype=torch.long)

    # Read norms (float64)
    raw_norms = _array("d")
    with open(norms_path, "rb") as f:
        raw_norms.fromfile(f, D)
    norms = torch.tensor(raw_norms.tolist(), dtype=torch.double)

    return repr_ints, norms


def basis_exists(params: Dict[str, Any], basis_dir: str = "./basis") -> bool:
    """Return True if all basis files for these parameters exist on disk."""
    paths = _filenames(params, basis_dir)
    return all(os.path.isfile(p) for p in paths)
