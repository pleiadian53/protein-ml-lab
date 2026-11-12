"""
frames.py — rigid-body frame (SE(3)) helpers, from scratch.

A *frame* is a per-residue local coordinate system: a rotation R in SO(3) and a
translation t in R^3. AlphaFold2's structure module represents a protein as a
"gas" of these frames (one per residue, placed on the backbone N–Cα–C atoms)
and refines them. Phase 4 (AF3) and Phase 5 (RFdiffusion) also operate on
frames, so these helpers live in `src/protml/` and are imported by the IPA
module, the FAPE loss, and the equivariance/invariance checks.

CONVENTIONS
-----------
A frame (R, t) maps a point expressed in the LOCAL frame to the GLOBAL frame:

    x_global = R @ x_local + t          (local_to_global)
    x_local  = R^T @ (x_global - t)     (global_to_local, the inverse)

Shapes are batched and broadcast-friendly:
    R : (..., 3, 3)     t : (..., 3)     points : (..., P, 3)
where the leading "..." dims of R/t broadcast against the points' frame dims.
We keep everything as plain tensor ops (no nn.Module) so the functions compose
cleanly and stay dtype-agnostic — run them in float64 for the precision checks.

WHY THIS MATTERS (the one idea)
-------------------------------
Equivariance/invariance proofs downstream hinge on a single algebraic fact:
applying a GLOBAL rigid motion T_g = (R_g, t_g) to every frame,

    (R_i, t_i)  ->  (R_g @ R_i,  R_g @ t_i + t_g),

leaves all *local* coordinates unchanged, and turns every *global* point into
R_g @ x + t_g. Distances between global points are therefore preserved, and a
point mapped back into its (transformed) local frame is identical. That is the
whole reason Invariant Point Attention is invariant — see `ipa.py`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def local_to_global(R: Tensor, t: Tensor, points: Tensor) -> Tensor:
    """Map points from a local frame to the global frame: R @ x + t.

    Args:
        R:      (..., 3, 3) rotation(s).
        t:      (..., 3) translation(s).
        points: (..., P, 3) points expressed in the local frame, where the
                leading dims match R/t (one frame per leading index, P points
                per frame).

    Returns:
        (..., P, 3) points in the global frame.
    """
    # einsum contracts the rotation's column index (j) with the point coords.
    rotated = torch.einsum("...ij,...pj->...pi", R, points)
    return rotated + t.unsqueeze(-2)


def global_to_local(R: Tensor, t: Tensor, points: Tensor) -> Tensor:
    """Inverse of :func:`local_to_global`: R^T @ (x - t).

    Args:
        R:      (..., 3, 3) rotation(s).
        t:      (..., 3) translation(s).
        points: (..., P, 3) points in the global frame.

    Returns:
        (..., P, 3) points expressed in the local frame.
    """
    centered = points - t.unsqueeze(-2)
    # Contract the rotation's ROW index (i) instead of the column => R^T.
    return torch.einsum("...ji,...pj->...pi", R, centered)


def invert(R: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Invert a frame: (R, t) -> (R^T, -R^T t)."""
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum("...ij,...j->...i", R_inv, t)
    return R_inv, t_inv


def compose(
    R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    """Compose two frames: applying (R2, t2) then (R1, t1).

    Returns (R1 @ R2, R1 @ t2 + t1), i.e. the frame whose action equals
    first (R2, t2) then (R1, t1).
    """
    R = torch.einsum("...ij,...jk->...ik", R1, R2)
    t = torch.einsum("...ij,...j->...i", R1, t2) + t1
    return R, t


def random_rotation(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw `n` proper rotation matrices in SO(3) via QR of a Gaussian matrix.

    QR of a random Gaussian matrix gives a (Haar-distributed up to sign)
    orthogonal Q; we fix the column signs and flip one column if needed so that
    det(Q) = +1 (a *proper* rotation, no reflection).

    Args:
        n:         number of rotations to draw.
        dtype:     float dtype (use float64 for precision checks).
        device:    target device.
        generator: optional RNG for reproducibility.

    Returns:
        (n, 3, 3) rotation matrices with det = +1.
    """
    a = torch.randn(n, 3, 3, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(a)
    # Make the decomposition sign-deterministic: enforce positive diagonal of R.
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(-2)
    # Ensure a proper rotation (det = +1) by flipping the last column where det < 0.
    det = torch.linalg.det(q)
    flip = torch.where(det < 0, -1.0, 1.0).to(dtype)
    q[..., :, -1] = q[..., :, -1] * flip.unsqueeze(-1)
    return q
