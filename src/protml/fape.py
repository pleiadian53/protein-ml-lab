"""
fape.py — Frame-Aligned Point Error (FAPE), AlphaFold2's structure loss, from scratch.

Reference: Jumper et al. 2021, Supplementary §1.9.2 (Algorithm 28).

WHAT FAPE IS (one sentence)
---------------------------
Instead of comparing predicted and true atoms in one global frame (which forces
a superposition, à la RMSD), FAPE compares them in EVERY residue's local frame
and averages — making it superposition-free and local by construction.

THE DEFINITION
--------------
Given predicted frames T_i = (R_i, t_i) and predicted points x_j, and the
corresponding ground-truth frames T_i^* and points x_j^*, for every (frame i,
point j) pair compute the distance between the two points *expressed in frame i*:

    d_ij = || T_i^{-1} ∘ x_j  −  (T_i^*)^{-1} ∘ x_j^* ||

then clamp and average:

    FAPE = (1/Z) · mean_{i,j} min(d_clamp, d_ij)          (AF2: Z = d_clamp = 10 Å)

WHY IT'S THE RIGHT LOSS
-----------------------
• Superposition-free / invariant: applying one global rigid motion to the whole
  prediction (frames AND points together) leaves every d_ij unchanged, because
  T_i^{-1} cancels the motion. So — unlike RMSD — no Kabsch alignment is needed;
  the loss is already invariant. (Proven numerically in the companion check.)
• Local: each residue frame scores all atoms relative to itself, so a single
  badly-placed domain does not corrupt the score everywhere (RMSD's failure mode).
• Chiral: frames encode handedness, so a mirror image is correctly penalised —
  a global *reflection* is NOT in the invariance group (only proper SE(3) is).
• Clamping bounds the gradient from gross early-training errors (AF2 clamps most
  examples at 10 Å).

This connects to Phase 0's metric zoo: lDDT is the superposition-free *evaluation*
metric; FAPE is the superposition-free *training* loss with the same spirit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from protml.frames import global_to_local


def fape_loss(
    R_pred: Tensor,   # (N_frames, 3, 3)
    t_pred: Tensor,   # (N_frames, 3)
    x_pred: Tensor,   # (N_pts, 3) predicted point positions (global frame)
    R_true: Tensor,   # (N_frames, 3, 3)
    t_true: Tensor,   # (N_frames, 3)
    x_true: Tensor,   # (N_pts, 3)
    *,
    clamp: float | None = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
) -> Tensor:
    """Frame-Aligned Point Error between a predicted and a true structure.

    Every point is expressed in every frame's local coordinates (for both the
    prediction and the ground truth) and the per-(frame, point) distances are
    clamped and averaged.

    Args:
        R_pred, t_pred: predicted frames (rotation, translation).
        x_pred:         predicted point positions in the global frame.
        R_true, t_true: ground-truth frames.
        x_true:         ground-truth point positions in the global frame.
        clamp:          distance clamp in the same units as coordinates (Å).
                        None disables clamping.
        length_scale:   divisor Z that non-dimensionalises the loss (AF2: 10 Å).
        eps:            added under the square root for a smooth gradient at 0.

    Returns:
        Scalar FAPE loss.
    """
    n_frames = R_pred.shape[0]
    n_pts = x_pred.shape[0]

    # Broadcast every point into every frame: (N_frames, N_pts, 3).
    xp = x_pred.unsqueeze(0).expand(n_frames, n_pts, 3)
    xt = x_true.unsqueeze(0).expand(n_frames, n_pts, 3)
    local_pred = global_to_local(R_pred, t_pred, xp)   # (N_frames, N_pts, 3)
    local_true = global_to_local(R_true, t_true, xt)

    # Per-(frame, point) distance, with eps under the sqrt for a smooth gradient.
    d = torch.sqrt((local_pred - local_true).pow(2).sum(dim=-1) + eps)  # (N_frames, N_pts)

    if clamp is not None:
        d = d.clamp(max=clamp)

    return d.mean() / length_scale


def naive_rmsd(x_pred: Tensor, x_true: Tensor) -> Tensor:
    """Global RMSD WITHOUT superposition — the thing FAPE improves on.

    Provided only as a contrast in the companion check: this blows up under a
    global rigid motion of the prediction, whereas FAPE does not.
    """
    return torch.sqrt((x_pred - x_true).pow(2).sum(dim=-1).mean())
