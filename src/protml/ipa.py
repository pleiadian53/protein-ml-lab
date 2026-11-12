"""
ipa.py — Invariant Point Attention (IPA), from scratch.

Reference: Jumper et al., "Highly accurate protein structure prediction with
AlphaFold", Nature 2021 — Supplementary Algorithm 22 (the structure module's
attention). This is a minimal, teaching-oriented reimplementation; it is not
weight-compatible with AF2 and omits engineering details (LayerNorm placement,
exact initialisation), but it is faithful to the part that matters: *why the
update is invariant*.

WHY THIS EXERCISE (read first)
------------------------------
EGNN (Phase 3) made E(n)-equivariance concrete on a graph by routing all
geometry through invariant distances and equivariant relative vectors. IPA is
the same idea, upgraded to attention and to *frames*: instead of a global
coordinate system it gives every residue its own local frame (R_i, t_i), and
attention reads geometry by placing "points" through those frames. The single
representation it returns is INVARIANT to a global rigid motion of the whole
structure — that invariance is the centerpiece of this exercise, and it is
proven by the same two algebraic facts as EGNN (see `frames.py`).

THE MECHANISM (three attention terms)
-------------------------------------
For residues i, j the attention logit combines three contributions:
  (1) SCALAR : an ordinary dot-product of query/key vectors built from the
      single representation s. Invariant because s is invariant.
  (2) PAIR   : a bias read from the pair representation z_ij. Invariant because
      z is invariant.
  (3) POINT  : query/key *points* are produced in each residue's LOCAL frame,
      then mapped to the GLOBAL frame and compared by squared distance
      ||T_i∘q_i^p − T_j∘k_j^p||². A global rigid motion preserves these
      distances, so this term is invariant too.
All three are invariant ⇒ the attention weights are invariant. The value
"points" are aggregated in the global frame and then mapped BACK into residue
i's local frame; expressed in its own frame, the result is invariant, and so is
the updated single representation. (Full derivation: `frames.py` docstring +
`examples/alphafold2/ipa_invariance_check.py`.)

NOTATION
--------
  N   : number of residues
  c_s : single-representation width
  c_z : pair-representation width (0 / None to disable the pair-bias term)
  H   : number of attention heads
  c   : per-head scalar channel
  Pq  : query/key points per head      Pv : value points per head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from protml.frames import global_to_local, local_to_global


class InvariantPointAttention(nn.Module):
    """One IPA block: (s, z, frames) -> updated single representation s'.

    The returned s' is invariant to any global rigid transform applied to the
    frames. The block does not itself update the frames — in AF2 a separate
    head predicts a frame increment from s'; here we isolate the attention so
    the invariance property is easy to state and test.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int | None = 16,
        n_head: int = 4,
        c_hidden: int = 16,
        n_qk_points: int = 4,
        n_v_points: int = 8,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.H = n_head
        self.c = c_hidden
        self.Pq = n_qk_points
        self.Pv = n_v_points

        # Scalar query/key/value projections from the single representation.
        self.to_q = nn.Linear(c_s, n_head * c_hidden)
        self.to_k = nn.Linear(c_s, n_head * c_hidden)
        self.to_v = nn.Linear(c_s, n_head * c_hidden)

        # Point query/key/value projections (3 coords per point), in local frame.
        self.to_qp = nn.Linear(c_s, n_head * n_qk_points * 3)
        self.to_kp = nn.Linear(c_s, n_head * n_qk_points * 3)
        self.to_vp = nn.Linear(c_s, n_head * n_v_points * 3)

        # Pair-bias projection (one scalar bias per head per residue pair).
        self.to_bias = nn.Linear(c_z, n_head, bias=False) if c_z else None

        # Per-head learned weight on the point term (kept positive via softplus),
        # exactly as in AF2 so each head can choose how much to trust geometry.
        self.head_weight = nn.Parameter(torch.zeros(n_head))

        # Output projection. Input = concat of:
        #   scalar out (H*c) + local point coords (H*Pv*3) + point norms (H*Pv)
        #   + pair-weighted out (H*c_z, if pair rep present).
        out_in = n_head * c_hidden + n_head * n_v_points * 3 + n_head * n_v_points
        if c_z:
            out_in += n_head * c_z
        self.to_out = nn.Linear(out_in, c_s)

    def forward(
        self,
        s: Tensor,            # (N, c_s) single representation (invariant)
        R: Tensor,            # (N, 3, 3) per-residue rotation
        t: Tensor,            # (N, 3) per-residue translation
        z: Tensor | None = None,  # (N, N, c_z) pair representation (invariant)
    ) -> Tensor:
        N = s.shape[0]
        H, c, Pq, Pv = self.H, self.c, self.Pq, self.Pv

        # --- 1. Scalar q/k/v -------------------------------------------------
        q = self.to_q(s).view(N, H, c)
        k = self.to_k(s).view(N, H, c)
        v = self.to_v(s).view(N, H, c)

        # --- 2. Point q/k/v in the LOCAL frame, then mapped to GLOBAL --------
        qp = self.to_qp(s).view(N, H * Pq, 3)
        kp = self.to_kp(s).view(N, H * Pq, 3)
        vp = self.to_vp(s).view(N, H * Pv, 3)
        # Each residue's frame places its own points into the global frame.
        qp_g = local_to_global(R, t, qp).view(N, H, Pq, 3)
        kp_g = local_to_global(R, t, kp).view(N, H, Pq, 3)
        vp_g = local_to_global(R, t, vp).view(N, H, Pv, 3)

        # --- 3. Three attention terms ---------------------------------------
        # (1) scalar dot product, scaled like standard attention.
        scalar = torch.einsum("ihc,jhc->ijh", q, k) / math.sqrt(c)

        # (3) point term: squared distance between i's query points and j's key
        # points, summed over points. Shapes: (N,1,H,Pq,3) - (1,N,H,Pq,3).
        diff = qp_g.unsqueeze(1) - kp_g.unsqueeze(0)      # (N, N, H, Pq, 3)
        sq_dist = diff.pow(2).sum(dim=(-1, -2))            # (N, N, H)
        gamma = F.softplus(self.head_weight)               # (H,) > 0
        w_c = math.sqrt(2.0 / (9.0 * Pq))                  # AF2 normalising const
        point = -0.5 * gamma.view(1, 1, H) * w_c * sq_dist

        # (2) pair bias.
        if self.to_bias is not None and z is not None:
            bias = self.to_bias(z)                         # (N, N, H)
            n_terms = 3
        else:
            bias = torch.zeros_like(scalar)
            n_terms = 2

        # Balance the contributions (AF2 weights the combined logit by 1/sqrt(#terms)).
        w_l = math.sqrt(1.0 / n_terms)
        logits = w_l * (scalar + bias + point)             # (N, N, H)
        attn = torch.softmax(logits, dim=1)                # softmax over j

        # --- 4. Aggregate outputs -------------------------------------------
        # Scalar output.
        o_scalar = torch.einsum("ijh,jhc->ihc", attn, v).reshape(N, H * c)

        # Point output: aggregate VALUE points in the GLOBAL frame, then map the
        # result back into residue i's own (local) frame. Because sum_j attn = 1,
        # a global translation cancels here — the local coords are invariant.
        o_pts_global = torch.einsum("ijh,jhpx->ihpx", attn, vp_g)   # (N,H,Pv,3)
        o_pts_local = global_to_local(
            R, t, o_pts_global.reshape(N, H * Pv, 3)
        ).view(N, H, Pv, 3)
        o_pts_flat = o_pts_local.reshape(N, H * Pv * 3)
        o_pts_norm = o_pts_local.norm(dim=-1).reshape(N, H * Pv)     # invariant scalars

        parts = [o_scalar, o_pts_flat, o_pts_norm]

        # Pair-weighted output (optional).
        if z is not None:
            o_pair = torch.einsum("ijh,ijc->ihc", attn, z).reshape(N, H * self.c_z)
            parts.append(o_pair)

        # --- 5. Project back to the single representation --------------------
        return self.to_out(torch.cat(parts, dim=-1))                # (N, c_s), invariant
