"""
egnn.py — An E(n)-Equivariant Graph Neural Network layer, from scratch.

Reference: Satorras, Hoogeboom, Welling, "E(n) Equivariant Graph Neural
Networks", ICML 2021 (arXiv:2102.09844).

WHY THIS EXERCISE (read first)
------------------------------
EGNN is the minimal model that makes E(n)-equivariance *concrete*. Once you can
write it from memory and prove its equivariance in two lines, the harder
machinery downstream is "the same idea with more bells":
  - AF2's Invariant Point Attention (IPA) = attention made invariant by
    expressing points in per-residue local frames (Phase 2).
  - AF3 / Boltz diffusion decoder = an equivariant denoiser over atomic
    coordinates; the noise is isotropic and the network is coordinate-
    equivariant, exactly like the position update below (Phase 4).

THE CORE IDEA (the whole paper in four lines)
---------------------------------------------
Split features into two kinds:
  - h_i : INVARIANT node features (type-0 / scalars). Unchanged by rotation.
  - x_i : EQUIVARIANT coordinates (type-1 / vectors). Rotate with the input.

Then obey two rules that GUARANTEE equivariance by construction:
  (R1) Messages depend on coordinates ONLY through the invariant squared
       distance  ||x_i - x_j||^2.  (A rotation/translation leaves it unchanged.)
  (R2) Coordinates are updated ONLY by adding multiples of the relative vectors
       (x_i - x_j).  (Translations cancel in the difference; rotations factor
       out of the whole sum — see the proof in `equivariance_check.py`.)

Everything else (the MLPs phi_e, phi_h, phi_x) operates on invariant scalars,
so it can be arbitrarily expressive without ever breaking equivariance.

NOTATION
--------
  N  : number of nodes (e.g. residues / atoms)
  d  : spatial dimension (3 for proteins)
  F  : node-feature width (the invariant channel)
Edges are given as a (2, E) index tensor `edge_index` = [src; dst], message
flows src -> dst (we aggregate incoming messages at dst).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EGNNLayer(nn.Module):
    """One E(n)-equivariant message-passing layer.

    Maps (h, x) -> (h', x') where h is invariant and x is equivariant.

    Args:
        feat_dim:   width F of the invariant node features h.
        hidden_dim: width of the internal MLPs.
        coord_agg:  'mean' or 'sum' aggregation for the coordinate update.
                    'mean' is more stable when node degree varies (recommended
                    for molecular graphs); the paper uses both.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        coord_agg: str = "mean",
    ) -> None:
        super().__init__()
        if coord_agg not in ("mean", "sum"):
            raise ValueError(f"coord_agg must be 'mean' or 'sum', got {coord_agg!r}")
        self.coord_agg = coord_agg

        # phi_e : edge MLP. Input = [h_i, h_j, ||x_i - x_j||^2]  (all invariant).
        # Output = the edge message m_ij (invariant).
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # phi_x : maps an edge message to a SCALAR weight for the coordinate
        # update. The scalar multiplies the relative vector (x_i - x_j), which
        # is what keeps the update equivariant (R2). No bias on the last layer
        # is a common choice so a zero message gives zero displacement.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # phi_h : node update MLP. Input = [h_i, aggregated messages] (invariant).
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(
        self,
        h: Tensor,          # (N, F)  invariant node features
        x: Tensor,          # (N, d)  equivariant coordinates
        edge_index: Tensor,  # (2, E) [src; dst], long
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]  # each (E,)

        # --- 1. Relative vectors and INVARIANT squared distance (R1) ---------
        rel = x[src] - x[dst]                       # (E, d) equivariant vector
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)  # (E, 1) INVARIANT scalar

        # --- 2. Edge messages: phi_e over invariant inputs only --------------
        edge_in = torch.cat([h[src], h[dst], dist2], dim=-1)  # (E, 2F+1)
        m_ij = self.edge_mlp(edge_in)                          # (E, hidden)

        # --- 3. Coordinate update: sum of scalar-weighted relative vectors ---
        # x'_i = x_i + agg_j  phi_x(m_ij) * (x_i - x_j)
        # phi_x(m_ij) is an invariant scalar; (x_i - x_j) is equivariant; their
        # product is equivariant, and summing equivariant vectors stays
        # equivariant. THIS is the line that does the geometric work.
        coord_w = self.coord_mlp(m_ij)              # (E, 1) invariant scalar
        coord_msg = coord_w * rel                   # (E, d) equivariant

        agg_coord = _scatter(coord_msg, dst, dim_size=x.size(0), reduce=self.coord_agg)
        x_out = x + agg_coord                        # (N, d)

        # --- 4. Node update: phi_h over invariant inputs only ----------------
        agg_msg = _scatter(m_ij, dst, dim_size=h.size(0), reduce="sum")  # (N, hidden)
        node_in = torch.cat([h, agg_msg], dim=-1)    # (N, F+hidden)
        h_out = h + self.node_mlp(node_in)           # residual; stays invariant

        return h_out, x_out


def _scatter(src: Tensor, index: Tensor, dim_size: int, reduce: str) -> Tensor:
    """Minimal scatter-add / scatter-mean over dim 0 (no external deps).

    Aggregates rows of `src` into `dim_size` buckets indexed by `index`.
    Kept dependency-free on purpose so the file is self-contained and readable;
    in real code use torch_scatter or torch_geometric.
    """
    out = src.new_zeros((dim_size, src.size(-1)))
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    if reduce == "mean":
        count = src.new_zeros((dim_size, 1))
        count.scatter_add_(0, index.unsqueeze(-1), torch.ones_like(index, dtype=src.dtype).unsqueeze(-1))
        out = out / count.clamp(min=1.0)
    return out


class EGNN(nn.Module):
    """A small stack of EGNN layers. Coordinates flow through equivariantly;
    invariant features are embedded in and read out at the ends."""

    def __init__(
        self,
        in_feat_dim: int,
        feat_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 4,
        coord_agg: str = "mean",
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_feat_dim, feat_dim)
        self.layers = nn.ModuleList(
            EGNNLayer(feat_dim, hidden_dim, coord_agg) for _ in range(n_layers)
        )

    def forward(
        self, h: Tensor, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return h, x
