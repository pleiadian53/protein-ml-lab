"""
02_contacts_from_attention.py — Recover a contact map from frozen ESM-2
attention, and score it against true contacts from a PDB structure.

This is the empirical counterpart to docs/plm/README.md §2.5: the coevolution
signal that MLM captures is linearly decodable from attention maps. ESM's own
contact head is logistic regression over symmetrized, background-corrected
attention; here we use the unsupervised version (APC-corrected mean attention)
so it runs without the regression weights, then score precision@L/k.

VERIFICATION BOUNDARY
---------------------
- VERIFIED IN-SANDBOX: symmetrization, average-product correction (APC),
  precision@k, and contact-from-distance logic, via `_selftest()` on synthetic
  tensors (run: `python 02_contacts_from_attention.py --selftest`).
- NOT RUN HERE (needs downloaded weights / network): the ESM forward pass and
  PDB parsing. Written against the real fair-esm + biotite APIs; run locally.

USAGE (local)
-------------
    pip install fair-esm biotite torch
    # score ESM attention contacts for a PDB chain:
    python 02_contacts_from_attention.py --model esm2_t12_35M_UR50D --pdb 1ubq --chain A
    # quick mechanism check, no weights:
    python 02_contacts_from_attention.py --selftest
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# Contact-from-attention core (VERIFIED IN-SANDBOX via _selftest)
# ----------------------------------------------------------------------------

def symmetrize(x: Tensor) -> Tensor:
    """Symmetrize the last two dims: contacts are undirected (i,j)==(j,i)."""
    return x + x.transpose(-1, -2)


def average_product_correction(x: Tensor) -> Tensor:
    """Average Product Correction (APC) — the standard background subtraction
    from DCA contact prediction. Removes per-position and global biases
    (phylogeny, conservation) that inflate scores independent of true contact.

        APC_ij = x_ij - (row_i * col_j) / total

    Operates on a square (L, L) matrix. This is the same correction plmDCA uses
    on coupling norms; here applied to attention.
    """
    if x.dim() != 2 or x.size(0) != x.size(1):
        raise ValueError(f"APC expects a square (L, L) matrix, got {tuple(x.shape)}")
    row = x.sum(dim=1, keepdim=True)          # (L, 1)
    col = x.sum(dim=0, keepdim=True)          # (1, L)
    total = x.sum()
    expected = (row * col) / total.clamp(min=1e-12)
    return x - expected


def attention_to_contacts(attentions: Tensor) -> Tensor:
    """Reduce per-head attention maps to a single (L, L) contact score matrix.

    Args:
        attentions: (n_layers, n_heads, L, L) attention from a frozen model,
            already stripped of special tokens (no <cls>/<eos> rows/cols).

    Returns:
        (L, L) symmetric, APC-corrected, zero-diagonal contact scores.

    The unsupervised reduction: average heads, symmetrize, APC. (ESM's supervised
    head instead learns a logistic-regression weight per head — see §2.5. The
    mean is a reasonable weight-free proxy for a mechanism demo.)
    """
    if attentions.dim() != 4:
        raise ValueError(f"expected (n_layers, n_heads, L, L), got {tuple(attentions.shape)}")
    mean_attn = attentions.mean(dim=(0, 1))   # (L, L) average over layers & heads
    scores = average_product_correction(symmetrize(mean_attn))
    scores.fill_diagonal_(0.0)
    return scores


def contacts_from_distances(dist: Tensor, threshold: float = 8.0) -> Tensor:
    """Binary contact map from a (L, L) pairwise distance matrix (Angstroms)."""
    contacts = (dist < threshold).to(torch.float32)
    contacts.fill_diagonal_(0.0)
    return contacts


def precision_at_k(
    scores: Tensor,
    true_contacts: Tensor,
    *,
    L: int,
    k_divisor: int = 5,
    min_separation: int = 6,
) -> float:
    """Precision@L/k for the top-scored long-range pairs.

    Standard contact-prediction metric: restrict to pairs with |i-j| >=
    min_separation (long-range contacts are the hard, informative ones), take
    the top L/k_divisor scored pairs, report fraction that are true contacts.
    """
    Lh = scores.size(0)
    iu, ju = torch.triu_indices(Lh, Lh, offset=min_separation)  # upper triangle, separated
    pair_scores = scores[iu, ju]
    pair_truth = true_contacts[iu, ju]

    n_top = max(1, L // k_divisor)
    n_top = min(n_top, pair_scores.numel())
    top = torch.topk(pair_scores, n_top).indices
    return pair_truth[top].mean().item()


class Scored(NamedTuple):
    scores: Tensor
    true_contacts: Tensor
    precision: float


# ----------------------------------------------------------------------------
# ESM + PDB glue (NOT RUN IN SANDBOX — written against real APIs)
# ----------------------------------------------------------------------------

def esm_attentions(sequence: str, model_name: str) -> Tensor:
    """Run a frozen ESM-2 and return (n_layers, n_heads, L, L) attention with
    special tokens stripped. Requires `fair-esm` and downloaded weights."""
    import esm  # type: ignore

    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([("query", sequence)])

    with torch.no_grad():
        out = model(toks, need_head_weights=True, return_contacts=False)
    # attentions: (B, layers, heads, T, T); T includes <cls> (pos 0) and <eos> (last).
    attn = out["attentions"][0]
    attn = attn[:, :, 1:-1, 1:-1]  # strip special tokens -> (layers, heads, L, L)
    return attn.contiguous()


def pdb_distance_matrix(pdb_id: str, chain: str) -> Tensor:
    """Cb-Cb (Ca for glycine) distance matrix for a PDB chain, in Angstroms.
    Requires `biotite` and network access to fetch the structure."""
    import biotite.database.rcsb as rcsb  # type: ignore
    import biotite.structure.io.pdbx as pdbx  # type: ignore
    import biotite.structure as struc  # type: ignore
    import numpy as np

    path = rcsb.fetch(pdb_id, "bcif")
    arr = pdbx.get_structure(pdbx.BinaryCIFFile.read(path), model=1)
    arr = arr[(arr.chain_id == chain) & struc.filter_amino_acids(arr)]

    # Representative atom per residue: Cb, or Ca for glycine (no Cb).
    res_ids = np.unique(arr.res_id)
    coords = []
    for rid in res_ids:
        res = arr[arr.res_id == rid]
        cb = res[res.atom_name == "CB"]
        rep = cb if cb.array_length() > 0 else res[res.atom_name == "CA"]
        coords.append(rep.coord[0])
    xyz = torch.tensor(np.stack(coords), dtype=torch.float32)
    diff = xyz[:, None, :] - xyz[None, :, :]
    return diff.norm(dim=-1)


# ----------------------------------------------------------------------------
# In-sandbox self-test of the verifiable core
# ----------------------------------------------------------------------------

def _selftest() -> None:
    torch.manual_seed(0)

    # symmetrize
    a = torch.randn(5, 5)
    assert torch.allclose(symmetrize(a), symmetrize(a).transpose(0, 1)), "symmetrize not symmetric"

    # APC removes a rank-1 background exactly. APC subtracts row*col/total, which
    # nulls any outer product v*v^T. Use a genuinely rank-1 *symmetric* background
    # (v v^T) -- NOT (r c^T + c r^T), which is rank-2 and APC won't fully null.
    v = (torch.rand(6, 1) + 0.5)
    bg = v * v.t()  # symmetric, rank 1
    corrected = average_product_correction(bg)
    assert corrected.abs().max() < 1e-5, f"APC should null a rank-1 bg, got {corrected.abs().max()}"

    # precision@k: construct scores that rank true contacts first -> precision 1.0
    L = 30
    true = torch.zeros(L, L)
    pairs = [(0, 10), (3, 20), (5, 25), (1, 15), (8, 29), (2, 18)]
    for i, j in pairs:
        true[i, j] = true[j, i] = 1.0
    scores = torch.rand(L, L) * 0.01
    scores = symmetrize(scores)
    for rank, (i, j) in enumerate(pairs):       # give true pairs dominant scores
        scores[i, j] = scores[j, i] = 100.0 - rank
    p = precision_at_k(scores, true, L=L, k_divisor=5, min_separation=6)  # top L/5 = 6 pairs
    assert p == 1.0, f"precision@L/5 should be 1.0 for perfectly-ranked scores, got {p}"

    # contacts_from_distances threshold + zero diagonal
    dist = torch.full((4, 4), 10.0)
    dist[0, 1] = dist[1, 0] = 5.0
    c = contacts_from_distances(dist, threshold=8.0)
    assert c[0, 1] == 1.0 and c[2, 3] == 0.0 and c[0, 0] == 0.0, "contact thresholding wrong"

    # end-to-end shape sanity on the attention reduction. (APC is symmetric in
    # exact arithmetic; allow float32 round-off ~1e-6 in the symmetry check.)
    fake_attn = torch.rand(6, 4, L, L)  # 6 layers, 4 heads
    s = attention_to_contacts(fake_attn)
    assert s.shape == (L, L), f"bad shape {tuple(s.shape)}"
    assert torch.allclose(s, s.t(), atol=1e-5), f"not symmetric: {(s - s.t()).abs().max()}"
    assert s.diag().abs().max() == 0.0, "diagonal not zeroed"

    print("[selftest] symmetrize / APC / precision@k / contacts / reduction: ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run in-sandbox verifiable core only")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D", help="any esm.pretrained name")
    ap.add_argument("--pdb", help="PDB id, e.g. 1ubq")
    ap.add_argument("--chain", default="A")
    ap.add_argument("--seq", help="override: provide sequence directly instead of from PDB")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.seq and not args.pdb:
        ap.error("provide --seq or --pdb (or --selftest)")

    # NOTE: from here on requires fair-esm weights (+ biotite/network for --pdb).
    if args.pdb and not args.seq:
        # In a full local run you'd also extract the sequence from the structure;
        # kept explicit here so the dependency is obvious.
        raise SystemExit(
            "Local run: pass --seq for the chain (sequence extraction from PDB "
            "left to your local pipeline), plus --pdb/--chain for true contacts."
        )

    attn = esm_attentions(args.seq, args.model)
    scores = attention_to_contacts(attn)

    if args.pdb:
        dist = pdb_distance_matrix(args.pdb, args.chain)
        true = contacts_from_distances(dist)
        p = precision_at_k(scores, true, L=len(args.seq))
        print(f"precision@L/5 (long-range, |i-j|>=6) = {p:.3f}  [{args.model}, {args.pdb}_{args.chain}]")
    else:
        print(f"contact scores computed: shape {tuple(scores.shape)} (no PDB -> no scoring)")


if __name__ == "__main__":
    main()
