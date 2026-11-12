"""
01_embed_and_probe.py — Embed protein sequences with a frozen ESM-2 and train a
linear probe on a per-sequence label.

This is the "embeddings as features" workflow that is ~80% of industrial pLM
use (docs/plm/README.md §5). The pedagogical point: a *linear* probe on frozen
embeddings is already strong — the self-supervised representation did the work,
not the classifier. If a linear head on frozen features works, the structure/
function signal is linearly present in the embedding.

VERIFICATION BOUNDARY
---------------------
- VERIFIED IN-SANDBOX: mean-pool aggregation, the linear-probe train/eval loop,
  and the standardization, via `_selftest()` on a synthetic linearly-separable
  dataset (run: `python 01_embed_and_probe.py --selftest`). The probe reaches
  ~100% on separable synthetic data, confirming the loop learns.
- NOT RUN HERE (needs downloaded weights): the ESM forward pass. Written
  against the real fair-esm API; run locally.

USAGE (local)
-------------
    pip install fair-esm torch
    python 01_embed_and_probe.py --model esm2_t12_35M_UR50D \\
        --fasta sequences.fasta --labels labels.csv
    python 01_embed_and_probe.py --selftest   # mechanism check, no weights
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch import Tensor


# ----------------------------------------------------------------------------
# Aggregation + probe (VERIFIED IN-SANDBOX via _selftest)
# ----------------------------------------------------------------------------

def mean_pool(token_embeddings: Tensor, lengths: Tensor) -> Tensor:
    """Mean-pool per-residue embeddings to one vector per sequence, ignoring
    padding and special tokens.

    Args:
        token_embeddings: (B, T, D) — includes <cls> at 0 and padding after the
            sequence. We assume special tokens already excluded by `lengths`.
        lengths: (B,) true residue counts (excluding <cls>/<eos>/pad).

    Returns:
        (B, D) sequence embeddings.

    Mean-pooling over residues is the standard sequence-level summary; the <cls>
    embedding is an alternative but mean-pool is more robust for ESM-2.
    """
    B, T, D = token_embeddings.shape
    # Residues live at positions 1 .. length (position 0 is <cls>).
    idx = torch.arange(T, device=token_embeddings.device)[None, :]      # (1, T)
    mask = (idx >= 1) & (idx <= lengths[:, None])                       # (B, T) bool
    mask_f = mask.unsqueeze(-1).to(token_embeddings.dtype)             # (B, T, 1)
    summed = (token_embeddings * mask_f).sum(dim=1)                    # (B, D)
    counts = mask_f.sum(dim=1).clamp(min=1.0)                          # (B, 1)
    return summed / counts


class LinearProbe(nn.Module):
    """A single linear layer — deliberately the simplest possible head, so that
    performance is attributable to the embedding, not the classifier."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def train_probe(
    features: Tensor,           # (N, D)
    labels: Tensor,             # (N,) long
    n_classes: int,
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[LinearProbe, float]:
    """Standardize features, train a linear probe, return (probe, val_accuracy).

    Standardization (zero mean, unit var per dim) matters: ESM embeddings have
    very different per-dimension scales and an un-normalized linear probe trains
    poorly.
    """
    torch.manual_seed(seed)
    N, D = features.shape

    perm = torch.randperm(N)
    n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mu = features[tr_idx].mean(0, keepdim=True)
    sd = features[tr_idx].std(0, keepdim=True).clamp(min=1e-6)
    feats = (features - mu) / sd

    probe = LinearProbe(D, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(feats[tr_idx])
        loss = loss_fn(logits, labels[tr_idx])
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        val_pred = probe(feats[val_idx]).argmax(dim=-1)
        acc = (val_pred == labels[val_idx]).float().mean().item()
    return probe, acc


# ----------------------------------------------------------------------------
# ESM glue (NOT RUN IN SANDBOX — written against real fair-esm API)
# ----------------------------------------------------------------------------

def embed_sequences(sequences: list[str], model_name: str, batch_size: int = 8) -> Tensor:
    """Mean-pooled ESM-2 embeddings, (N, D). Requires fair-esm + weights."""
    import esm  # type: ignore

    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers  # last layer

    out: list[Tensor] = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        _, _, toks = bc([(f"s{i+j}", s) for j, s in enumerate(chunk)])
        lengths = torch.tensor([len(s) for s in chunk])
        with torch.no_grad():
            res = model(toks, repr_layers=[repr_layer], return_contacts=False)
        tok_emb = res["representations"][repr_layer]  # (B, T, D)
        out.append(mean_pool(tok_emb, lengths))
    return torch.cat(out, dim=0)


# ----------------------------------------------------------------------------
# In-sandbox self-test of the verifiable core
# ----------------------------------------------------------------------------

def _selftest() -> None:
    torch.manual_seed(0)

    # mean_pool: position 0 is <cls> and must be ignored; padding past length too.
    B, T, D = 2, 6, 4
    emb = torch.zeros(B, T, D)
    emb[0, 1:4] = torch.tensor([1.0, 2.0, 3.0, 4.0])   # 3 real residues, value 1..4
    emb[0, 0] = 999.0                                   # <cls>, must be excluded
    emb[0, 4:] = -999.0                                 # padding, must be excluded
    emb[1, 1:3] = torch.tensor([2.0, 2.0, 2.0, 2.0])
    lengths = torch.tensor([3, 2])
    pooled = mean_pool(emb, lengths)
    assert torch.allclose(pooled[0], torch.tensor([1.0, 2.0, 3.0, 4.0])), f"pool0 {pooled[0]}"
    assert torch.allclose(pooled[1], torch.tensor([2.0, 2.0, 2.0, 2.0])), f"pool1 {pooled[1]}"

    # probe loop learns a linearly-separable synthetic task -> high val accuracy.
    N, Dd, C = 200, 16, 3
    centers = torch.randn(C, Dd) * 4.0
    y = torch.randint(0, C, (N,))
    X = centers[y] + torch.randn(N, Dd) * 0.5  # tight clusters -> separable
    _, acc = train_probe(X, y, C, epochs=150, val_frac=0.25, seed=1)
    assert acc > 0.95, f"probe should nail separable data, got val acc {acc:.3f}"

    print(f"[selftest] mean_pool exact; linear probe val_acc={acc:.3f} on separable data: ALL PASS")


def _read_fasta(path: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--fasta", help="sequences, FASTA")
    ap.add_argument("--labels", help="CSV: id,label (integer-encodable)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not (args.fasta and args.labels):
        ap.error("provide --fasta and --labels (or --selftest)")

    import csv
    seqs = _read_fasta(args.fasta)
    with open(args.labels) as fh:
        label_map = {row[0]: row[1] for row in csv.reader(fh) if row}
    classes = sorted(set(label_map.values()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    ids = [i for i in seqs if i in label_map]
    sequences = [seqs[i] for i in ids]
    labels = torch.tensor([cls_to_idx[label_map[i]] for i in ids])

    feats = embed_sequences(sequences, args.model)  # needs weights
    _, acc = train_probe(feats, labels, len(classes))
    print(f"linear-probe val accuracy = {acc:.3f}  [{args.model}, {len(ids)} seqs, {len(classes)} classes]")


if __name__ == "__main__":
    main()
