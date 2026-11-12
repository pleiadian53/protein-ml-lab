# Phase 3 — Equivariance & Geometric Deep Learning

> **Status: 🎯 active.** The goal is to make E(n)-equivariance *concrete*: write
> an equivariant layer from memory and prove its equivariance in two lines.

## Why this phase

EGNN is the minimal model that makes equivariance tangible. Everything
downstream is "the same idea with more bells":

- **AF2 Invariant Point Attention (IPA)** — attention made invariant by
  expressing points in per-residue local frames (Phase 2).
- **AF3 / Boltz diffusion decoder** — an equivariant denoiser over atomic
  coordinates, exactly like the EGNN coordinate update.

## Core idea (the whole EGNN paper in four lines)

Split node state into two kinds:

- `h_i` — **invariant** features (scalars). Unchanged by rotation/translation.
- `x_i` — **equivariant** coordinates (vectors). Transform with the input.

Two rules guarantee equivariance *by construction*:

1. Messages depend on coordinates **only** through the invariant squared
   distance `||x_i - x_j||^2`.
2. Coordinates update **only** by adding multiples of relative vectors
   `(x_i - x_j)`.

## Code pointers

- Layer implementation: [`src/protml/egnn.py`](../../src/protml/egnn.py)
- Numerical proof + verification: [`examples/equivariance/equivariance_check.py`](../../examples/equivariance/equivariance_check.py)

## The whiteboard proof

The two load-bearing facts (full derivation lives in the header of
`equivariance_check.py`):

1. **Distance invariance** — `||(Rx_i + t) - (Rx_j + t)||^2 = ||x_i - x_j||^2`,
   so every MLP input is unchanged ⇒ `h'` is invariant.
2. **Relative-vector equivariance** — `(Rx_i + t) - (Rx_j + t) = R(x_i - x_j)`,
   so `x'_i → R x'_i + t`.

Stacking layers preserves both properties, so the whole network is E(3)-equivariant.

## Run the check

```bash
pip install torch
PYTHONPATH=src python examples/equivariance/equivariance_check.py
```

Verifies E(3)-equivariance to float64 precision (~1e-16).
