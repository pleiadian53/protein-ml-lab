# protein-ml-lab

A **depth-study sandbox for the full protein-ML stack** — protein language
models (ESM), structure prediction (AlphaFold lineage), and generative design
(RFdiffusion, ProteinMPNN, Chroma) — plus the cross-cutting frontier topics
in biology foundation models: mechanistic interpretability, agentic systems
over bio FMs, and scaling / RL post-training.

Focus is **first-principles depth**: clean derivations, small from-scratch
implementations that numerically verify the key properties (equivariance,
invariance, frame-aligned losses), and reproducible drivers built against the
real model APIs. Heavy training runs on a remote pod (A40+); everything in
this repo is sized to smoke-test locally on a laptop.

## Scope

The three layers of the field:

| Layer | What it answers | Representative models |
|---|---|---|
| **Representation** | How do we encode sequence/structure for a network? | ESM-2, ESM-3, ESMFold |
| **Prediction** | sequence (+MSA) → 3D structure | AlphaFold 2 / 3, Boltz, Chai |
| **Design** | target → new molecule | RFdiffusion, ProteinMPNN, Chroma, ProGen2, EvoDiff |

Cross-cutting tracks: equivariance / geometric DL, mechanistic interpretability,
agentic & reasoning systems over bio FMs, scaling and RL / post-training.
Imaging/microscopy foundation models are intentionally **out of scope** here —
a different modality.

## Status

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Structural biology vocabulary | ✅ draft — [`docs/vocab/`](docs/vocab/) |
| 1 | Protein language models (ESM) | 🎯 active — [`docs/plm/`](docs/plm/), [`examples/plm/`](examples/plm/) |
| 2 | AlphaFold 2 internals (Evoformer, IPA, FAPE) | 🎯 active — [`src/protml/ipa.py`](src/protml/ipa.py), [`src/protml/fape.py`](src/protml/fape.py), [`src/protml/frames.py`](src/protml/frames.py); checks verified |
| 3 | Equivariance & geometric DL | ✅ verified — [`docs/equivariance/`](docs/equivariance/), [`src/protml/egnn.py`](src/protml/egnn.py) |
| 4 | AlphaFold 3 & the diffusion turn | 📝 planned |
| 5 | Generative protein design (RFdiffusion, ProteinMPNN, Chroma) | 📝 planned |
| 6 | Drug-discovery applications | 📝 planned |
| 7 | Mechanistic interpretability of bio FMs | 📝 planned |
| 8 | Agentic & reasoning systems over bio FMs | 📝 planned |
| 9 | Scaling & post-training (RL, reward modeling) | 📝 planned |

Status legend: ✅ verified · 🎯 active · 📝 planned.

## Code organization

- `src/protml/` — reusable library code (importable modules: `egnn`, `frames`,
  `ipa`, `fape`, `compute`, …).
- `examples/<topic>/` — driver scripts that **import from `src/protml/`** and
  exercise the library end-to-end. The dependency arrow is always
  `examples → src/`, never the reverse.
- `notebooks/<topic>/` — intuition / EDA notebooks, mirroring `examples/<topic>/`.
- `docs/<topic>/` — per-phase write-ups: theory, derivations, results.

A capability that matures beyond depth-study can graduate into a stable
**application** under `src/protml/applications/<name>/` with its own write-up
under `docs/applications/<name>.md`, and from there to a **product** under
`docs/products/<name>/` once it has a stable public API, a benchmark, and a
test suite. Whether a given application graduates here or is spun out to its
own repo is decided per-application.

## Environment

Local smoke-test environment is managed with **mamba**; heavy training stays
on a remote pod (A40+ via SkyPilot / RunPod) and is not provisioned by this
repo today.

```bash
mamba env create -f environment.yml
mamba activate protml
```

The bio toolchain (MMseqs2, HHsuite, JackHMMER) needed for MSA construction is
deferred until Phase 2 actually needs it — install on demand:

```bash
mamba install -c bioconda mmseqs2 hhsuite hmmer
```

## Quickstart — verified numerical checks

The from-scratch geometric checks all run on CPU in seconds and confirm the
load-bearing equivariance / invariance properties to float64 precision.

```bash
# Phase 3 — EGNN E(3)-equivariance check
PYTHONPATH=src python examples/equivariance/equivariance_check.py

# Phase 2 — IPA single-rep invariance under global SE(3)
PYTHONPATH=src python examples/alphafold2/ipa_invariance_check.py

# Phase 2 — FAPE properties (rigid-invariant, chirality-sensitive)
PYTHONPATH=src python examples/alphafold2/fape_check.py
```

Reference results on this hardware: EGNN equivariance to ~1e-16, IPA invariance
to ~5e-16, FAPE rigid-invariant while un-superposed RMSD diverges to ~14 Å.

## Phase 1 drivers (ESM)

```bash
python examples/plm/01_embed_and_probe.py --selftest
python examples/plm/02_contacts_from_attention.py --selftest
```

Real runs need `fair-esm` (+ `biotite` for PDB contacts) and downloaded
weights; the ESM forward pass is written against the real API but not run in
the build sandbox. See [`docs/plm/README.md`](docs/plm/README.md) for the full
write-up.

## Compute & model tiers

Models are sized into three tiers per where they can run:

| Tier | Examples | Where it runs |
|---|---|---|
| **small** | EGNN / IPA / FAPE checks; ESM-2 8M / 35M embed + probe | Laptop (CPU/MPS), seconds |
| **medium** | ESM-2 150M / 650M frozen forward; small LoRA fine-tune | Laptop borderline; pod recommended |
| **large** | ESM-2 3B / 15B, ESMFold, AlphaFold 2 / 3, RFdiffusion | Pod only (CUDA, A40+) |

Check your local feasibility:

```bash
PYTHONPATH=src python examples/ops/compute_check.py
```
