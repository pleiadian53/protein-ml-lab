# protein-ml-lab

protein-ml-lab develops new methods for modeling and designing proteins, built on the geometry that structure imposes. Protein structure lives on SE(3), the group of rotations and translations in three dimensions, and the state-of-the-art models for structure prediction and for protein and antibody design are built around how that group acts: equivariant layers, residue frames, frame-aligned losses. The lab works across the stack, from protein language models (ESM) to structure prediction in the AlphaFold lineage to generative design (RFdiffusion, ProteinMPNN, Chroma), together with the cross-cutting frontier questions in biology foundation models: mechanistic interpretability, agentic systems built over them, and scaling and RL post-training.

**Action operators against a known group.** The same mathematics, a Lie group acting on a state space with its Lie algebra supplying the infinitesimal view, underlies the action operators of two sibling projects. The idea originated in [GRL](https://github.com/pleiadian53/GRL), and [ssl-lab](https://github.com/pleiadian53/ssl-lab) uses it to model gene perturbations as actions on a cell's latent state. There the operators are learned from data, and nothing guarantees that they form a group. Here the group is given by physics, which makes this the place to develop action-operator theory and implementation against a case where the right answer is known: whether an operator learner recovers SO(3) or SE(3) structure when it is present, and which tools carry over to learned operators, from the exponential map and composition by the Baker–Campbell–Hausdorff formula to representation theory and equivariant layers.

Methods are built from first principles: derivations written out, small implementations that numerically verify the properties they rely on (equivariance, invariance, frame-aligned losses), and drivers written against the real model APIs. Heavy training runs on a remote GPU pod (A40 or larger); everything in this repository is sized to smoke-test on a laptop.

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

A capability that matures beyond a first implementation can graduate into a stable
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
