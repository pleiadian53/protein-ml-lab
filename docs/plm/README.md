# Phase 1 — Protein Language Models (ESM)

> **Status: 🎯 active.** Builds directly on transformer/MLM priors — the new
> content is *why* masked language modeling on raw sequence learns 3D
> structure, and the lineage connecting it to pre-deep-learning contact
> prediction. Derivations first, empirical drivers in `examples/plm/`.

---

## 0. Where pLMs sit

The **representation** layer of the stack. A protein language model maps a raw
amino-acid sequence to per-residue embeddings, trained self-supervised on
hundreds of millions of unlabeled sequences (UniRef). No MSA, no structure
labels at training time. Downstream tasks — contact prediction, variant effect,
fitness, structure (ESMFold) — are built on these embeddings.

The central tension from Phase 0: **PDB (~220k structures) is ~1000× smaller
than UniRef (~250M sequences).** Supervised structure learning is data-starved;
self-supervised sequence learning is data-rich. pLMs convert the abundant
unlabeled signal into representations that transfer to the scarce labeled task.

## 1. The model: ESM-2 in one paragraph

ESM-2 (Lin et al., 2023) is a BERT-style encoder transformer over a vocabulary
of the 20 amino acids plus special tokens (`<cls>`, `<eos>`, `<pad>`, `<mask>`).
Trained with **masked language modeling (MLM)**: mask ~15% of residues, predict
them from context. Scaled 8M → 15B parameters. The headline empirical claim:
**structural information emerges in the weights purely from sequence MLM** —
contact maps are linearly decodable from attention, and a folding head
(ESMFold) reaches near-AF2 accuracy without any MSA. ESM-2 also swaps learned
positional embeddings for **RoPE** (rotary), which matters for length
generalization (and connects to your ViT positional-encoding thread).

## 2. Why MLM on sequence learns structure — the derivation chain

This is the centerpiece derivation. Three links: **MLM ≈ pseudolikelihood ≈
Potts model (DCA) → contacts.**

### 2.1 The biological premise: coevolution

(From Phase 0.) Residues in 3D contact coevolve: a destabilizing mutation at
position *i* is compensated by a mutation at its spatial neighbor *j*. So the
*joint* distribution over sequences `P(x_1, ..., x_L)` carries spatial
information in its **pairwise dependencies**. Any model that captures those
pairwise dependencies has implicitly captured contacts.

### 2.2 The classical route: Potts model / Direct Coupling Analysis

Pre-deep-learning contact prediction (Marks 2011, Morcos 2011) fits a Potts
model — a pairwise Markov Random Field — to the columns of an MSA:

```
P(x) = (1/Z) exp(  Σ_i h_i(x_i)  +  Σ_{i<j} J_ij(x_i, x_j)  )
```

- `h_i(·)`  : per-position field (single-site preferences), 20-vector.
- `J_ij(·,·)`: pairwise coupling, a 20×20 matrix per pair (i,j).
- `Z`       : partition function (intractable — this is the whole problem).

The learned couplings `J_ij` (after average-product correction to remove
phylogenetic/entropic bias) give contact scores: large `‖J_ij‖` ⇒ residues
i,j in contact. This *worked* and powered pre-AF contact prediction. Two
limitations: it needs a deep MSA per protein, and `Z` is intractable so you fit
by approximation — which is exactly where pseudolikelihood enters.

### 2.3 The bridge: pseudolikelihood

The intractable `Z` is dodged by maximizing **pseudolikelihood** instead of
likelihood — replace the joint by the product of per-site conditionals:

```
PL(x) = Π_i  P(x_i | x_{\i})          (x_{\i} = all positions except i)
```

For the Potts model, each conditional is a clean softmax with **no partition
function over the whole sequence** — only a local normalization over 20 amino
acids at site i:

```
P(x_i = a | x_{\i}) = softmax_a (  h_i(a) + Σ_{j≠i} J_ij(a, x_j)  )
```

Pseudolikelihood is a consistent estimator (exact in the large-data limit) and
made Potts-model contact prediction (plmDCA) practical. **Key observation:
this conditional is exactly a masked-token prediction** — "given all other
residues, predict residue i."

### 2.4 The modern route: MLM is amortized, nonlinear pseudolikelihood

MLM trains a network to model precisely the same conditional
`P(x_i | x_{\i})` — that *is* the masked-prediction objective. Two differences
from plmDCA, both upgrades:

1. **Amortized.** plmDCA fits a fresh `{h, J}` per protein family (per MSA).
   ESM learns *one* set of weights that produces the conditionals for *any*
   sequence via a forward pass — no per-protein MSA, no per-protein fitting.
   This is why ESMFold needs no MSA at inference.
2. **Nonlinear / higher-order.** Potts couplings are strictly pairwise and
   linear in the one-hot features. The transformer's attention composes
   multi-residue, context-dependent interactions — it can represent
   dependencies a pairwise MRF cannot.

> **The short version.** "MLM is an amortized, nonlinear generalization of
> the pseudolikelihood objective that made Potts-model DCA tractable. Both model
> `P(residue_i | rest)`; coevolutionary (pairwise) structure in that conditional
> is what encodes contacts, which is why both recover contact maps — but the
> transformer amortizes across all proteins in one set of weights and captures
> higher-order, context-dependent couplings, removing the per-protein MSA that
> DCA required."

### 2.5 Why this means attention maps ≈ contact maps

If the pairwise dependency structure lives in the model, *where* is it? Largely
in the **attention maps**. A symmetrized, background-corrected combination of
attention heads is linearly predictive of contacts: ESM's contact head is
literally a **logistic regression over the (symmetrized) attention maps** of the
frozen transformer. This is the empirical object the `02_*` driver reproduces.

A subtlety worth carrying: not every head encodes contacts; a *sparse subset*
does, and a small supervised regression selects/weights them. That's why "look
at attention" is true but "any head = contacts" is false.

## 3. ESMFold and the MSA-free trade-off (Phase 1 ↔ Phase 2 hinge)

ESMFold = frozen ESM-2 embeddings → a folding trunk (a trimmed AF2-style
structure module). ~6× faster than AF2 at inference; modestly lower accuracy on
proteins with deep MSAs.

The natural question — *when does MSA-free win?* — answer: **orphan proteins
and fast-mutating regions** with few/no homologs. AF2's accuracy degrades as the
MSA gets shallow (less coevolution signal); ESMFold has no MSA to be shallow, so
on orphans it can win outright. The two approaches are complementary, not
strictly ranked. This is also the conceptual handoff to Phase 2: the *folding
trunk* ESMFold borrows is exactly AF2's structure module, which we dissect next.

## 4. ESM-3 (frontier, brief)

ESM-3 (Hayes et al., 2024) is multimodal: **sequence + structure + function as
discrete tokens**, jointly masked-trained, so you can prompt in any modality and
generate the others (sequence→structure, structure→sequence, partial→completed).
It blurs the representation/prediction/design layers the three-layer model
separates. Know it exists, know "structure is tokenized via a learned
VQ-VAE-style codebook so it shares the masked-token objective with sequence,"
and that it's the current "foundation model for biology" banner.

## 5. The empirical drivers (use-case-gated, in `examples/plm/`)

Per the project convention, reusable code lives in `src/protml/` and is imported
by these drivers; the dependency arrow is `examples/ → src/`, never the reverse.
Anything prototyped alongside a driver is lifted to `src/protml/` as soon as a
second consumer wants it.

- **`01_embed_and_probe.py`** — load an ESM-2 checkpoint (model-agnostic, pick at
  runtime), embed sequences, train a linear probe on a per-sequence label. The
  "embeddings as features" workflow that is ~80% of industrial pLM use. Shows
  that a *linear* probe on frozen embeddings is already strong — the
  representation did the work.
- **`02_contacts_from_attention.py`** — extract the symmetrized attention maps
  from a frozen ESM-2, recover a contact-map prediction, and compare to true
  contacts derived from a PDB structure (precision@L/5, the standard metric).
  This is §2.5 made empirical — *see* the coevolution signal in the attention.

> **Verification status.** The contact-extraction math (symmetrization, APC
> background correction, precision@k), the probe training loop, and all
> argument/IO plumbing are unit-tested in-sandbox against synthetic tensors (see
> the `_selftest` blocks). The ESM forward pass itself is written against the
> real `fair-esm` API but was **not** run here (no weight access in the build
> sandbox) — run locally to verify end-to-end. This is the verification boundary
> called out when the drivers were commissioned.

## 6. Derivations to be able to do cold

1. Write the Potts model and its pseudolikelihood conditional; show the
   conditional is a per-site softmax with no global `Z`.
2. State precisely how MLM's objective relates to that conditional, and name the
   two ways MLM generalizes it (amortization, nonlinearity).
3. Explain why coevolution (pairwise dependency) is the quantity that encodes
   contacts, and why that makes both DCA and pLMs recover contact maps.
4. Explain the ESM contact head as logistic regression on attention, and why
   only a subset of heads carries the signal.
5. Explain when ESMFold (MSA-free) beats AF2 (MSA-based) and why.

## 7. Reading

1. Rives et al. 2021 (ESM-1b) — "structure emerges from scale" framing.
2. Lin et al. 2023 (ESM-2 / ESMFold) — the model + MSA-free folding.
3. Hayes et al. 2024 (ESM-3) — multimodal tokens.
4. Background (optional, for the DCA link): Marks et al. 2011; Ekeberg et al.
   2013 (plmDCA — the pseudolikelihood-for-Potts paper that makes §2.3 concrete).

## 8. Connections

- **← Phase 0:** coevolution, MSA, contact map, precision@L/k metric.
- **← ViT prior:** RoPE positional encoding; attention-map interpretation.
- **→ Phase 2 (AF2):** ESMFold's folding trunk *is* AF2's structure module; the
  MSA-vs-MSA-free trade-off frames why AF2 leans on the MSA so hard.
- **→ Phase 4 (AF3):** ESM-3's tokenized structure + masked generation is a
  discrete-token cousin of AF3's continuous diffusion over coordinates.
