# Phase 2 — AlphaFold2 Internals

> **Status: 🎯 active.** Theory + from-scratch IPA & FAPE implementations are
> in place and numerically verified ([`src/protml/ipa.py`](../../src/protml/ipa.py),
> [`src/protml/fape.py`](../../src/protml/fape.py),
> [`src/protml/frames.py`](../../src/protml/frames.py); checks under
> [`examples/alphafold2/`](../../examples/alphafold2/) confirm SE(3) invariance to
> ~5e-16 and rigid-invariant FAPE while un-superposed RMSD diverges to ~14 Å).
> Evoformer remains conceptual study — its triangle machinery is documented here
> with the load-bearing identities, and a from-scratch tri-attention check is a
> reasonable next code artifact if depth is wanted.

Reading: Jumper et al. 2021 (the AF2 paper) + its Supplementary Information,
which is where the algorithms actually live.

---

## 0. Where AF2 sits — the big picture

AF2 takes a target sequence plus its MSA (and optional templates) and predicts
a full 3D structure end-to-end. Two ideas carry it:

1. **A dual representation.** Throughout the network the protein lives as
   *two* tensors that are continuously cross-updated: a **single
   representation** `s ∈ ℝ^{L × c_s}` (one feature vector per residue) and a
   **pair representation** `z ∈ ℝ^{L × L × c_z}` (one feature vector per
   residue pair). MSA information flows into both via the Evoformer.
2. **A "residue gas" of frames.** The structure module never predicts xyz
   coordinates directly. It predicts, per residue, a **rigid frame**
   `T_i = (R_i, t_i) ∈ SE(3)` placed on the backbone N–Cα–C atoms, refines
   that gas over a few rounds of Invariant Point Attention, and only at the
   end reads atom positions out of the frames + side-chain torsion angles.

These two ideas — pair representation + frame gas — are what every downstream
prediction model (AF3, Boltz, Chai, ESMFold's folding trunk) inherits.

Pipeline overview:

```
sequence + MSA + templates
    │
    ▼
  EVOFORMER  (~48 blocks)
    │  alternately updates (s, z) using MSA row/col attention,
    │  outer-product mean (MSA → pair), and the triangle ops on pair
    ▼
  STRUCTURE MODULE  (8 IPA blocks)
    │  iterates a "gas" of per-residue SE(3) frames; reads atom positions
    │  + side-chain torsions at the end
    ▼
  RECYCLING  (≤ 3 outer iterations feeding (s_final, z_final, x_final) back)
    │
    ▼
  final 3D structure + pLDDT + PAE
```

## 1. Inputs and representations

- **Target sequence** `(L,)` — amino-acid tokens.
- **MSA representation** `m ∈ ℝ^{N_seq × L × c_m}` — `N_seq` aligned homologs
  (~10³ for proteins with deep MSAs). AF2 keeps a "main" MSA cluster and a
  separate "extra" MSA stack for efficiency; both feed the Evoformer.
- **Single representation** `s ∈ ℝ^{L × c_s}` (`c_s ≈ 384`) — distilled from
  the first row of the MSA representation when the Evoformer ends, and is
  what the structure module consumes.
- **Pair representation** `z ∈ ℝ^{L × L × c_z}` (`c_z ≈ 128`) — the
  "working memory" for geometry. Everything geometric — distances,
  contact probability, attention biases — is mediated through `z`.
- **Templates** (optional) — homologous structures projected into a template
  pair representation that's added into `z`. Helpful when MSAs are shallow.

The Phase-0 vocab covers MSA construction (HHblits / JackHMMER / MMseqs2)
and the coevolution rationale; that's the upstream of `m`.

## 2. Evoformer — the dual-track block

Each Evoformer block does two things in alternation: refine `m` using `z`
as a bias, and refine `z` using `m` as a signal. A block has roughly:

**MSA stack** (acts on `m`)
- **Row attention with pair bias** — for each MSA row (a sequence), do
  attention along the residue axis with an additive bias `b_{ij} = Wz_{ij}`
  read from the pair representation. This is the channel through which
  3D geometry shapes the MSA representation.
- **Column attention** — attention across sequences at the same column
  (residue), so within-column statistics (the MSA's coevolution signal) can
  inform per-residue features.
- **Transition (MLP).**

**Communication: MSA → pair**
- **Outer product mean** — collapse the MSA representation into a pair
  update via an outer product of two MLP projections of `m`, averaged over
  sequences. This is where coevolution flows from `m` into `z`:
  `z_{ij} += W₁(m_{·i}) ⊗ W₂(m_{·j})` (mean over `·`).

**Pair stack** (acts on `z`)
- **Triangle multiplicative update — outgoing edges.**
  `z_{ij} += W_out · sum_k  proj_a(z_{ik}) ⊙ proj_b(z_{jk})`
- **Triangle multiplicative update — incoming edges.**
  `z_{ij} += W_in · sum_k  proj_a(z_{ki}) ⊙ proj_b(z_{kj})`
- **Triangle self-attention — starting node** (attention over j for fixed i
  with bias from `z_{ik}` and `z_{jk}`).
- **Triangle self-attention — ending node** (the symmetric version).
- **Transition (MLP).**

### Why "triangles"? — the load-bearing geometric identity

The pair representation `z` plays the role of a learned distance map; for
real distances, the **triangle inequality** holds:

```
d(i, j)  ≤  d(i, k) + d(k, j)   for all k.
```

A pair update that touches `z_{ij}` ought to be consistent with every other
edge through a third node `k`. The two triangle multiplicative updates and
the two triangle self-attentions are exactly the four ways to route signal
through a third vertex `k` (i→k→j vs. k→i, k→j, etc.) — the symmetry of
the pair tensor across these four operators is what lets the network *enforce
geometric consistency without ever building an explicit distance matrix*.
This is the single most distinctive idea in AF2; without it the pair stack
would just be a 2D image, and the network would have no way to communicate
"if i is close to k and k is close to j, then i should be close-ish to j."

## 3. Structure module — the residue gas

The structure module turns `(s, z)` into 3D atom coordinates. Eight identical
blocks iterate the following on a gas of per-residue frames
`T_i = (R_i, t_i)`:

1. **Invariant Point Attention (IPA)** updates `s` using `z` and the current
   frames. IPA is the structural module's analogue of the Evoformer's
   attention: a standard attention plus a *point* term that reads geometry
   through the frames. See §3.1 — this is the centerpiece and is implemented
   from scratch here, with a numerical proof of its invariance.
2. **Frame update** — predict an SE(3) increment `(ΔR_i, Δt_i)` from `s` and
   compose it with the current frame: `T_i ← T_i ∘ ΔT_i`. The rotation
   increment is parameterised as a unit quaternion (small-angle friendly,
   automatically proper).
3. **Side-chain torsions** — predict the χ angles per residue from `s` and
   place side-chain atoms by rotation about the backbone bonds.
4. **Auxiliary FAPE loss** on each intermediate block (deep supervision).

After the eight blocks, the **final** atom coordinates are read out from the
frames and torsions.

### 3.1 Invariant Point Attention (IPA), in detail

For each residue `i, j` IPA combines three attention terms; the magic is in
where they get their geometry from.

| Term | Comes from | Why it's invariant |
|---|---|---|
| **Scalar** `q_i · k_j` | linear projections of `s` | `s` is itself invariant (no spatial index) |
| **Pair bias** `b_{ij}` | linear projection of `z_{ij}` | `z` is invariant |
| **Point distance** `Σ_p ‖T_i ∘ q_i^p − T_j ∘ k_j^p‖²` | query/key *points* produced in the local frame and mapped to the global frame | global rigid motion preserves distances |

All three terms are invariant ⇒ the attention weights are invariant. The
*value* points are aggregated in the global frame and then read back into
residue `i`'s own (transformed) local frame; because `Σ_j attn_{ij} = 1`,
the translation in any global rigid motion cancels exactly once and the
local-frame readout — and hence the updated `s` — is invariant.

**The two-box proof + verification** lives in the example driver
[`examples/alphafold2/ipa_invariance_check.py`](../../examples/alphafold2/ipa_invariance_check.py)
(verified to **~5.6e-16** in float64 on this hardware; a non-rigid scaling
control shifts the output by 0.46, so the test bites).

The conceptual relationship to EGNN (Phase 3) is exact: where EGNN routes
geometry through **invariant distances** and updates **equivariant
coordinates**, IPA routes geometry through points mapped via per-residue
frames and updates **invariant features**. Both rely on the same two
algebraic facts — distance invariance and the cancellation of translations
in mean-like aggregations.

## 4. FAPE — the structure loss

**Frame-Aligned Point Error**: instead of comparing predicted and true atoms
in one global frame (RMSD's mistake, which needs Kabsch superposition), FAPE
compares them in *every residue's local frame* and averages.

```
d_ij  = ‖ T_i^{-1} ∘ x_j  −  (T_i^*)^{-1} ∘ x_j^* ‖

FAPE  = (1/Z) · mean_{i,j} min(d_clamp, d_ij)            (AF2: Z = d_clamp = 10 Å)
```

The four properties that earn FAPE its place over RMSD:

1. **Superposition-free / rigid-invariant.** Applying one global rigid motion
   to the entire prediction (frames *and* points together) leaves every `d_ij`
   unchanged, because `T_i^{-1}` cancels the motion. **Verified**:
   [`fape_check.py`](../../examples/alphafold2/fape_check.py) shows FAPE
   stays at the 1e-3 eps-floor under a random rigid motion while un-superposed
   RMSD jumps to ~14 Å.
2. **Local.** Each frame scores all atoms relative to itself, so a single
   badly-placed domain doesn't corrupt the score everywhere — RMSD's failure
   mode on multi-domain proteins.
3. **Chiral.** Frames encode handedness, so a mirror image (an improper
   "rigid" motion) is correctly penalised — only **proper** SE(3) is in the
   invariance group. Verified: a coordinate flip drives FAPE to 0.74.
4. **Clamped.** Distances are capped at 10 Å so gross early-training errors
   don't blow up the gradient.

FAPE is the *training* loss with the same spirit as the *evaluation* metric
**lDDT** (Phase 0 vocab §6): both are local and superposition-free.

## 5. Recycling

The Evoformer + structure module + (in AF3) diffusion together form one
"forward pass." AF2 then runs the whole network **up to 4 times** (3
recycling iterations), feeding `(s_final, z_final, atom-coords)` from
iteration `n` back into iteration `n+1` as additional inputs (after small
projections). Two things to know:

- **Stop-gradient through recycles.** Only the *final* iteration receives
  gradient; the earlier passes use `detach()` on their outputs. This makes
  recycling cheap to add (no quadratic memory in recycle count) and avoids
  unrolling a huge graph.
- **Why it helps.** Each pass gives the pair representation a stronger
  "geometric prior" (the previous structure's coordinates discretised into
  distance bins), which then biases the next Evoformer + IPA pass. Empirically
  it accounts for several lDDT points; conceptually it is iterative refinement.

## 6. Other heads and auxiliary losses

The structure FAPE loss is the headline, but AF2's training objective is a
weighted sum of several heads:

- **Distogram head** — bin the `L × L` pair distances into 64 bins and cross-
  entropy against the true binned distances. Auxiliary; gives `z` an explicit
  geometric supervision signal.
- **Masked MSA head** — BERT-style: mask MSA tokens, predict them. Auxiliary;
  trains `m` to be a generative model of homologous sequences (the same
  conditional as the ESM derivation in Phase 1).
- **pLDDT head** — predicts the per-residue lDDT (a *confidence*, 0–100) from
  the final single representation. This is the per-residue confidence that
  colours every AF2 picture.
- **PAE head** — predicts the `L × L` "if I align on residue i, how wrong is
  residue j?" matrix. Tells you about inter-domain / inter-chain confidence.
- **Auxiliary FAPE** on every intermediate structure-module block (deep
  supervision).

## 7. The MSA's central role, and the AF2 ↔ ESMFold hinge

AF2's accuracy depends strongly on **MSA depth**. The Evoformer's pair stack
extracts coevolution; with few homologs the pair representation is poorly
informed, and accuracy degrades. This is the "**orphan-protein**" weakness
Phase 1 identified, and the exact slot where ESMFold (MSA-free, with a frozen
ESM-2 trunk) can win: no MSA to be shallow.

The **ESMFold folding trunk** is a trimmed AF2 structure module — same
IPA-based residue-gas refinement, just fed by frozen ESM-2 embeddings
instead of a freshly-built Evoformer. So everything in §3 transfers directly
to ESMFold; the difference is what stands in for the Evoformer.

## 8. Derivations to be able to do cold

The centerpiece set — the load-bearing identities of Phase 2:

1. Write IPA's three attention terms and prove the resulting single-rep
   update is invariant to a global SE(3) on the frame gas. (Two-box proof
   in [`ipa_invariance_check.py`](../../examples/alphafold2/ipa_invariance_check.py).)
2. Write FAPE from scratch and prove it is invariant to a global rigid motion
   of the prediction. Explain why this means "no Kabsch alignment needed,"
   and why a *reflection* is correctly penalised.
3. State the four triangle ops on the pair representation and explain which
   geometric identity (the triangle inequality on a learned distance map)
   they encode.
4. Sketch the AF2 forward pipeline (Evoformer → structure module → recycling)
   and name the four auxiliary heads with their losses.
5. Explain the recycling stop-gradient: why detach the earlier passes, why
   not backprop through all four.

## 9. Code in this phase

| File | What it is |
|---|---|
| [`src/protml/frames.py`](../../src/protml/frames.py) | Rigid-body frame helpers (`local_to_global`, `global_to_local`, `invert`, `compose`, `random_rotation`). Used by IPA, FAPE, and both checks. |
| [`src/protml/ipa.py`](../../src/protml/ipa.py) | Faithful-but-minimal IPA module: three attention terms, point features in local frames, output read back into the local frame. |
| [`src/protml/fape.py`](../../src/protml/fape.py) | FAPE loss + a `naive_rmsd` contrast for the check. |
| [`examples/alphafold2/ipa_invariance_check.py`](../../examples/alphafold2/ipa_invariance_check.py) | Two-box invariance proof in the docstring + numerical verification (~5.6e-16 in float64). |
| [`examples/alphafold2/fape_check.py`](../../examples/alphafold2/fape_check.py) | Demonstrates the four FAPE properties next to naive RMSD. |

Run them:

```bash
PYTHONPATH=src python examples/alphafold2/ipa_invariance_check.py
PYTHONPATH=src python examples/alphafold2/fape_check.py
```

## 10. Reading

1. Jumper et al. 2021 — "Highly accurate protein structure prediction with
   AlphaFold." *Nature*. The paper is short; the **Supplementary Information**
   is where the algorithms live (Alg 1–32). Algorithms 22 (IPA) and 28 (FAPE)
   are the two to read carefully.
2. Lin et al. 2023 (ESMFold half) — for the comparison in §7.
3. *Illustrated AlphaFold* by Elana Simon (online walkthrough) — useful as a
   second-pass explanation if any part of the supplement is unclear.
4. Boltz / Chai code (skim) — open reimplementations; useful to look at the
   structure module if a from-scratch Evoformer block becomes the next
   exercise.

## 11. Connections

- **← Phase 0:** torsion angles, frames, MSA, lDDT — Phase 0 vocab §3 / §5 /
  §6 are the prerequisites.
- **← Phase 1:** MLM/coevolution → why MSAs carry geometry; ESMFold's
  folding trunk is exactly AF2's structure module.
- **← Phase 3:** EGNN equivariance and IPA invariance share the same two
  algebraic facts (distance invariance, translation cancellation in
  mean-like aggregation). Read [`docs/equivariance/README.md`](../equivariance/README.md)
  in tandem with §3.
- **→ Phase 4 (AF3):** Evoformer becomes Pairformer (similar triangle ops,
  no MSA stack); structure module is replaced by a diffusion decoder over
  atom coordinates. The pair representation + the frame mental model carry
  over almost verbatim.
- **→ Phase 5 (design):** RFdiffusion is equivariant diffusion *over frames*
  — the design analogue of Phase 4's atom-diffusion, and a direct cousin of
  the structure module here.
