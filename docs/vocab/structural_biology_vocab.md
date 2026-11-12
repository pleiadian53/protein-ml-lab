# Phase 0 — Structural Biology Vocabulary

> **Status: ✅ first draft.** Prerequisite for all later phases. Goal is
> *fluent passive recognition* — you should read AF2/AF3/ESM papers without
> tripping on terms, and recognize them when a colleague uses them casually.
> Not memorization; comprehension. Budget 3–4 hours.
>
> Pitched for a strong ML/stats reader: the biology is explained, the math
> intuition is assumed. Where a term has an ML analogue, it's flagged → like this.

---

## 1. The structural hierarchy

A protein is a chain of amino acids that folds into a 3D shape. Four levels:

- **Primary structure** — the linear sequence of amino acids. The "string."
  Length `L` is the number of residues. → this is the token sequence.
- **Secondary structure** — local, recurring backbone motifs stabilized by
  hydrogen bonds. The two dominant ones: **α-helix** (coiled) and **β-sheet**
  (extended strands lying alongside each other). Plus loops/coils connecting
  them. Often coarse-grained into a 3-state (H/E/C) or 8-state (DSSP) label per
  residue. → a per-token local-structure label.
- **Tertiary structure** — the full 3D fold of a single chain. The set of 3D
  coordinates of all atoms.
- **Quaternary structure** — how multiple chains assemble into a complex
  (e.g. hemoglobin = 4 chains). AF-Multimer / AF3 / Boltz predict this.

**Why folding is hard (Levinthal's paradox, useful to know):** a chain has
astronomically many possible conformations, yet folds reliably in
microseconds–seconds. The sequence determines the fold (Anfinsen's dogma), but
the map is the prediction problem.

## 2. Amino acids, backbone, side-chain

- **Amino acid / residue** — one unit in the chain. "Residue" is the word used
  once it's polymerized into the chain (it has lost a water molecule in the
  bond — the "residue" of the reaction). 20 standard types, each a single
  letter (A, R, N, D, C, …). → vocabulary size 20 (+ a few special tokens).
- **Backbone (main chain)** — the repeating **N – Cα – C(=O)** unit, identical
  for every residue. This is the structural skeleton.
- **Cα ("C-alpha")** — the central carbon of each residue. Hugely important:
  it's the single point most often used to represent a residue's position. A
  "Cα-only" structure is `L × 3` coordinates. → think of it as the residue's
  positional embedding *in 3D space*.
- **Side-chain (R group)** — the variable part hanging off Cα that defines
  which of the 20 amino acids this is. Ranges from a single H (glycine) to
  large aromatic rings (tryptophan).
- **Residue identity vs conformation** — *which* amino acid (sequence) is
  separate from *where its atoms are* (structure). AF2 predicts backbone frames
  first, then side-chain placement.

## 3. Torsion (dihedral) angles — why geometry, not just coordinates

Bond lengths and bond angles in a protein are nearly fixed; the degrees of
freedom that matter are **rotations about bonds** — the torsion (dihedral)
angles. This is the single most important geometric fact for understanding why
the field cares about equivariance and frames.

- **Backbone torsions: φ (phi), ψ (psi), ω (omega).** Rotations about the
  three backbone bonds per residue. ω is almost always ~180° (planar peptide
  bond), so the real backbone freedom is essentially (φ, ψ) per residue.
- **Ramachandran plot** — the 2D (φ, ψ) density. Only certain regions are
  sterically allowed; α-helices and β-sheets occupy distinct basins. A classic
  sanity check on predicted structures.
- **Side-chain torsions: χ₁ … χ₄ (chi angles).** Rotations within the
  side-chain. Predicting these = "rotamer" / side-chain packing.

> **Key consequence.** A protein's conformation can be described *either* by a
> set of torsion angles (intrinsic, rotation/translation-invariant) *or* by
> Cartesian atom coordinates (extrinsic, changes under global rigid motion).
> Networks predict in Cartesian space but the *content* is rotation/translation
> invariant — which is exactly what equivariance machinery (Phase 3) and local
> frames (IPA, Phase 2) exist to handle.

## 4. Spatial-relationship representations

- **Contact map** — binary `L × L` matrix; entry (i,j) = 1 if residues i and j
  are "in contact" (typically Cβ–Cβ distance < 8 Å). Symmetric. The classic
  intermediate prediction target pre-AlphaFold. → an adjacency matrix.
- **Distance map / distogram** — `L × L` of pairwise distances, or
  (in AF2) a *distribution* over binned distances per pair (the "distogram"
  auxiliary head). → soft, probabilistic contact map.
- **Pair representation** — in AF2/AF3, an `L × L × c` tensor of learned
  features per residue pair. The network's working memory for geometry. →
  generalized, learned distogram.
- **Local frame / rigid body** — each residue gets a coordinate frame: a
  rotation `R ∈ SO(3)` + translation `t ∈ ℝ³`, placed on its backbone atoms.
  AF2's structure module predicts these frames ("residue gas" of free-floating
  frames that get refined). → per-token SE(3) pose.

## 5. Evolutionary information — the MSA

- **Homologs** — proteins related by common evolutionary ancestry, with similar
  sequence and usually similar structure.
- **MSA (Multiple Sequence Alignment)** — stack the target sequence with many
  homologs, aligned column-by-column (insertions/deletions handled with gaps).
  An `N_seq × L` array of amino-acid tokens.
- **Coevolution — the central signal.** If two residues are in contact in 3D, a
  mutation at one is often compensated by a mutation at the other to preserve
  the structure. So **correlated columns in the MSA imply spatial contact.**
  This is the evolutionary signal AF2 exploits, and it's *why* MSA depth
  drives accuracy.
- **Potts model / Direct Coupling Analysis (DCA)** — the pre-deep-learning way
  to extract contacts: fit a Markov-random-field (Potts) model to the MSA
  columns, read off direct couplings. → a pairwise MRF; MLM on sequences is a
  modern amortized cousin (see Phase 1). Worth knowing for the "what came
  before AlphaFold" question.
- **MSA tooling** — built with **HHblits**, **JackHMMER**, or the fast
  **MMseqs2** (the latter powers ColabFold). MSA construction is a real chunk of
  AF2's wall-clock cost.
- **Orphan proteins** — few or no detectable homologs → shallow/empty MSA →
  AF2 struggles. This is precisely where MSA-free models (ESMFold) have an edge:
  no MSA to be shallow.

## 6. Evaluation metrics — and why each exists

You will be asked "why not just RMSD?" Know the trade-offs.

- **RMSD (Root-Mean-Square Deviation)** — after optimal rigid superposition
  (the **Kabsch algorithm**), the RMS distance between corresponding atoms (Å).
  *Lower is better.* Weaknesses: (a) requires global alignment, so a single
  badly-placed domain wrecks the whole score (non-local); (b) not
  length-normalized, so it's hard to compare across proteins.
- **TM-score (Template Modeling score)** — ∈ (0, 1], length-normalized,
  *higher is better*. Down-weights large deviations so it's not dominated by a
  few outliers. Rule of thumb: **> 0.5 ⇒ same fold**, < 0.17 ⇒ random. The
  standard "did we get the global topology right" metric.
- **GDT-TS (Global Distance Test – Total Score)** — fraction of residues within
  a set of distance cutoffs (1/2/4/8 Å) after superposition, averaged. The
  historical **CASP** ranking metric. Higher better, 0–100.
- **lDDT (local Distance Difference Test)** — *superposition-free* and *local*:
  for each atom, what fraction of its local interatomic distances are preserved
  (within tolerance) vs the reference. 0–100, higher better. **lDDT is what AF2
  effectively optimizes / reports per-residue.** Because it's local, it isn't
  fooled by global mis-superposition and gives a per-residue confidence map.
- **pLDDT** — AF2's **predicted** lDDT, output per residue: the model's own
  confidence (0–100). Colors the famous AF2 structure pictures. Low pLDDT
  regions often correspond to intrinsically disordered regions.
- **PAE (Predicted Aligned Error)** — AF2's `L × L` predicted error map: "if I
  align on residue i, how wrong is residue j?" Tells you about relative
  domain placement / inter-chain confidence. → the off-diagonal companion to
  pLDDT.

> **The short version:** RMSD = global + outlier-sensitive + not
> normalized; TM-score = global topology, normalized; lDDT = local,
> superposition-free, per-residue, and the one AF2 cares about. FAPE (Phase 2)
> is the *training* loss and is frame-local by construction.

## 7. Data resources

- **PDB (Protein Data Bank)** — the ground truth: ~220k+ *experimentally*
  determined structures (X-ray crystallography, cryo-EM, NMR). The training
  signal. Small relative to sequence databases — this scarcity is *why*
  evolutionary/self-supervised signals matter so much.
- **UniProt / UniRef** — sequence databases: ~250M+ protein sequences, no
  structures. UniRef clusters them at identity thresholds (UniRef50/90/100).
  → the unlabeled corpus for self-supervised pretraining (ESM trains here).
- **AFDB (AlphaFold Protein Structure Database)** — ~200M+ *predicted*
  structures released by DeepMind/EMBL-EBI. Predictions, not ground truth, but
  vast coverage.
- **CASP (Critical Assessment of Structure Prediction)** — the biennial blind
  competition. AF2 won CASP14 (2020) decisively (median GDT-TS ~92), which is
  the event that "solved" single-domain prediction. CASP is the canonical
  benchmark framing in the field.
- **CAMEO / CASP15+** — ongoing/continuous assessments; CASP15 (2022) featured
  the next wave incl. complexes and ligands.

## 8. One-screen glossary (fast recall)

| Term | One-liner |
|------|-----------|
| Residue | one amino acid in the chain |
| Cα | central carbon; the residue's representative 3D point |
| Backbone | repeating N–Cα–C(=O) skeleton |
| Side-chain | variable R group; defines the amino acid |
| φ/ψ/ω | backbone torsion angles (ω ≈ 180°) |
| χ₁–χ₄ | side-chain torsion angles (rotamers) |
| Secondary structure | α-helix / β-sheet / loop (local motifs) |
| Contact map | L×L binary "within 8 Å" matrix |
| Distogram | L×L binned distance distribution |
| Pair representation | L×L×c learned geometric features (AF2/3) |
| Local frame | per-residue SE(3) pose (R, t) |
| MSA | aligned stack of homologous sequences |
| Coevolution | correlated MSA columns ⇒ spatial contact |
| Potts/DCA | pre-DL contact extraction from MSA |
| Orphan protein | few homologs ⇒ shallow MSA ⇒ MSA-free wins |
| RMSD | global superposed atom error (Å), outlier-sensitive |
| TM-score | normalized global topology, >0.5 = same fold |
| GDT-TS | CASP ranking metric (distance-cutoff fractions) |
| lDDT | local, superposition-free; what AF2 optimizes |
| pLDDT | AF2's predicted per-residue confidence |
| PAE | AF2's predicted pairwise alignment error |
| PDB | ~220k experimental structures (ground truth) |
| UniRef | ~250M sequences (SSL pretraining corpus) |
| AFDB | ~200M AlphaFold-predicted structures |
| CASP | biennial blind benchmark; AF2 won CASP14 |

---

## Self-check (can you answer these cold?)

1. Why does correlated mutation between two MSA columns imply a 3D contact?
2. A model scores great lDDT but mediocre RMSD. What likely happened?
3. Why are orphan proteins the case where ESMFold can beat AF2?
4. What are the *actual* degrees of freedom in a backbone conformation, and
   why does that motivate predicting frames/torsions rather than raw xyz?
5. Why is PDB (~220k) being ~1000× smaller than UniRef (~250M) the central
   tension that shapes the whole field?

> Answers are in the sections above; if any of these is shaky, re-read that
> section before moving to Phase 1.
