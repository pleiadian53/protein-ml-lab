# Action Operators in Protein ML — the *Given* Pole

> **Status: 🌱 new.** Conceptual study, grounded in the already-verified frame/IPA/FAPE machinery of [Phase 2](../alphafold2/README.md) and the equivariance work of [Phase 3](../equivariance/README.md). No new code is required to *understand* this phase — it re-reads code that already exists; a from-scratch SE(3) flow sampler is a reasonable next artifact if depth is wanted.

This phase connects protein-ML structure modeling to the **action-operator** framework developed in the sibling project [ssl-lab](https://pleiadian53.github.io/ssl-lab/) (which in turn builds on [GRL](https://github.com/pleiadian53/GRL)). The connection is not an analogy — it is the *same construction*, evaluated at the opposite end of one dial.

## The one idea

The action-operator framework says: **an action is not a label, it is an operator $\hat O : \mathcal{S} \to \mathcal{S}$** the model constructs and applies to a state. In ssl-lab the operator acts on a *learned latent* and its structure is *discovered from data*. Protein ML is the other pole — the **given** pole — where the operator's structure is handed to us by physics:

- the state is a set of **rigid frames**, one per residue, $T_i = (R_i, t_i) \in SE(3)$;
- the operators are honest **$SE(3)$ group elements** — rotations and translations — not free matrices;
- **equivariance is *demanded*** (a hard symmetry the architecture must respect by construction), not *learned* as a soft penalty.

> **The dial, in one line.** ssl-lab: operator structure **learned**, equivariance **learned** (free / named generators on a behavioral latent). Protein ML: operator structure **given** by $SE(3)$, equivariance **demanded** (the $\mathfrak{se}(3)$ generators, fixed). Same machinery — $f = \exp(M)$ acting on a state — with the generator algebra swapped. ssl-lab's [operator gallery](https://pleiadian53.github.io/ssl-lab/action_operator/02-operator-gallery/) calls this the `SE3Basis`; this phase is what that basis *means* for real proteins.

## Why this is the tightest instance of the framework

In the behavioral setting the real-world operator $\hat O_\theta$ is inaccessible and its latent shadow $f_\theta$ must be learned, with only an approximate, *associational* link between them. In proteins there is no such gap: the operator **is** the rigid motion, written down exactly. The frame $T_i$ that AlphaFold places on a residue is literally the operator that carries the canonical local frame to its position in space, $x_{\text{global}} = R_i  x_{\text{local}} + t_i$. So the protein case is where the formalism is *cleanest* — the operators are group elements you can compose, invert, and exponentiate exactly, and equivariance is a theorem you can check to machine precision (as [Phase 2](../alphafold2/README.md) already does, to $\sim 5\times 10^{-16}$).

## Reading order

1. **[Residue frames as $SE(3)$ operators](01-residue-frames-as-se3-operators.md)** — each frame is an action operator; the structure module is a *rollout* of $SE(3)$ operators; equivariance demanded, checked from scratch.
2. **[Generative structure as a flow on $SE(3)$](02-generative-structure-as-a-flow-on-se3.md)** — RFdiffusion / FrameDiff / FrameFlow: evolve frames from noise by *integrating a flow on the group* — the protein twin of ssl-lab's temporal rollout.

## Notation

| Symbol | Meaning |
|---|---|
| $\hat O : \mathcal{S} \to \mathcal{S}$ | an **action operator** — a function that transforms a state |
| $T_i = (R_i, t_i) \in SE(3)$ | a **rigid frame** on residue $i$: rotation $R_i \in SO(3)$, translation $t_i \in \mathbb{R}^3$ |
| $\mathfrak{se}(3),\ \mathfrak{so}(3)$ | the **Lie algebras** (generators) of $SE(3)$, $SO(3)$ |
| $R = \exp(\hat\omega)$ | a rotation as the exponential of a skew-symmetric generator $\hat\omega \in \mathfrak{so}(3)$ |
| $\Delta(SE(3))$ | a **distribution over** rigid motions (e.g. $\mathrm{IGSO}(3) \times \mathcal{N}(\mathbb{R}^3)$) |

## Code this builds on (already in the repo, numerically verified)

- [`src/protml/frames.py`](../../src/protml/frames.py) — rigid frames $T_i = (R_i, t_i)$ and their composition/inverse.
- [`src/protml/ipa.py`](../../src/protml/ipa.py) — Invariant Point Attention: the equivariance-by-construction layer.
- [`src/protml/fape.py`](../../src/protml/fape.py) — the frame-aligned loss.
- [`examples/alphafold2/`](../../examples/alphafold2/) — $SE(3)$-invariance checks ($\sim 5\times 10^{-16}$); [`examples/equivariance/`](../../examples/equivariance/) — the EGNN equivariance proof.

## The sibling pole

- **[ssl-lab — Action Operators](https://pleiadian53.github.io/ssl-lab/action_operator/00-from-actions-to-operators/)** (foundation) and **[Operator World Models](https://pleiadian53.github.io/ssl-lab/operator_world_models/)** (synthesis) — the *learned* pole, where the operator acts on a behavioral latent and equivariance is discovered from data. This phase is its mirror image.
