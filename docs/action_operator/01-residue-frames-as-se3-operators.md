# Residue Frames as $SE(3)$ Operators

> **Status: 🌱 new.** Re-reads the verified frame/IPA machinery of [Phase 2](../alphafold2/README.md) through the action-operator lens. Prerequisite: the [phase overview](README.md), and ideally ssl-lab's [operator gallery](https://pleiadian53.github.io/ssl-lab/action_operator/02-operator-gallery/) (the `SE3Basis`).

The [overview](README.md) claimed that a protein structure model is an *action operator* model where the operators are honest $SE(3)$ group elements. This chapter makes that literal: a residue frame **is** an action operator, the structure module is a **rollout** of such operators, and the equivariance the framework asks for is here a theorem you can check to machine precision.

## The whole idea in four lines

1. AlphaFold's structure module never predicts $xyz$ directly. It predicts, per residue, a **rigid frame** $T_i = (R_i, t_i) \in SE(3)$ — a rotation $R_i \in SO(3)$ and a translation $t_i \in \mathbb{R}^3$ placed on the backbone.
2. That frame *is the operator* that carries a residue's canonical local coordinates into global space: $x_{\text{global}} = R_i  x_{\text{local}} + t_i$. Placing a residue = **applying an $SE(3)$ operator** to the canonical frame.
3. The module **iterates**: starting from a trivial "black-hole" frame, it repeatedly composes a predicted update onto the current frame. Predicting a structure is a **rollout of $SE(3)$ operators**.
4. Every step is built so that rotating the whole input rotates the whole output the same way — **equivariance by construction**, not by penalty.

## A frame is an operator

Write the canonical residue (idealized N–Cα–C geometry) at the origin. The frame $T_i = (R_i, t_i)$ acts on any local point $x$ by

$$
T_i \cdot x = R_i  x + t_i.
$$

This is exactly the affine-on-the-left form of an action operator $\hat O(s) = As + b$ — the *affine operator* from ssl-lab's [foundation](https://pleiadian53.github.io/ssl-lab/action_operator/00-from-actions-to-operators/), here with $A = R_i$ constrained to $SO(3)$ and $b = t_i$. Two frames compose by the $SE(3)$ group law,

$$
T_2 \cdot (T_1 \cdot x) = (R_2 R_1)  x + (R_2 t_1 + t_2),
\qquad T_2 \circ T_1 = (R_2 R_1,\ R_2 t_1 + t_2),
$$

and every frame is invertible, $T^{-1} = (R^{\top}, -R^{\top} t)$. So frames form a **group** under composition — which is precisely the action-operator claim that *operators compose, and a skill is a sequence of operators*, instantiated on rigid motions. (See [`src/protml/frames.py`](../../src/protml/frames.py) for `compose` and `inverse`.)

## The generator: $\exp$ from $\mathfrak{se}(3)$

Where does the operator come from? The same place as in the [gallery](https://pleiadian53.github.io/ssl-lab/action_operator/02-operator-gallery/): a generator, exponentiated. A rotation is the exponential of a **skew-symmetric** generator,

$$
R = \exp(\hat\omega), \qquad
\hat\omega = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix} \in \mathfrak{so}(3),
$$

with $\omega \in \mathbb{R}^3$ the axis-angle vector. The full rigid motion is $\exp$ of an element of $\mathfrak{se}(3)$ — six generators in all, three rotational ($\mathfrak{so}(3)$) and three translational. This is exactly ssl-lab's `SE3Basis`: the policy emits six coefficients $(\omega_x, \omega_y, \omega_z, t_x, t_y, t_z)$, and $\exp$ of their combination is a guaranteed-valid rigid transform.

> **The dial, made concrete.** Free generators $\Rightarrow \exp(M) \in GL(D)$ (any near-identity dynamics — ssl-lab's behavioral pole). Skew-symmetric generators $\Rightarrow \exp(M) \in SO(3)$ (a rotation, nothing else). $\mathfrak{se}(3)$ generators $\Rightarrow$ full rigid motion. *Constraining the generator algebra is how you choose the operator class* — and proteins want the rigid one, because that is the symmetry the physics has.

## The structure module is a rollout

AlphaFold's structure module runs several iterations (≈8). Each one predicts a small frame update $\Delta T_i$ and **composes it onto the running frame**:

```
T_i^(0)  →  T_i^(1)  →  ⋯  →  T_i^(K)
        ΔT^(0)     ΔT^(1)        (each ΔT a small SE(3) operator)
```

with $T_i^{(k+1)} = T_i^{(k)} \circ \Delta T_i^{(k)}$. This is structurally identical to ssl-lab's temporal rollout $z_{t} \to z_{t+1} \to \cdots$: a one-step operator, iterated, with the running state fed back in. The difference is only the state and the algebra — a latent vector advanced by $\exp(M)$ there, a rigid frame refined by an $SE(3)$ increment here.

## Equivariance, demanded — the whiteboard proof

In the behavioral pole, equivariance ($E(\hat O(s)) = f(E(s))$) is a *loss term* the model is nudged toward. In the protein pole it is a **hard constraint the architecture satisfies by construction**, because the physics is exactly symmetric: rotate and translate the whole protein and it is the same protein.

Formally, a global rigid motion $g = (Q, c) \in SE(3)$ acts on every frame by left-composition, $T_i \mapsto g \circ T_i$. Invariant Point Attention (IPA) is built so its scalar outputs are **invariant** and its predicted frame updates are **equivariant** under this action. The load-bearing trick is to express every point in a residue's *own* local frame before computing anything:

$$
g \circ T_i \ \text{acting, then} \ (g \circ T_i)^{-1}(g \cdot p) = T_i^{-1} g^{-1} g  p = T_i^{-1} p,
$$

so the global motion $g$ **cancels**: quantities computed in local frames do not see it. That is the two-line proof that frame-relative attention is invariant — the protein analogue of EGNN's "depend on coordinates only through $\lVert x_i - x_j \rVert^2$" trick from [Phase 3](../equivariance/README.md). The from-scratch check in [`examples/alphafold2/`](../../examples/alphafold2/) confirms it numerically: apply a random $g$, and IPA outputs move by exactly $g$ (or not at all, for invariants) to $\sim 5\times 10^{-16}$, while a naive coordinate baseline's un-superposed RMSD diverges to $\sim 14$ Å.

> **Why this matters for the framework.** The commuting square that ssl-lab must *train* toward, the protein model gets *for free* — because $\hat O_\theta$ is a known group element and the encoder is built to respect the group. The price is rigidity (you can only express rigid motions); the payoff is exactness (equivariance to machine precision, no data needed to learn it).

## Code pointers

- [`src/protml/frames.py`](../../src/protml/frames.py) — $T = (R, t)$, `compose`, `inverse`, and the $\exp$/log maps to $\mathfrak{se}(3)$.
- [`src/protml/ipa.py`](../../src/protml/ipa.py) — Invariant Point Attention; the equivariance-by-construction layer.
- [`examples/alphafold2/`](../../examples/alphafold2/) — the numerical $SE(3)$-invariance proof.

## Recap, and where next

A residue frame is an $SE(3)$ action operator; frames compose as a group; each is $\exp$ of an $\mathfrak{se}(3)$ generator (ssl-lab's `SE3Basis`); the structure module is a rollout of such operators; and equivariance is a built-in theorem rather than a learned penalty. We have, in other words, the *deterministic* operator picture — one structure, refined.

The next chapter makes it *generative*: instead of refining toward a single answer, **evolve frames from noise by integrating a flow on $SE(3)$** — RFdiffusion / FrameDiff / FrameFlow — which is the protein twin of ssl-lab's conditioned rollout. Continue to **[Generative structure as a flow on $SE(3)$](02-generative-structure-as-a-flow-on-se3.md)**.
