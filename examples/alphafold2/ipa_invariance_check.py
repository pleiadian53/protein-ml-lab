"""
ipa_invariance_check.py — numerically verify that Invariant Point Attention is
invariant to a global rigid motion of the protein.

Run it:
    PYTHONPATH=src python examples/alphafold2/ipa_invariance_check.py

This is the Phase-2 analogue of the Phase-3 `equivariance_check.py`. There the
centerpiece was a two-line proof that an EGNN layer is E(3)-EQUIVARIANT. Here
the centerpiece is the two-box proof that IPA's single-representation update
is SE(3)-INVARIANT, confirmed to float64 precision (~1e-15).

THE WHITEBOARD PROOF
====================
Apply one global rigid motion T_g = (R_g, t_g) to every residue frame:

        (R_i, t_i)  ->  (R_g R_i,  R_g t_i + t_g)        for all i.

The single rep s and the pair rep z are scalar/pair features — they do NOT carry
a spatial index, so they are unchanged. We show every input to the attention is
unchanged, hence s' is unchanged.

Box 1 — the attention logits are invariant.
  Three terms make up each logit a_ij:
    • scalar  q_i·k_j   : built from s ⇒ unchanged.
    • pair    b_ij      : built from z ⇒ unchanged.
    • point   Σ_p ||T_i∘q_i^p − T_j∘k_j^p||² :
        a query point placed in the global frame transforms as
            T_i∘q  =  R_i q + t_i
                   -> R_g R_i q + R_g t_i + t_g  =  R_g (T_i∘q) + t_g,
        and likewise for the key point. Their difference loses the +t_g:
            (R_g a + t_g) − (R_g b + t_g) = R_g (a − b),
        and ||R_g (a−b)||² = ||a−b||² because R_g is orthogonal. ⇒ unchanged.
  All three terms unchanged ⇒ logits unchanged ⇒ attention weights unchanged.

Box 2 — the aggregated output, read in the local frame, is invariant.
  Value points are aggregated in the global frame with weights that sum to 1:
        o_i^global = Σ_j a_ij (T_j∘v_j^p)  ->  R_g o_i^global + t_g
  (the t_g survives exactly once because Σ_j a_ij = 1). Reading it back into
  residue i's OWN — now transformed — frame:
        o_i^local' = (R_g R_i)^T ( R_g o_i^global + t_g − (R_g t_i + t_g) )
                   = R_i^T R_g^T R_g (o_i^global − t_i)
                   = R_i^T (o_i^global − t_i)  =  o_i^local.
  The local point coords (and therefore their norms) are unchanged, so every
  input to the output projection is unchanged ⇒ s' is unchanged. ∎

The script confirms this, and — as a control — shows that a NON-rigid map
(anisotropic scaling) DOES change the output, so the test actually has teeth.
"""

from __future__ import annotations

import torch

from protml.frames import random_rotation
from protml.ipa import InvariantPointAttention


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64  # float64 so the invariance shows at ~1e-12, not ~1e-6

    N, c_s, c_z = 11, 32, 16
    ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, n_head=4).to(dtype).eval()

    # Random inputs: invariant single/pair reps + a random "residue gas" of frames.
    s = torch.randn(N, c_s, dtype=dtype)
    z = torch.randn(N, N, c_z, dtype=dtype)
    R = random_rotation(N, dtype=dtype)
    t = torch.randn(N, 3, dtype=dtype)

    with torch.no_grad():
        s_out = ipa(s, R, t, z)

        # --- Global rigid motion applied to every frame --------------------
        Rg = random_rotation(1, dtype=dtype)[0]      # (3, 3)
        tg = torch.randn(3, dtype=dtype)             # (3,)
        R_moved = torch.einsum("ij,njk->nik", Rg, R)         # R_g R_i
        t_moved = torch.einsum("ij,nj->ni", Rg, t) + tg      # R_g t_i + t_g
        s_out_moved = ipa(s, R_moved, t_moved, z)

        rigid_err = (s_out - s_out_moved).abs().max().item()

        # --- Negative control: a NON-rigid map should change the output ----
        scale = torch.tensor([1.0, 1.7, 0.4], dtype=dtype)   # anisotropic scaling
        t_scaled = t * scale
        s_out_scaled = ipa(s, R, t_scaled, z)
        nonrigid_diff = (s_out - s_out_scaled).abs().max().item()

        # --- Also confirm it holds with no pair representation -------------
        ipa_nopair = InvariantPointAttention(c_s=c_s, c_z=None, n_head=4).to(dtype).eval()
        s1 = ipa_nopair(s, R, t)
        s2 = ipa_nopair(s, R_moved, t_moved)
        rigid_err_nopair = (s1 - s2).abs().max().item()

    print("Invariant Point Attention — invariance check")
    print(f"  residues N = {N}, c_s = {c_s}, c_z = {c_z}, dtype = {dtype}")
    print(f"  max|s' − s'(global rigid motion)|        = {rigid_err:.3e}   (want ~0)")
    print(f"  max|s' − s'(global rigid motion)| no pair = {rigid_err_nopair:.3e}   (want ~0)")
    print(f"  max|s' − s'(non-rigid scaling)|          = {nonrigid_diff:.3e}   (want >0)")

    tol = 1e-9
    assert rigid_err < tol, f"IPA not invariant under rigid motion: {rigid_err}"
    assert rigid_err_nopair < tol, f"IPA (no pair) not invariant: {rigid_err_nopair}"
    assert nonrigid_diff > 1e-3, "non-rigid control failed to change output"
    print("\nPASS — IPA single-rep update is SE(3)-invariant; non-rigid control bites.")


if __name__ == "__main__":
    main()
