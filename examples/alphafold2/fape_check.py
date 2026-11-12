"""
fape_check.py — show the two properties that make FAPE the right structure loss:
it is invariant to a global rigid motion of the prediction (superposition-free),
and it is sensitive to genuine structural error (and to reflections).

Run it:
    PYTHONPATH=src python examples/alphafold2/fape_check.py

The punchline is the contrast with a naive (un-superposed) RMSD: move the whole
predicted structure by a random rotation+translation and RMSD explodes while
FAPE stays at zero, because FAPE reads every point in a local frame that moves
with the prediction.
"""

from __future__ import annotations

import torch

from protml.fape import fape_loss, naive_rmsd
from protml.frames import local_to_global, random_rotation


def main() -> None:
    torch.manual_seed(0)
    dtype = torch.float64

    N = 16  # treat one Cα point per residue, and one frame per residue
    R_true = random_rotation(N, dtype=dtype)
    t_true = torch.randn(N, 3, dtype=dtype) * 5.0
    x_true = torch.randn(N, 3, dtype=dtype) * 5.0

    # Perfect prediction.
    R_pred, t_pred, x_pred = R_true.clone(), t_true.clone(), x_true.clone()
    fape_perfect = fape_loss(R_pred, t_pred, x_pred, R_true, t_true, x_true)
    rmsd_perfect = naive_rmsd(x_pred, x_true)

    # Apply ONE global rigid motion to the whole prediction (frames + points).
    Rg = random_rotation(1, dtype=dtype)[0]
    tg = torch.randn(3, dtype=dtype) * 10.0
    R_moved = torch.einsum("ij,njk->nik", Rg, R_true)
    t_moved = torch.einsum("ij,nj->ni", Rg, t_true) + tg
    x_moved = torch.einsum("ij,nj->ni", Rg, x_true) + tg
    fape_moved = fape_loss(R_moved, t_moved, x_moved, R_true, t_true, x_true)
    rmsd_moved = naive_rmsd(x_moved, x_true)

    # Genuine structural error: perturb the predicted points.
    x_noisy = x_true + 0.5 * torch.randn(N, 3, dtype=dtype)
    fape_noisy = fape_loss(R_true, t_true, x_noisy, R_true, t_true, x_true)

    # Reflection control: mirror the prediction (flip z). A global reflection is
    # NOT a proper rigid motion, so FAPE should penalise it.
    mirror = torch.tensor([1.0, 1.0, -1.0], dtype=dtype)
    R_ref = R_true * mirror.view(1, 1, 3)
    t_ref = t_true * mirror
    x_ref = x_true * mirror
    fape_reflected = fape_loss(R_ref, t_ref, x_ref, R_true, t_true, x_true)

    print("Frame-Aligned Point Error (FAPE) — properties check")
    print(f"  residues/points N = {N}, dtype = {dtype}")
    print(f"  {'case':<34s}{'FAPE':>14s}{'naive RMSD':>16s}")
    print(f"  {'perfect prediction':<34s}{fape_perfect.item():>14.3e}{rmsd_perfect.item():>16.3e}")
    print(f"  {'+ global rigid motion of pred':<34s}{fape_moved.item():>14.3e}{rmsd_moved.item():>16.3e}")
    print(f"  {'+ 0.5A point noise':<34s}{fape_noisy.item():>14.3e}{'-':>16s}")
    print(f"  {'+ global reflection (mirror)':<34s}{fape_reflected.item():>14.3e}{'-':>16s}")

    # FAPE is invariant to the rigid motion (matches perfect, ~eps-level), while
    # naive RMSD does not.
    assert torch.allclose(fape_moved, fape_perfect, atol=1e-9), "FAPE not rigid-invariant"
    assert rmsd_moved > 1.0, "control: un-superposed RMSD should blow up under motion"
    assert fape_noisy > fape_perfect + 1e-3, "FAPE insensitive to point noise"
    assert fape_reflected > fape_perfect + 1e-3, "FAPE should penalise a reflection"
    print("\nPASS — FAPE is rigid-invariant (superposition-free) yet sensitive to"
          " real error and chirality; naive RMSD is not motion-invariant.")


if __name__ == "__main__":
    main()
