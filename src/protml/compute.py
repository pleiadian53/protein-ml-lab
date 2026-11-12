"""
compute.py — hardware detection + small/medium/large model-tier feasibility.

The convention this codifies
----------------------------
Models in this project fall into three tiers, sized to where they can run:

  SMALL  — laptop loop. The from-scratch toys (EGNN equivariance, IPA
           invariance, FAPE check) and the smallest ESM-2 checkpoints
           (8M / 35M params) for embed / probe / contact-extraction.
           Runs on CPU/MPS in seconds–minutes with <4 GB RAM.

  MEDIUM — borderline on a 16 GB laptop. ESM-2 150M / 650M frozen forward,
           small LoRA-style fine-tunes. Inference on M1 works but is slow;
           a pod is recommended for training.

  LARGE  — pod-only. ESM-2 3B / 15B, ESMFold, AlphaFold 2 / 3, RFdiffusion,
           Chroma. Needs CUDA + 24 GB+ VRAM and runs hours-to-days.

The CLI driver `examples/ops/compute_check.py` prints the local hardware and
which tier(s) are feasible on it.

Adapted from the agentic-spliceai `ops_compute_check.py` (general
hardware-detection core kept; task estimators replaced with the protein-ML
tiers above). Stdlib only — no heavy deps so the check runs on any machine.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class TierFeasibility:
    """Whether a model-size tier fits on the detected hardware."""

    feasible: bool
    required_ram_gb: float
    required_vram_gb: float
    examples: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass
class ComputeReport:
    """Detected hardware plus per-tier feasibility."""

    device: str           # "cuda", "mps", or "cpu"
    device_name: str
    vram_gb: float        # 0.0 for CPU / MPS (MPS shares system RAM)
    ram_gb: float
    disk_free_gb: float
    tiers: dict[str, TierFeasibility] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Hardware detection
# -----------------------------------------------------------------------------

def _detect_cuda() -> tuple[str, float]:
    """Return (gpu_name, vram_gb) via nvidia-smi, or ('', 0.0) on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        name, mem_mib = out.stdout.strip().splitlines()[0].split(",", 1)
        return name.strip(), float(mem_mib.strip()) / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return "", 0.0


def _detect_system_ram() -> float:
    """Return system RAM in GB (macOS sysctl / Linux /proc/meminfo / fallback)."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(out.stdout.strip()) / (1024 ** 3)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except FileNotFoundError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        return 0.0


def _detect_apple_chip() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Apple Silicon (unknown)"


def detect_hardware(output_path: str = ".") -> ComputeReport:
    """Detect device + memory + free disk at `output_path`."""
    ram_gb = _detect_system_ram()
    disk_free_gb = shutil.disk_usage(output_path).free / (1024 ** 3)

    gpu_name, vram_gb = _detect_cuda()
    if gpu_name:
        return ComputeReport(
            device="cuda", device_name=gpu_name,
            vram_gb=vram_gb, ram_gb=ram_gb, disk_free_gb=disk_free_gb,
        )
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ComputeReport(
            device="mps", device_name=_detect_apple_chip() or "Apple Silicon",
            vram_gb=0.0, ram_gb=ram_gb, disk_free_gb=disk_free_gb,
        )
    return ComputeReport(
        device="cpu", device_name=platform.processor() or "Unknown CPU",
        vram_gb=0.0, ram_gb=ram_gb, disk_free_gb=disk_free_gb,
    )


# -----------------------------------------------------------------------------
# Tier feasibility — protein-ML small / medium / large
# -----------------------------------------------------------------------------

# Rough VRAM/RAM targets and representative workloads per tier.
_TIERS = {
    "small": {
        "required_ram_gb": 4.0,
        "required_vram_gb": 0.0,
        "examples": [
            "EGNN equivariance check (Phase 3)",
            "IPA invariance & FAPE checks (Phase 2)",
            "ESM-2 8M / 35M — embed + linear probe (Phase 1)",
        ],
    },
    "medium": {
        "required_ram_gb": 12.0,
        "required_vram_gb": 8.0,
        "examples": [
            "ESM-2 150M / 650M — frozen forward, contact-from-attention",
            "Small LoRA fine-tune on ESM-2 150M",
        ],
    },
    "large": {
        "required_ram_gb": 32.0,
        "required_vram_gb": 24.0,
        "examples": [
            "ESM-2 3B / 15B — inference and fine-tuning",
            "ESMFold (≈3B-class) — full structure inference",
            "AlphaFold 2 / 3 — inference and training",
            "RFdiffusion / Chroma — generative design at scale",
        ],
    },
}


def _check_tier(report: ComputeReport, name: str) -> TierFeasibility:
    spec = _TIERS[name]
    ram_need = spec["required_ram_gb"]
    vram_need = spec["required_vram_gb"]

    feasible = True
    notes: list[str] = []

    if report.device == "cuda":
        if report.vram_gb >= vram_need:
            notes.append(f"VRAM OK: {report.vram_gb:.0f} GB available "
                         f"(~{vram_need:.0f} GB recommended)")
        else:
            feasible = False
            notes.append(f"insufficient VRAM: {report.vram_gb:.0f} GB available, "
                         f"~{vram_need:.0f} GB recommended")
    else:
        # CPU or MPS: no dedicated VRAM; the LARGE tier is pod-only.
        if name == "large":
            feasible = False
            notes.append("LARGE tier is pod-only — run on a CUDA host (A40+)")
        elif report.ram_gb >= ram_need:
            notes.append(f"RAM OK on {report.device}: {report.ram_gb:.0f} GB available "
                         f"(~{ram_need:.0f} GB recommended)")
        else:
            feasible = False
            notes.append(f"insufficient RAM on {report.device}: "
                         f"{report.ram_gb:.0f} GB available, ~{ram_need:.0f} GB recommended")

    return TierFeasibility(
        feasible=feasible,
        required_ram_gb=ram_need,
        required_vram_gb=vram_need,
        examples=list(spec["examples"]),
        notes=notes,
    )


def check_compute(output_path: str = ".", tiers: list[str] | None = None) -> ComputeReport:
    """Detect hardware and report feasibility for each model-size tier."""
    path = Path(output_path)
    report = detect_hardware(output_path=str(path if path.exists() else Path(".")))
    for name in (tiers or list(_TIERS.keys())):
        if name in _TIERS:
            report.tiers[name] = _check_tier(report, name)
    return report


def print_report(report: ComputeReport) -> None:
    """Human-readable summary for the CLI driver."""
    print()
    print("=" * 70)
    print("protein-ml-lab — compute report")
    print("=" * 70)
    print(f"  Device:    {report.device_name} ({report.device})")
    if report.vram_gb > 0:
        print(f"  VRAM:      {report.vram_gb:.1f} GB")
    print(f"  RAM:       {report.ram_gb:.1f} GB")
    print(f"  Disk free: {report.disk_free_gb:.1f} GB")
    print()
    print(f"  {'Tier':<8s}{'Feasible':<12s}{'RAM need':<12s}{'VRAM need':<12s}")
    print("  " + "-" * 60)
    for name, tier in report.tiers.items():
        status = "yes" if tier.feasible else "NO"
        print(f"  {name:<8s}{status:<12s}{tier.required_ram_gb:<12.0f}{tier.required_vram_gb:<12.0f}")
    print()
    for name, tier in report.tiers.items():
        print(f"  [{name}]")
        for note in tier.notes:
            print(f"    - {note}")
        for ex in tier.examples:
            print(f"    · {ex}")
        print()
