"""
compute_check.py — print the local hardware and per-tier feasibility for the
small/medium/large protein-ML model tiers.

Run it:
    PYTHONPATH=src python examples/ops/compute_check.py

Thin CLI driver around `protml.compute.check_compute` — the dependency arrow
points examples/ → src/, never the other way.
"""

from __future__ import annotations

import argparse

from protml.compute import check_compute, print_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect compute and report small/medium/large tier feasibility.",
    )
    parser.add_argument(
        "--tier", type=str, default=None, choices=["small", "medium", "large"],
        help="Check a single tier only (default: all three).",
    )
    parser.add_argument(
        "--output-path", type=str, default=".",
        help="Path used to estimate free disk space (default: current directory).",
    )
    args = parser.parse_args()

    tiers = [args.tier] if args.tier else None
    report = check_compute(output_path=args.output_path, tiers=tiers)
    print_report(report)


if __name__ == "__main__":
    main()
