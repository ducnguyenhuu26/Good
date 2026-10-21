"""Refuse a scientific summary until complete per-track score files exist."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"results_dir={args.results_dir}")
        return
    scores = list(args.results_dir.glob("*/**/scores.csv"))
    memory = list(args.results_dir.glob("*/**/memory_adaptation.csv"))
    if not scores or not memory:
        parser.error("complete locked C1/C2 score files and C3 memory-adaptation files are missing; smoke artifacts cannot be summarized as science")
    parser.error("hierarchical paired statistical aggregation is not implemented; scientific summary is blocked")


if __name__ == "__main__":
    main()
