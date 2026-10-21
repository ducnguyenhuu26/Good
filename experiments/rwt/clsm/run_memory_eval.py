"""Memory-adaptation evaluator release gate; requires a full CLSM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("mem",))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"env={args.env} track={args.track} checkpoint={args.checkpoint}")
        return
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        parser.error(f"checkpoint cannot be loaded safely: {error}")
    if payload.get("checkpoint_kind") != "clsm_joint_gem_locked":
        parser.error("reference-only checkpoint is not a full CLSM memory model")
    parser.error("causal integrated memory update and matched adaptation evaluator are not implemented; C3 is blocked")


if __name__ == "__main__":
    main()
