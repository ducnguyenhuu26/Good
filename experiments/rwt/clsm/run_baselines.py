"""Train the matched public-input pooled MLP baseline.

Other required baseline families remain explicit gaps; this command never
relabels a simplified adapter as a switching or published native method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import yaml

from rwt.clsm.baselines import PooledGaussianMLP
from rwt.clsm.config import load_config
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import git_source_state, source_fingerprint
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.public_data import load_public_split


def _flatten(sequences):
    return tuple(torch.cat([getattr(sequence, key) for sequence in sequences])
                 for key in ("obs", "action", "next_obs"))


@torch.no_grad()
def _validation_nll(model, sequences):
    model.eval()
    values = []
    for sequence in sequences:
        values.append(-model.log_prob(sequence.obs, sequence.action, sequence.next_obs))
    return float(torch.cat(values).mean())


def train(env_id: str, track: str, config: dict, output_dir: Path, seed: int) -> dict:
    schema = mujoco_nominal_schema(env_id)
    data_dir = output_dir / track / env_id / "data"
    train_sequences, train_hash = load_public_split(data_dir, "train")
    validation_sequences, validation_hash = load_public_split(data_dir, "validation")
    reference_path = output_dir / track / env_id / "clsm" / str(seed) / "reference_nominal.npz"
    if not reference_path.exists():
        raise FileNotFoundError(f"matched CLSM nominal reference dataset missing: {reference_path}")
    with np.load(reference_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"obs", "action", "next_obs"}:
            raise ValueError("reference data has nonpublic columns")
        reference = tuple(torch.from_numpy(arrays[key]).to(torch.float64)
                          for key in ("obs", "action", "next_obs"))
    torch.manual_seed(seed)
    model = PooledGaussianMLP(schema.observation_dim, schema.action_dim,
                              config["network"]["hidden_dim"],
                              config["emission"]["variance_floor"],
                              config["emission"]["variance_ceiling"])
    model.fit_normalization(*reference)
    generator = torch.Generator().manual_seed(seed + 500000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr_dynamics"],
                                 weight_decay=config["training"]["weight_decay"])
    code_reference = []
    for _ in range(config["training"]["reference_epochs"]):
        for indices in torch.randperm(len(reference[0]), generator=generator).split(
                min(config["training"]["batch_size"], len(reference[0]))):
            mean = model(reference[0][indices], reference[1][indices])[0]
            loss = ((mean - reference[2][indices]) / model.delta_scale).square().mean()
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("nonfinite pooled reference initialization")
            optimizer.step()
        code_reference.append(float(loss.detach()))
    with torch.no_grad():
        residual = model(*reference[:2])[0] - reference[2]
        model.log_variance.copy_(residual.square().mean(0).clamp(
            model.variance_floor, model.variance_ceiling).log())
    train_obs, train_action, train_next = _flatten(train_sequences)
    run_dir = output_dir / track / env_id / "pooled_mlp" / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_val = _validation_nll(model, validation_sequences)
    best_val = baseline_val
    commit, source_dirty = git_source_state()
    data_hashes = {"train": train_hash, "validation": validation_hash,
                   "reference": _sha256(reference_path)}
    def save(path, outer):
        torch.save({"kind": "pooled_mlp", "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(), "outer_iteration": outer,
                    "best_validation_nll": best_val, "config": config,
                    "data_hashes": data_hashes, "git_commit": commit,
                    "source_dirty": source_dirty,
                    "source_fingerprint": source_fingerprint()}, path)
    best_path = run_dir / "checkpoint_best.pt"
    latest_path = run_dir / "checkpoint_latest.pt"
    save(best_path, 0)
    rows = []
    total_steps = config["training"]["gradient_steps_per_iteration"]
    for outer in range(config["training"]["outer_gem_iterations"]):
        losses = []
        model.train()
        for _ in range(total_steps):
            indices = torch.randint(len(train_obs), (config["training"]["batch_size"],),
                                    generator=generator)
            loss = -model.log_prob(train_obs[indices], train_action[indices], train_next[indices]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite pooled training loss")
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite pooled gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        val = _validation_nll(model, validation_sequences)
        if val < best_val:
            best_val = val
            save(best_path, outer + 1)
        rows.append({"outer_iteration": outer + 1, "train_nll": np.mean(losses),
                     "validation_nll": val, "best_validation_nll": best_val})
        save(latest_path, outer + 1)
    with (run_dir / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    report = {"status": "smoke_pooled_mlp" if config.get("mode") == "smoke" else "pooled_mlp",
              "validation_nll_before_nonstationary_training": baseline_val,
              "best_validation_nll": best_val, "data_hashes": data_hashes,
              "reference_loss": code_reference,
              "checkpoint_best": str(best_path), "parameter_count": sum(p.numel() for p in model.parameters()),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "git_commit": commit, "source_dirty": source_dirty,
              "source_fingerprint": source_fingerprint()}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp", "evol", "mem"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="pooled_mlp", choices=("pooled_mlp",))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        parser.error("config and CLI track disagree")
    seed = config["seed"] if args.seed is None else args.seed
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track, "seed": seed,
                          "method": args.method, "config": config}))
        return
    if args.check_data:
        root = args.output_dir / args.track / args.env / "data"
        for split in ("train", "validation"):
            _, digest = load_public_split(root, split)
            print(json.dumps({"split": split, "sha256": digest}))
        return
    result = train(args.env, args.track, config, args.output_dir, seed)
    print(json.dumps({"status": result["status"],
                      "best_validation_nll": result["best_validation_nll"]}))


if __name__ == "__main__":
    main()
