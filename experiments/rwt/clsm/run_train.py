"""CLSM-R2 pilot runner with explicit data, reference, train and evaluate stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gymnasium
import mujoco
import numpy as np
import scipy
import torch

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.support import DisjointPartition
from rwt.clsm.serialization import git_source_state, source_fingerprint
from rwt.clsm.train import train_reference
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import R2StationaryExcitationCollector
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld
from experiments.rwt.clsm.public_data import load_public_split


def collect_reference(env_id: str, n: int, seed: int, schema):
    """Collect known-nominal reference transitions under independent excitation."""
    obs_rows, actions, next_rows = [], [], []
    world = 0
    while len(obs_rows) < n:
        length = min(500, n - len(obs_rows))
        env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * length, schema)
        try:
            obs, _ = env.reset(seed=seed + world)
            collector = R2StationaryExcitationCollector(schema.action_low, schema.action_high,
                                                        seed + 10000 + world)
            for _ in range(length):
                action = collector.act(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                obs_rows.append(obs.copy()); actions.append(action.copy()); next_rows.append(next_obs.copy())
                obs = next_obs
                if terminated or truncated:
                    break
        finally:
            env.close()
        world += 1
    return np.asarray(obs_rows), np.asarray(actions), np.asarray(next_rows)


def _resolve_output(args, seed):
    return args.output_dir / args.track / args.env / "r2" / str(seed)


def _load_reference(path, config, data_path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_r2_checkpoint(payload, config)
    if payload.get("data_sha256") != _sha256(data_path):
        raise ValueError("nominal reference data hash differs from its checkpoint")
    return payload


def validate_r2_checkpoint(payload, config=None):
    if payload.get("architecture_revision") != 2:
        raise ValueError("checkpoint architecture_revision mismatch: expected CLSM-R2 revision 2")
    if config is not None and payload.get("config") != config:
        raise ValueError("checkpoint resolved configuration mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp", "evol", "mem"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    stage = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "check-data", "reference-only", "train", "evaluate"):
        stage.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        parser.error("config track and --track disagree")
    if args.device != "cpu":
        parser.error("R2 CPU path is the only validated execution device")
    if config.get("environment_family") == "maniskill_pd":
        parser.error("PushCube-v1 PD adapter has not passed a local controller/physics audit")
    seed = config["seed"] if args.seed is None else args.seed
    output = _resolve_output(args, seed)
    reference_data = output / "reference_nominal.npz"
    reference_checkpoint = output / "checkpoint_reference_r2.pt"
    data_root = args.output_dir / args.track / args.env / "data"

    if args.dry_run:
        print(json.dumps({"architecture_revision": 2, "env": args.env, "track": args.track,
                          "device": args.device, "config": config,
                          "stages": ["--check-data", "--reference-only", "--train", "--evaluate"]}, indent=2))
        return
    if args.check_data:
        if not (data_root / "manifest.json").is_file():
            raise FileNotFoundError(f"missing public dataset manifest: {data_root / 'manifest.json'}")
        results = {}
        for split in ("train", "validation", "test"):
            sequences, digest = load_public_split(data_root, split)
            results[split] = {"worlds": len(sequences), "transitions": sum(len(x.obs) for x in sequences),
                              "sha256": digest}
        print(json.dumps(results, indent=2)); return

    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    schema = mujoco_nominal_schema(args.env)
    parts = build_clsm(schema, config)
    if args.reference_only:
        n = config["transitions"]["train"] if config["mode"] == "smoke" else max(5000, config["transitions"]["train"])
        obs, action, next_obs = collect_reference(args.env, n, seed, schema)
        np.savez_compressed(reference_data, obs=obs, action=action, next_obs=next_obs)
        curve = train_reference(parts.law, torch.from_numpy(obs), torch.from_numpy(action),
                                torch.from_numpy(next_obs), epochs=config["training"]["reference_epochs"],
                                lr=config["training"]["lr_dynamics"], seed=seed,
                                batch_size=config["training"]["batch_size"],
                                weight_decay=config["training"].get("weight_decay", 1e-4),
                                gradient_clip_norm=config["training"]["gradient_clip_norm"])
        payload = {"architecture_revision": 2, "law": parts.law.state_dict(), "config": config,
                   "seed": seed, "data_sha256": _sha256(reference_data),
                   "curve": curve, "source_fingerprint": source_fingerprint(),
                   "git_commit": git_source_state()[0], "source_dirty": git_source_state()[1],
                   "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                                "numpy": np.__version__, "scipy": scipy.__version__}}
        torch.save(payload, reference_checkpoint)
        report = {"status": "nominal_reference_fit", "architecture_revision": 2,
                  "transitions": len(obs), "initial_normalized_mse": curve[0],
                  "final_normalized_mse": curve[-1], "reference_sha256": payload["data_sha256"],
                  "checkpoint": str(reference_checkpoint)}
        (output / "reference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); return

    if config["mode"] != "pilot":
        parser.error("R2 finite partition train/evaluate stages require the registered mode: pilot dataset")

    if not reference_checkpoint.is_file() or not reference_data.is_file():
        raise FileNotFoundError("run --reference-only before --train or --evaluate")
    reference = _load_reference(reference_checkpoint, config, reference_data)
    parts.law.load_state_dict(reference["law"])
    model_path = output / "checkpoint_r2.pt"

    if args.train:
        sequences, train_hash = load_public_split(data_root, "train")
        worlds = [(seq.obs, seq.action, seq.next_obs) for seq in sequences]
        candidates = DisjointPartition.enumerate(schema.action_dim)
        per_world_scores = np.zeros((len(candidates), len(worlds)), dtype=np.float64)
        for candidate_index, candidate in enumerate(candidates):
            for world_index, world in enumerate(worlds):
                per_world_scores[candidate_index, world_index] = float(DisjointPartition.profile_score(
                    candidate, [world], parts.law,
                    gh_nodes=config["partition"]["quadrature_nodes"]))
        scores = [(float(row.sum()), candidate) for row, candidate in zip(per_world_scores, candidates)]
        scores.sort(key=lambda item: item[0], reverse=True)
        bootstrap_count = config["partition"]["train_world_bootstrap_replicates"]
        rng = np.random.default_rng(seed + 610000)
        winner_counts = np.zeros(len(candidates), dtype=np.int64)
        for _ in range(bootstrap_count):
            sample = rng.integers(0, len(worlds), size=len(worlds))
            winner_counts[int(per_world_scores[:, sample].sum(1).argmax())] += 1
        winner_order = np.argsort(-winner_counts)
        parts.law.mechanics.set_partition(scores[0][1])
        payload = {"architecture_revision": 2, "law": parts.law.state_dict(),
                   "evolution": parts.evolution.state_dict(), "hierarchy": parts.hierarchy.state_dict(),
                   "partition_scores": [score for score, _ in scores],
                   "selected_partition": scores[0][1].int().tolist(),
                   "bootstrap_winner_counts": winner_counts.tolist(),
                   "train_sha256": train_hash, "reference_sha256": reference["data_sha256"],
                   "config": config, "source_fingerprint": source_fingerprint(),
                   "source_dirty": git_source_state()[1], "git_commit": git_source_state()[0]}
        torch.save(payload, model_path)
        report = {"status": "r2_train_profile_m_step", "architecture_revision": 2,
                  "train_sha256": train_hash, "worlds": len(worlds),
                  "partition_candidates": len(scores), "selected_partition": payload["selected_partition"],
                  "best_to_second_log_evidence_gap": scores[0][0] - scores[1][0],
                  "bootstrap_replicates": bootstrap_count,
                  "bootstrap_winner_frequency": float(winner_counts[int(winner_order[0])] / bootstrap_count),
                  "bootstrap_top_candidates": [{"candidate_index": int(idx), "wins": int(winner_counts[idx])}
                                                for idx in winner_order[:5]],
                  "initialization_seed_used": seed,
                  "full_generalized_em": "NOT RUN; this stage fits nominal anchor and train-only finite partition profile"}
        (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); return

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if payload.get("architecture_revision") != 2 or payload.get("config") != config:
        raise ValueError("model checkpoint is incompatible with CLSM-R2")
    if payload.get("source_fingerprint") != source_fingerprint():
        raise ValueError("model source fingerprint differs from current R2 implementation")
    parts.law.load_state_dict(payload["law"])
    sequences, test_hash = load_public_split(data_root, "test")
    total_nll, total_steps = 0.0, 0
    for seq in sequences:
        # Report a public marginal profile diagnostic with the frozen training partition.
        score = DisjointPartition.profile_score(parts.law.mechanics.group_mask.bool(),
                                               [(seq.obs, seq.action, seq.next_obs)], parts.law,
                                               gh_nodes=config["partition"]["quadrature_nodes"])
        total_nll -= float(score); total_steps += len(seq.obs)
    report = {"status": "held_out_public_profile_evaluation", "architecture_revision": 2,
              "test_sha256": test_hash, "worlds": len(sequences), "transitions": total_steps,
              "mean_profile_nll": total_nll / total_steps,
              "claim_limit": "diagnostic likelihood only; no C1/C2/C3 claim"}
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
