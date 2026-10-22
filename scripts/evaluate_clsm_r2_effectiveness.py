"""Held-out one-step predictive diagnostic for the CLSM-R2 CPU pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from scipy.special import ndtri

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.inference_dp import AgeLawFilter
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import source_fingerprint
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.public_data import load_public_split


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[indices].mean(1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--config", type=Path, default=Path("experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm_r2_pilot"))
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--max-worlds", type=int, default=0, help="0 evaluates all held-out worlds")
    parser.add_argument("--max-steps", type=int, default=0, help="0 evaluates each full held-out world")
    args = parser.parse_args()
    if args.max_worlds < 0 or args.max_steps < 0:
        parser.error("max-worlds and max-steps must be nonnegative")
    torch.set_num_threads(1)
    config = load_config(args.config)
    data_root = args.output_dir / config["track"] / args.env / "data"
    run_root = args.output_dir / config["track"] / args.env / "r2" / str(args.seed)
    evaluator_root = args.output_dir.parent / "clsm_evaluator" / config["track"] / args.env
    sequences, public_hash = load_public_split(data_root, "test")
    data_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    evaluator_manifest = json.loads((evaluator_root / "manifest.json").read_text(encoding="utf-8"))
    evaluator_path = Path(evaluator_manifest["test"]["evaluator_file"])
    if _sha256(evaluator_path) != evaluator_manifest["test"]["evaluator_sha256"]:
        raise ValueError("test evaluator file hash does not match its manifest")
    with np.load(evaluator_path, allow_pickle=False) as oracle:
        hidden = {key: oracle[key].copy() for key in ("code", "motor_1", "motor_2", "damping")}

    model_payload = torch.load(run_root / "checkpoint_r2.pt", map_location="cpu", weights_only=True)
    if model_payload.get("architecture_revision") != 2 or model_payload.get("config") != config:
        raise ValueError("test checkpoint architecture or resolved config mismatch")
    model_parts = build_clsm(mujoco_nominal_schema(args.env), config)
    model_parts.law.load_state_dict(model_payload["law"])
    model_parts.law.eval(); model_parts.evolution.eval(); model_parts.hierarchy.eval()
    obs_dim = model_parts.law.schema.observation_dim
    offsets = np.cumsum([0] + [len(seq.obs) for seq in sequences])
    if offsets[-1] != len(hidden["code"]):
        raise ValueError("public and evaluator test transitions are not aligned")
    world_count = len(sequences) if args.max_worlds == 0 else min(args.max_worlds, len(sequences))

    per_world = {name: [] for name in ("r2_filter", "nominal_only", "oracle_code_strength")}
    code_correct, code_total = 0, 0
    strength_abs_errors = {"motor_1": [], "motor_2": [], "damping": []}
    with torch.no_grad():
        for world_index, sequence in enumerate(sequences[:world_count]):
            start, end = int(offsets[world_index]), int(offsets[world_index + 1])
            step_count = len(sequence.obs) if args.max_steps == 0 else min(args.max_steps, len(sequence.obs))
            end = start + step_count
            codes = hidden["code"][start:end]
            strengths = np.column_stack((hidden["motor_1"][start:end], hidden["motor_2"][start:end],
                                         hidden["damping"][start:end]))
            # Pilot worlds are stationary by construction; fail if the evaluator file says otherwise.
            if len(np.unique(codes)) != 1 or not np.allclose(strengths, strengths[0]):
                raise ValueError("effectiveness evaluator expects stationary test worlds")
            world_nll = {name: 0.0 for name in per_world}
            filt = AgeLawFilter(model_parts.law, model_parts.evolution,
                                model_parts.hierarchy.new_world(),
                                config["filter"]["max_world_length"],
                                config["initial_law_information"])
            true_code = int(codes[0])
            true_xi = np.zeros(3, dtype=np.float64)
            for j, (lo, hi) in enumerate(((.50, .85), (.50, .85), (1.30, 2.00))):
                pct = np.clip((strengths[0, j] - lo) / (hi - lo), 1e-9, 1.0 - 1e-9)
                true_xi[j] = ndtri(pct)
            xi_true_t = torch.as_tensor(true_xi[None, :], dtype=torch.float64)
            code_true_t = torch.tensor([true_code], dtype=torch.long)
            for t in range(step_count):
                obs, action, nxt = sequence.obs[t], sequence.action[t], sequence.next_obs[t]
                result = filt.observe(obs, action, nxt)
                world_nll["r2_filter"] -= result["predictive_log_likelihood"] / obs_dim
                pred_code = int(np.argmax(result["code_posterior"]))
                code_correct += int(pred_code == true_code); code_total += 1
                active_coord = {1: (0, "motor_1"), 2: (1, "motor_2"), 4: (2, "damping")}.get(true_code)
                if active_coord is not None:
                    coord, label = active_coord
                    estimate_xi = filt.xi_mean[true_code, coord]
                    lo, hi = ((.50, .85) if coord < 2 else (1.30, 2.00))
                    estimate = lo + (hi - lo) * torch.special.ndtr(estimate_xi)
                    strength_abs_errors[label].append(abs(float(estimate) - strengths[0, coord]))
                log_nominal = model_parts.law.log_prob(obs[None, :], action[None, :], nxt[None, :],
                    torch.zeros(1, dtype=torch.long), torch.zeros((1, 3), dtype=torch.float64))[0]
                log_oracle = model_parts.law.log_prob(obs[None, :], action[None, :], nxt[None, :],
                    code_true_t, xi_true_t)[0]
                world_nll["nominal_only"] -= float(log_nominal) / obs_dim
                world_nll["oracle_code_strength"] -= float(log_oracle) / obs_dim
                if t + 1 < step_count:
                    filt.evolve()
            for name in per_world:
                per_world[name].append(world_nll[name] / step_count)

    score = {name: float(np.mean(values)) for name, values in per_world.items()}
    paired_difference = np.asarray(per_world["r2_filter"]) - np.asarray(per_world["nominal_only"])
    report = {
        "evaluation": "held_out_one_step_predictive_nll_per_observation_dimension",
        "architecture_revision": 2,
        "source_fingerprint": source_fingerprint(),
        "checkpoint_source_fingerprint": model_payload.get("source_fingerprint"),
        "checkpoint_training_dirty": model_payload.get("source_dirty"),
        "env": args.env,
        "test_public_sha256": public_hash,
        "test_evaluator_sha256": evaluator_manifest["test"]["evaluator_sha256"],
        "worlds": world_count, "steps_per_world_max": args.max_steps or None,
        "transitions": int(sum(min(len(seq.obs), args.max_steps) if args.max_steps else len(seq.obs)
                                for seq in sequences[:world_count])),
        "world_level_mean_nll": score,
        "r2_minus_nominal_nll": float(paired_difference.mean()),
        "paired_world_bootstrap_95pct_ci": _bootstrap_ci(paired_difference, args.seed + 88000),
        "worlds_r2_better_than_nominal": int((paired_difference < 0).sum()),
        "code_accuracy_after_each_observation": code_correct / code_total,
        "mean_absolute_physical_multiplier_error": {
            key: float(np.mean(values)) if values else None for key, values in strength_abs_errors.items()},
        "notes": [
            "Oracle code/strength is evaluator-only and is used solely for the separately named upper bound.",
            "The R2 decoder is nominal-anchor-trained with a train-profiled partition; full generalized EM was not run.",
            "These stationary pilot scores do not establish C1 composition, C2 future-law, or C3 memory effectiveness.",
        ],
    }
    horizon_tag = f"steps_{args.max_steps}" if args.max_steps else "full_worlds"
    path = run_root / f"effectiveness_evaluation_{horizon_tag}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
