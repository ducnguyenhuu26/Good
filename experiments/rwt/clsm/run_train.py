"""CLSM reference initialization and structured approximate GEM training CLI."""
from __future__ import annotations

import argparse
import csv
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
import yaml

from rwt.clsm.belief import LawFilter
from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import git_source_state, source_fingerprint, verify_checkpoint_source
from rwt.clsm.train import (approximate_estep, collect_emission_statistics,
                            generalized_em_mstep, make_mstep_optimizer,
                            train_reference)
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld
from experiments.rwt.clsm.public_data import load_public_split


def collect_reference(env_id: str, n: int, seed: int):
    schema = mujoco_nominal_schema(env_id)
    obs_rows, actions, next_rows = [], [], []
    world = 0
    while len(obs_rows) < n:
        env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * min(500, n - len(obs_rows)), schema)
        try:
            obs, _ = env.reset(seed=seed + world)
            collector = SmoothActionCollector(schema.action_low, schema.action_high, seed + 10000 + world)
            for _ in range(len(env.schedule)):
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


def _fresh_filter(parts, config):
    return LawFilter(parts.law, parts.evolution, parts.hierarchy.new_memory(),
                     config["filter"]["components"], config["filter"]["max_dropped_mass"],
                     config["filter"]["covariance_jitter_initial"],
                     config["filter"]["covariance_jitter_max"])


def _prequential_validation(parts, sequences, config) -> tuple[float, float]:
    evidence, count, worst_loss = 0., 0, 0.
    parts.law.eval(); parts.evolution.eval(); parts.hierarchy.eval()
    for sequence in sequences:
        filt = _fresh_filter(parts, config)
        for t in range(len(sequence.obs)):
            result = filt.observe(sequence.obs[t], sequence.action[t], sequence.next_obs[t])
            evidence += result["log_predictive"]
            count += 1
            if t + 1 < len(sequence.obs):
                ancestry = filt.evolve()
                worst_loss = max(worst_loss, ancestry.dropped_mass)
    return -evidence / count, worst_loss


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _save_checkpoint(path: Path, parts, optimizer, generator, config, data_hashes,
                     outer: int, best_nll: float, versions, commit, source_dirty):
    state = {"kind": "clsm_approximate_gem", "law": parts.law.state_dict(),
             "evolution": parts.evolution.state_dict(),
             "hierarchy": parts.hierarchy.state_dict(),
             "optimizer": optimizer.state_dict(), "training_generator": generator.get_state(),
             "torch_rng_state": torch.get_rng_state(), "config": config,
             "data_hashes": data_hashes, "outer_iteration": outer,
             "best_validation_nll": best_nll, "versions": versions,
             "git_commit": commit, "source_dirty": source_dirty,
             "source_fingerprint": source_fingerprint()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, parts, optimizer, generator, config, data_hashes):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("kind") != "clsm_approximate_gem" or state["config"] != config or
            state["data_hashes"] != data_hashes):
        raise ValueError("checkpoint architecture/config/public data hashes disagree")
    verify_checkpoint_source(state)
    parts.law.load_state_dict(state["law"])
    parts.evolution.load_state_dict(state["evolution"])
    parts.hierarchy.load_state_dict(state["hierarchy"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["training_generator"])
    torch.set_rng_state(state["torch_rng_state"])
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp","evol","mem"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        parser.error("config track and --track disagree")
    if config.get("environment_family") == "maniskill_pd":
        parser.error("PushCube-v1 PD adapter has not passed a local ManiSkill controller/physics audit")
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track, "device": args.device,
                          "reference_only": args.reference_only, "config": config}, indent=2))
        return
    if args.device == "cuda":
        parser.error("CUDA filter execution is not validated; use --device cpu")
    if args.resume and args.evaluate_only:
        parser.error("choose either --resume or --evaluate-only")
    seed = config["seed"] if args.seed is None else args.seed
    output = args.output_dir / args.track / args.env / ("reference_only" if args.reference_only else "clsm") / str(seed)
    data_path = output / "reference_nominal.npz"
    if args.check_data:
        if args.reference_only:
            if not data_path.exists():
                parser.error(f"missing nominal reference data: {data_path}")
            print(data_path)
        else:
            root = args.output_dir / args.track / args.env / "data"
            for split in ("train", "validation"):
                sequences, digest = load_public_split(root, split)
                print(json.dumps({"split": split, "worlds": len(sequences), "sha256": digest}))
        return
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    parts = build_clsm(mujoco_nominal_schema(args.env), config)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__}
    commit, source_dirty = git_source_state()
    if args.reference_only:
        if args.resume or args.evaluate_only:
            parser.error("reference-only checkpoint resume/evaluation is not supported")
        n = config["transitions"]["train"] if config.get("mode") == "smoke" else 10000
        obs, action, next_obs = collect_reference(args.env, n, seed)
        np.savez_compressed(data_path, obs=obs, action=action, next_obs=next_obs)
        curve = train_reference(parts.law, torch.from_numpy(obs), torch.from_numpy(action),
                                torch.from_numpy(next_obs), epochs=config["training"]["reference_epochs"],
                                lr=config["training"]["lr_dynamics"], seed=seed,
                                batch_size=config["training"]["batch_size"],
                                weight_decay=config["training"]["weight_decay"],
                                gradient_clip_norm=config["training"]["gradient_clip_norm"])
        checkpoint = output / "checkpoint_reference.pt"
        torch.save({"state_dict": parts.law.state_dict(), "config": config, "seed": seed,
                    "data_sha256": _sha256(data_path), "versions": versions, "git_commit": commit,
                    "source_dirty": source_dirty,
                    "source_fingerprint": source_fingerprint(),
                    "torch_rng_state": torch.get_rng_state()}, checkpoint)
        report = {"status": "reference_initialization_only", "train_loss": curve,
                  "checkpoint": str(checkpoint), "data_sha256": _sha256(data_path),
                  "checkpoint_sha256": _sha256(checkpoint), "versions": versions,
                  "git_commit": commit}
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "checkpoint": str(checkpoint), "final_loss": curve[-1]}))
        return

    data_root = args.output_dir / args.track / args.env / "data"
    train_sequences, train_hash = load_public_split(data_root, "train")
    validation_sequences, validation_hash = load_public_split(data_root, "validation")
    train_ids = set(json.loads((data_root / "manifest.json").read_text())["splits"]["train"]["world_ids"])
    val_ids = set(json.loads((data_root / "manifest.json").read_text())["splits"]["validation"]["world_ids"])
    if train_ids & val_ids:
        raise RuntimeError("training and validation world IDs overlap")
    optimizer = make_mstep_optimizer(parts.law, parts.evolution, parts.hierarchy, config)
    generator = torch.Generator().manual_seed(seed + 400000)
    latest, best = output / "checkpoint_latest.pt", output / "checkpoint_best.pt"
    curve_path = output / "training_curve.csv"
    inference_diagnostic_path = output / "inference_diagnostics.csv"
    evolution_diagnostic_path = output / "evolution_diagnostics.csv"
    hashes = {"train": train_hash, "validation": validation_hash}
    if args.resume or args.evaluate_only:
        if not data_path.exists():
            raise FileNotFoundError(f"missing nominal reference data for checkpoint provenance: {data_path}")
        hashes["reference"] = _sha256(data_path)
    start, best_nll = 0, float("inf")
    rows = []
    inference_rows = []
    evolution_rows = []
    if args.resume or args.evaluate_only:
        path = best if args.evaluate_only else latest
        state = _load_checkpoint(path, parts, optimizer, generator, config, hashes)
        start, best_nll = state["outer_iteration"], state["best_validation_nll"]
        if curve_path.exists():
            with curve_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        if args.resume:
            for path, target in ((inference_diagnostic_path, inference_rows),
                                 (evolution_diagnostic_path, evolution_rows)):
                if path.exists():
                    with path.open(newline="", encoding="utf-8") as handle:
                        target.extend(csv.DictReader(handle))
        if args.evaluate_only:
            val_nll, dropped = _prequential_validation(parts, validation_sequences, config)
            print(json.dumps({"validation_prequential_nll": val_nll, "max_dropped_mass": dropped,
                              "checkpoint": str(path)}))
            return
    else:
        n = config["transitions"]["train"] if config.get("mode") == "smoke" else 10000
        obs, action, next_obs = collect_reference(args.env, n, seed)
        np.savez_compressed(data_path, obs=obs, action=action, next_obs=next_obs)
        hashes["reference"] = _sha256(data_path)
        reference_curve = train_reference(parts.law, torch.from_numpy(obs), torch.from_numpy(action),
                                          torch.from_numpy(next_obs), epochs=config["training"]["reference_epochs"],
                                          lr=config["training"]["lr_dynamics"], seed=seed,
                                          batch_size=config["training"]["batch_size"],
                                          weight_decay=config["training"]["weight_decay"],
                                          gradient_clip_norm=config["training"]["gradient_clip_norm"])
        (output / "reference_curve.json").write_text(json.dumps(reference_curve), encoding="utf-8")
        val_nll, dropped = _prequential_validation(parts, validation_sequences, config)
        if dropped > config["filter"]["max_dropped_mass"]:
            raise RuntimeError("initial validation mixture mass loss exceeds configured gate")
        best_nll = val_nll
        _save_checkpoint(best, parts, optimizer, generator, config, hashes, 0, best_nll, versions, commit, source_dirty)
        _save_checkpoint(latest, parts, optimizer, generator, config, hashes, 0, best_nll, versions, commit, source_dirty)
    hashes["reference"] = _sha256(data_path)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    (output / "environment_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output / "data_manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    for outer in range(start, config["training"]["outer_gem_iterations"]):
        parts.law.eval(); parts.evolution.eval(); parts.hierarchy.eval()
        posteriors = [approximate_estep(_fresh_filter(parts, config), seq.obs, seq.action, seq.next_obs)
                      for seq in train_sequences]
        for world_index, posterior in enumerate(posteriors):
            inference_rows.extend({"outer_iteration": outer + 1, "world_index": world_index, **row}
                                  for row in posterior.inference_diagnostics)
            evolution_rows.extend({"outer_iteration": outer + 1, "world_index": world_index, **row}
                                  for row in posterior.evolution_diagnostics)
        _write_csv(inference_diagnostic_path, inference_rows)
        _write_csv(evolution_diagnostic_path, evolution_rows)
        lost = max(p.max_dropped_mass for p in posteriors)
        if lost > config["filter"]["max_dropped_mass"]:
            raise RuntimeError("training mixture mass loss exceeds configured gate")
        train_nll = -sum(float(p.log_predictive.sum()) for p in posteriors) / sum(len(s.obs) for s in train_sequences)
        stats = collect_emission_statistics(train_sequences, posteriors)
        parts.law.train(); parts.evolution.train(); parts.hierarchy.train()
        mstep = generalized_em_mstep(parts.law, parts.evolution, parts.hierarchy,
                                      stats, posteriors, config, optimizer=optimizer, generator=generator)
        val_nll, val_lost = _prequential_validation(parts, validation_sequences, config)
        if val_lost > config["filter"]["max_dropped_mass"] or not np.isfinite(val_nll):
            raise RuntimeError("validation predictive likelihood or mixture mass gate failed")
        improved = val_nll < best_nll
        if improved:
            best_nll = val_nll
            _save_checkpoint(best, parts, optimizer, generator, config, hashes, outer + 1, best_nll, versions, commit, source_dirty)
        row = {"outer_iteration": outer + 1, "train_prequential_nll": train_nll,
               "validation_prequential_nll": val_nll, "best_validation_nll": best_nll,
               "train_dropped_mass": lost, "validation_dropped_mass": val_lost,
               "mean_emission_nll": np.mean([m["emission_nll"] for m in mstep]),
               "mean_evolution_nll": np.mean([m["evolution_nll"] for m in mstep]),
               "mean_hierarchy_nll": np.mean([m["hierarchy_nll"] for m in mstep]),
               "mean_gradient_norm": np.mean([m["gradient_norm"] for m in mstep])}
        rows.append(row)
        _write_csv(curve_path, rows)
        _save_checkpoint(latest, parts, optimizer, generator, config, hashes, outer + 1, best_nll, versions, commit, source_dirty)
        print(json.dumps({"outer_iteration": outer + 1, "validation_prequential_nll": val_nll,
                          "best": improved, "max_dropped_mass": max(lost, val_lost)}))
    report = {"status": "smoke_approximate_gem" if config.get("mode") == "smoke" else "approximate_gem",
              "best_validation_nll": best_nll, "checkpoint_best": str(best),
              "checkpoint_latest": str(latest), "data_hashes": hashes,
              "git_commit": commit, "source_dirty": source_dirty,
              "source_fingerprint": source_fingerprint(),
              "full_evaluation_gate": "closed_long_world_mass_loss_and_incomplete_baselines_maniskill_panel"}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
