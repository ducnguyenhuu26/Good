"""Shared frozen-protocol helpers for Q1 feasibility A/B/C scripts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1.datasets import rollout, sample_environment, stage_pair
from q1.simulator import DynamicsParams, apply_program, response


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(file.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(sha256_file(file)))
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def load_protocol(config_path: str | Path, out_path: str | Path, *, initialize: bool = False):
    cfg_path = ROOT / config_path
    out = ROOT / out_path
    if initialize:
        if out.exists() and any(out.iterdir()):
            permitted = {"config.resolved.yaml", "protocol_manifest.json", "original_split_manifest.json"}
            present = {p.name for p in out.iterdir()}
            if not present.issubset(permitted):
                raise FileExistsError(f"refusing to overwrite nonempty feasibility output: {out}")
        out.mkdir(parents=True, exist_ok=True)
    elif not (out / "protocol_manifest.json").is_file():
        raise FileNotFoundError(f"run Experiment A first; missing frozen protocol manifest: {out / 'protocol_manifest.json'}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not initialize:
        cfg = yaml.safe_load((out / "config.resolved.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((out / "protocol_manifest.json").read_text(encoding="utf-8"))
        return cfg, manifest, out
    split_path = ROOT / cfg["split_manifest"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_cfg = yaml.safe_load((ROOT / "configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest_sources=[]
    saved_seeds=set()
    for source in sorted((ROOT/"results/q1").rglob("split_manifest.json")):
        if out in source.parents:
            continue
        data=json.loads(source.read_text(encoding="utf-8"))
        env_seeds=data.get("environment_seeds",{})
        saved={int(s) for values in env_seeds.values() for s in values}
        saved_seeds.update(saved)
        manifest_sources.append({"path":source.relative_to(ROOT).as_posix(),
            "sha256":sha256_file(source),"environment_seed_count":len(saved)})
    from q1.diagnostics.feasibility_abc import sample_idlike_manifest, split_training_background_indices
    idlike = sample_idlike_manifest(seed=cfg["idlike_manifest_seed"], count=cfg["idlike_backgrounds"],
                                    train_ranges=split_cfg["parameter_ranges"],
                                    forbidden_parameter_seeds=saved_seeds)
    inner_n = int(cfg["c_direct_effect"]["training_inner_validation_backgrounds"])
    fit_idx,inner_idx=split_training_background_indices(count=len(split["environment_ids"]["train"]),
        seed=int(cfg["seed"]+1701),fit_count=int(cfg["c_direct_effect"]["training_fit_backgrounds"]),validation_count=inner_n)
    if len(fit_idx) != int(cfg["c_direct_effect"]["training_fit_backgrounds"]):
        raise ValueError("not enough original train backgrounds for the frozen C fit split")
    b_fit_idx,b_val_idx=split_training_background_indices(count=len(split["environment_ids"]["train"]),
        seed=int(cfg["seed"]+2701),fit_count=int(cfg["b_probe"]["inner_train_backgrounds"]),
        validation_count=int(cfg["b_probe"]["inner_validation_backgrounds"]))
    validation_ids = split["environment_ids"]["validation"][:int(cfg["development_ood_backgrounds"])]
    manifest = {
        "protocol_label": "DEVELOPMENT_DIAGNOSTIC",
        "idlike": idlike,
        "development_ood": {"environment_ids": validation_ids,
            "environment_seeds": split["environment_seeds"]["validation"][:len(validation_ids)],
            "parameter_region": "validation", "label": "DEVELOPMENT_OOD_PREVIOUSLY_INSPECTED"},
        "c_training_fit": {"indices": fit_idx,
            "environment_ids": [split["environment_ids"]["train"][i] for i in fit_idx],
            "environment_seeds": [split["environment_seeds"]["train"][i] for i in fit_idx]},
        "c_training_inner_validation": {"indices": inner_idx,
            "environment_ids": [split["environment_ids"]["train"][i] for i in inner_idx],
            "environment_seeds": [split["environment_seeds"]["train"][i] for i in inner_idx]},
        "b_inner_train": {"indices": b_fit_idx,
            "environment_ids": [split["environment_ids"]["train"][i] for i in b_fit_idx]},
        "b_inner_validation": {"indices": b_val_idx,
            "environment_ids": [split["environment_ids"]["train"][i] for i in b_val_idx]},
        "original_split_manifest_sha256": sha256_file(split_path),
        "original_split_manifest": cfg["split_manifest"],
        "test_split_evaluated_or_used": False,
        "test_split_names_rejected_by_scripts": ["test_id", "test_background_ood"],
        "train_background_parameter_seeds": split["environment_seeds"]["train"],
        "saved_environment_parameter_seeds_for_collision_check_only": sorted(saved_seeds),
        "saved_split_manifests_for_collision_check_only": manifest_sources,
        "seed_protocol": {"base": cfg["seed"], "idlike_manifest": cfg["idlike_manifest_seed"],
                          "training_seeds": cfg["c_direct_effect"]["optimizer_seeds"]},
    }
    config_file=out / "config.resolved.yaml"
    if config_file.exists():
        if yaml.safe_load(config_file.read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("existing resolved config differs; refusing to overwrite frozen config")
    else:
        config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest_file=out / "protocol_manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("existing protocol manifest differs; refusing to overwrite frozen split")
    else:
        write_json(manifest_file, manifest)
    split_file=out / "original_split_manifest.json"
    if split_file.exists():
        if json.loads(split_file.read_text(encoding="utf-8")) != split:
            raise RuntimeError("existing source split manifest differs; refusing to overwrite")
    else:
        split_file.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return cfg, manifest, out


def params_for(manifest: dict, regime: str, index: int, split_manifest: dict,
               parameter_ranges: dict) -> DynamicsParams:
    if regime == "IDLIKE_DEVELOPMENT":
        p = manifest["idlike"]["backgrounds"][index]
        return DynamicsParams(np.asarray(p["B"]), p["d"], p["b"], p["kappa"])
    if regime == "DEVELOPMENT_OOD":
        return sample_environment(split_manifest, "validation", index, parameter_ranges)
    if regime == "TRAIN":
        return sample_environment(split_manifest, "train", index, parameter_ranges)
    raise ValueError(f"forbidden or unknown regime {regime!r}")


def support_tensor(old_traj, normalizer, device):
    raw = np.c_[old_traj.states[:-1], old_traj.actions,
                old_traj.states[1:, 2:] - old_traj.states[:-1, 2:]]
    t = torch.as_tensor(raw, dtype=torch.float32, device=device)
    return torch.cat((normalizer.normalize_x(t[:, :6]), normalizer.normalize_dv(t[:, 6:])), dim=-1)[None]


def encode_old_context(model, normalizer, old_traj, device):
    model.eval()
    with torch.no_grad():
        return model.encode(support_tensor(old_traj, normalizer, device))[0]


def probe_rows(traj):
    return np.c_[traj.states[:-1], traj.actions]


def get_stage_data(params, program, *, seed: int, n_support: int, n_adapt: int,
                   n_query: int, dt: float, sigma: float):
    old, adaptation, query = stage_pair(params, tuple(program), seed=seed,
        n_support=n_support, n_query=max(int(n_adapt), int(n_query)), dt=dt, sigma=sigma)
    return old, adaptation, query


def eval_truth(params, program, probes, *, dt):
    return response(params, probes, dt=dt), response(apply_program(params, tuple(program)), probes, dt=dt)


def predict_m2(model, normalizer, context, probes, device):
    x = torch.as_tensor(probes, dtype=torch.float32, device=device)
    c = torch.as_tensor(context, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        pred = normalizer.denormalize_dv(model.executor(
            normalizer.normalize_x(x), c.expand(len(x), -1)))
    return pred.cpu().numpy().astype(np.float64)
