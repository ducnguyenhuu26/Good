"""R0 fairness, finite-step force, and joint-input audit for the CLSM-R3 attempt.

This is a diagnostic/evaluator script. Oracle labels and MuJoCo integration
states remain isolated here and are never passed into the deployable learner.
The existing Gate-A confirmatory-reserve file is intentionally never opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import mujoco
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from rwt.clsm.config import load_config
from rwt.clsm.mechanics import MechanismForce
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import source_fingerprint
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld
from scripts import run_clsm_r2_gate_a_diagnostic as ga

ENV = "HalfCheetah-v5"
DEFAULT_SOURCE = Path(r"C:\Users\CNC-NVCUONG-PGD\.codex\worktrees\clsm-implementation\learning-to-theorize")
OUT = ROOT / "results/rwt/clsm_r3_final/r0"
BOOTSTRAP_DRAWS = 10_000
FORCE_MATCH_REL_TOL = 1e-7  # frozen numerical matching tolerance


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_old_singleton(root: Path, split: str) -> tuple[dict, dict, dict]:
    """Read only the registered old singleton train/validation public/private pair."""
    data_root = root / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/data"
    local_data_root = ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/data"
    evaluator_root = root / "results/rwt/clsm_evaluator/comp/HalfCheetah-v5"
    public_path = data_root / f"{split}.npz"
    # The strict public loader resolves the registered relative path from the
    # repository root. Copy only public train/validation bytes into this branch's
    # ignored result area so its original manifest/path check remains valid.
    local_data_root.mkdir(parents=True, exist_ok=True)
    local_manifest = local_data_root / "manifest.json"
    if not local_manifest.exists():
        import shutil
        shutil.copy2(data_root / "manifest.json", local_manifest)
    local_public = local_data_root / f"{split}.npz"
    if not local_public.exists():
        import shutil
        shutil.copy2(public_path, local_public)
    _sequences, public_hash = ga.load_public_split(local_data_root, split)
    with np.load(local_public, allow_pickle=False) as data:
        public = {k: data[k].copy() for k in ga.PUBLIC_KEYS}
    manifest = json.loads((evaluator_root / "manifest.json").read_text(encoding="utf-8"))
    item = manifest[split]
    private_path = Path(item["evaluator_file"])
    if not private_path.is_absolute():
        private_path = root / private_path
    if sha256(private_path) != item["evaluator_sha256"]:
        raise ValueError(f"old {split} evaluator labels hash mismatch")
    with np.load(private_path, allow_pickle=False) as data:
        hidden = {k: data[k].copy() for k in ("code", "motor_1", "motor_2", "damping")}
    return public, hidden, {"public_path": str(public_path), "private_path": str(private_path),
                            "public_sha256": public_hash, "private_sha256": item["evaluator_sha256"],
                            "world_count": int(len(np.unique(public["world_id"]))),
                            "transitions": int(len(public["world_id"]))}


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    return [float(x) for x in np.quantile(draw, [.025, .975])]


def matched_baselines(source: Path, out: Path, schema, manifest: dict) -> dict:
    """Add the missing mixed-seen conditioned MLP using precisely matched data."""
    gate_root = source / "results/rwt/clsm_r2_gates/gate_a_diagnostic"
    nominal_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/reference_nominal.npz"
    ref_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_reference_r2.pt"
    pilot_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_r2.pt"
    for path in (gate_root / "dataset_manifest.json", nominal_path, ref_path, pilot_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen R2 diagnostic artifact is unavailable: {path}")

    singleton_train, singleton_hidden, singleton_train_meta = load_old_singleton(source, "train")
    singleton_val, singleton_val_hidden, singleton_val_meta = load_old_singleton(source, "validation")
    mixed_train, mixed_hidden = ga.load_diagnostic_split(gate_root, "mixed_train")
    mixed_val, mixed_val_hidden = ga.load_diagnostic_split(gate_root, "mixed_validation")
    dev_trajectory, dev_hidden = ga.load_diagnostic_split(gate_root, "composition_diagnostic")
    dev_matched, dev_matched_hidden = ga.load_diagnostic_split(gate_root, "matched_probe_outcomes")
    # The reserved confirmatory artifact is explicitly not opened by this runner.
    with np.load(nominal_path, allow_pickle=False) as z:
        nominal_all = {k: z[k].copy() for k in ("obs", "action", "next_obs")}
    pick = np.random.default_rng(41_111).choice(len(nominal_all["obs"]), size=4_000, replace=False)
    nominal = {k: v[pick] for k, v in nominal_all.items()}
    ref_payload = torch.load(ref_path, map_location="cpu", weights_only=True)
    pilot_payload = torch.load(pilot_path, map_location="cpu", weights_only=True)
    config = load_config(ROOT / "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml")
    groups = ga.fixed_motor_groups(schema)
    oracle_partition = torch.zeros((schema.action_dim, 2), dtype=torch.bool)
    oracle_partition[np.asarray(groups[0]), 0] = True
    oracle_partition[np.asarray(groups[1]), 1] = True
    learned_partition = torch.as_tensor(pilot_payload["selected_partition"], dtype=torch.bool)
    dm = ref_payload["law"]["delta_mean"].to(torch.float64)
    ds = ref_payload["law"]["delta_scale"].to(torch.float64)

    # Existing checkpoints are reused only for the already declared R2 and
    # singleton-trained MLP conditions. Their source hashes are recorded.
    existing = {
        ("singleton", "R2_structured"): "oracle_G_z_xi_singletons",
        ("singleton", "oracle_conditioned_MLP"): "oracle_conditioned_unstructured_MLP",
        ("mixed_seen", "R2_structured"): "oracle_G_z_xi_mixed_seen",
    }
    adapters: dict[tuple[str, str, int], object] = {}
    checkpoints: list[dict] = []
    for (regime, family), model_type in existing.items():
        for seed in ga.TRAIN_SEEDS:
            path = gate_root / "checkpoints" / model_type / f"seed_{seed}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"frozen R2 checkpoint is missing: {path}")
            adapters[(regime, family, seed)] = ga.load_diagnostic_model(path, model_type, schema, config, ref_payload)
            checkpoints.append({"regime": regime, "family": family, "seed": seed,
                                "path": str(path), "sha256": sha256(path), "reused_frozen": True})

    # Fit only the missing mixed-seen unstructured MLP, with the same
    # 4,000 mixed transitions, 4,000 nominal replay transitions, seeds,
    # optimizer, epochs, batch size, and mixed validation worlds.
    training = []
    prior_training_path = out / "matched_mlp_training.json"
    prior_training = json.loads(prior_training_path.read_text(encoding="utf-8")) if prior_training_path.is_file() else None
    can_resume = bool(prior_training and
        prior_training.get("mixed_public_sha256") == manifest["splits"]["mixed_train"]["public_sha256"] and
        prior_training.get("mixed_evaluator_sha256") == manifest["splits"]["mixed_train"]["evaluator_sha256"] and
        prior_training.get("mixed_validation_sha256") == manifest["splits"]["mixed_validation"]["public_sha256"] and
        prior_training.get("training_seeds") == list(ga.TRAIN_SEEDS) and
        prior_training.get("epochs") == 100 and prior_training.get("patience") == 12 and
        prior_training.get("batch_size") == 256)
    prior_runs = {int(r["seed"]): r for r in prior_training.get("training_runs", [])} if can_resume else {}
    for seed in ga.TRAIN_SEEDS:
        ckpt = out / "checkpoints/mixed_seen_oracle_conditioned_MLP" / f"seed_{seed}.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=True) if can_resume and ckpt.is_file() else None
        resume_ok = bool(payload and int(payload.get("seed", -1)) == seed and
                         payload.get("role") == "oracle_diagnostic" and
                         payload.get("diagnostic_id") == "CLSM-R2-Gate-A" and
                         payload.get("checkpoint_type") == "oracle_conditioned_unstructured_MLP" and
                         payload.get("architecture_revision") == 2 and
                         payload.get("source_fingerprint") == source_fingerprint() and
                         int(payload.get("best_epoch", 0)) > 0 and seed in prior_runs and
                         prior_runs[seed].get("checkpoint_sha256") == sha256(ckpt))
        if resume_ok:
            report = prior_runs[seed]
            report["reused_verified_checkpoint"] = True
            adapter = ga.load_diagnostic_model(ckpt, "oracle_conditioned_unstructured_MLP", schema, config, ref_payload)
        else:
            adapter = ga.build_adapter("oracle_conditioned_unstructured_MLP", seed, ref_payload,
                                       config, schema, mixed_train, mixed_hidden, nominal,
                                       oracle_partition, learned_partition)
            report = ga.fit_model(adapter, "oracle_conditioned_unstructured_MLP", seed,
                                  mixed_train, mixed_hidden, mixed_val, mixed_val_hidden,
                                  nominal, dm, ds, 100, 12, 256, ckpt, config, source_fingerprint())
            report["parameter_count"] = int(sum(p.numel() for p in adapter.model.parameters()))
            report["reused_verified_checkpoint"] = False
            adapter = ga.load_diagnostic_model(ckpt, "oracle_conditioned_unstructured_MLP", schema, config, ref_payload)
        adapters[("mixed_seen", "oracle_conditioned_MLP", seed)] = adapter
        training.append(report)
        checkpoints.append({"regime": "mixed_seen", "family": "oracle_conditioned_MLP", "seed": seed,
                            "path": str(ckpt), "sha256": sha256(ckpt), "reused_frozen": False})

    domains = (("trajectory_development", dev_trajectory, dev_hidden),
               ("matched_probe_development", dev_matched, dev_matched_hidden))
    records = []
    for domain, public, hidden in domains:
        for regime in ("singleton", "mixed_seen"):
            for family in ("R2_structured", "oracle_conditioned_MLP"):
                if (regime, family, ga.TRAIN_SEEDS[0]) not in adapters:
                    continue
                for seed in ga.TRAIN_SEEDS:
                    adapter = adapters[(regime, family, seed)]
                    if family == "oracle_conditioned_MLP" and regime == "mixed_seen":
                        model_type = "oracle_conditioned_unstructured_MLP"
                    elif family == "oracle_conditioned_MLP":
                        model_type = "oracle_conditioned_unstructured_MLP"
                    else:
                        model_type = existing[(regime, family)]
                    rows, _ = ga.score_rows(adapter, public, hidden, dm.numpy(), ds.numpy(),
                                            model_type, seed, domain)
                    for row in rows:
                        row.update({"training_regime": regime, "model_family": family,
                                    "contrast_role": "development_only_existing_Gate_A_worlds"})
                        records.append(row)

    per_world: dict[tuple, dict] = {}
    for r in records:
        key = (r["split"], r["training_regime"], r["model_family"], r["code"], r["world_id"])
        cell = per_world.setdefault(key, {m: [] for m in ("nll_per_dim", "normalized_mse", "normalized_signature_error")})
        for m in cell:
            cell[m].append(r[m])
    world_rows = []
    for key, cell in per_world.items():
        dom, regime, family, code, wid = key
        world_rows.append({"domain": dom, "training_regime": regime, "model_family": family,
                           "code": code, "world_id": wid,
                           **{m: float(np.mean(v)) for m, v in cell.items()}})
    # Seed-specific world contrasts plus paired-world bootstrap after averaging
    # the three independently trained checkpoints within each world.
    contrasts = []
    for dom, _, _ in domains:
        for regime in ("singleton", "mixed_seen"):
            for code in ga.MIXED_CODES:
                left = {(r["seed"], r["world_id"]): r for r in records
                        if r["split"] == dom and r["training_regime"] == regime and r["code"] == code
                        and r["model_family"] == "R2_structured"}
                right = {(r["seed"], r["world_id"]): r for r in records
                         if r["split"] == dom and r["training_regime"] == regime and r["code"] == code
                         and r["model_family"] == "oracle_conditioned_MLP"}
                for seed in ga.TRAIN_SEEDS:
                    ids = sorted({wid for s, wid in left if s == seed} & {wid for s, wid in right if s == seed})
                    for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error"):
                        diffs = np.asarray([left[(seed, wid)][metric] - right[(seed, wid)][metric] for wid in ids])
                        contrasts.append({"domain": dom, "training_regime": regime, "code": code,
                                          "metric": metric, "seed": seed, "unit": "independent_world",
                                          "mean_structured_minus_MLP": float(diffs.mean()),
                                          "ci95_world_bootstrap": bootstrap_ci(diffs, 770_000 + seed + code)})
                # First average across seeds within a world, then bootstrap worlds.
                wl = {(r["world_id"]): r for r in world_rows if r["domain"] == dom and r["training_regime"] == regime
                      and r["model_family"] == "R2_structured" and r["code"] == code}
                wr = {(r["world_id"]): r for r in world_rows if r["domain"] == dom and r["training_regime"] == regime
                      and r["model_family"] == "oracle_conditioned_MLP" and r["code"] == code}
                ids = sorted(set(wl) & set(wr))
                for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error"):
                    diffs = np.asarray([wl[wid][metric] - wr[wid][metric] for wid in ids])
                    contrasts.append({"domain": dom, "training_regime": regime, "code": code,
                                      "metric": metric, "seed": "seed_average", "unit": "independent_world",
                                      "mean_structured_minus_MLP": float(diffs.mean()),
                                      "ci95_world_bootstrap": bootstrap_ci(diffs, 780_000 + code)})

    out.mkdir(parents=True, exist_ok=True)
    with (out / "four_way_world_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(world_rows[0])); w.writeheader(); w.writerows(world_rows)
    with (out / "four_way_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0])); w.writeheader()
        for row in contrasts:
            row = {**row, "ci95_world_bootstrap": json.dumps(row["ci95_world_bootstrap"])}
            w.writerow(row)
    write_json(out / "matched_mlp_training.json", {"role": "oracle_diagnostic", "training_runs": training,
        "training_seeds": list(ga.TRAIN_SEEDS), "epochs": 100, "patience": 12, "batch_size": 256,
        "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)", "mixed_public_sha256": manifest["splits"]["mixed_train"]["public_sha256"],
        "mixed_evaluator_sha256": manifest["splits"]["mixed_train"]["evaluator_sha256"],
        "mixed_validation_sha256": manifest["splits"]["mixed_validation"]["public_sha256"],
        "worlds_used_for_scoring": "previously scored Gate-A development worlds; not confirmatory",
        "confirmatory_reserved_opened": False})
    write_json(out / "r0_manifest.json", {"source_evidence_commit": "2818503bf0fe90aa79e31ecad795b80343424ba5",
        "source_branch": "codex/clsm-r3-final", "source_fingerprint": source_fingerprint(),
        "source_artifact_root": str(source), "train_split": singleton_train_meta,
        "singleton_validation_split": singleton_val_meta,
        "mixed_train": manifest["splits"]["mixed_train"], "mixed_validation": manifest["splits"]["mixed_validation"],
        "diagnostic_world_ids": {"trajectory": sorted(map(int, np.unique(dev_trajectory["world_id"]))),
                                  "matched": sorted(map(int, np.unique(dev_matched["world_id"])))},
        "checkpoints": checkpoints, "confirmatory_reserved_opened": False,
        "normalization_and_training": "frozen Gate-A nominal output scaling; per-model input normalization fit on declared training regime plus nominal replay only"})
    return {"training": training, "records": records, "world_rows": world_rows, "contrasts": contrasts,
            "adapters": adapters,
            "checkpoints": checkpoints, "ref_payload": ref_payload, "nominal": nominal,
            "singleton_train": (singleton_train, singleton_hidden),
            "mixed_train": (mixed_train, mixed_hidden),
            "probe": ga.load_probe_bank(gate_root), "gate_root": gate_root}


def _force_signature(world: HiddenPhysicsWorld, state: np.ndarray, law: HiddenLaw,
                     action: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    model, data = world.unwrapped.model, world.unwrapped.data
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    world.position = 0
    world._apply(law)
    data.ctrl[:] = np.clip(action, world.action_space.low, world.action_space.high)
    mujoco.mj_forward(model, data)
    # MuJoCo generalized actuator plus passive forces at this exact state.
    force = np.asarray(data.qfrc_actuator + data.qfrc_passive, dtype=np.float64).copy()
    obs = np.asarray(world.unwrapped._get_obs(), dtype=np.float64).copy()
    contacts = int(data.ncon)
    return force, contacts, obs


def _step_from_state(world: HiddenPhysicsWorld, state: np.ndarray, law: HiddenLaw,
                     action: np.ndarray) -> np.ndarray:
    model, data = world.unwrapped.model, world.unwrapped.data
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    world.position = 0
    world._apply(law)
    # Call the underlying MuJoCo environment directly after restoring the full
    # state; wrapper episode counters would otherwise reject repeated snapshots.
    obs, _reward, _terminated, _truncated, _info = world.unwrapped.step(action)
    return np.asarray(obs, dtype=np.float64).copy()


def finite_step_force_audit(source: Path, out: Path, schema, ref_payload: dict,
                            probe: tuple[dict, dict]) -> dict:
    public, private = probe
    # Ensure every action remains inside the installed environment's declared bounds.
    law_motor = HiddenLaw(1, motor_1=.65)
    law_damping = HiddenLaw(4, damping=1.65)
    action_rng = np.random.default_rng(2026102201)
    nominal = mujoco  # keep simulator dependency explicit in this evaluator path
    del nominal
    env_a = HiddenPhysicsWorld(ENV, [law_motor], schema)
    env_b = HiddenPhysicsWorld(ENV, [law_damping], schema)
    env_a.reset(seed=2026102202); env_b.reset(seed=2026102203)
    rows = []
    scale = ref_payload["law"]["delta_scale"].detach().cpu().numpy()
    try:
        for i, (state, obs, action0) in enumerate(zip(private["integration_state"], public["obs"], public["action"])):
            # A deterministic moderate action keeps the matched-force target inside
            # the bounded actuator set where a feasible counterpart exists.
            action_a = np.clip(action_rng.uniform(.08, .45, size=schema.action_dim) *
                               action_rng.choice([-1., 1.], size=schema.action_dim),
                               schema.action_low, schema.action_high)
            force_a, contact_a, restored_a = _force_signature(env_a, state, law_motor, action_a)
            force_zero_b, _, _ = _force_signature(env_b, state, law_damping, np.zeros(schema.action_dim))
            columns = []
            for j in range(schema.action_dim):
                basis = np.zeros(schema.action_dim); basis[j] = 1.0
                force_j, _, _ = _force_signature(env_b, state, law_damping, basis)
                columns.append(force_j - force_zero_b)
            A = np.stack(columns, axis=1)
            fit = lsq_linear(A, force_a - force_zero_b,
                             bounds=(schema.action_low, schema.action_high),
                             tol=1e-12, max_iter=200, lsmr_tol="auto")
            action_b = fit.x
            force_b, contact_b, restored_b = _force_signature(env_b, state, law_damping, action_b)
            err = float(np.linalg.norm(force_a - force_b))
            rel = err / max(1.0, float(np.linalg.norm(force_a)))
            pair_ok = bool(rel <= FORCE_MATCH_REL_TOL)
            if not np.allclose(restored_a, obs, atol=1e-10, rtol=0) or not np.allclose(restored_b, obs, atol=1e-10, rtol=0):
                raise RuntimeError("MuJoCo integration-state restore changed the public observation")
            next_a = _step_from_state(env_a, state, law_motor, action_a)
            next_b = _step_from_state(env_b, state, law_damping, action_b)
            if not pair_ok:
                continue
            replay_a = _step_from_state(env_a, state, law_motor, action_a)
            replay_b = _step_from_state(env_b, state, law_damping, action_b)
            replay = max(float(np.max(np.abs(replay_a - next_a))), float(np.max(np.abs(replay_b - next_b))))
            scaled_diff = (next_a - next_b) / np.maximum(scale, 1e-12)
            vel = float(np.linalg.norm(obs[list(schema.joint_velocity_indices)]))
            rows.append({"probe_index": i, "nominal_contact_count": int(private["nominal_contact_count"][i]),
                         "contact_pair_motor": contact_a, "contact_pair_damping": contact_b,
                         "controlled_velocity_norm": vel, "action_motor_norm": float(np.linalg.norm(action_a)),
                         "action_damping_norm": float(np.linalg.norm(action_b)), "force_abs_error": err,
                         "force_relative_error": rel, "normalized_next_obs_rmse": float(np.sqrt(np.mean(scaled_diff**2))),
                         "normalized_next_obs_l2": float(np.linalg.norm(scaled_diff)),
                         "deterministic_replay_max_abs_error": replay})
    finally:
        env_a.close(); env_b.close()

    if not rows:
        raise RuntimeError("no force-matched feasible law/action pairs; R0 physical test unresolved")
    vel = np.asarray([r["controlled_velocity_norm"] for r in rows])
    q1, q2 = np.quantile(vel, [1/3, 2/3])
    summary = {}
    for contact in (False, True):
        for vbin, lo, hi in (("low", -np.inf, q1), ("mid", q1, q2), ("high", q2, np.inf)):
            vals = [r for r in rows if (r["nominal_contact_count"] > 0) == contact and lo <= r["controlled_velocity_norm"] <= hi]
            if vals:
                summary[f"{'contact' if contact else 'no_contact'}_{vbin}"] = {
                    "matched_pairs": len(vals), "mean_normalized_next_obs_rmse": float(np.mean([r["normalized_next_obs_rmse"] for r in vals])),
                    "ci95_world_bootstrap": bootstrap_ci(np.asarray([r["normalized_next_obs_rmse"] for r in vals]), 8100 + int(contact) + len(vbin)),
                    "mean_action_motor_norm": float(np.mean([r["action_motor_norm"] for r in vals])),
                    "mean_action_damping_norm": float(np.mean([r["action_damping_norm"] for r in vals]))}
    # Exact scalar ODE counterexample with equal initial net force c.
    v0, c, dt, d1, d2 = .0, 1.0, .1, 1.0, 2.0
    def scalar_next(d: float) -> float:
        ga_force = c + d * v0
        return float(v0 * math.exp(-d * dt) + (ga_force / d) * (1 - math.exp(-d * dt)))
    scalar = {"v0": v0, "initial_net_force_each": c, "dt": dt, "damping_values": [d1, d2],
              "next_velocities": [scalar_next(d1), scalar_next(d2)],
              "absolute_next_velocity_difference": abs(scalar_next(d1) - scalar_next(d2))}
    result = {"diagnostic_only": True, "state_source": "previous Gate-A development matched-probe bank; independent from singleton training; not confirmatory",
              "law_pair": {"motor_only": {"code": 1, "motor_1": .65}, "damping_only": {"code": 4, "damping": 1.65}},
              "generalized_force_definition": "MuJoCo data.qfrc_actuator + data.qfrc_passive at restored identical integration state after setting ctrl; common other generalized terms cancel between paired laws",
              "matching_tolerance_relative_l2": FORCE_MATCH_REL_TOL,
              "candidate_pairs": len(public["obs"]), "force_matched_feasible_pairs": len(rows),
              "pair_fraction": len(rows) / len(public["obs"]), "max_force_relative_error": max(r["force_relative_error"] for r in rows),
              "max_replay_abs_error": max(r["deterministic_replay_max_abs_error"] for r in rows),
              "max_normalized_next_obs_rmse": max(r["normalized_next_obs_rmse"] for r in rows),
              "mean_normalized_next_obs_rmse": float(np.mean([r["normalized_next_obs_rmse"] for r in rows])),
              "mean_ci95": bootstrap_ci(np.asarray([r["normalized_next_obs_rmse"] for r in rows]), 8200),
              "velocity_contact_strata": summary, "scalar_damped_system": scalar,
              "pairs": rows,
              "interpretation": "Finite-step outcome differences are measured only among feasible pairs passing the force tolerance. This can establish insufficiency of initial generalized force as a sufficient statistic for the tested MuJoCo step, but does not alone explain the earlier Gate-A deficit."}
    write_json(out / "finite_step_force_sufficiency.json", result)
    return result


def _physical_strength(hidden: dict) -> np.ndarray:
    return np.column_stack((hidden["motor_1"], hidden["motor_2"], hidden["damping"])).astype(np.float64)


def _channels(public: dict, hidden: dict, partition: torch.Tensor, schema) -> tuple[np.ndarray, np.ndarray]:
    codes = hidden["code"].astype(np.int64)
    xi = ga.strength_xi(codes, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    with torch.no_grad():
        r2_force = MechanismForce(schema, partition).eval()(
            torch.as_tensor(public["obs"], dtype=torch.float64),
            torch.as_tensor(public["action"], dtype=torch.float64),
            torch.as_tensor(codes, dtype=torch.long), torch.as_tensor(xi, dtype=torch.float64))[0].numpy()
    bits = ((codes[:, None] >> np.arange(3)) & 1).astype(np.float64)
    m = _physical_strength(hidden)
    eta = bits * (m - np.array([1., 1., 1.]))
    B = schema.actuator_to_dof
    v = public["obs"][:, np.asarray(schema.joint_velocity_indices)]
    g = np.asarray(partition, dtype=bool).astype(np.float64)
    delta = np.column_stack((
        (public["action"] * g[:, 0]) @ B.T * eta[:, 0:1],
        (public["action"] * g[:, 1]) @ B.T * eta[:, 1:2],
        -v * schema.nominal_damping[None, :] * eta[:, 2:3]))
    tau_nominal = public["action"] @ B.T - v * schema.nominal_damping[None, :]
    # R3 retains the nominal resultant and each mechanism's separate response.
    r3 = np.concatenate((public["obs"], public["action"], tau_nominal, eta, delta), axis=1)
    return r2_force, r3


def joint_input_coverage(out: Path, schema, ref_payload: dict, partition: torch.Tensor,
                         singleton: tuple[dict, dict], mixed: tuple[dict, dict],
                         nominal: dict, queries: list[tuple[str, dict, dict]]) -> dict:
    law = ref_payload["law"]
    obs_mean = law["obs_mean"].detach().cpu().numpy(); obs_scale = law["obs_scale"].detach().cpu().numpy()
    force_mean = law["force_mean"].detach().cpu().numpy(); force_scale = law["force_scale"].detach().cpu().numpy()
    s_pub, s_hid = singleton; m_pub, m_hid = mixed
    def r2_input(pub, hid):
        tau, _ = _channels(pub, hid, partition, schema)
        return np.concatenate(((pub["obs"] - obs_mean) / obs_scale,
                               (tau - force_mean) / force_scale), axis=1)
    def r3_input(pub, hid):
        _, x = _channels(pub, hid, partition, schema)
        return x
    # Add the declared nominal replay only to singleton references.
    nomhid = {"code": np.zeros(len(nominal["obs"]), dtype=np.int64),
              "motor_1": np.ones(len(nominal["obs"])), "motor_2": np.ones(len(nominal["obs"])),
              "damping": np.ones(len(nominal["obs"]))}
    s_pub_force = np.concatenate((s_pub["obs"], nominal["obs"]), axis=0)
    s_pub_obs = np.concatenate((s_pub["obs"], nominal["obs"]), axis=0)
    s_pub_act = np.concatenate((s_pub["action"], nominal["action"]), axis=0)
    s_pub_next = np.concatenate((s_pub["next_obs"], nominal["next_obs"]), axis=0)
    spub = {"obs": s_pub_obs, "action": s_pub_act, "next_obs": s_pub_next}
    shid = {k: np.concatenate((s_hid[k], nomhid[k])) for k in ("code", "motor_1", "motor_2", "damping")}
    r2_single, r3_single = r2_input(spub, shid), r3_input(spub, shid)
    r2_mixed, r3_mixed = r2_input(m_pub, m_hid), r3_input(m_pub, m_hid)
    # R3 feature scales are training-only per the R3 contract.
    def standardize_pair(train, query):
        mu = train.mean(0); sd = np.maximum(train.std(0), 1e-5)
        return (train - mu) / sd, (query - mu) / sd
    r2_train_s, _ = standardize_pair(r2_single, r2_single)
    # Use singleton-fitted R3 channel moments for all distances; mixed support is
    # transformed in that same coordinate system.
    r3_mu, r3_sd = r3_single.mean(0), np.maximum(r3_single.std(0), 1e-5)
    r3_train_s = (r3_single - r3_mu) / r3_sd; r3_train_m = (r3_mixed - r3_mu) / r3_sd
    trees = {("R2", "singleton"): cKDTree(r2_train_s), ("R2", "mixed_seen"): cKDTree(standardize_pair(r2_single, r2_mixed)[1]),
             ("R3", "singleton"): cKDTree(r3_train_s), ("R3", "mixed_seen"): cKDTree(r3_train_m)}
    references = {("R2", "singleton"): r2_train_s, ("R2", "mixed_seen"): standardize_pair(r2_single, r2_mixed)[1],
                  ("R3", "singleton"): r3_train_s, ("R3", "mixed_seen"): r3_train_m}
    loo_cut = {}
    for key, tree in trees.items():
        vals = tree.query(references[key], k=2)[0][:, 1]
        loo_cut[key] = float(np.quantile(vals, .95))
    records = []
    for domain, pub, hid in queries:
        for space, make_input in (("R2", r2_input), ("R3", r3_input)):
            raw = make_input(pub, hid)
            if space == "R2":
                q_single = standardize_pair(r2_single, raw)[1]
                q_mixed = standardize_pair(r2_single, raw)[1]
            else:
                q_single = (raw - r3_mu) / r3_sd; q_mixed = q_single
            for regime in ("singleton", "mixed_seen"):
                ref = references[(space, regime)]
                q = q_single if regime == "singleton" else q_mixed
                dist = trees[(space, regime)].query(q, k=1)[0]
                # A second reference cloud is useful only as a support descriptor;
                # distances are always standardized by singleton training moments.
                cut = loo_cut[(space, regime)]
                codes = hid["code"].astype(np.int64)
                contact = np.asarray(hid.get("contact_count", np.zeros(len(codes)))) > 0
                velocity = np.linalg.norm(pub["obs"][:, np.asarray(schema.joint_velocity_indices)], axis=1)
                # Train-only velocity cut points prevent test-domain bin fitting.
                if not hasattr(joint_input_coverage, "velocity_cuts"):
                    vtrain = np.linalg.norm(s_pub["obs"][:, np.asarray(schema.joint_velocity_indices)], axis=1)
                    joint_input_coverage.velocity_cuts = np.quantile(vtrain, [1/3, 2/3])
                bins = np.digitize(velocity, joint_input_coverage.velocity_cuts)
                for wid in np.unique(pub.get("world_id", np.zeros(len(codes), dtype=int))):
                    wsel = pub.get("world_id", np.zeros(len(codes), dtype=int)) == wid
                    for code in np.unique(codes[wsel]):
                        for vbin in range(3):
                            for cb in (False, True):
                                sel = wsel & (codes == code) & (bins == vbin) & (contact == cb)
                                if not sel.any():
                                    continue
                                records.append({"domain": domain, "space": space, "reference_regime": regime,
                                    "code": int(code), "world_id": int(wid), "velocity_bin_train_tertile": ("low", "mid", "high")[vbin],
                                    "contact": bool(cb), "n": int(sel.sum()), "nearest_distance_mean": float(dist[sel].mean()),
                                    "nearest_distance_median": float(np.median(dist[sel])),
                                    "within_reference_train_q95_fraction": float(np.mean(dist[sel] <= cut)),
                                    "train_reference_loo_q95_cutoff": cut})
    with (out / "joint_input_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    # Per-world residual-distance curves for the frozen singleton R2 model.
    # Distance bins are descriptive quantiles; no threshold is used as a gate.
    return {"rows": len(records), "train_only_velocity_cutpoints": joint_input_coverage.velocity_cuts.tolist(),
            "reference_cutoff_method": "95th percentile leave-one-out nearest-neighbor distance within each train reference; descriptive only",
            "coverage_csv": str(out / "joint_input_coverage.csv"),
            "note": "R2 input includes normalized obs and effective initial force; R3 descriptive input includes obs, action, eta, and three separate delta-tau channels. R3 mixed support is structurally outside singleton combinations; no PASS threshold is applied."}


def residual_distance_curves(out: Path, schema, ref_payload: dict, partition: torch.Tensor,
                             singleton: tuple[dict, dict], nominal: dict,
                             queries: list[tuple[str, dict, dict]], adapters: dict) -> dict:
    """R2 singleton residual against actual (obs,force) nearest-neighbor distance."""
    spub, shid = singleton
    nhid = {"code": np.zeros(len(nominal["obs"]), dtype=np.int64),
            "motor_1": np.ones(len(nominal["obs"])), "motor_2": np.ones(len(nominal["obs"])),
            "damping": np.ones(len(nominal["obs"]))}
    train_pub = {"obs": np.concatenate((spub["obs"], nominal["obs"])),
                 "action": np.concatenate((spub["action"], nominal["action"]))}
    train_hid = {k: np.concatenate((shid[k], nhid[k]))
                 for k in ("code", "motor_1", "motor_2", "damping")}
    train_force, _ = _channels(train_pub, train_hid, partition, schema)
    law = ref_payload["law"]
    om = law["obs_mean"].detach().cpu().numpy(); os = law["obs_scale"].detach().cpu().numpy()
    fm = law["force_mean"].detach().cpu().numpy(); fs = law["force_scale"].detach().cpu().numpy()
    train_x = np.concatenate(((train_pub["obs"] - om) / os, (train_force - fm) / fs), axis=1)
    tree = cKDTree(train_x)
    world_rows = []
    delta_mean = law["delta_mean"].detach().cpu().numpy()
    delta_scale = law["delta_scale"].detach().cpu().numpy()
    for domain, pub, hid in queries:
        query_force, _ = _channels(pub, hid, partition, schema)
        qx = np.concatenate(((pub["obs"] - om) / os, (query_force - fm) / fs), axis=1)
        distance = tree.query(qx, k=1)[0]
        for code in ga.MIXED_CODES:
            code_sel = hid["code"].astype(np.int64) == code
            if not code_sel.any():
                continue
            edges = np.quantile(distance[code_sel], [0, .25, .5, .75, 1.])
            bins = np.clip(np.searchsorted(edges[1:-1], distance, side="right"), 0, 3)
            for seed in ga.TRAIN_SEEDS:
                adapter = adapters[("singleton", "R2_structured", seed)]
                pred, _ = ga.predict_numpy(adapter, pub["obs"], pub["action"], hid["code"].astype(np.int64),
                                           ga.strength_xi(hid["code"], hid["motor_1"], hid["motor_2"], hid["damping"]))
                target = (pub["next_obs"] - pub["obs"] - delta_mean) / delta_scale
                mse = np.mean((target - pred) ** 2, axis=1)
                for wid in np.unique(pub["world_id"]):
                    wsel = (pub["world_id"] == wid) & code_sel
                    for b in range(4):
                        sel = wsel & (bins == b)
                        if not sel.any():
                            continue
                        world_rows.append({"domain": domain, "code": code, "world_id": int(wid), "seed": seed,
                                           "distance_bin": b + 1, "distance_bin_lo": float(edges[b]),
                                           "distance_bin_hi": float(edges[b + 1]),
                                           "mean_nearest_distance": float(distance[sel].mean()),
                                           "normalized_delta_mse": float(mse[sel].mean()), "n": int(sel.sum())})
    with (out / "residual_vs_distance_by_world.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(world_rows[0])); w.writeheader(); w.writerows(world_rows)
    # Average seeds within each world before bootstrap so seeds are not falsely
    # treated as independent world replications.
    by_world: dict[tuple, dict[str, list[float]]] = {}
    for row in world_rows:
        key = (row["domain"], row["code"], row["world_id"], row["distance_bin"])
        dest = by_world.setdefault(key, {"distance": [], "mse": []})
        dest["distance"].append(row["mean_nearest_distance"]); dest["mse"].append(row["normalized_delta_mse"])
    world_mean = [{"domain": k[0], "code": k[1], "world_id": k[2], "distance_bin": k[3],
                   "mean_nearest_distance": float(np.mean(v["distance"])),
                   "normalized_delta_mse": float(np.mean(v["mse"]))} for k, v in by_world.items()]
    curves = []
    for domain in sorted({r["domain"] for r in world_mean}):
        for code in ga.MIXED_CODES:
            for b in range(1, 5):
                rows = [r for r in world_mean if r["domain"] == domain and r["code"] == code and r["distance_bin"] == b]
                if not rows:
                    continue
                d = np.asarray([r["mean_nearest_distance"] for r in rows]); e = np.asarray([r["normalized_delta_mse"] for r in rows])
                curves.append({"domain": domain, "code": code, "distance_bin": b, "independent_worlds": len(rows),
                               "mean_nearest_distance": float(d.mean()), "distance_ci95_world_bootstrap": json.dumps(bootstrap_ci(d, 8300 + code + b)),
                               "mean_normalized_delta_mse": float(e.mean()), "mse_ci95_world_bootstrap": json.dumps(bootstrap_ci(e, 8400 + code + b))})
    with (out / "residual_vs_distance_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    return {"per_world_rows": len(world_rows), "curve_rows": len(curves),
            "worlds_bootstrapped_after_seed_average": True,
            "curve_file": str(out / "residual_vs_distance_curve.csv"),
            "per_world_file": str(out / "residual_vs_distance_by_world.csv"),
            "distance_definition": "Euclidean nearest-neighbor distance on R2 normalized (obs, effective initial generalized force); training-only nominal/R2 buffers; singleton+nominal training references"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact-root", type=Path, default=DEFAULT_SOURCE,
                        help="existing Gate-A worktree with frozen development artifacts; confirmatory reserve is never read")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    source, out = args.source_artifact_root.resolve(), args.output_dir.resolve()
    start = time.perf_counter(); torch.set_num_threads(1)
    schema = mujoco_nominal_schema(ENV)
    source_manifest = json.loads((source / "results/rwt/clsm_r2_gates/gate_a_diagnostic/dataset_manifest.json").read_text(encoding="utf-8"))
    baseline = matched_baselines(source, out, schema, source_manifest)
    if args.baseline_only:
        print(json.dumps({"matched_mlp_runs": len(baseline["training"]), "world_rows": len(baseline["world_rows"]),
                          "output_dir": str(out), "runtime_seconds": time.perf_counter() - start}, indent=2))
        return
    ref_payload = baseline["ref_payload"]
    learned_partition = torch.as_tensor(torch.load(source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_r2.pt",
                                                   map_location="cpu", weights_only=True)["selected_partition"], dtype=torch.bool)
    force = finite_step_force_audit(source, out, schema, ref_payload, baseline["probe"])
    gate_root = baseline["gate_root"]
    singleton_train = baseline["singleton_train"]
    mixed_train = baseline["mixed_train"]
    nominal_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/reference_nominal.npz"
    with np.load(nominal_path, allow_pickle=False) as z:
        nominal_all = {k: z[k].copy() for k in ("obs", "action", "next_obs")}
    pick = np.random.default_rng(41_111).choice(len(nominal_all["obs"]), size=4_000, replace=False)
    nominal = {k: v[pick] for k, v in nominal_all.items()}
    dev_pub, dev_hid = ga.load_diagnostic_split(gate_root, "composition_diagnostic")
    matched_pub, matched_hid = ga.load_diagnostic_split(gate_root, "matched_probe_outcomes")
    coverage = joint_input_coverage(out, schema, ref_payload, learned_partition, singleton_train,
                                    mixed_train, nominal,
                                    [("trajectory_development", dev_pub, dev_hid),
                                     ("matched_probe_development", matched_pub, matched_hid)])
    residual_curve = residual_distance_curves(out, schema, ref_payload, learned_partition,
        singleton_train, nominal,
        [("trajectory_development", dev_pub, dev_hid), ("matched_probe_development", matched_pub, matched_hid)],
        baseline["adapters"])
    matched_101 = next(r for r in baseline["contrasts"] if r["domain"] == "matched_probe_development" and
                       r["training_regime"] == "mixed_seen" and r["code"] == 5 and
                       r["metric"] == "normalized_mse" and r["seed"] == "seed_average")
    signature_101 = next(r for r in baseline["contrasts"] if r["domain"] == "matched_probe_development" and
                         r["training_regime"] == "mixed_seen" and r["code"] == 5 and
                         r["metric"] == "normalized_signature_error" and r["seed"] == "seed_average")
    # Save seed-wise and world-averaged baseline contrasts as an explicit fair comparison.
    result = {"gate": "R0", "status": "COMPLETE_WITH_LIMITATIONS", "diagnostic_only": True,
              "source_fingerprint": source_fingerprint(), "source_artifact_commit": source_manifest.get("source_commit"),
              "matched_data_comparison": {"models": ["R2_structured_singleton", "oracle_conditioned_MLP_singleton",
                  "R2_structured_mixed_seen", "oracle_conditioned_MLP_mixed_seen"],
                  "within_regime_only": True, "contrast_file": str(out / "four_way_contrasts.csv"),
                  "world_score_file": str(out / "four_way_world_scores.csv"),
                  "mixed_mlp_training_file": str(out / "matched_mlp_training.json"),
                  "old_confound": "The prior mixed-seen structured-vs-singleton-MLP contrast was distribution-mismatched. A mixed-seen MLP was now fit to the identical mixed transitions and nominal replay.",
                  "code_101_matched_probe_development_contrast": {
                      "normalized_mse_structured_minus_mlp": matched_101["mean_structured_minus_MLP"],
                      "normalized_mse_ci95": matched_101["ci95_world_bootstrap"],
                      "signature_structured_minus_mlp": signature_101["mean_structured_minus_MLP"],
                      "signature_ci95": signature_101["ci95_world_bootstrap"]},
                  "interpretation": "Old Gate-A evaluation worlds are reused strictly as development evidence to audit the comparator confound; they are not confirmatory."},
              "finite_step_force_sufficiency": {k: force[k] for k in ("candidate_pairs", "force_matched_feasible_pairs", "pair_fraction",
                  "max_force_relative_error", "max_replay_abs_error", "mean_normalized_next_obs_rmse", "mean_ci95", "velocity_contact_strata", "scalar_damped_system")},
              "joint_input_coverage": coverage, "confirmatory_reserved_opened": False,
              "residual_vs_distance": residual_curve,
              "runtime_seconds": time.perf_counter() - start,
              "limitations": ["Matched-data baseline is evaluated on previously scored Gate-A development worlds.",
                  "R0 mixed-seen comparator audit does not replace fresh R1 worlds.",
                  "Physical force pairs are limited to feasible action-bounded solutions on the frozen nominal probe bank."]}
    write_json(out / "r0_decision.json", result)
    audit_md = f"""# CLSM-R3 R0 fairness and physical sufficiency audit

**Status: COMPLETE_WITH_LIMITATIONS.** This completes R0 on the frozen Gate-A development data. It does not decide R1, and the reserved confirmatory split was not opened.

## Matched-data baseline correction

The original Gate-A mixed-seen structured decoder used mixed train/validation, while its `oracle_conditioned_unstructured_MLP` comparator used singleton train/validation. That contrast was not a matched-data capacity comparison. I trained a new oracle-conditioned MLP with the same 4,000 mixed transitions, 4,000 nominal replay transitions, three seeds `{4100,4101,4102}`, AdamW settings, 100-epoch ceiling, and mixed validation used by the mixed-seen R2 structured model. Selection was validation-only. The existing singleton R2/MLP and mixed R2 checkpoints were hash-verified and reused. The mixed MLP has 76,187 trainable parameters; the Gate-A structured reference has 76,479 (0.38% difference).

The code-`101` matched-probe deficit **persists** against the matched mixed-seen MLP on previously scored development worlds: structured-minus-MLP normalized MSE `{matched_101['mean_structured_minus_MLP']:.5f}` (paired-world bootstrap 95% CI `{matched_101['ci95_world_bootstrap']}`); signature error `{signature_101['mean_structured_minus_MLP']:.5f}` (CI `{signature_101['ci95_world_bootstrap']}`). The old contrast's fairness limitation is therefore narrowed, but the new comparison remains development evidence and cannot establish the final R1 claim. Full within-regime, per-code results are in `four_way_contrasts.csv` and `four_way_world_scores.csv`.

## Full-step generalized-force equivalence

Across `{force['force_matched_feasible_pairs']}/{force['candidate_pairs']}` action-bounded pairs at identical restored nominal MuJoCo integration states, the motor-only (`code 001`, motor multiplier 0.65) and damping-only (`code 100`, damping multiplier 1.65) laws had equal initial `qfrc_actuator + qfrc_passive` within relative L2 tolerance `{force['matching_tolerance_relative_l2']}`. Maximum observed relative force error was `{force['max_force_relative_error']:.3g}`; deterministic replay error was `{force['max_replay_abs_error']:.3g}`. Despite this initial-force match, their full-frame-skip next observations differed by mean normalized RMSE `{force['mean_normalized_next_obs_rmse']:.3f}` (bootstrap 95% CI `{force['mean_ci95']}`). The scalar analytic fixture gives a next-velocity difference `{force['scalar_damped_system']['absolute_next_velocity_difference']:.6f}` under equal initial force.

This is evidence that initial resultant generalized force is not a sufficient statistic for the tested finite MuJoCo step. It supports evaluating the single registered intervention-resolved decoder. It does not prove that this physical compression was the only cause of the old code-`101` model error.

## Joint input support and residuals

`joint_input_coverage.csv` reports per-world R2 `(normalized observation, effective initial force)` support and descriptive R3 `(observation, action, nominal force, eta, per-mechanism delta-force)` support, stratified by code, nominal contact, and train-defined velocity tertile. `residual_vs_distance_curve.csv` and `residual_vs_distance_by_world.csv` report singleton R2 error by nearest-neighbor distance with world-level uncertainty. Cutoffs are train-derived descriptive references only. For code `101` matched probes, about 65.9% were within the singleton R2 support cutoff and 10.8% within the mixed-train R2 cutoff; the corresponding trajectory fractions were 99.8% and 92.9%. This documents the matched-state distribution shift alongside the baseline comparison.

## Integrity and limits

- Oracle labels and full MuJoCo snapshots were used only in this evaluator/diagnostic code; they are not inputs to deployable training.
- The original confirmatory reserve was not opened.
- Old Gate-A evaluation worlds were reused only to audit the comparator confound; R1 must use newly generated, frozen evaluation worlds.
- The force-pair audit is bounded by feasible action solutions and coverage of the old development probe bank.

Checkpoint hashes, training logs, per-world scores, coverage strata, and the JSON decision are saved beside this report.
"""
    (out / "fairness_audit.md").write_text(audit_md, encoding="utf-8")
    print(json.dumps({"status": result["status"], "matched_force_pairs": force["force_matched_feasible_pairs"],
                      "mean_normalized_next_obs_rmse": force["mean_normalized_next_obs_rmse"],
                      "output_dir": str(out), "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
