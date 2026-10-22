"""Run the isolated, oracle-labeled CLSM-R2 Gate A diagnostic.

This evaluator/trainer deliberately reads private labels and MuJoCo snapshots.
All outputs/checkpoints are tagged oracle_diagnostic and cannot be loaded by
the deployable CLSM-R2 loaders.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
import mujoco
import numpy as np
import torch
from scipy.special import ndtri
from torch import nn

from rwt.clsm.config import load_config
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import source_fingerprint
from rwt.clsm.support import DisjointPartition
from experiments.rwt.clsm.build_datasets import PUBLIC_KEYS, _sha256
from experiments.rwt.clsm.collectors import R2StationaryExcitationCollector
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld, fixed_motor_groups
from experiments.rwt.clsm.public_data import load_public_split

ENV_ID = "HalfCheetah-v5"
SINGLETON_CODES = (0, 1, 2, 4)
MIXED_CODES = (3, 5, 6, 7)
TRAIN_SEEDS = (4100, 4101, 4102)
BOOTSTRAP_DRAWS = 10_000
PROBE_COUNT = 128
MIXED_TRAIN_WORLDS_PER_CODE = 20
MIXED_VAL_WORLDS_PER_CODE = 8
EVAL_WORLDS_PER_CODE = 40
CONFIRM_WORLDS_PER_CODE = 8
WORLD_LENGTH = 50


def digest(path: Path) -> str:
    return _sha256(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def physical_draw(code: int, rng: np.random.Generator) -> tuple[float, float, float]:
    return (float(rng.uniform(.50, .85)), float(rng.uniform(.50, .85)),
            float(rng.uniform(1.30, 2.00)))


def strength_xi(code: np.ndarray, motor_1: np.ndarray, motor_2: np.ndarray,
                damping: np.ndarray) -> np.ndarray:
    physical = np.column_stack((motor_1, motor_2, damping)).astype(np.float64)
    lower = np.array([.50, .50, 1.30]); upper = np.array([.85, .85, 2.00])
    probability = np.clip((physical - lower) / (upper - lower), 1e-7, 1 - 1e-7)
    xi = ndtri(probability)
    bits = ((code[:, None].astype(np.int64) >> np.arange(3)) & 1).astype(bool)
    xi[~bits] = 0.0
    return xi


def _empty_public() -> dict[str, list]:
    return {key: [] for key in PUBLIC_KEYS}


def collect_world_set(env_id: str, schema, codes: tuple[int, ...], worlds_per_code: int,
                      length: int, seed_base: int, world_base: int) -> tuple[dict, dict, dict]:
    public = _empty_public()
    hidden = {key: [] for key in ("code", "motor_1", "motor_2", "damping", "contact_count")}
    world_meta: list[dict] = []
    rng = np.random.default_rng(seed_base)
    world_number = 0
    for code in codes:
        for rep in range(worlds_per_code):
            world_id = world_base + world_number
            world_number += 1
            m1, m2, damp = physical_draw(code, rng)
            law = HiddenLaw(code, m1, m2, damp, False)
            env = HiddenPhysicsWorld(env_id, [law] * length, schema)
            actual_steps = 0
            try:
                obs, _ = env.reset(seed=seed_base + world_id)
                collector = R2StationaryExcitationCollector(schema.action_low, schema.action_high,
                                                            seed_base + 2_000_000 + world_id)
                for step in range(length):
                    action = collector.act(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    values = (obs.copy(), action.copy(), reward, next_obs.copy(), terminated,
                              truncated, world_id, step)
                    for key, value in zip(PUBLIC_KEYS, values):
                        public[key].append(value)
                    hidden["code"].append(code)
                    hidden["motor_1"].append(m1)
                    hidden["motor_2"].append(m2)
                    hidden["damping"].append(damp)
                    hidden["contact_count"].append(int(env.unwrapped.data.ncon))
                    obs = next_obs
                    actual_steps += 1
                    if terminated or truncated:
                        break
            finally:
                env.close()
            world_meta.append({"world_id": world_id, "code": code, "transitions": actual_steps,
                               "motor_1": m1, "motor_2": m2, "damping": damp,
                               "seed": seed_base + world_id})
    return ({k: np.asarray(v) for k, v in public.items()},
            {k: np.asarray(v) for k, v in hidden.items()},
            {"worlds": world_meta, "world_count": len(world_meta),
             "transitions": len(public["reward"])})


def save_split(root: Path, name: str, public: dict, hidden: dict, meta: dict) -> dict:
    public_dir, private_dir = root / "public", root / "evaluator_only"
    public_dir.mkdir(parents=True, exist_ok=True); private_dir.mkdir(parents=True, exist_ok=True)
    public_path, private_path = public_dir / f"{name}.npz", private_dir / f"{name}_oracle.npz"
    np.savez_compressed(public_path, **public)
    hidden = {**hidden, "world_id": public["world_id"], "world_step": public["world_step"]}
    np.savez_compressed(private_path, **hidden)
    return {"public_path": str(public_path), "public_sha256": digest(public_path),
            "evaluator_path": str(private_path), "evaluator_sha256": digest(private_path),
            "world_count": meta["world_count"], "transitions": meta["transitions"],
            "worlds": meta["worlds"]}


def collect_probe_states(env_id: str, schema, count: int, seed: int) -> tuple[dict, dict]:
    env = gym.make(env_id)
    public = {"obs": [], "action": []}
    private = {"integration_state": [], "nominal_contact_count": []}
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(env.unwrapped.model, spec)
    try:
        obs, _ = env.reset(seed=seed)
        collector = R2StationaryExcitationCollector(schema.action_low, schema.action_high, seed + 99)
        while len(public["obs"]) < count:
            state = np.empty(state_size, dtype=np.float64)
            mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, state, spec)
            action = collector.act(obs)
            public["obs"].append(obs.copy()); public["action"].append(action.copy())
            private["integration_state"].append(state)
            private["nominal_contact_count"].append(int(env.unwrapped.data.ncon))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + len(public["obs"]) + 1)
    finally:
        env.close()
    return ({k: np.asarray(v) for k, v in public.items()},
            {k: np.asarray(v) for k, v in private.items()})


def restore_and_step(world: HiddenPhysicsWorld, state: np.ndarray, action: np.ndarray,
                     law: HiddenLaw, probe_index: int) -> tuple[np.ndarray, int, np.ndarray]:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(world.unwrapped.model, world.unwrapped.data, state, spec)
    mujoco.mj_forward(world.unwrapped.model, world.unwrapped.data)
    world.position = probe_index
    world._apply(law)
    restored_obs = np.asarray(world.unwrapped._get_obs(), dtype=np.float64).copy()
    next_obs, _, _, _, _ = world.step(action)
    return np.asarray(next_obs, dtype=np.float64), int(world.unwrapped.data.ncon), restored_obs


def generate_matched_probe_outcomes(root: Path, schema, probe_public: dict, probe_private: dict,
                                    worlds_per_code: int, seed: int, world_base: int) -> dict:
    public_rows = {"obs": [], "action": [], "next_obs": [], "world_id": [], "world_step": []}
    hidden_rows = {k: [] for k in ("code", "motor_1", "motor_2", "damping", "contact_count")}
    world_records = []
    rng = np.random.default_rng(seed)
    nprobe = len(probe_public["obs"])
    for code_offset, code in enumerate(MIXED_CODES):
        for rep in range(worlds_per_code):
            world_id = world_base + code_offset * 10_000 + rep
            m1, m2, damp = physical_draw(code, rng)
            law = HiddenLaw(code, m1, m2, damp, False)
            world = HiddenPhysicsWorld(ENV_ID, [law] * nprobe, schema)
            try:
                world.reset(seed=seed + world_id)
                for index in range(nprobe):
                    obs = probe_public["obs"][index]
                    nxt, contact, restored_obs = restore_and_step(
                        world, probe_private["integration_state"][index], probe_public["action"][index], law, index)
                    if not np.allclose(restored_obs, obs, atol=1e-10, rtol=0):
                        raise RuntimeError("matched-probe state restore failed observation check")
                    public_rows["obs"].append(obs.copy())
                    public_rows["action"].append(probe_public["action"][index].copy())
                    public_rows["next_obs"].append(nxt.copy())
                    public_rows["world_id"].append(world_id)
                    public_rows["world_step"].append(index)
                    hidden_rows["code"].append(code); hidden_rows["motor_1"].append(m1)
                    hidden_rows["motor_2"].append(m2); hidden_rows["damping"].append(damp)
                    hidden_rows["contact_count"].append(contact)
            finally:
                world.close()
            world_records.append({"world_id": world_id, "code": code, "motor_1": m1,
                                  "motor_2": m2, "damping": damp})
    public = {k: np.asarray(v) for k, v in public_rows.items()}
    hidden = {k: np.asarray(v) for k, v in hidden_rows.items()}
    # Check deterministic physical replay on eight fixed states for each active set.
    replay_error = 0.0
    replay_records = []
    for code in MIXED_CODES:
        m1, m2, damp = physical_draw(code, np.random.default_rng(seed + code))
        law = HiddenLaw(code, m1, m2, damp, False)
        left = HiddenPhysicsWorld(ENV_ID, [law] * 8, schema)
        right = HiddenPhysicsWorld(ENV_ID, [law] * 8, schema)
        try:
            left.reset(seed=seed + 1); right.reset(seed=seed + 1)
            errors = []
            for i in range(8):
                yl, _, _ = restore_and_step(left, probe_private["integration_state"][i],
                                            probe_public["action"][i], law, i)
                yr, _, _ = restore_and_step(right, probe_private["integration_state"][i],
                                            probe_public["action"][i], law, i)
                errors.append(float(np.max(np.abs(yl - yr))))
            replay_error = max(replay_error, max(errors))
            replay_records.append({"code": code, "max_abs_replay_error": max(errors)})
        finally:
            left.close(); right.close()
    record = save_split(root, "matched_probe_outcomes", public, hidden,
                        {"world_count": len(world_records), "transitions": len(public["world_id"]),
                         "worlds": world_records})
    record["deterministic_replay_max_abs_error"] = replay_error
    record["deterministic_replay_by_code"] = replay_records
    return record


def load_existing_split(split: str) -> tuple[dict, dict, dict]:
    data_root = ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/data"
    evaluator_root = ROOT / "results/rwt/clsm_evaluator/comp/HalfCheetah-v5"
    _, public_hash = load_public_split(data_root, split)
    manifest = json.loads((evaluator_root / "manifest.json").read_text(encoding="utf-8"))
    private_path = Path(manifest[split]["evaluator_file"])
    if digest(private_path) != manifest[split]["evaluator_sha256"]:
        raise ValueError(f"{split} evaluator file hash mismatch")
    with np.load(data_root / f"{split}.npz", allow_pickle=False) as f:
        public = {k: f[k].copy() for k in PUBLIC_KEYS}
    with np.load(private_path, allow_pickle=False) as f:
        hidden = {k: f[k].copy() for k in ("code", "motor_1", "motor_2", "damping")}
    if len(hidden["code"]) != len(public["world_id"]):
        raise ValueError("public/evaluator transition counts do not align")
    meta = {"public_sha256": public_hash, "evaluator_sha256": manifest[split]["evaluator_sha256"],
            "world_count": len(np.unique(public["world_id"])), "transitions": len(public["world_id"]),
            "source_paths": [str(data_root / f"{split}.npz"), str(private_path)]}
    return public, hidden, meta


def prepare_data(root: Path, schema) -> dict:
    started = time.perf_counter()
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split, entry in manifest["splits"].items():
            if split in ("singleton_train", "singleton_validation"):
                continue
            if digest(Path(entry["public_path"])) != entry["public_sha256"] or digest(Path(entry["evaluator_path"])) != entry["evaluator_sha256"]:
                raise ValueError(f"cached {split} data hash mismatch")
        return manifest
    singleton_train = load_existing_split("train")
    singleton_val = load_existing_split("validation")
    splits = {
        "singleton_train": {"public_sha256": singleton_train[2]["public_sha256"],
                             "evaluator_sha256": singleton_train[2]["evaluator_sha256"],
                             "public_path": singleton_train[2]["source_paths"][0],
                             "evaluator_path": singleton_train[2]["source_paths"][1],
                             "world_count": singleton_train[2]["world_count"],
                             "transitions": singleton_train[2]["transitions"]},
        "singleton_validation": {"public_sha256": singleton_val[2]["public_sha256"],
                                  "evaluator_sha256": singleton_val[2]["evaluator_sha256"],
                                  "public_path": singleton_val[2]["source_paths"][0],
                                  "evaluator_path": singleton_val[2]["source_paths"][1],
                                  "world_count": singleton_val[2]["world_count"],
                                  "transitions": singleton_val[2]["transitions"]},
    }
    designs = [
        ("mixed_train", MIXED_CODES, MIXED_TRAIN_WORLDS_PER_CODE, 41_100, 300_000),
        ("mixed_validation", MIXED_CODES, MIXED_VAL_WORLDS_PER_CODE, 42_100, 400_000),
        ("composition_diagnostic", MIXED_CODES, EVAL_WORLDS_PER_CODE, 43_100, 500_000),
        ("composition_confirmatory_reserved", MIXED_CODES, CONFIRM_WORLDS_PER_CODE, 44_100, 600_000),
    ]
    for name, codes, nworld, seed, wid in designs:
        public, hidden, meta = collect_world_set(ENV_ID, schema, codes, nworld, WORLD_LENGTH, seed, wid)
        splits[name] = save_split(root, name, public, hidden, meta)
    probe_public, probe_private = collect_probe_states(ENV_ID, schema, PROBE_COUNT, 45_100)
    public_probe_path = root / "public" / "matched_probe_bank.npz"
    private_probe_path = root / "evaluator_only" / "matched_probe_bank_state.npz"
    np.savez_compressed(public_probe_path, **probe_public)
    np.savez_compressed(private_probe_path, **probe_private)
    splits["matched_probe_bank"] = {"public_path": str(public_probe_path),
                                    "public_sha256": digest(public_probe_path),
                                    "evaluator_path": str(private_probe_path),
                                    "evaluator_sha256": digest(private_probe_path),
                                    "world_count": 0, "transitions": PROBE_COUNT}
    matched = generate_matched_probe_outcomes(root, schema, probe_public, probe_private,
                                              EVAL_WORLDS_PER_CODE, 46_100, 700_000)
    splits["matched_probe_outcomes"] = matched
    all_ids = {}
    for name, item in splits.items():
        if name in ("singleton_train", "singleton_validation"):
            continue
        path = Path(item["public_path"])
        with np.load(path, allow_pickle=False) as f:
            if "world_id" in f.files:
                all_ids[name] = sorted(np.unique(f["world_id"]).astype(int).tolist())
    manifest = {
        "protocol": "docs/clsm_r2_gate_a_protocol.md", "env": ENV_ID,
        "architecture_revision": 2, "source_commit": _git("rev-parse", "HEAD"),
        "source_fingerprint": source_fingerprint(), "bit_codes": {"011": 3, "101": 5, "110": 6, "111": 7},
        "singletons": [0, 1, 2, 4], "physical_strength_bounds": {"motor": [.50, .85], "damping": [1.30, 2.00]},
        "fixed_world_length": WORLD_LENGTH, "world_ids": all_ids, "splits": splits,
        "train_seed": 41_100, "validation_seed": 42_100, "diagnostic_seed": 43_100,
        "reserved_confirmatory_seed": 44_100, "probe_seed": 45_100,
        "matched_outcome_seed": 46_100, "matched_probe_count": PROBE_COUNT,
        "runtime_seconds": {"data_generation": time.perf_counter() - started},
        "oracle_labels_in_public_data": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def _git(*args: str) -> str:
    import subprocess
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def features(public: dict, hidden: dict, conditioned: bool) -> np.ndarray:
    base = np.concatenate((public["obs"], public["action"]), axis=1)
    if not conditioned:
        return base.astype(np.float64)
    code = hidden["code"].astype(np.int64)
    bits = ((code[:, None] >> np.arange(3)) & 1).astype(np.float64)
    xi = strength_xi(code, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    return np.concatenate((base, bits, xi), axis=1).astype(np.float64)


class GaussianMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width, dtype=torch.float64), nn.SiLU(),
                                 nn.Linear(width, width, dtype=torch.float64), nn.SiLU(),
                                 nn.Linear(width, output_dim, dtype=torch.float64))
        self.log_variance = nn.Parameter(torch.zeros(output_dim, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        return mean, self.log_variance.exp().clamp(1e-5, 1e2).expand_as(mean)


class ModelAdapter:
    def __init__(self, model_type: str, model: nn.Module, x_mean: torch.Tensor | None,
                 x_scale: torch.Tensor | None, delta_mean: torch.Tensor,
                 delta_scale: torch.Tensor, code: torch.Tensor | None = None):
        self.model_type = model_type; self.model = model
        self.x_mean = x_mean; self.x_scale = x_scale
        self.delta_mean = delta_mean; self.delta_scale = delta_scale; self.code = code

    def predict(self, obs: torch.Tensor, action: torch.Tensor, code: torch.Tensor,
                xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons",
                               "oracle_G_z_xi_mixed_seen"):
            mean, var = self.model(obs, action, code, xi)
            return ((mean - obs - self.delta_mean) / self.delta_scale,
                    var / self.delta_scale.square())
        raw = torch.cat((obs, action), dim=-1)
        if self.model_type == "oracle_conditioned_unstructured_MLP":
            bits = ((code[:, None] >> torch.arange(3, device=code.device)) & 1).to(obs.dtype)
            raw = torch.cat((raw, bits, xi), dim=-1)
        x = (raw - self.x_mean) / self.x_scale
        pred, var = self.model(x)
        return pred, var


def _mlp_width(input_dim: int, output_dim: int, target_parameters: int) -> int:
    choices = range(180, 281)
    return min(choices, key=lambda width: abs(((input_dim + 1) * width + (width + 1) * width +
                                               (width + 1) * output_dim + output_dim) - target_parameters))


def _nll(y: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return .5 * ((y - mean).square() / variance.clamp_min(1e-8) +
                 variance.clamp_min(1e-8).log() + math.log(2 * math.pi)).mean()


def load_joined(split: str) -> tuple[dict, dict]:
    public, hidden, _ = load_existing_split(split)
    return public, hidden


def build_adapter(model_type: str, seed: int, ref_payload: dict, config: dict,
                  schema, train_public: dict, train_hidden: dict, nominal: dict,
                  partition_oracle: torch.Tensor, partition_learned: torch.Tensor) -> ModelAdapter:
    torch.manual_seed(seed)
    obs_dim, action_dim = schema.observation_dim, schema.action_dim
    delta_mean = ref_payload["law"]["delta_mean"].clone().to(torch.float64)
    delta_scale = ref_payload["law"]["delta_scale"].clone().to(torch.float64)
    if model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons",
                      "oracle_G_z_xi_mixed_seen"):
        law = ExecutableLaw(schema, config["network"]["hidden_dim"],
                            config["emission"]["variance_floor"], config["emission"]["variance_ceiling"])
        law.load_state_dict(ref_payload["law"])
        chosen = partition_learned if model_type.startswith("learned_G") else partition_oracle
        law.mechanics.set_partition(chosen)
        law.train()
        return ModelAdapter(model_type, law, None, None, delta_mean, delta_scale)
    condition = model_type == "oracle_conditioned_unstructured_MLP"
    train_x = features(train_public, train_hidden, condition)
    nominal_hidden = {"code": np.zeros(len(nominal["obs"]), dtype=np.int64),
                      "motor_1": np.ones(len(nominal["obs"])), "motor_2": np.ones(len(nominal["obs"])),
                      "damping": np.ones(len(nominal["obs"]))}
    nominal_public = {"obs": nominal["obs"], "action": nominal["action"]}
    nom_x = features(nominal_public, nominal_hidden, condition)
    all_x = np.concatenate((train_x, nom_x), axis=0)
    x_mean = torch.as_tensor(all_x.mean(0), dtype=torch.float64)
    x_scale = torch.as_tensor(all_x.std(0), dtype=torch.float64).clamp_min(1e-5)
    if model_type == "oracle_conditioned_unstructured_MLP":
        input_dim = obs_dim + action_dim + 6
    else:
        input_dim = obs_dim + action_dim
    target_law = ExecutableLaw(schema, config["network"]["hidden_dim"],
                               config["emission"]["variance_floor"], config["emission"]["variance_ceiling"])
    target_params = sum(p.numel() for p in target_law.parameters())
    width = _mlp_width(input_dim, obs_dim, target_params)
    model = GaussianMLP(input_dim, obs_dim, width)
    model.train()
    return ModelAdapter(model_type, model, x_mean, x_scale, delta_mean, delta_scale)


def tensor_public(public: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.as_tensor(public[k], dtype=torch.float64) for k in ("obs", "action", "next_obs"))


def tensor_hidden(hidden: dict) -> tuple[torch.Tensor, torch.Tensor]:
    code = torch.as_tensor(hidden["code"], dtype=torch.long)
    xi = torch.as_tensor(strength_xi(hidden["code"], hidden["motor_1"], hidden["motor_2"], hidden["damping"]), dtype=torch.float64)
    return code, xi


def fit_model(adapter: ModelAdapter, model_type: str, seed: int, train_pub: dict, train_hidden: dict,
              val_pub: dict, val_hidden: dict, nominal: dict, delta_mean: torch.Tensor,
              delta_scale: torch.Tensor, epochs: int, patience: int, batch_size: int,
              out_checkpoint: Path, config: dict, source_fp: str) -> dict:
    obs, action, next_obs = tensor_public(train_pub)
    code, xi = tensor_hidden(train_hidden)
    nlaw = len(obs)
    nobs, naction, nnext = tensor_public(nominal)
    ncode = torch.zeros(len(nobs), dtype=torch.long)
    nxi = torch.zeros((len(nobs), 3), dtype=torch.float64)
    all_obs = torch.cat((obs, nobs)); all_action = torch.cat((action, naction))
    all_next = torch.cat((next_obs, nnext)); all_code = torch.cat((code, ncode)); all_xi = torch.cat((xi, nxi))
    target = (all_next - all_obs - delta_mean) / delta_scale
    if model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
        adapter.model.mechanics.set_partition(adapter.model.mechanics.group_mask.bool())
    # Validation is domain-specific: singleton models see only singleton validation;
    # the mixed-seen positive control sees only fresh mixed validation.
    vo, va, vn = tensor_public(val_pub)
    vc, vxi = tensor_hidden(val_hidden)
    vtarget = (vn - vo - delta_mean) / delta_scale
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 971)
    best_val = math.inf; best_epoch = -1; best_state = None; stale = 0; history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        adapter.model.train()
        order = rng.permutation(len(all_obs))
        train_losses = []
        for begin in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[begin:begin + batch_size], dtype=torch.long)
            if model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
                mean_raw, var_raw = adapter.model(all_obs[idx], all_action[idx], all_code[idx], all_xi[idx])
                mean = (mean_raw - all_obs[idx] - delta_mean) / delta_scale
                var = var_raw / delta_scale.square()
            else:
                raw_pub = {"obs": all_obs[idx].numpy(), "action": all_action[idx].numpy()}
                raw_hid = {"code": all_code[idx].numpy(), "motor_1": np.ones(len(idx)),
                           "motor_2": np.ones(len(idx)), "damping": np.ones(len(idx))}
                if model_type == "oracle_conditioned_unstructured_MLP":
                    raw_hid["motor_1"] = .50 + .35 * torch.special.ndtr(all_xi[idx, 0]).numpy()
                    raw_hid["motor_2"] = .50 + .35 * torch.special.ndtr(all_xi[idx, 1]).numpy()
                    raw_hid["damping"] = 1.30 + .70 * torch.special.ndtr(all_xi[idx, 2]).numpy()
                xraw = torch.as_tensor(features(raw_pub, raw_hid, model_type == "oracle_conditioned_unstructured_MLP"), dtype=torch.float64)
                x = (xraw - adapter.x_mean) / adapter.x_scale
                mean, var = adapter.model(x)
            loss = _nll(target[idx], mean, var)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 5.0)
            optimizer.step(); train_losses.append(float(loss.detach()))
        adapter.model.eval()
        with torch.no_grad():
            if model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
                val_raw, vv_raw = adapter.model(vo, va, vc, vxi)
                val_mean = (val_raw - vo - delta_mean) / delta_scale
                val_var = vv_raw / delta_scale.square()
            else:
                vp = {"obs": vo.numpy(), "action": va.numpy()}
                vh = {"code": vc.numpy(), "motor_1": np.ones(len(vc)), "motor_2": np.ones(len(vc)), "damping": np.ones(len(vc))}
                if model_type == "oracle_conditioned_unstructured_MLP":
                    vh["motor_1"] = .50 + .35 * torch.special.ndtr(vxi[:, 0]).numpy()
                    vh["motor_2"] = .50 + .35 * torch.special.ndtr(vxi[:, 1]).numpy()
                    vh["damping"] = 1.30 + .70 * torch.special.ndtr(vxi[:, 2]).numpy()
                vx = (torch.as_tensor(features(vp, vh, model_type == "oracle_conditioned_unstructured_MLP")) - adapter.x_mean) / adapter.x_scale
                val_mean, val_var = adapter.model(vx)
            val_loss = float(_nll(vtarget, val_mean, val_var))
        history.append({"epoch": epoch + 1, "train_nll": float(np.mean(train_losses)), "validation_nll": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_epoch = epoch + 1
            best_state = copy.deepcopy(adapter.model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint selected")
    adapter.model.load_state_dict(best_state); adapter.model.eval()
    metadata = {"role": "oracle_diagnostic", "diagnostic_id": "CLSM-R2-Gate-A",
                "checkpoint_type": model_type, "architecture_revision": 2,
                "source_fingerprint": source_fp, "source_commit": _git("rev-parse", "HEAD"),
                "seed": seed, "config": config, "best_epoch": best_epoch,
                "best_validation_nll": best_val, "input_mean": adapter.x_mean,
                "input_scale": adapter.x_scale, "delta_mean": delta_mean,
                "delta_scale": delta_scale, "partition": (adapter.model.mechanics.group_mask.detach().cpu()
                                                               if hasattr(adapter.model, "mechanics") else None),
                "state_dict": adapter.model.state_dict()}
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, out_checkpoint)
    return {"model": model_type, "seed": seed, "best_epoch": best_epoch,
            "best_validation_nll": best_val, "epochs_run": len(history),
            "runtime_seconds": time.perf_counter() - start, "checkpoint": str(out_checkpoint),
            "checkpoint_sha256": digest(out_checkpoint), "training_history": history}


def load_diagnostic_model(path: Path, model_type: str, schema, config: dict,
                          ref_payload: dict) -> ModelAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("role") != "oracle_diagnostic" or payload.get("diagnostic_id") != "CLSM-R2-Gate-A":
        raise ValueError("Gate A loader only accepts tagged oracle_diagnostic checkpoints")
    if payload.get("checkpoint_type") != model_type or payload.get("architecture_revision") != 2:
        raise ValueError("diagnostic checkpoint identity/version mismatch")
    dm = payload["delta_mean"].to(torch.float64); ds = payload["delta_scale"].to(torch.float64)
    if model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
        model = ExecutableLaw(schema, config["network"]["hidden_dim"],
                              config["emission"]["variance_floor"], config["emission"]["variance_ceiling"])
        model.load_state_dict(payload["state_dict"])
        return ModelAdapter(model_type, model.eval(), None, None, dm, ds)
    input_dim = schema.observation_dim + schema.action_dim + (6 if model_type == "oracle_conditioned_unstructured_MLP" else 0)
    target_params = sum(p.numel() for p in ExecutableLaw(schema, config["network"]["hidden_dim"]).parameters())
    width = _mlp_width(input_dim, schema.observation_dim, target_params)
    model = GaussianMLP(input_dim, schema.observation_dim, width)
    model.load_state_dict(payload["state_dict"])
    return ModelAdapter(model_type, model.eval(), payload["input_mean"].to(torch.float64),
                        payload["input_scale"].to(torch.float64), dm, ds)


def predict_numpy(adapter: ModelAdapter, obs: np.ndarray, action: np.ndarray,
                  code: np.ndarray, xi: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    means, variances = [], []
    with torch.no_grad():
        for begin in range(0, len(obs), batch):
            end = min(len(obs), begin + batch)
            o = torch.as_tensor(obs[begin:end], dtype=torch.float64)
            a = torch.as_tensor(action[begin:end], dtype=torch.float64)
            c = torch.as_tensor(code[begin:end], dtype=torch.long)
            x = torch.as_tensor(xi[begin:end], dtype=torch.float64)
            m, v = adapter.predict(o, a, c, x)
            means.append(m.numpy()); variances.append(v.numpy())
    return np.concatenate(means), np.concatenate(variances)


def score_rows(adapter: ModelAdapter, public: dict, hidden: dict, delta_mean: np.ndarray,
               delta_scale: np.ndarray, model: str, seed: int, split_name: str) -> tuple[list[dict], np.ndarray]:
    code = hidden["code"].astype(np.int64)
    xi = strength_xi(code, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    mean, var = predict_numpy(adapter, public["obs"], public["action"], code, xi)
    target = (public["next_obs"] - public["obs"] - delta_mean) / delta_scale
    residual = target - mean
    nll = .5 * (residual ** 2 / np.maximum(var, 1e-8) + np.log(np.maximum(var, 1e-8)) + math.log(2 * math.pi)).mean(1)
    mse = np.mean(residual ** 2, axis=1)
    rows = []
    for world_id in np.unique(public["world_id"]):
        select = public["world_id"] == world_id
        rows.append({"model": model, "seed": seed, "split": split_name,
                     "code": int(code[select][0]), "world_id": int(world_id),
                     "n_transitions": int(select.sum()), "nll_per_dim": float(nll[select].mean()),
                     "normalized_mse": float(mse[select].mean()),
                     "normalized_signature_error": float(np.sqrt(mse[select].mean()))})
    return rows, residual


def metric_ci(world_values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(world_values)
    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    means = world_values[draws].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def paired_ci(left: np.ndarray, right: np.ndarray, seed: int) -> dict:
    diff = left - right
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    boot = diff[idx].mean(1)
    return {"mean_difference": float(diff.mean()), "ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
            "n_worlds": int(n), "worlds_left_worse": int(np.sum(diff > 0))}


def aggregate_scores(rows: list[dict], root: Path) -> tuple[dict, list[dict]]:
    models = sorted({r["model"] for r in rows})
    summary: dict = {"per_composition": {}, "paired_comparisons": {}, "macro_average": {}}
    world_rows = []
    for model in models:
        for code in MIXED_CODES:
            selected = [r for r in rows if r["model"] == model and r["code"] == code]
            ids = sorted({r["world_id"] for r in selected})
            for world_id in ids:
                matches = [r for r in selected if r["world_id"] == world_id]
                world_rows.append({"model": model, "code": code, "world_id": world_id,
                                   **{metric: float(np.mean([m[metric] for m in matches]))
                                      for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error")}})
    for model in models:
        summary["per_composition"][model] = {}
        for code in MIXED_CODES:
            cells = [r for r in world_rows if r["model"] == model and r["code"] == code]
            summary["per_composition"][model][str(code)] = {metric: {
                "mean": float(np.mean([r[metric] for r in cells])),
                "world_bootstrap_ci95": metric_ci(np.asarray([r[metric] for r in cells]), 88_000 + code)}
                for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error")}
    comparisons = [
        ("oracle_G_z_xi_singletons", "oracle_conditioned_unstructured_MLP"),
        ("oracle_G_z_xi_singletons", "oracle_G_z_xi_mixed_seen"),
        ("oracle_G_z_xi_mixed_seen", "oracle_conditioned_unstructured_MLP"),
        ("learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_singletons"),
        ("oracle_G_z_xi_singletons", "pooled_MLP"),
    ]
    for left_model, right_model in comparisons:
        key = f"{left_model}_minus_{right_model}"
        summary["paired_comparisons"][key] = {}
        for code in MIXED_CODES:
            left = {r["world_id"]: r for r in world_rows if r["model"] == left_model and r["code"] == code}
            right = {r["world_id"]: r for r in world_rows if r["model"] == right_model and r["code"] == code}
            ids = sorted(set(left) & set(right))
            summary["paired_comparisons"][key][str(code)] = {
                metric: paired_ci(np.asarray([left[i][metric] for i in ids]),
                                  np.asarray([right[i][metric] for i in ids]),
                                  seed=99_000 + code + len(metric))
                for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error")}
            l_mse = np.asarray([left[i]["normalized_mse"] for i in ids])
            r_mse = np.asarray([right[i]["normalized_mse"] for i in ids])
            summary["paired_comparisons"][key][str(code)]["severe_mse_ratio_gt_2"] = {
                "fraction": float(np.mean(l_mse > 2 * np.maximum(r_mse, 1e-12))),
                "world_bootstrap_ci95": metric_ci((l_mse > 2 * np.maximum(r_mse, 1e-12)).astype(float), 101_000 + code)}
    for model in models:
        means = []
        for code in MIXED_CODES:
            means.append(np.mean([r["normalized_mse"] for r in world_rows if r["model"] == model and r["code"] == code]))
        summary["macro_average"][model] = {"normalized_mse": float(np.mean(means))}
    with (root / "heldout_world_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(world_rows[0]))
        writer.writeheader(); writer.writerows(world_rows)
    return summary, world_rows


def fisher_for_adapter(adapter: ModelAdapter, obs: np.ndarray, action: np.ndarray,
                       hidden: dict, delta_mean: torch.Tensor, delta_scale: torch.Tensor) -> np.ndarray:
    from torch.func import jacrev, vmap
    codes = hidden["code"].astype(np.int64)
    xi_np = strength_xi(codes, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    o = torch.as_tensor(obs, dtype=torch.float64); a = torch.as_tensor(action, dtype=torch.float64)
    x = torch.as_tensor(xi_np, dtype=torch.float64)
    c = torch.as_tensor(codes, dtype=torch.long)
    if adapter.model_type in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
        def fn(one_o, one_a, one_x):
            one_c = c[0:1]
            mean, var = adapter.model(one_o[None], one_a[None], one_c, one_x[None])
            return (mean[0] - one_o - delta_mean) / delta_scale
        def var_fn(one_o, one_a, one_x):
            _, var = adapter.model(one_o[None], one_a[None], c[0:1], one_x[None])
            return var[0] / delta_scale.square()
    else:
        def fn(one_o, one_a, one_x):
            code_bits = ((c[0] >> torch.arange(3)) & 1).to(torch.float64)
            raw = torch.cat((one_o, one_a))
            if adapter.model_type == "oracle_conditioned_unstructured_MLP":
                raw = torch.cat((raw, code_bits, one_x))
            pred, _ = adapter.model(((raw - adapter.x_mean) / adapter.x_scale)[None])
            return pred[0]
        def var_fn(one_o, one_a, one_x):
            code_bits = ((c[0] >> torch.arange(3)) & 1).to(torch.float64)
            raw = torch.cat((one_o, one_a))
            if adapter.model_type == "oracle_conditioned_unstructured_MLP":
                raw = torch.cat((raw, code_bits, one_x))
            _, var = adapter.model(((raw - adapter.x_mean) / adapter.x_scale)[None])
            return var[0]
    jac = vmap(jacrev(fn, argnums=2))(o, a, x).detach().cpu().numpy()
    var = vmap(var_fn)(o, a, x).detach().cpu().numpy()
    active = [i for i in range(3) if int(codes[0]) & (1 << i)]
    info = np.einsum("ndq,nd,ndr->qr", jac[:, :, active], 1 / np.maximum(var, 1e-8),
                     jac[:, :, active])
    full = np.zeros((3, 3), dtype=np.float64)
    full[np.ix_(active, active)] = info
    return full


def fisher_analysis(model_adapters: dict, probe_public: dict, outcome_public: dict,
                    outcome_hidden: dict, dm: torch.Tensor, ds: torch.Tensor,
                    root: Path) -> dict:
    by_model: dict = {}
    codes = outcome_hidden["code"].astype(np.int64)
    for model in ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons", "oracle_G_z_xi_mixed_seen"):
        by_model[model] = {}
        adapter = model_adapters[model][0]  # predeclared seed 4100 for the Fisher characterization
        for code in MIXED_CODES:
            world_ids = np.unique(outcome_public["world_id"][codes == code])
            matrices = []
            for world_id in world_ids:
                idx = np.flatnonzero(outcome_public["world_id"] == world_id)
                mat = fisher_for_adapter(adapter, outcome_public["obs"][idx], outcome_public["action"][idx],
                                         {k: outcome_hidden[k][idx] for k in outcome_hidden}, dm, ds)
                matrices.append(mat)
            arr = np.stack(matrices)
            avg = arr.mean(0)
            active = [i for i in range(3) if code & (1 << i)]
            eigs = np.linalg.eigvalsh(avg[np.ix_(active, active)])
            sv = np.linalg.svd(avg[np.ix_(active, active)], compute_uv=False)
            rank = int(np.linalg.matrix_rank(avg[np.ix_(active, active)], tol=max(sv[0] * 1e-8, 1e-12)))
            corr = np.zeros((3, 3), dtype=np.float64)
            block = avg[np.ix_(active, active)]
            std = np.sqrt(np.maximum(np.diag(block), 1e-30))
            corr[np.ix_(active, active)] = block / np.outer(std, std)
            ci = {}
            for i in range(3):
                for j in range(3):
                    if i not in active or j not in active:
                        continue
                    ci[f"{i},{j}"] = metric_ci(arr[:, i, j], 110_000 + code * 10 + i * 3 + j)
            by_model[model][str(code)] = {
                "world_count": len(world_ids), "fisher_3x3_sum_over_128_probes_mean_over_worlds": avg.tolist(),
                "fisher_element_world_bootstrap_ci95": ci, "sensitivity_correlation": corr.tolist(),
                "active_coordinates": active, "eigenvalues_active": eigs.tolist(),
                "minimum_singular_value_active": float(sv[-1]),
                "condition_number_active": float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
                "rank_conditional_on_fitted_decoder": rank,
                "rank_tolerance": float(max(sv[0] * 1e-8, 1e-12)),
                "interpretation": "decoder-conditional information only; not global partition or simulator identifiability",
            }
    result = {"probe_bank_sha256": digest(root / "public/matched_probe_bank.npz"),
              "models": by_model, "probe_count": PROBE_COUNT,
              "definition": "sum_probe J^T Sigma_normalized^-1 J, inactive coordinates embedded as zeros"}
    write_json(root / "full_fisher.json", result)
    return result


def residual_tables(residual_records: list[dict], root: Path, velocity_cuts: list[float]) -> None:
    path = root / "per_dimension_residuals_by_regime.csv"
    fields = ["model", "seed", "split", "code", "obs_dimension", "velocity_regime", "contact_regime",
              "count", "signed_bias", "rmse"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for rec in residual_records:
            vel = rec["public"]["obs"][:, rec["velocity_indices"]]
            speed = np.linalg.norm(vel, axis=1)
            velocity_bin = np.digitize(speed, velocity_cuts)
            contacts = rec["hidden"]["contact_count"]
            contact_bin = (contacts > 0).astype(int)
            residual = rec["residual"]
            for d in range(residual.shape[1]):
                for vb in range(3):
                    for cb in (0, 1):
                        idx = (velocity_bin == vb) & (contact_bin == cb)
                        if not idx.any():
                            continue
                        values = residual[idx, d]
                        writer.writerow({"model": rec["model"], "seed": rec["seed"], "split": rec["split"],
                                         "code": rec["code"], "obs_dimension": d,
                                         "velocity_regime": ("low", "mid", "high")[vb],
                                         "contact_regime": ("no_contact", "contact")[cb],
                                         "count": int(len(values)), "signed_bias": float(values.mean()),
                                         "rmse": float(np.sqrt(np.mean(values ** 2)))})


def make_residual_plot(rows: list[dict], root: Path) -> None:
    import matplotlib.pyplot as plt
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, code in zip(axes, MIXED_CODES):
        for model in models:
            vals = [r["normalized_mse"] for r in rows if r["model"] == model and r["code"] == code]
            if vals:
                ax.scatter(np.full(len(vals), models.index(model)), vals, s=9, alpha=.35)
                ax.plot(models.index(model), np.mean(vals), marker="_", markersize=18, color="black")
        ax.set_title(f"code {code:03b}"); ax.set_xticks(range(len(models)), models, rotation=90, fontsize=7)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Per-world normalized delta MSE")
    fig.tight_layout(); fig.savefig(root / "heldout_per_world_normalized_mse.png", dpi=160); plt.close(fig)


def decide_gate_a(summary: dict, fisher: dict, replay_error: float, root: Path) -> dict:
    comp = summary["paired_comparisons"]
    primary_single = comp["oracle_G_z_xi_singletons_minus_oracle_conditioned_unstructured_MLP"]
    mixed_vs_unstructured = comp["oracle_G_z_xi_mixed_seen_minus_oracle_conditioned_unstructured_MLP"]
    single_vs_mixed = comp["oracle_G_z_xi_singletons_minus_oracle_G_z_xi_mixed_seen"]
    learned_vs_oracle = comp["learned_G_oracle_z_xi_singletons_minus_oracle_G_z_xi_singletons"]
    decision = "INCONCLUSIVE"
    reason = "The criteria are not decisively satisfied by paired independent-world evidence."
    representation_fail = any(mixed_vs_unstructured[str(code)][metric]["ci95"][0] > 0
                               for code in MIXED_CODES for metric in ("normalized_mse", "normalized_signature_error"))
    per_code_fail = [int(code) for code in MIXED_CODES
                     if primary_single[str(code)]["normalized_mse"]["ci95"][0] > 0
                     and primary_single[str(code)]["normalized_signature_error"]["ci95"][0] > 0
                     and single_vs_mixed[str(code)]["normalized_mse"]["ci95"][0] > 0]
    # Capacity and zero-shot gaps are kept distinct. An absent mixed-seen baseline contrast is inconclusive.
    if replay_error > 1e-10:
        reason = "Matched-state simulator replay exceeded 1e-10; matched-probe evidence is invalid."
    elif representation_fail:
        decision = "FAIL"
        reason = "The oracle-G mixed-seen structured positive control is worse than the oracle-conditioned unstructured baseline on at least one registered composition metric with paired-world 95% CI above zero."
    elif per_code_fail:
        decision = "FAIL"
        reason = f"Mixed-seen structured fits better, while singleton-trained structured is worse than both the mixed-seen control and oracle-conditioned MLP on codes {per_code_fail} with paired-world CIs above zero."
    elif any(learned_vs_oracle[str(code)][metric]["ci95"][0] > 0
             for code in MIXED_CODES for metric in ("normalized_mse", "normalized_signature_error")):
        decision = "FAIL"
        reason = "Train-selected support is materially worse than oracle support on the held-out compositions with paired-world confidence intervals excluding zero."
    else:
        all_noninferior = all(primary_single[str(code)][metric]["ci95"][1] <= 0
                              for code in MIXED_CODES for metric in ("normalized_mse", "normalized_signature_error"))
        mixed_capacity_supported = all(mixed_vs_unstructured[str(code)][metric]["ci95"][1] <= 0
                                       for code in MIXED_CODES for metric in ("normalized_mse", "normalized_signature_error"))
        no_support_penalty = all(learned_vs_oracle[str(code)][metric]["ci95"][1] <= 0
                                 for code in MIXED_CODES for metric in ("normalized_mse", "normalized_signature_error"))
        full_fisher = all(fisher["models"]["oracle_G_z_xi_singletons"][str(code)]["rank_conditional_on_fitted_decoder"]
                          == int(code.bit_count()) for code in MIXED_CODES)
        if all_noninferior and mixed_capacity_supported and no_support_penalty and full_fisher:
            decision = "PASS"
            reason = "On the registered probe family, singleton-trained structured is not worse than oracle-conditioned MLP by paired-world CIs for MSE/signature; mixed-seen capacity and learned-support controls pass, with full conditional active-set Fisher rank."
    result = {"gate": "A", "status": decision, "reason": reason,
              "registered_protocol": "docs/clsm_r2_gate_a_protocol.md",
              "simulator_replay_max_abs_error": replay_error,
              "zero_shot_codes_with_clear_mse_and_signature_deterioration": per_code_fail,
              "mixed_seen_vs_unstructured_summary": mixed_vs_unstructured,
              "limitations": ["Decision applies only to HalfCheetah-v5 declared intervention family and sampled coverage.",
                              "One nominal-reference initialization lineage; three diagnostic training seeds.",
                              "Confirmatory worlds were generated and reserved without scoring."],
              "next_action": "If FAIL, inspect residual structure and coverage before revising force-response representation; if PASS, proceed to Gate B; if INCONCLUSIVE, increase diagnostic worlds as justified by observed intervals."}
    write_json(root / "decision.json", result)
    (root / "decision.md").write_text(f"# Gate A decision\n\n**Status: {decision}.** {reason}\n\nSee `heldout_composition.json`, `full_fisher.json`, and `dataset_manifest.json` for evidence. This decision is scoped to the registered HalfCheetah-v5 family and diagnostic coverage.\n", encoding="utf-8")
    return result


def run(root: Path, args) -> dict:
    torch.set_num_threads(1)
    config = load_config(ROOT / "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml")
    schema = mujoco_nominal_schema(ENV_ID)
    data_manifest = prepare_data(root, schema)
    train_pub, train_hidden = load_joined("train")
    val_single_pub, val_single_hidden = load_joined("validation")
    mixed_train_pub, mixed_train_hidden = load_diagnostic_split(root, "mixed_train")
    mixed_val_pub, mixed_val_hidden = load_diagnostic_split(root, "mixed_validation")
    test_pub, test_hidden = load_diagnostic_split(root, "composition_diagnostic")
    probe_pub, probe_private = load_probe_bank(root)
    matched_pub, matched_hidden = load_diagnostic_split(root, "matched_probe_outcomes")
    nominal_npz = np.load(ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/reference_nominal.npz")
    nominal = {k: nominal_npz[k].copy() for k in ("obs", "action", "next_obs")}
    nominal_npz.close()
    nominal_choice = np.random.default_rng(41_111).choice(len(nominal["obs"]), size=4_000, replace=False)
    nominal = {k: value[nominal_choice] for k, value in nominal.items()}
    ref_payload = torch.load(ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_reference_r2.pt",
                             map_location="cpu", weights_only=True)
    pilot_payload = torch.load(ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_r2.pt",
                               map_location="cpu", weights_only=True)
    oracle_groups = fixed_motor_groups(schema)
    oracle_partition = torch.zeros((schema.action_dim, 2), dtype=torch.bool)
    oracle_partition[np.asarray(oracle_groups[0]), 0] = True
    oracle_partition[np.asarray(oracle_groups[1]), 1] = True
    learned_partition = torch.as_tensor(pilot_payload["selected_partition"], dtype=torch.bool)
    delta_mean = ref_payload["law"]["delta_mean"].to(torch.float64)
    delta_scale = ref_payload["law"]["delta_scale"].to(torch.float64)
    models = ("oracle_G_z_xi_singletons", "learned_G_oracle_z_xi_singletons",
              "oracle_G_z_xi_mixed_seen", "oracle_conditioned_unstructured_MLP", "pooled_MLP")
    training = []
    model_adapters: dict[str, list[ModelAdapter]] = {name: [] for name in models}
    eval_rows = []; residual_recs = []; matched_rows = []; matched_residual_recs = []
    source_fp = source_fingerprint()
    for model_type in models:
        if model_type == "oracle_G_z_xi_mixed_seen":
            fit_pub, fit_hidden = mixed_train_pub, mixed_train_hidden
            valid_pub, valid_hidden = mixed_val_pub, mixed_val_hidden
        else:
            fit_pub, fit_hidden = train_pub, train_hidden
            valid_pub, valid_hidden = val_single_pub, val_single_hidden
        for seed in TRAIN_SEEDS:
            adapter = build_adapter(model_type, seed, ref_payload, config, schema, fit_pub, fit_hidden,
                                    nominal, oracle_partition, learned_partition)
            ckpt = root / "checkpoints" / model_type / f"seed_{seed}.pt"
            training_report = fit_model(adapter, model_type, seed, fit_pub, fit_hidden, valid_pub,
                                        valid_hidden, nominal, delta_mean, delta_scale,
                                        args.epochs, args.patience, args.batch_size, ckpt, config, source_fp)
            training.append(training_report)
            # Reload the tagged artifact through the isolated diagnostic loader.
            loaded = load_diagnostic_model(ckpt, model_type, schema, config, ref_payload)
            model_adapters[model_type].append(loaded)
            rows, residual = score_rows(loaded, test_pub, test_hidden, delta_mean.numpy(), delta_scale.numpy(),
                                        model_type, seed, "composition_trajectory")
            eval_rows.extend(rows)
            residual_recs.append({"model": model_type, "seed": seed, "split": "composition_trajectory",
                                  "code": None, "public": test_pub, "hidden": test_hidden,
                                  "residual": residual, "velocity_indices": schema.joint_velocity_indices})
            prows, presidual = score_rows(loaded, matched_pub, matched_hidden, delta_mean.numpy(),
                                          delta_scale.numpy(), model_type, seed, "matched_state_action")
            matched_rows.extend(prows)
            matched_residual_recs.append({"model": model_type, "seed": seed, "split": "matched_state_action",
                                         "code": None, "public": matched_pub, "hidden": matched_hidden,
                                         "residual": presidual, "velocity_indices": schema.joint_velocity_indices})
            print(f"fit/eval {model_type} seed={seed} best_epoch={training_report['best_epoch']} "
                  f"val_nll={training_report['best_validation_nll']:.6f}", flush=True)
    write_json(root / "oracle_decoder_training.json", {
        "role": "oracle_diagnostic", "training_runs": training,
        "models": list(models), "training_seeds": list(TRAIN_SEEDS),
        "singleton_train_sha256": data_manifest["splits"]["singleton_train"]["public_sha256"],
        "mixed_train_sha256": data_manifest["splits"]["mixed_train"]["public_sha256"],
        "nominal_reference_sha256": _sha256(ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/reference_nominal.npz"),
        "source_fingerprint": source_fp,
        "warning": "All checkpoints in this file are privileged oracle_diagnostic models, not deployable CLSM-R2."})
    summary, world_rows = aggregate_scores(eval_rows, root)
    write_json(root / "heldout_composition.json", {
        "scope": "trajectory occupancy, 4 independent held-out composition codes, no adaptation",
        "codes": {f"{code:03b}": code for code in MIXED_CODES},
        "models": summary["per_composition"], "paired_comparisons": summary["paired_comparisons"],
        "macro_average_normalized_mse": summary["macro_average"],
        "worlds_per_code": EVAL_WORLDS_PER_CODE, "training_seeds_averaged_within_world": list(TRAIN_SEEDS),
        "confirmatory_scores": "reserved and not opened"})
    # Reuse same aggregator without overwriting the trajectory-world table.
    tmp_root = root / "matched_metrics"; tmp_root.mkdir(exist_ok=True)
    matched_summary, _ = aggregate_scores(matched_rows, tmp_root)
    write_json(root / "matched_probe_scores.json", {
        "scope": "fixed common matched state/action snapshots; oracle diagnostic only",
        "probe_count": PROBE_COUNT, "models": matched_summary["per_composition"],
        "paired_comparisons": matched_summary["paired_comparisons"],
        "macro_average_normalized_mse": matched_summary["macro_average"],
        "worlds_per_code": EVAL_WORLDS_PER_CODE})
    train_obs = train_pub["obs"]
    speed = np.linalg.norm(train_obs[:, schema.joint_velocity_indices], axis=1)
    cuts = np.quantile(speed, [1 / 3, 2 / 3]).tolist()
    residual_tables(residual_recs + matched_residual_recs, root, cuts)
    make_residual_plot(world_rows, root)
    fisher = fisher_analysis(model_adapters, probe_pub, matched_pub, matched_hidden,
                             delta_mean, delta_scale, root)
    decision = decide_gate_a(summary, fisher, data_manifest["splits"]["matched_probe_outcomes"]["deterministic_replay_max_abs_error"], root)
    result = {"gate_A": decision, "dataset_manifest": data_manifest,
              "training_runs": len(training), "trajectory_world_rows": len(world_rows),
              "matched_probe_world_rows": len(matched_world_rows or []),
              "fisher_model_seed": TRAIN_SEEDS[0], "runtime_seconds": None}
    write_json(root / "gate_A_summary.json", result)
    return result


def load_diagnostic_split(root: Path, name: str) -> tuple[dict, dict]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["splits"][name]
    pub_path, priv_path = Path(entry["public_path"]), Path(entry["evaluator_path"])
    if digest(pub_path) != entry["public_sha256"] or digest(priv_path) != entry["evaluator_sha256"]:
        raise ValueError(f"{name}: public or evaluator hash mismatch")
    with np.load(pub_path, allow_pickle=False) as f:
        public = {k: f[k].copy() for k in f.files}
    with np.load(priv_path, allow_pickle=False) as f:
        hidden = {k: f[k].copy() for k in f.files}
    if "code" in public or "motor_1" in public or len(hidden["code"]) != len(public["world_id"]):
        raise ValueError("diagnostic public/private split boundary is invalid")
    return public, hidden


def load_probe_bank(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["splits"]["matched_probe_bank"]
    pub_path, priv_path = Path(entry["public_path"]), Path(entry["evaluator_path"])
    if digest(pub_path) != entry["public_sha256"] or digest(priv_path) != entry["evaluator_sha256"]:
        raise ValueError("probe bank hash mismatch")
    with np.load(pub_path, allow_pickle=False) as f: public = {k: f[k].copy() for k in f.files}
    with np.load(priv_path, allow_pickle=False) as f: private = {k: f[k].copy() for k in f.files}
    return public, private


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm_r2_gates/gate_a_diagnostic"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    schema = mujoco_nominal_schema(ENV_ID)
    manifest = prepare_data(root, schema)
    if args.prepare_only:
        print(json.dumps({"dataset_manifest": str(root / "dataset_manifest.json"),
                          "split_counts": {k: {"worlds": v["world_count"], "transitions": v["transitions"]}
                                           for k, v in manifest["splits"].items()},
                          "composition_codes": {f"{c:03b}": c for c in MIXED_CODES},
                          "probe_count": PROBE_COUNT,
                          "reserved_confirmatory": "generated but not scored"}, indent=2))
        return 0
    result = run(root, args)
    print(json.dumps(result["gate_A"], indent=2))
    return 0 if result["gate_A"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
