"""Frozen R1 experiment for the CLSM-R3 final decoder attempt.

All training uses registered train/validation sets. Fresh physical worlds and
matched probes are generated and hashed before fitting, but their evaluator
labels are loaded only after all R3 checkpoints and rules are frozen.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
import yaml

try:
    import psutil
except ImportError:
    psutil = None

from rwt.clsm.config import load_config
from rwt.clsm.config_r3 import load_r3_config
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.dynamics_r3 import InterventionResolvedLaw
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import source_fingerprint
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld
from scripts import run_clsm_r2_gate_a_diagnostic as ga

OUT = ROOT / "results/rwt/clsm_r3_final/r1"
SOURCE = Path(r"C:\Users\CNC-NVCUONG-PGD\.codex\worktrees\clsm-implementation\learning-to-theorize")
GATE_ROOT = SOURCE / "results/rwt/clsm_r2_gates/gate_a_diagnostic"
ENV = "HalfCheetah-v5"
SEEDS = (4100, 4101, 4102)
MIXED_CODES = (3, 5, 6, 7)
SINGLETON_CODES = (0, 1, 2, 4)
BOOTSTRAP_DRAWS = 10_000


def process_peak_rss_bytes():
    if psutil is None:
        return None
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def attach_resource_metrics(rows, wall_start: float, cpu_start: float):
    worlds = len({r["world_id"] for r in rows})
    wall_per_world = (time.perf_counter() - wall_start) / max(worlds, 1)
    cpu_per_world = (time.process_time() - cpu_start) / max(worlds, 1)
    peak_rss = process_peak_rss_bytes()
    for row in rows:
        row["amortized_scoring_wall_seconds_per_world"] = wall_per_world
        row["amortized_scoring_cpu_seconds_per_world"] = cpu_per_world
        row["process_peak_rss_bytes_at_score"] = peak_rss


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def tensor_data(public: dict, hidden: dict):
    obs, action, nxt = ga.tensor_public(public)
    code, xi = ga.tensor_hidden(hidden)
    return obs, action, nxt, code, xi


def load_public_and_oracle(root: Path, name: str):
    public, hidden = ga.load_diagnostic_split(root, name)
    return public, hidden


def load_r1_split(root: Path):
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["split"]
    pub_path, evaluator_path = Path(entry["public_path"]), Path(entry["evaluator_path"])
    if sha256(pub_path) != entry["public_sha256"] or sha256(evaluator_path) != entry["evaluator_sha256"]:
        raise ValueError("R1 fresh split checksum mismatch")
    with np.load(pub_path, allow_pickle=False) as z:
        public = {k: z[k].copy() for k in z.files}
    with np.load(evaluator_path, allow_pickle=False) as z:
        hidden = {k: z[k].copy() for k in z.files}
    if any(k in public for k in ("code", "motor_1", "motor_2", "damping")):
        raise ValueError("R1 evaluator labels leaked into public split")
    if len(hidden["code"]) != len(public["world_id"]):
        raise ValueError("R1 evaluator/public row count mismatch")
    return public, hidden


def check_deterministic_replay(schema, probe_pub: dict, probe_priv: dict) -> float:
    error = 0.0
    for code in MIXED_CODES:
        m1, m2, damping = ga.physical_draw(code, np.random.default_rng(46_000 + code))
        law = HiddenLaw(code, m1, m2, damping, False)
        left = HiddenPhysicsWorld(ENV, [law], schema)
        right = HiddenPhysicsWorld(ENV, [law], schema)
        try:
            left.reset(seed=46_100 + code); right.reset(seed=46_100 + code)
            y_left, _, _ = ga.restore_and_step(left, probe_priv["integration_state"][0],
                                                probe_pub["action"][0], law, 0)
            y_right, _, _ = ga.restore_and_step(right, probe_priv["integration_state"][0],
                                                 probe_pub["action"][0], law, 0)
            error = max(error, float(np.max(np.abs(y_left - y_right))))
        finally:
            left.close(); right.close()
    return error


def load_registered(source: Path, schema):
    split_root = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/data"
    data_manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    evaluator_root = source / "results/rwt/clsm_evaluator/comp/HalfCheetah-v5"
    evaluator_manifest = json.loads((evaluator_root / "manifest.json").read_text(encoding="utf-8"))

    def load_singleton(split):
        item = data_manifest["splits"][split]
        public_path = source / item["public_file"]
        if sha256(public_path) != item["public_sha256"]:
            raise ValueError(f"registered {split} public data hash mismatch")
        private_path = source / evaluator_manifest[split]["evaluator_file"]
        if sha256(private_path) != evaluator_manifest[split]["evaluator_sha256"]:
            raise ValueError(f"registered {split} evaluator labels hash mismatch")
        with np.load(public_path, allow_pickle=False) as z:
            public = {k: z[k].copy() for k in ga.PUBLIC_KEYS}
        with np.load(private_path, allow_pickle=False) as z:
            hidden = {k: z[k].copy() for k in ("code", "motor_1", "motor_2", "damping")}
        if len(public["world_id"]) != len(hidden["code"]):
            raise ValueError(f"registered {split} public/evaluator row mismatch")
        meta = {"public_path": str(public_path), "private_path": str(private_path),
                "public_sha256": item["public_sha256"],
                "private_sha256": evaluator_manifest[split]["evaluator_sha256"],
                "world_count": int(len(np.unique(public["world_id"]))),
                "transitions": int(len(public["world_id"]))}
        return public, hidden, meta

    train_pub, train_hid, train_meta = load_singleton("train")
    val_pub, val_hid, val_meta = load_singleton("validation")
    # Load exactly the Gate-A diagnostic mixed train/validation split.
    mixed_train, mixed_train_hid = load_public_and_oracle(GATE_ROOT, "mixed_train")
    mixed_val, mixed_val_hid = load_public_and_oracle(GATE_ROOT, "mixed_validation")
    nominal_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/reference_nominal.npz"
    ref_path = source / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_reference_r2.pt"
    if not nominal_path.is_file() or not ref_path.is_file():
        raise FileNotFoundError("registered nominal reference artifacts are missing")
    with np.load(nominal_path, allow_pickle=False) as z:
        nominal_all = {k: z[k].copy() for k in ("obs", "action", "next_obs")}
    pick = np.random.default_rng(41_111).choice(len(nominal_all["obs"]), 4000, replace=False)
    nominal = {k: v[pick] for k, v in nominal_all.items()}
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    return ((train_pub, train_hid, val_pub, val_hid),
            (mixed_train, mixed_train_hid, mixed_val, mixed_val_hid),
            nominal, ref, {"singleton_train": train_meta, "singleton_validation": val_meta,
                           "mixed_train_public_sha256": sha256(GATE_ROOT / "public/mixed_train.npz"),
                           "mixed_train_oracle_sha256": sha256(GATE_ROOT / "evaluator_only/mixed_train_oracle.npz"),
                           "mixed_validation_public_sha256": sha256(GATE_ROOT / "public/mixed_validation.npz"),
                           "mixed_validation_oracle_sha256": sha256(GATE_ROOT / "evaluator_only/mixed_validation_oracle.npz"),
                           "nominal_reference_sha256": sha256(nominal_path),
                           "nominal_checkpoint_sha256": sha256(ref_path)})


def make_nominal_probe(schema, root: Path):
    path = root / "probe_bank"
    path.mkdir(parents=True, exist_ok=True)
    pub_path, priv_path = path / "public.npz", path / "integration_state_evaluator_only.npz"
    if not pub_path.exists() or not priv_path.exists():
        pub, priv = ga.collect_probe_states(ENV, schema, 128, 45_101)
        np.savez_compressed(pub_path, **pub)
        np.savez_compressed(priv_path, **priv)
    with np.load(pub_path, allow_pickle=False) as z:
        pub = {k: z[k].copy() for k in z.files}
    with np.load(priv_path, allow_pickle=False) as z:
        priv = {k: z[k].copy() for k in z.files}
    return pub, priv, {"public_sha256": sha256(pub_path), "evaluator_sha256": sha256(priv_path),
                       "count": len(pub["obs"]), "seed": 45_101}


def generate_fresh_eval(schema, root: Path, probe_pub: dict, probe_priv: dict):
    """Create the fresh 40-world/code trajectory and fixed-state test bank."""
    root.mkdir(parents=True, exist_ok=True)
    trajectory_root = root / "fresh_trajectory"
    probe_root = root / "fresh_matched_probes"
    traj_manifest = trajectory_root / "split_manifest.json"
    probe_manifest = probe_root / "split_manifest.json"
    if not traj_manifest.is_file():
        pub, priv, meta = ga.collect_world_set(ENV, schema, MIXED_CODES, 40, 50, 45_102, 900_000)
        manifest = ga.save_split(trajectory_root, "test", pub, priv, meta)
        ga.write_json(traj_manifest, {"split": manifest, "seed": 45_102,
                                      "role": "r1_fresh_confirmatory_evaluator"})
    if not probe_manifest.is_file():
        pub = {k: [] for k in ("obs", "action", "next_obs", "world_id", "world_step")}
        hidden = {k: [] for k in ("code", "motor_1", "motor_2", "damping", "contact_count")}
        worlds = []
        rng = np.random.default_rng(45_103)
        world_no = 0
        for code in MIXED_CODES:
            for rep in range(40):
                wid = 1_000_000 + world_no
                world_no += 1
                m1, m2, damping = ga.physical_draw(code, rng)
                law = HiddenLaw(code, m1, m2, damping, False)
                env = HiddenPhysicsWorld(ENV, [law] * len(probe_pub["obs"]), schema)
                try:
                    env.reset(seed=45_103 + wid)
                    for index, (obs, action, state) in enumerate(zip(
                            probe_pub["obs"], probe_pub["action"], probe_priv["integration_state"])):
                        nxt, contact, restored = ga.restore_and_step(env, state, action, law, index)
                        if not np.allclose(restored, obs, atol=1e-10, rtol=0):
                            raise RuntimeError("fresh R1 probe state restore mismatch")
                        pub["obs"].append(obs.copy()); pub["action"].append(action.copy())
                        pub["next_obs"].append(nxt.copy()); pub["world_id"].append(wid)
                        pub["world_step"].append(index)
                        hidden["code"].append(code); hidden["motor_1"].append(m1)
                        hidden["motor_2"].append(m2); hidden["damping"].append(damping)
                        hidden["contact_count"].append(contact)
                finally:
                    env.close()
                worlds.append({"world_id": wid, "code": code, "motor_1": m1,
                               "motor_2": m2, "damping": damping})
        pub = {k: np.asarray(v) for k, v in pub.items()}
        hidden = {k: np.asarray(v) for k, v in hidden.items()}
        meta = {"worlds": worlds, "world_count": len(worlds), "transitions": len(pub["world_id"])}
        probe_root.mkdir(parents=True, exist_ok=True)
        record = ga.save_split(probe_root, "test", pub, hidden, meta)
        replay_error = check_deterministic_replay(schema, probe_pub, probe_priv)
        record["deterministic_replay_max_abs_error"] = replay_error
        ga.write_json(probe_manifest, {"split": record, "seed": 45_103,
                                       "role": "r1_fresh_confirmatory_evaluator",
                                       "probe_bank_sha256": sha256(root / "probe_bank/public.npz")})
    manifest = {"trajectory": json.loads(traj_manifest.read_text(encoding="utf-8")),
                "matched_probes": json.loads(probe_manifest.read_text(encoding="utf-8"))}
    return manifest


def load_f0(schema, ref_payload: dict, config: dict):
    f0 = ExecutableLaw(schema, config["network"]["hidden_dim"],
                       config["emission"]["variance_floor"], config["emission"]["variance_ceiling"])
    f0.load_state_dict(ref_payload["law"], strict=True)
    f0.eval()
    for p in f0.parameters():
        p.requires_grad_(False)
    return f0


def select_residual_width(schema, target_count: int, low: int, high: int) -> tuple[int, dict]:
    hdim = schema.observation_dim + schema.action_dim + len(schema.controlled_dof_indices)
    cdim = 3 + 3 * len(schema.controlled_dof_indices)
    inp = hdim + cdim
    out = schema.observation_dim
    options = []
    for width in range(low, high + 1):
        count = (inp + 1) * width + (width + 1) * width + (width + 1) * out + out
        options.append((abs(count - target_count), width, count))
    _, width, count = min(options)
    return width, {"width": width, "candidate_trainable_parameters": count,
                   "matched_oracle_mlp_parameters": int(target_count),
                   "absolute_difference": int(abs(count - target_count)),
                   "relative_difference": float(abs(count - target_count) / target_count)}


def _normalized_target(nxt, obs, dm, ds):
    return (nxt - obs - dm) / ds


def _nll(target, mean, var):
    var = var.clamp(1e-8, 1e2)
    return .5 * ((target - mean).square() / var + var.log() + math.log(2 * math.pi)).mean()


def fit_r3(schema, f0, partition, ref_payload, train_pub, train_hid,
           val_pub, val_hid, nominal, seed: int, out_path: Path,
           epochs=100, patience=12, batch_size=256, width=224):
    torch.manual_seed(seed)
    dm = ref_payload["law"]["delta_mean"].to(torch.float64)
    ds = ref_payload["law"]["delta_scale"].to(torch.float64)
    model = InterventionResolvedLaw(schema, copy.deepcopy(f0), partition, width, dm, ds).double()
    to, ta, tn, tc, tx = tensor_data(train_pub, train_hid)
    no = torch.as_tensor(nominal["obs"], dtype=torch.float64)
    na = torch.as_tensor(nominal["action"], dtype=torch.float64)
    nnxt = torch.as_tensor(nominal["next_obs"], dtype=torch.float64)
    nc = torch.zeros(len(no), dtype=torch.long)
    nx = torch.zeros((len(no), 3), dtype=torch.float64)
    obs = torch.cat((to, no)); action = torch.cat((ta, na))
    nxt = torch.cat((tn, nnxt)); code = torch.cat((tc, nc)); xi = torch.cat((tx, nx))
    target = _normalized_target(nxt, obs, dm, ds)
    model.fit_channel_normalization(obs, action, code, xi)
    vo, va, vn, vc, vxi = tensor_data(val_pub, val_hid)
    vtarget = _normalized_target(vn, vo, dm, ds)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 971)
    best = math.inf; best_epoch = -1; best_state = None; stale = 0; history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train(); order = rng.permutation(len(obs)); losses = []
        for begin in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[begin:begin + batch_size], dtype=torch.long)
            mean, variance = model(obs[ids], action[ids], code[ids], xi[ids])
            loss = _nll(target[ids], mean, variance)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vm, vv = model(vo, va, vc, vxi)
            vloss = float(_nll(vtarget, vm, vv))
        history.append({"epoch": epoch + 1, "train_nll": float(np.mean(losses)), "validation_nll": vloss})
        if vloss < best - 1e-6:
            best = vloss; best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("R3 did not produce a finite validation checkpoint")
    model.load_state_dict(best_state); model.eval()
    payload = {"role": "oracle_diagnostic", "diagnostic_id": "CLSM-R3-R1",
               "architecture_revision": 3, "architecture_id": model.architecture_id,
               "schema_env": schema.env_id, "seed": seed, "training_regime": "",
               "best_epoch": best_epoch, "best_validation_nll": best,
               "model_state_dict": model.state_dict(), "metadata": model.checkpoint_metadata(),
               "source_fingerprint": source_fingerprint(), "source_commit": ga._git("rev-parse", "HEAD"),
               "reference_nominal_sha256": None}
    out_path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, out_path)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_validation_nll": best,
                   "epochs_run": len(history), "runtime_seconds": time.perf_counter() - start,
                   "checkpoint": str(out_path), "checkpoint_sha256": sha256(out_path),
                   "trainable_parameters": model.trainable_parameter_count(),
                   "training_history": history}


def load_r3(path: Path, schema, f0, partition, width, ref_payload):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    from rwt.clsm.dynamics_r3 import validate_r3_checkpoint
    validate_r3_checkpoint(payload)
    dm = ref_payload["law"]["delta_mean"].to(torch.float64)
    ds = ref_payload["law"]["delta_scale"].to(torch.float64)
    model = InterventionResolvedLaw(schema, copy.deepcopy(f0), partition, width, dm, ds).double()
    model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
    return model


def score_model(model, family: str, seed: int, public: dict, hidden: dict,
                dm: np.ndarray, ds: np.ndarray, width_info: dict, ref_payload: dict,
                config: dict, schema, f0, partition):
    if family.startswith("R3"):
        pass
    else:
        return ga.score_rows(model, public, hidden, dm, ds, family, seed, "test")[0]
    code = hidden["code"].astype(np.int64)
    xi = ga.strength_xi(code, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    obs = public["obs"]; action = public["action"]
    mean, variance = [], []
    with torch.no_grad():
        for start in range(0, len(obs), 512):
            stop = min(len(obs), start + 512)
            m, v = model(torch.as_tensor(obs[start:stop], dtype=torch.float64),
                         torch.as_tensor(action[start:stop], dtype=torch.float64),
                         torch.as_tensor(code[start:stop], dtype=torch.long),
                         torch.as_tensor(xi[start:stop], dtype=torch.float64))
            mean.append(m.numpy()); variance.append(v.numpy())
    mean = np.concatenate(mean); variance = np.concatenate(variance)
    target = (public["next_obs"] - obs - dm) / ds
    mse = ((target - mean) ** 2).mean(1)
    nll = .5 * (((target - mean) ** 2) / np.maximum(variance, 1e-8) +
                np.log(np.maximum(variance, 1e-8)) + math.log(2 * math.pi)).mean(1)
    rows = []
    for wid in np.unique(public["world_id"]):
        mask = public["world_id"] == wid
        rows.append({"model": family, "seed": seed, "split": "test",
                     "code": int(code[mask][0]), "world_id": int(wid),
                     "n_transitions": int(mask.sum()), "nll_per_dim": float(nll[mask].mean()),
                     "normalized_mse": float(mse[mask].mean()),
                     "normalized_signature_error": float(np.sqrt(mse[mask].mean()))})
    return rows


def bootstrap_ci(values: np.ndarray, seed: int, one_sided=False):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    q = [0.0, .95] if one_sided else [.025, .975]
    return [float(x) for x in np.quantile(draws, q)]


def average_seed_world(rows, family, code, metric, domain):
    relevant = [r for r in rows if r["model"] == family and r["code"] == code and r["domain"] == domain]
    by_world = {}
    for r in relevant:
        by_world.setdefault(r["world_id"], []).append(float(r[metric]))
    return {wid: float(np.mean(vals)) for wid, vals in by_world.items()}


def paired_summary(rows, left, right, code, metric, domain, seed, relative=False):
    l = average_seed_world(rows, left, code, metric, domain)
    r = average_seed_world(rows, right, code, metric, domain)
    ids = sorted(set(l) & set(r))
    diff = np.asarray([l[i] - r[i] for i in ids])
    base = float(np.mean([r[i] for i in ids]))
    seed_sensitivity = {}
    for train_seed in SEEDS:
        ls = {r["world_id"]: float(r[metric]) for r in rows if r["model"] == left and r["code"] == code and
              r["domain"] == domain and r["seed"] == train_seed}
        rs = {r["world_id"]: float(r[metric]) for r in rows if r["model"] == right and r["code"] == code and
              r["domain"] == domain and r["seed"] == train_seed}
        common = sorted(set(ls) & set(rs))
        seed_sensitivity[str(train_seed)] = float(np.mean([ls[i] - rs[i] for i in common])) if common else None
    return {"left": left, "right": right, "code": int(code), "metric": metric,
            "domain": domain, "n_worlds": len(ids), "mean_difference": float(diff.mean()),
            "paired_world_ci95": bootstrap_ci(diff, seed),
            "paired_world_upper_one_sided_95": bootstrap_ci(diff, seed, True)[1],
            "right_mean": base, "noninferiority_margin_5pct": .05 * base,
            "training_seed_mean_differences": seed_sensitivity}


def aggregate_and_decide(rows, worlds_manifest):
    comparisons = []
    for regime, r3, mlp in (("mixed_seen", "R3_mixed_seen", "oracle_conditioned_MLP_mixed_seen"),
                            ("singleton", "R3_singleton", "oracle_conditioned_MLP_singleton")):
        for domain in ("trajectory", "matched_probe"):
            for code in MIXED_CODES:
                for metric in ("normalized_mse", "normalized_signature_error"):
                    comparisons.append(paired_summary(rows, r3, mlp, code, metric, domain,
                                                      800_000 + code * 7 + len(metric) + len(domain)))
    # Singleton R3 must strictly improve macro normalized MSE on matched probes.
    macro_diffs = {}
    for metric in ("normalized_mse", "normalized_signature_error"):
        per_code = []
        for code in MIXED_CODES:
            l = average_seed_world(rows, "R3_singleton", code, metric, "matched_probe")
            r = average_seed_world(rows, "oracle_conditioned_MLP_singleton", code, metric, "matched_probe")
            ids = sorted(set(l) & set(r)); per_code.append(np.asarray([l[i] - r[i] for i in ids]))
        # Stratified paired bootstrap preserves the four registered code strata.
        rng = np.random.default_rng(808_000 + len(metric))
        samples = np.empty(BOOTSTRAP_DRAWS)
        for b in range(BOOTSTRAP_DRAWS):
            code_means = []
            for values in per_code:
                code_means.append(float(values[rng.integers(0, len(values), len(values))].mean()))
            samples[b] = np.mean(code_means)
        macro_diffs[metric] = {"mean_difference": float(np.mean([x.mean() for x in per_code])),
                               "paired_stratified_world_ci95": [float(x) for x in np.quantile(samples, [.025, .975])]}
    # Trajectory meaningful-regression checks against both singleton references.
    regression = []
    for right in ("R2_singleton", "oracle_conditioned_MLP_singleton"):
        per_code = []
        base_per_code = []
        for code in MIXED_CODES:
            l = average_seed_world(rows, "R3_singleton", code, "normalized_mse", "trajectory")
            r = average_seed_world(rows, right, code, "normalized_mse", "trajectory")
            ids = sorted(set(l) & set(r)); per_code.append(np.asarray([l[i] - r[i] for i in ids]))
            base_per_code.append(np.asarray([r[i] for i in ids]))
        rng = np.random.default_rng(809_000 + len(right)); samples = np.empty(BOOTSTRAP_DRAWS)
        for b in range(BOOTSTRAP_DRAWS):
            diffs = []; bases = []
            for d, base in zip(per_code, base_per_code):
                idx = rng.integers(0, len(d), len(d)); diffs.append(float(d[idx].mean())); bases.append(float(base[idx].mean()))
            samples[b] = np.mean(diffs) - .05 * np.mean(bases)
        regression.append({"against": right, "mean_difference": float(np.mean([x.mean() for x in per_code])),
                           "macro_baseline_mean": float(np.mean([x.mean() for x in base_per_code])),
                           "meaningful_deterioration_margin": float(.05 * np.mean([x.mean() for x in base_per_code])),
                           "difference_minus_margin_ci95": [float(x) for x in np.quantile(samples, [.025, .975])]})
    mixed_ok = True; singleton_ni_ok = True
    for c in comparisons:
        if c["domain"] != "matched_probe":
            continue
        ni = c["paired_world_upper_one_sided_95"] < c["noninferiority_margin_5pct"]
        if c["left"] == "R3_mixed_seen": mixed_ok &= ni
        else: singleton_ni_ok &= ni
    macro_improvement = macro_diffs["normalized_mse"]["paired_stratified_world_ci95"][1] < 0
    no_trajectory_regression = all(x["difference_minus_margin_ci95"][0] <= 0 for x in regression)
    # The report also checks numerical/replay and variance diagnostics after scoring.
    passed = mixed_ok and singleton_ni_ok and macro_improvement and no_trajectory_regression
    per_composition = {}
    for family in sorted({r["model"] for r in rows}):
        per_composition[family] = {}
        for domain in ("trajectory", "matched_probe"):
            for code in MIXED_CODES:
                metrics = {}
                for metric in ("nll_per_dim", "normalized_mse", "normalized_signature_error"):
                    values = average_seed_world(rows, family, code, metric, domain)
                    ordered = np.asarray([values[wid] for wid in sorted(values)])
                    metrics[metric] = {"mean": float(ordered.mean()),
                                       "world_bootstrap_ci95": bootstrap_ci(ordered, 810_000 + code + len(family) + len(domain))}
                per_composition[family][f"{domain}/code_{code}"] = {"n_worlds": len(ordered), **metrics}
    return {"comparisons": comparisons, "per_composition": per_composition, "singleton_macro": macro_diffs,
            "trajectory_regression_checks": regression,
            "criteria": {"mixed_seen_all_code_mse_and_signature_noninferiority": bool(mixed_ok),
                         "singleton_all_code_mse_and_signature_noninferiority": bool(singleton_ni_ok),
                         "singleton_macro_mse_ci_upper_below_zero": bool(macro_improvement),
                         "no_meaningful_trajectory_regression_over_5pct": bool(no_trajectory_regression)},
            "status": "PASS" if passed else "FAIL"}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    config_r3 = load_r3_config(ROOT / "experiments/rwt/clsm/configs/clsm_r3_final.yaml")
    schema = mujoco_nominal_schema(ENV)
    source_commit = ga._git("rev-parse", "HEAD")
    reference_config = load_config(ROOT / "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml")
    (singleton, mixed, nominal, ref_payload, input_hashes) = load_registered(SOURCE, schema)
    f0 = load_f0(schema, ref_payload, reference_config)
    groups = ga.fixed_motor_groups(schema)
    partition = torch.zeros((schema.action_dim, 2), dtype=torch.bool)
    partition[np.asarray(groups[0]), 0] = True; partition[np.asarray(groups[1]), 1] = True
    target_mlp = GATE_ROOT / "checkpoints/oracle_conditioned_unstructured_MLP/seed_4100.pt"
    if not target_mlp.is_file():
        raise FileNotFoundError("R0 mixed/singleton MLP reference checkpoint is missing")
    mlp_payload = torch.load(target_mlp, map_location="cpu", weights_only=True)
    target_params = sum(p.numel() for p in ga.load_diagnostic_model(
        target_mlp, "oracle_conditioned_unstructured_MLP", schema, reference_config, ref_payload).model.parameters())
    width, width_report = select_residual_width(schema, target_params,
                                               *config_r3["decoder"]["residual_width_search"])
    probe_pub, probe_priv, probe_info = make_nominal_probe(schema, OUT)
    eval_manifest = generate_fresh_eval(schema, OUT, probe_pub, probe_priv)
    # Freeze the complete candidate, data hashes, and decision rules before fitting.
    freeze = {"gate": "R1", "candidate_id": config_r3["candidate_id"],
              "source_commit_at_freeze": source_commit, "source_fingerprint": source_fingerprint(),
              "config": config_r3, "config_sha256": sha256(ROOT / "experiments/rwt/clsm/configs/clsm_r3_final.yaml"),
              "reference_config_sha256": sha256(ROOT / "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml"),
              "input_hashes": input_hashes, "probe_bank": probe_info,
              "fresh_eval_manifest": eval_manifest, "residual_width_selection": width_report,
              "criteria": {"mixed_seen_each_code_mse_and_signature_upper_95_one_sided_below_5pct_mlp": True,
                           "singleton_each_code_mse_and_signature_upper_95_one_sided_below_5pct_mlp": True,
                           "singleton_macro_probe_mse_95_ci_upper_below_zero": True,
                           "trajectory_regression_material_if_lower_ci_of_r3_minus_baseline_exceeds_5pct": True},
              "training_seeds": list(SEEDS), "checkpoint_rule": "regime-specific validation standardized Gaussian NLL",
              "oracle_role": "diagnostic only; never deployable learner feature",
              "old_confirmatory_opened": False}
    freeze_path = OUT / "preregistration.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous != freeze:
            raise RuntimeError("existing frozen R1 preregistration differs; refusing to overwrite")
    else:
        write_json(freeze_path, freeze)

    all_ckpt = {}
    training_runs = []
    for regime, data in (("singleton", singleton), ("mixed_seen", mixed)):
        train_pub, train_hid, val_pub, val_hid = data
        for seed in SEEDS:
            ckpt = OUT / "checkpoints" / regime / f"seed_{seed}.pt"
            if not ckpt.is_file():
                model, fit = fit_r3(schema, f0, partition, ref_payload,
                                    train_pub, train_hid, val_pub, val_hid, nominal,
                                    seed, ckpt, width=width)
                payload = torch.load(ckpt, map_location="cpu", weights_only=True)
                payload["training_regime"] = regime
                payload["reference_nominal_sha256"] = input_hashes["nominal_checkpoint_sha256"]
                payload["training_data_sha256"] = (input_hashes["singleton_train"]["public_sha256"]
                    if regime == "singleton" else input_hashes["mixed_train_public_sha256"])
                torch.save(payload, ckpt)
                fit["checkpoint_sha256"] = sha256(ckpt)
            else:
                checkpoint_payload = torch.load(ckpt, map_location="cpu", weights_only=True)
                fit = {"seed": seed, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
                       "best_epoch": checkpoint_payload.get("best_epoch"),
                       "best_validation_nll": checkpoint_payload.get("best_validation_nll"),
                       "trainable_parameters": checkpoint_payload.get("metadata", {}).get("trainable_parameter_count"),
                       "runtime_seconds": None, "reused_existing_checkpoint": True}
            training_runs.append({"regime": regime, **fit})
            all_ckpt[(regime, seed)] = ckpt
    write_json(OUT / "training.json", {"training_runs": training_runs, "width_report": width_report})
    # A frozen fit now exists, so the registered prior confirmatory reserve may be opened.
    old_confirm_pub, old_confirm_hid = load_public_and_oracle(GATE_ROOT, "composition_confirmatory_reserved")
    # Evaluator-only labels for fresh worlds are opened only here, after all R3 fits are frozen.
    trajectory_pub, trajectory_hid = load_r1_split(OUT / "fresh_trajectory")
    probe_test_pub, probe_test_hid = load_r1_split(OUT / "fresh_matched_probes")
    old_confirm = (old_confirm_pub, old_confirm_hid)
    eval_domains = (("trajectory", trajectory_pub, trajectory_hid),
                    ("matched_probe", probe_test_pub, probe_test_hid),
                    ("old_confirmatory_supplementary", old_confirm_pub, old_confirm_hid))
    diagnostic_root = SOURCE / "results/rwt/clsm_r2_gates/gate_a_diagnostic"
    baseline_adapters = {}
    for regime, family, model_type in (
        ("singleton", "R2_singleton", "oracle_G_z_xi_singletons"),
        ("singleton", "oracle_conditioned_MLP_singleton", "oracle_conditioned_unstructured_MLP"),
        ("mixed_seen", "R2_mixed_seen", "oracle_G_z_xi_mixed_seen"),
        ("mixed_seen", "oracle_conditioned_MLP_mixed_seen", "oracle_conditioned_unstructured_MLP")):
        for seed in SEEDS:
            if family.endswith("MLP_mixed_seen"):
                path = OUT.parent / "r0/checkpoints/mixed_seen_oracle_conditioned_MLP" / f"seed_{seed}.pt"
            else:
                path = diagnostic_root / "checkpoints" / model_type / f"seed_{seed}.pt"
            baseline_adapters[(family, seed)] = ga.load_diagnostic_model(path, model_type, schema,
                                                                          reference_config, ref_payload)
    rows = []
    dm = ref_payload["law"]["delta_mean"].numpy(); ds = ref_payload["law"]["delta_scale"].numpy()
    evaluation_wall_start = time.perf_counter()
    evaluation_cpu_start = time.process_time()
    for domain, public, hidden in eval_domains:
        for regime, r3_family, r2_family, mlp_family in (
            ("singleton", "R3_singleton", "R2_singleton", "oracle_conditioned_MLP_singleton"),
            ("mixed_seen", "R3_mixed_seen", "R2_mixed_seen", "oracle_conditioned_MLP_mixed_seen")):
            for seed in SEEDS:
                wall_start = time.perf_counter(); cpu_start = time.process_time()
                r3 = load_r3(all_ckpt[(regime, seed)], schema, f0, partition, width, ref_payload)
                new_rows = score_r3_rows(r3, r3_family, seed, public, hidden, dm, ds)
                attach_resource_metrics(new_rows, wall_start, cpu_start)
                for row in new_rows:
                    row["domain"] = domain; rows.append(row)
                for family in (r2_family, mlp_family):
                    wall_start = time.perf_counter(); cpu_start = time.process_time()
                    adapter = baseline_adapters[(family, seed)]
                    baseline_rows = ga.score_rows(adapter, public, hidden, dm, ds, adapter.model_type,
                                                  seed, domain)[0]
                    codes = hidden["code"].astype(np.int64)
                    xis = ga.strength_xi(codes, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
                    predicted, predicted_var = ga.predict_numpy(adapter, public["obs"], public["action"], codes, xis)
                    target = (public["next_obs"] - public["obs"] - dm) / ds
                    standardized_sq = (target - predicted) ** 2 / np.maximum(predicted_var, 1e-8)
                    covered = np.abs(target - predicted) <= 1.6448536269514722 * np.sqrt(np.maximum(predicted_var, 1e-8))
                    for row in baseline_rows:
                        mask = public["world_id"] == row["world_id"]
                        row["standardized_residual_variance_ratio"] = float(standardized_sq[mask].mean())
                        row["marginal_90pct_coverage"] = float(covered[mask].mean())
                        row["model"] = family; row["domain"] = domain; rows.append(row)
                    attach_resource_metrics(rows[-len(baseline_rows):], wall_start, cpu_start)
    import csv
    score_path = OUT / "per_world_seed_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    # Contract criteria are applied to fresh trajectory and matched-probe worlds only.
    primary_rows = [r for r in rows if r["domain"] in ("trajectory", "matched_probe")]
    result = aggregate_and_decide(primary_rows, eval_manifest)
    result["old_confirmatory_supplementary"] = {"opened_after_freeze": True,
        "world_count": int(len(np.unique(old_confirm_pub["world_id"]))),
        "per_world_scores_included": True,
        "does_not_change_registered_decision": True}
    result["frozen_width_report"] = width_report
    result["runtime_seconds_total"] = (float(sum(r["runtime_seconds"] for r in training_runs))
                                       if all(r.get("runtime_seconds") is not None for r in training_runs)
                                       else None)
    calibration = {}
    for code in MIXED_CODES:
        for domain in ("trajectory", "matched_probe"):
            cells = [r for r in primary_rows if r["model"] == "R3_mixed_seen" and
                     r["code"] == code and r["domain"] == domain]
            calibration[f"{domain}/code_{code}"] = {
                "mean_seed_and_world_standardized_residual_variance_ratio": float(np.mean([r["standardized_residual_variance_ratio"] for r in cells])),
                "mean_seed_and_world_marginal_90pct_coverage": float(np.mean([r["marginal_90pct_coverage"] for r in cells])),
                "worlds": int(len({r["world_id"] for r in cells}))}
    result["mixed_seen_variance_calibration_descriptive"] = calibration
    result["training_runtime_note"] = "Per-seed runtime was not retained after checkpoint resume; validation NLL and best epoch are available in checkpoint payloads. No runtime claim is made."
    result["evaluation_resource_use"] = {
        "wall_seconds": float(time.perf_counter() - evaluation_wall_start),
        "process_cpu_seconds": float(time.process_time() - evaluation_cpu_start),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "per_world_fields": ["amortized_scoring_wall_seconds_per_world",
                             "amortized_scoring_cpu_seconds_per_world"],
        "rss_semantics": "process peak working set observed during evaluation; stored on every per-world score row",
        "per_world_time_semantics": "model-call time divided by number of independent worlds in that score batch"}
    result["status"] = result["status"] if result["status"] == "FAIL" else "PASS_PENDING_DIAGNOSTICS"
    # Independently replay deterministic samples and assess standardized residual variance.
    result["deterministic_replay_max_abs_error"] = check_deterministic_replay(schema, probe_pub, probe_priv)
    result["source_commit"] = source_commit
    result["per_world_seed_scores"] = str(score_path)
    result["training_runs"] = training_runs
    write_json(OUT / "r1_decision.json", result)
    write_report(result, training_runs, freeze, OUT / "decision.md")
    return result


def score_r3_rows(model, family, seed, public, hidden, dm, ds):
    code = hidden["code"].astype(np.int64)
    xi = ga.strength_xi(code, hidden["motor_1"], hidden["motor_2"], hidden["damping"])
    obs = public["obs"]; action = public["action"]
    means = []; variances = []
    with torch.no_grad():
        for start in range(0, len(obs), 512):
            end = min(len(obs), start + 512)
            m, v = model(torch.as_tensor(obs[start:end], dtype=torch.float64),
                         torch.as_tensor(action[start:end], dtype=torch.float64),
                         torch.as_tensor(code[start:end], dtype=torch.long),
                         torch.as_tensor(xi[start:end], dtype=torch.float64))
            means.append(m.numpy()); variances.append(v.numpy())
    mean = np.concatenate(means); variance = np.concatenate(variances)
    target = (public["next_obs"] - obs - dm) / ds
    mse = ((target - mean) ** 2).mean(1)
    variance_ratio = ((target - mean) ** 2 / np.maximum(variance, 1e-8)).mean(1)
    coverage90 = (np.abs(target - mean) <= 1.6448536269514722 * np.sqrt(np.maximum(variance, 1e-8))).mean(1)
    nll = .5 * (((target - mean) ** 2) / np.maximum(variance, 1e-8) +
                np.log(np.maximum(variance, 1e-8)) + math.log(2 * math.pi)).mean(1)
    rows = []
    for wid in np.unique(public["world_id"]):
        mask = public["world_id"] == wid
        rows.append({"model": family, "seed": seed, "split": "test", "domain": "",
                     "code": int(code[mask][0]), "world_id": int(wid), "n_transitions": int(mask.sum()),
                     "nll_per_dim": float(nll[mask].mean()), "normalized_mse": float(mse[mask].mean()),
                     "normalized_signature_error": float(np.sqrt(mse[mask].mean())),
                     "standardized_residual_variance_ratio": float(variance_ratio[mask].mean()),
                     "marginal_90pct_coverage": float(coverage90[mask].mean())})
    return rows


def write_report(result, training_runs, freeze, path: Path):
    lines = ["# CLSM-R3 R1 decisive decoder gate", "",
             f"**Decision: {result['status']}**", "",
             "R1 scores use four new mixed compositions, independent physical worlds, and a fixed-state matched probe bank. The previously scored Gate-A diagnostic remains development evidence. Oracle labels were evaluator-only and were opened after the checkpoint freeze.", "",
             "## Registered decision criteria", "",
             "| Criterion | Result |", "|---|---:|"]
    for key, value in result["criteria"].items():
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")
    lines += ["", "## Training and capacity", "",
              f"Residual width: {freeze['residual_width_selection']['width']}; R3 trainable parameters: {freeze['residual_width_selection']['candidate_trainable_parameters']}; oracle MLP target: {freeze['residual_width_selection']['matched_oracle_mlp_parameters']}.",
              f"Training runs: {len(training_runs)}; seeds: {', '.join(map(str, SEEDS))}. Checkpoint selection used only regime-specific validation standardized Gaussian NLL.",
              "", "## Paired world intervals", "",
              "The table reports paired world-bootstrap 95% CIs, and the registered one-sided upper bounds are in `r1_decision.json`.",
              "", "| Domain | Training regime | Code | Metric | R3 − matched MLP | 95% CI | 5% margin | One-sided upper |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in result["comparisons"]:
        lines.append(f"| {c['domain']} | {c['left']} | {c['code']} | {c['metric']} | {c['mean_difference']:.6g} | [{c['paired_world_ci95'][0]:.6g}, {c['paired_world_ci95'][1]:.6g}] | {c['noninferiority_margin_5pct']:.6g} | {c['paired_world_upper_one_sided_95']:.6g} |")
    lines += ["", "## Per composition predictive metrics", "",
              "Means and 95% world-bootstrap CIs average training seeds within each independent world before bootstrapping. Seed-specific contrasts are retained in the decision JSON.",
              "", "| Model | Domain | Code | NLL / dim (95% CI) | Normalized MSE (95% CI) | Signature RMSE (95% CI) | Worlds |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, groups in result["per_composition"].items():
        for key, metrics in groups.items():
            domain, code_txt = key.split("/code_")
            nll = metrics["nll_per_dim"]; mse = metrics["normalized_mse"]; sig = metrics["normalized_signature_error"]
            lines.append(f"| {model} | {domain} | {code_txt} | {nll['mean']:.5g} [{nll['world_bootstrap_ci95'][0]:.5g}, {nll['world_bootstrap_ci95'][1]:.5g}] | {mse['mean']:.5g} [{mse['world_bootstrap_ci95'][0]:.5g}, {mse['world_bootstrap_ci95'][1]:.5g}] | {sig['mean']:.5g} [{sig['world_bootstrap_ci95'][0]:.5g}, {sig['world_bootstrap_ci95'][1]:.5g}] | {metrics['n_worlds']} |")
    lines += ["", "## Singleton macro and trajectory check", "",
              "```json", json.dumps({"singleton_macro": result["singleton_macro"],
                                     "trajectory_regression_checks": result["trajectory_regression_checks"]}, indent=2), "```",
              "", "## Mixed-seen variance calibration", "",
              "The ratios and marginal coverage below are descriptive; no post-hoc threshold was used to alter the registered decision.",
              "", "```json", json.dumps(result["mixed_seen_variance_calibration_descriptive"], indent=2), "```",
              "", "## Training checkpoints", "",
              "| Regime | Seed | Best epoch | Best validation standardized NLL | Trainable parameters | Runtime seconds |", "|---|---:|---:|---:|---:|---:|"]
    for run in result["training_runs"]:
        lines.append(f"| {run['regime']} | {run['seed']} | {run.get('best_epoch', 'n/a')} | {run.get('best_validation_nll', 'n/a')} | {run.get('trainable_parameters', 'n/a')} | {run.get('runtime_seconds', 'not retained')} |")
    lines += ["", result["training_runtime_note"],
              "", "## CPU scoring and memory", "",
              "```json", json.dumps(result["evaluation_resource_use"], indent=2), "```",
              "", "## Limits", "",
              "This is an oracle-conditioned diagnostic of representation and compositional transfer. It does not establish that an unlabeled learner recovers the mechanism posterior. A failed locked criterion halts the CLSM-R3 family attempt; later inference and generalized-EM gates must not be run after R1 failure.",
              "", f"Fresh dataset and code freeze manifest: `{Path(freeze['fresh_eval_manifest']['trajectory']['split']['public_path']).parent.parent}`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    print(json.dumps(run(), indent=2))
