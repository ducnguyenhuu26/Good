"""Evaluator-only matched MuJoCo snapshots for projected transition-law scores."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.rwt.clsm.audit_physics import _restore_snapshot, _snapshot
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld


CALIBRATION_SEED = 12041


@dataclass
class ProbeBank:
    obs: np.ndarray
    action: np.ndarray
    states: np.ndarray
    warmstarts: np.ndarray


def build_probe_bank(env_id: str, count: int, seed: int) -> ProbeBank:
    """Collect independent nominal public probes and evaluator-only snapshots."""
    if count < 2:
        raise ValueError("at least two independent calibration probes required")
    rows, actions, states, warmstarts = [], [], [], []
    episode = 0
    while len(rows) < count:
        env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * (count + 1))
        try:
            obs, _ = env.reset(seed=seed + episode)
            collector = SmoothActionCollector(env.schema.action_low, env.schema.action_high,
                                              seed + 10000 + episode)
            while len(rows) < count:
                action = collector.act(obs)
                snapshot, warmstart = _snapshot(env.unwrapped.model, env.unwrapped.data)
                rows.append(obs.copy()); actions.append(action.copy())
                states.append(snapshot.copy()); warmstarts.append(warmstart.copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        episode += 1
    return ProbeBank(np.asarray(rows), np.asarray(actions),
                     np.asarray(states), np.asarray(warmstarts))


def save_probe_bank(path: Path, bank: ProbeBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=bank.obs, action=bank.action,
                        states=bank.states, warmstarts=bank.warmstarts)


def load_probe_bank(path: Path) -> ProbeBank:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"obs", "action", "states", "warmstarts"}:
            raise ValueError("invalid evaluator probe bank")
        return ProbeBank(*(arrays[name].copy() for name in ("obs", "action", "states", "warmstarts")))


def fixed_probe_bank(env_id: str, count: int, evaluator_root: Path) -> tuple[ProbeBank, Path]:
    """One predeclared calibration bank shared across methods and world seeds."""
    path = evaluator_root / "probes" / env_id / f"bank_{count}_{CALIBRATION_SEED}.npz"
    if path.exists():
        bank = load_probe_bank(path)
        if len(bank.obs) != count:
            raise ValueError("stored probe-bank size mismatch")
    else:
        bank = build_probe_bank(env_id, count, CALIBRATION_SEED)
        save_probe_bank(path, bank)
    return bank, path


def oracle_signature(env_id: str, bank: ProbeBank, law: HiddenLaw,
                     delta_scale: torch.Tensor) -> torch.Tensor:
    """Counterfactual law on fixed full-state snapshots; no learner access."""
    env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * 2)
    deltas = []
    try:
        env.reset(seed=0)
        model, data = env.unwrapped.model, env.unwrapped.data
        for obs, action, state, warmstart in zip(bank.obs, bank.action, bank.states, bank.warmstarts):
            env._apply(law)
            _restore_snapshot(model, data, (state, warmstart))
            restored = env.unwrapped._get_obs().copy()
            if not np.allclose(restored, obs, atol=1e-5, rtol=0):
                raise RuntimeError("public probe observation changed on state restoration")
            next_obs = env.env.step(action)[0]
            deltas.append((next_obs - obs) / delta_scale.detach().cpu().numpy())
        env._restore()
        return torch.as_tensor(np.asarray(deltas).reshape(-1), dtype=torch.float64)
    finally:
        env.close()
