"""Read-only force-path and prior-support diagnostics on a pinned CLSM law.

The current filter/trainer is not replayed. Model-defining source files must
match the checkpoint's commit before using its law weights here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import mujoco_nominal_schema
from experiments.rwt.clsm.oracle_probes import fixed_probe_bank, oracle_signature
from experiments.rwt.clsm.physics_wrappers import HiddenLaw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(checkpoint: Path, reference: Path, evaluator_root: Path,
          pooled_checkpoint: Path | None = None) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "clsm_approximate_gem":
        raise ValueError("fitted CLSM checkpoint required")
    unchanged = subprocess.run([
        "git", "diff", "--quiet", payload["git_commit"], "HEAD", "--",
        "src/rwt/clsm/mechanics.py", "src/rwt/clsm/dynamics.py",
        "src/rwt/clsm/schemas.py", "src/rwt/clsm/factory.py",
    ], check=False)
    if unchanged.returncode != 0:
        raise RuntimeError("model-defining source differs from pinned checkpoint")
    env_id = "HalfCheetah-v5"
    parts = build_clsm(mujoco_nominal_schema(env_id), payload["config"])
    parts.law.load_state_dict(payload["law"])
    parts.law.eval()
    law = parts.law
    support = law.mechanics.W.detach()
    with np.load(reference) as data:
        obs = torch.as_tensor(data["obs"][:8], dtype=torch.float64)
        action = torch.as_tensor(data["action"][:8], dtype=torch.float64)
    fisher = torch.zeros((3, 3), dtype=torch.float64)
    variance = law.log_variance.exp().detach()
    for observation, control in zip(obs, action):
        def prediction(theta):
            return law(observation[None], control[None], torch.tensor([7]), theta[None])[0][0]
        jacobian = torch.autograd.functional.jacobian(
            prediction, torch.full((3,), -2., dtype=torch.float64))
        fisher += jacobian.T @ torch.diag(variance.reciprocal()) @ jacobian
    bank, bank_path = fixed_probe_bank(env_id, 64, evaluator_root)
    probe_obs = torch.as_tensor(bank.obs, dtype=torch.float64)
    probe_action = torch.as_tensor(bank.action, dtype=torch.float64)
    def signature(code: int) -> torch.Tensor:
        theta = torch.full((len(probe_obs), 3), -2., dtype=torch.float64)
        with torch.no_grad():
            mean = law(probe_obs, probe_action,
                       torch.full((len(probe_obs),), code), theta)[0]
        return ((mean - probe_obs) / law.delta_scale).flatten()
    oracle_1 = oracle_signature(env_id, bank, HiddenLaw(1, .7, .7, 1.5), law.delta_scale)
    oracle_2 = oracle_signature(env_id, bank, HiddenLaw(2, .7, .7, 1.5), law.delta_scale)
    rng = np.random.default_rng(41)
    theta = rng.normal(-2., 1.5, (100000, 3))
    intensity = .12 + np.logaddexp(0, theta)
    gain = np.exp(-intensity[:, 0, None] * support[:, 0].numpy())
    own_gain = gain[:, :3].mean(1)
    damping = np.exp(intensity[:, 2])
    report = {
        "status": "forensic_pinned_law_diagnostic_only",
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_commit": payload["git_commit"],
        "model_defining_source_unchanged": True,
        "probe_sha256": _sha256(bank_path),
        "support_matrix": support.numpy().tolist(),
        "support_cosine": float(support[:, 0] @ support[:, 1]),
        "fisher_eigenvalues_8_reference_transitions": torch.linalg.eigvalsh(fisher).tolist(),
        "fisher_condition_8_reference_transitions": float(torch.linalg.cond(fisher)),
        "learned_signature_distance_code1_vs2_theta_minus2": float(torch.linalg.vector_norm(signature(1) - signature(2))),
        "oracle_signature_distance_code1_vs2_fixed_strength": float(torch.linalg.vector_norm(oracle_1 - oracle_2)),
        "code1_prior_own_group_gain_quantiles_05_50_95": np.quantile(own_gain, [.05, .5, .95]).tolist(),
        "code1_prior_own_group_gain_in_physical_range_fraction": float(np.mean((own_gain >= .5) & (own_gain <= .85))),
        "code1_prior_wrong_group_median_weakening": float(np.median(1 - gain[:, 3:].mean(1))),
        "damping_prior_multiplier_quantiles_05_50_95": np.quantile(damping, [.05, .5, .95]).tolist(),
        "damping_prior_in_physical_range_fraction": float(np.mean((damping >= 1.3) & (damping <= 2.))),
    }
    if pooled_checkpoint is not None:
        pooled = torch.load(pooled_checkpoint, map_location="cpu", weights_only=True)
        report["pooled_reference_hash_equal"] = (
            pooled["data_hashes"]["reference"] == payload["data_hashes"]["reference"])
        report["pooled_delta_scale_max_abs_difference"] = float((
            pooled["model"]["delta_scale"] - payload["law"]["delta_scale"]).abs().max())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--evaluator-root", required=True, type=Path)
    parser.add_argument("--pooled-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.reference, args.evaluator_root,
                   args.pooled_checkpoint)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
