"""Evaluator-only audit for a frozen R2 support pilot; never imported by learners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from scipy.special import ndtri

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import mujoco_nominal_schema
from experiments.rwt.clsm.build_datasets import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    root = args.output_dir / config["track"] / args.env
    run = root / "r2" / str(seed)
    public_root = root / "data"
    evaluator_root = args.output_dir.parent / "clsm_evaluator" / config["track"] / args.env
    public_manifest = json.loads((public_root / "manifest.json").read_text(encoding="utf-8"))
    evaluator_manifest = json.loads((evaluator_root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(public_root / "train.npz", allow_pickle=False) as public:
        obs = public["obs"].copy(); action = public["action"].copy()
    evaluator_path = Path(evaluator_manifest["train"]["evaluator_file"])
    if _sha256(evaluator_path) != evaluator_manifest["train"]["evaluator_sha256"]:
        raise ValueError("evaluator-only training oracle hash mismatch")
    with np.load(evaluator_path, allow_pickle=False) as oracle:
        code = oracle["code"].copy()
        strength = np.column_stack((oracle["motor_1"], oracle["motor_2"], oracle["damping"]))
    if len(code) != len(action) or len(obs) != len(action):
        raise ValueError("public/evaluator training rows do not align")

    payload = torch.load(run / "checkpoint_r2.pt", map_location="cpu", weights_only=True)
    schema = mujoco_nominal_schema(args.env)
    model = build_clsm(schema, config).law
    model.load_state_dict(payload["law"])
    model.eval()
    x = torch.as_tensor(obs, dtype=torch.float64)
    a = torch.as_tensor(action, dtype=torch.float64)
    theta = np.zeros((len(code), 3), dtype=np.float64)
    for dim, (low, high) in enumerate(((.50, .85), (.50, .85), (1.30, 2.00))):
        pct = np.clip((strength[:, dim] - low) / (high - low), 1e-9, 1 - 1e-9)
        theta[:, dim] = ndtri(pct)
    theta_t = torch.as_tensor(theta, dtype=torch.float64)
    code_t = torch.as_tensor(code, dtype=torch.long)
    fisher = np.zeros(3, dtype=np.float64)
    epsilon = .01
    with torch.no_grad():
        for dim, mode in enumerate((1, 2, 4)):
            idx = np.flatnonzero(code == mode)
            if not len(idx):
                continue
            rows = torch.as_tensor(idx, dtype=torch.long)
            plus, minus = theta_t[rows].clone(), theta_t[rows].clone()
            plus[:, dim] += epsilon; minus[:, dim] -= epsilon
            mean_plus, variance = model(x[rows], a[rows], code_t[rows], plus)
            mean_minus, _ = model(x[rows], a[rows], code_t[rows], minus)
            jac = (mean_plus - mean_minus) / (2 * epsilon)
            fisher[dim] = float((jac.square() / variance).sum())
    action_cov = np.cov(action.T)
    action_eigen = np.linalg.eigvalsh(action_cov)
    counts = {str(c): int((code == c).sum()) for c in (0, 1, 2, 4)}
    strength_sd = {str(c): [float(strength[code == c, j].std()) for j in range(3) if c & (1 << j)]
                   for c in (1, 2, 4)}
    chosen = np.asarray(payload["selected_partition"], dtype=np.int8)
    oracle_group = np.zeros_like(chosen)
    oracle_group[:schema.action_dim // 2, 0] = 1
    oracle_group[schema.action_dim // 2:, 1] = 1
    match = bool(np.array_equal(chosen, oracle_group) or np.array_equal(chosen, oracle_group[:, ::-1]))
    report = {
        "gate": "0_and_train_only_support_audit",
        "public_train_sha256": public_manifest["splits"]["train"]["public_sha256"],
        "evaluator_train_sha256": evaluator_manifest["train"]["evaluator_sha256"],
        "transitions": int(len(code)), "worlds": len(public_manifest["splits"]["train"]["world_ids"]),
        "regime_transition_counts": counts, "active_strength_standard_deviations": strength_sd,
        "action_covariance_eigenvalues": action_eigen.tolist(),
        "action_covariance_condition_number": float(action_eigen[-1] / action_eigen[0]),
        "normalized_sensitivity_fisher_diagonal": fisher.tolist(),
        "fisher_min_eigenvalue": float(fisher.min()),
        "fisher_condition_number": float(fisher.max() / fisher.min()) if np.all(fisher > 0) else None,
        "fisher_local_rank": int(np.linalg.matrix_rank(np.diag(fisher))),
        "selected_partition_matches_evaluator_partition_up_to_label_swap": match,
        "partition_score_gap": json.loads((run / "training_report.json").read_text())["best_to_second_log_evidence_gap"],
        "interpretation": "evaluator labels used only after support freeze for this diagnostic",
    }
    path = run / "gate0_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
