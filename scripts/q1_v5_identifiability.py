"""CPU Fisher-information audit for log quadratic drag under fixed policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q1.datasets import rollout
from q1.simulator import sample_params
from q1_v5.identifiability import efficient_information
from q1_v5.model import DT, SIGMA, fit_theta, jacobian, response
from scripts.q1_v5_experiment import train_shared


def xy(trajectory):
    return np.c_[trajectory.states[:-1], trajectory.actions], trajectory.delta_v


def c_coordinates(c):
    w = np.asarray(c["weights"], dtype=float)
    s = np.asarray(c["slopes"], dtype=float)
    return np.r_[np.log(w[:2] / w[2]), np.log(s)]


def from_c_coordinates(z):
    z = np.asarray(z)
    logits = np.r_[z[:2], 0.0]
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return {"weights": weights, "slopes": np.exp(z[2:])}


def joint_jacobian(x, theta, c, eps=1e-5):
    jt = jacobian(x, theta, c)
    base = c_coordinates(c)
    jc = []
    for i in range(len(base)):
        step = np.zeros_like(base)
        step[i] = eps
        yp = response(x, theta, from_c_coordinates(base + step)).reshape(-1)
        ym = response(x, theta, from_c_coordinates(base - step)).reshape(-1)
        jc.append((yp - ym) / (2 * eps))
    return np.column_stack((jt, np.column_stack(jc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[20261024444, 20261024618])
    ap.add_argument("--backgrounds", type=int, default=16)
    ap.add_argument("--train-backgrounds", type=int, default=40)
    ap.add_argument("--output", default="results/q1/q1_v5_identifiability")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"will not overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], {"protocol": "Q1_FIXED_POLICY_identifiability_diagnostic", "cases": []}
    for seed in args.seeds:
        train = []
        for bi in range(args.train_backgrounds):
            p = sample_params(np.random.default_rng(seed + bi * 97 + 33), "train")
            t = rollout(p, n=512, seed=seed + bi * 811 + 191, sigma=SIGMA)
            x, y = xy(t)
            train.append((x, y))
        c, _, train_loss = train_shared(train)
        for regime, region, offset in (("IDLIKE", "train", 0), ("OOD", "test_background_ood", 3_000_000)):
            for bi in range(args.backgrounds):
                bg_seed = seed + 9_000_000 + offset + bi * 1987
                p = sample_params(np.random.default_rng(bg_seed), region)
                traj = rollout(p, n=96, seed=bg_seed + 99, sigma=SIGMA)
                x, y = xy(traj)
                manifest["cases"].append({"seed": seed, "regime": regime, "background": bi,
                                          "parameter_seed": bg_seed, "trajectory_seed": bg_seed + 99})
                for n in (5, 10, 20, 50, 96):
                    theta, cov, mse = fit_theta(x[:n], y[:n], c)
                    J = joint_jacobian(x[:n], theta, c)
                    info = efficient_information(J, 5, sigma=SIGMA)
                    # Offline-only calibration diagnostic: simulator truth is not passed to fitting.
                    b_est = float(np.exp(theta[5]))
                    b_se = b_est * float(info["crlb_se"])
                    v = x[:n, 2:4]
                    qdrag = np.linalg.norm(v, axis=1, keepdims=True) * v
                    vflat, qflat = v.ravel(), qdrag.ravel()
                    collin = float(vflat @ qflat / max(np.linalg.norm(vflat) * np.linalg.norm(qflat), 1e-30))
                    rows.append({"seed": seed, "regime": regime, "background": bi, "N_old": n,
                                 "b_true_offline": p.b, "b_hat": b_est, "abs_error_offline": abs(b_est-p.b),
                                 "b_se_crlb_local": b_se, "fit_mse": mse,
                                 "I_eff_logb": info["efficient_information"],
                                 "logb_crlb_se": info["crlb_se"], "nuisance_rank": info["nuisance_rank"],
                                 "nuisance_columns": info["nuisance_columns"],
                                 "nuisance_condition": info["nuisance_condition"],
                                 "svd_tolerance": info["svd_tolerance"],
                                 "velocity_rms": float(np.sqrt(np.mean(v**2))),
                                 "velocity_qdrag_cosine": collin,
                                 "action_rms": float(np.sqrt(np.mean(x[:n,4:6]**2))),
                                 "quadratic_signal_to_noise": float(np.linalg.norm(J[:,5]) / SIGMA)})
        print(f"finished seed={seed}; train_loss={train_loss:.6g}", flush=True)
    with (out / "fisher_per_background.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = {}
    for regime in ("IDLIKE", "OOD"):
        for n in (5, 10, 20, 50, 96):
            group = [r for r in rows if r["regime"] == regime and r["N_old"] == n]
            summaries[f"{regime}|N={n}"] = {
                "n": len(group),
                "median_I_eff_logb": float(np.median([r["I_eff_logb"] for r in group])),
                "median_local_logb_se": (float(np.median([r["logb_crlb_se"] for r in group]))
                                          if np.isfinite([r["logb_crlb_se"] for r in group]).all() else None),
                "local_logb_se_unbounded_count": sum(not np.isfinite(r["logb_crlb_se"]) for r in group),
                "median_abs_b_error_offline": float(np.median([r["abs_error_offline"] for r in group])),
                "median_velocity_rms": float(np.median([r["velocity_rms"] for r in group])),
                "median_velocity_qdrag_cosine": float(np.median([r["velocity_qdrag_cosine"] for r in group])),
                "median_action_rms": float(np.median([r["action_rms"] for r in group])),
            }
    prov = {"seeds": args.seeds, "train_backgrounds": args.train_backgrounds,
            "development_backgrounds_per_regime_per_seed": args.backgrounds,
            "old_support_curve": [5, 10, 20, 50, 96], "dt": DT, "sigma": SIGMA,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interpretation": "Local linearized fixed-policy diagnostic; information is for log(b). Shared action-response nuisance is included; simulator truth appears only in offline calibration columns."}
    (out / "fisher_summary.json").write_text(json.dumps({"summary": summaries, "provenance": prov}, indent=2, allow_nan=False))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
