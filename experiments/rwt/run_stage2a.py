"""CPU-first Stage 2A learned compositional dynamics experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rwt.dynamics import default_regimes
from rwt.environment import RecurrentDynamicsEnv
from rwt.schedule import make_schedule
from rwt.stage2a import (
    ALL_THEORY_CODES, TRAIN_THEORY_CODES, CompositionalDynamics,
    IndependentExpertsDynamics, MonolithicDynamics, TheoryBelief,
    TransitionArrays, save_checkpoint, train_latent_model,
    train_privileged_factor_diagnostic,
)


def _action(rng: np.random.Generator, max_norm: float = 0.75) -> np.ndarray:
    value = rng.normal(size=2)
    value /= max(np.linalg.norm(value), 1e-12)
    return value * rng.uniform(0.0, max_norm)


def collect(split: str, seeds: list[int]) -> tuple[TransitionArrays, list[dict[str, object]]]:
    """Collect blinded transition arrays plus evaluator-only sidecar records."""
    states, actions, next_states = [], [], []
    evaluator: list[dict[str, object]] = []
    for seed in seeds:
        schedule = make_schedule(split, seed, min_dwell=16, max_dwell=22)
        env = RecurrentDynamicsEnv(schedule, noise_std=0.025)
        state, _ = env.reset(seed=seed + (0 if split == "train" else 100_000))
        rng = np.random.default_rng(seed + 50_000)
        for t in range(schedule.horizon):
            action = _action(rng, env.max_action_norm)
            next_state, reward, terminated, truncated, info = env.step(action)
            states.append(state.copy()); actions.append(action.copy()); next_states.append(next_state.copy())
            # This evaluator-only sidecar is never passed to training/inference.
            evaluator.append({"seed": seed, "step": t, "regime": schedule.regime_name_at(t),
                              "change_point": t in schedule.change_steps(), "state": state.copy(),
                              "action": action.copy(), "next_state": next_state.copy()})
            state = next_state
            if terminated or truncated:
                break
        env.close()
    data = TransitionArrays(np.asarray(states, np.float32), np.asarray(actions, np.float32), np.asarray(next_states, np.float32))
    return data, evaluator


def _candidate_tensors(model, kind, state, action):
    if kind == "compositional":
        means = state[:, None, :] + model.predict_codes(state, action)
        scales = model.log_scale.clamp(-6, 2).exp().view(1, 1, -1).expand_as(means)
    elif kind == "experts":
        means = model.predict_codes(state, action)
        scales = model.log_scale.clamp(-6, 2).exp()[None, :, :].expand_as(means)
    else:
        mean, scale = model.predict(state, action)
        means, scales = mean[:, None, :], scale[:, None, :]
    return means, scales


def _normal_log_prob(target, means, scales):
    z = (target[:, None, :] - means) / scales
    return -0.5 * (z.square() + 2 * scales.log() + np.log(2 * np.pi)).sum(-1)


def evaluate(model, kind: str, rows: list[dict[str, object]], *, horizon: int = 5) -> list[dict[str, object]]:
    """Causal one-step prediction and short open-loop rollouts per regime."""
    model.eval()
    by_episode: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["seed"]), []).append(row)
    out = []
    codes = ALL_THEORY_CODES if kind == "compositional" else TRAIN_THEORY_CODES if kind == "experts" else ((0, 0),)
    with torch.no_grad():
        for seed, episode in by_episode.items():
            belief = TheoryBelief(codes, hazard=0.02)
            ll_by_step = []
            predictions = []
            for row in episode:
                s = torch.as_tensor(np.asarray(row["state"])[None], dtype=torch.float32)
                a = torch.as_tensor(np.asarray(row["action"])[None], dtype=torch.float32)
                sn = torch.as_tensor(np.asarray(row["next_state"])[None], dtype=torch.float32)
                means, scales = _candidate_tensors(model, kind, s, a)
                logp = _normal_log_prob(sn, means, scales)[0].cpu().numpy().astype(np.float64)
                weights = belief.predict_weights()
                log_mix = np.log(weights.clip(1e-300)) + logp
                max_log = np.max(log_mix)
                nll = -(max_log + np.log(np.exp(log_mix - max_log).sum()))
                pred = np.einsum("k,kd->d", weights, means[0].cpu().numpy())
                predictions.append((pred, float(nll), logp, weights.copy()))
                belief.update(logp)
            # Score all predictions by evaluator-only regime labels.
            for i, row in enumerate(episode):
                pred, nll, logp, prior_weights = predictions[i]
                target = np.asarray(row["next_state"])
                mse = float(np.square(pred - target).mean())
                regime = str(row["regime"])
                roll_errors = []
                # Roll out from observed anchor using posterior available before
                # the anchor transition; score only horizons staying in regime.
                for length in (1, 5, horizon):
                    if length > horizon:
                        continue
                    current = np.asarray(row["state"], dtype=np.float32).copy()
                    weights = prior_weights.copy()
                    end = min(i + length, len(episode))
                    actual_end = i
                    for j in range(i, end):
                        if str(episode[j]["regime"]) != regime:
                            break
                        state_t = torch.as_tensor(current[None], dtype=torch.float32)
                        action_t = torch.as_tensor(np.asarray(episode[j]["action"])[None], dtype=torch.float32)
                        means, _ = _candidate_tensors(model, kind, state_t, action_t)
                        current = np.einsum("k,kd->d", weights, means[0].cpu().numpy()).astype(np.float32)
                        actual_end = j + 1
                    if actual_end == i + length:
                        truth = np.asarray(episode[actual_end - 1]["next_state"])
                        roll_errors.append((length, float(np.square(current - truth).mean())))
                out.append({"seed": seed, "step": row["step"], "regime": regime,
                            "change_point": int(bool(row["change_point"])), "mse": mse,
                            "nll": nll, "rollout": roll_errors})
    return out


def oracle_metrics(rows):
    regimes = default_regimes()
    result = []
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row["seed"]), []).append(row)
    for episode in by_episode.values():
        for i, row in enumerate(episode):
            regime = str(row["regime"])
            parameters = regimes[regime].parameters
            s = np.asarray(row["state"]); a = np.asarray(row["action"]); truth = np.asarray(row["next_state"])
            pred = parameters.predict_state(s, a)
            roll = [(1, float(np.square(pred - truth).mean()))]
            current = s.copy(); last = i
            for j in range(i, min(i + 5, len(episode))):
                if str(episode[j]["regime"]) != regime:
                    break
                current = parameters.predict_state(current, np.asarray(episode[j]["action"]))
                last = j + 1
            if last == i + 5:
                roll.append((5, float(np.square(current - np.asarray(episode[last - 1]["next_state"])).mean())))
            # This is a deterministic expected-transition oracle; its NLL is not
            # reported because this baseline does not estimate the noise law.
            result.append({"regime": regime, "mse": float(np.square(pred - truth).mean()), "nll": None, "rollout": roll})
    return result


def evaluate_privileged_factor_diagnostic(model, rows):
    """Oracle-labeled diagnostic: maps evaluator regime labels to bit codes."""
    code_index = {"A": 0, "B": 1, "D": 2, "C": 3}
    output = []
    model.eval()
    with torch.no_grad():
        by_episode = {}
        for row in rows:
            by_episode.setdefault(int(row["seed"]), []).append(row)
        for episode in by_episode.values():
            for i, row in enumerate(episode):
                regime = str(row["regime"])
                code = code_index[regime]
                s = np.asarray(row["state"], np.float32)
                a = np.asarray(row["action"], np.float32)
                truth = np.asarray(row["next_state"], np.float32)
                state = torch.as_tensor(s[None]); action = torch.as_tensor(a[None])
                pred = (state + model.predict_codes(state, action)[:, code])[0].cpu().numpy()
                scale = model.log_scale.clamp(-6, 2).exp().cpu().numpy()
                nll = float((0.5 * (((truth - pred) / scale) ** 2 + 2 * np.log(scale) + np.log(2 * np.pi))).sum())
                current = s.copy(); last = i
                for j in range(i, min(i + 5, len(episode))):
                    if str(episode[j]["regime"]) != regime:
                        break
                    ss = torch.as_tensor(current[None])
                    aa = torch.as_tensor(np.asarray(episode[j]["action"], np.float32)[None])
                    current = (ss + model.predict_codes(ss, aa)[:, code])[0].cpu().numpy()
                    last = j + 1
                roll = [(1, float(np.square(pred - truth).mean()))]
                if last == i + 5:
                    roll.append((5, float(np.square(current - np.asarray(episode[last - 1]["next_state"])).mean())))
                output.append({"seed": row["seed"], "step": row["step"], "regime": regime,
                               "change_point": int(bool(row["change_point"])),
                               "mse": float(np.square(pred - truth).mean()), "nll": nll, "rollout": roll})
    return output


def aggregate(metrics):
    names = sorted({str(row["regime"]) for row in metrics})
    rows = []
    for name in names:
        subset = [row for row in metrics if row["regime"] == name]
        item = {"regime": name, "n": len(subset)}
        for metric in ("mse", "nll"):
            values = [float(row[metric]) for row in subset if row[metric] is not None]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else None
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        rollout = [(length, value) for row in subset for length, value in row["rollout"]]
        for length in (1, 5):
            values = [value for h, value in rollout if h == length]
            item[f"rollout_mse_h{length}"] = float(np.mean(values)) if values else None
        rows.append(item)
    id_mean = np.mean([r["mse_mean"] for r in rows if r["regime"] in {"A", "B", "D"}])
    for row in rows:
        row["generalization_gap_vs_train_regimes"] = row["mse_mean"] - id_mean if row["mse_mean"] is not None else None
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rwt/stage2a")
    parser.add_argument("--train-seeds", default="42,43,44,45,46,47,48,49,50,51,52,53")
    parser.add_argument("--validation-seeds", default="142,143,144")
    parser.add_argument("--evaluation-seeds", default="242,243,244,245,246,247")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, train_rows = collect("train", [int(x) for x in args.train_seeds.split(",")])
    validation, val_rows = collect("train", [int(x) for x in args.validation_seeds.split(",")])
    evaluation, eval_rows = collect("evaluation", [int(x) for x in args.evaluation_seeds.split(",")])
    np.savez_compressed(
        args.output_dir / "training_transitions.npz",
        train_state=train.state, train_action=train.action, train_next_state=train.next_state,
        validation_state=validation.state, validation_action=validation.action,
        validation_next_state=validation.next_state,
    )
    # Keep only blinded numerical arrays in the model fit call.
    assert len(train.state) > 0 and not any(row["regime"] == "C" for row in val_rows)
    assert all(set(np.unique([row["regime"] for row in eval_rows])) == {"A", "B", "C", "D"} for _ in [0])
    configs = {
        "compositional": {"state_dim": 4, "action_dim": 2, "hidden_dim": args.hidden_dim},
        "experts": {"state_dim": 4, "action_dim": 2, "hidden_dim": args.hidden_dim},
        "monolithic": {"state_dim": 4, "action_dim": 2, "hidden_dim": max(12, int(args.hidden_dim * np.sqrt(3)))},
    }
    models = {}
    histories = []
    params = {}
    for i, kind in enumerate(("monolithic", "experts", "compositional")):
        torch.manual_seed(2026 + i)
        cls = {"monolithic": MonolithicDynamics, "experts": IndependentExpertsDynamics,
               "compositional": CompositionalDynamics}[kind]
        model = cls(**configs[kind])
        history = train_latent_model(model, train, validation, model_kind=kind, epochs=args.epochs,
                                     batch_size=args.batch_size, seed=6026 + i, device=args.device)
        models[kind] = model
        params[kind] = sum(p.numel() for p in model.parameters())
        histories += [{"model": kind, **row} for row in history]
        save_checkpoint(args.output_dir / f"{kind}.pt", model, model_kind=kind, config=configs[kind])
    # Explicit privileged-label diagnostic, kept outside all main method results.
    torch.manual_seed(2030)
    diagnostic_model = CompositionalDynamics(**configs["compositional"])
    diagnostic_history = train_privileged_factor_diagnostic(
        diagnostic_model, train, np.asarray([row["regime"] for row in train_rows]),
        validation, np.asarray([row["regime"] for row in val_rows]),
        epochs=args.epochs, batch_size=args.batch_size, seed=6030, device=args.device,
    )
    histories += [{"model": "privileged_factor_diagnostic", **row} for row in diagnostic_history]
    metrics = []
    for kind, model in models.items():
        for row in evaluate(model, kind, eval_rows):
            metrics.append({"model": kind, **row})
    oracle = oracle_metrics(eval_rows)
    summaries = []
    for kind in models:
        summaries += [{"model": kind, **row} for row in aggregate([r for r in metrics if r["model"] == kind])]
    summaries += [{"model": "oracle_reference", **row} for row in aggregate(oracle)]
    diagnostic_metrics = evaluate_privileged_factor_diagnostic(diagnostic_model, eval_rows)
    summaries += [{"model": "privileged_factor_diagnostic", **row} for row in aggregate(diagnostic_metrics)]
    save_checkpoint(args.output_dir / "privileged_factor_diagnostic.pt", diagnostic_model,
                    model_kind="compositional", config=configs["compositional"])
    write_csv(args.output_dir / "training_curves.csv", histories)
    write_csv(args.output_dir / "per_regime_metrics.csv", summaries)
    write_csv(args.output_dir / "evaluation_trace.csv", [
        {k: v for k, v in r.items() if k != "rollout"} for r in metrics
    ])
    write_csv(args.output_dir / "privileged_factor_diagnostic.csv", [
        {k: v for k, v in r.items() if k != "rollout"} for r in diagnostic_metrics
    ])
    pd_rows = [{"model": name, "parameter_count": count} for name, count in params.items()]
    write_csv(args.output_dir / "model_budget.csv", pd_rows)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({"train_seeds": args.train_seeds, "validation_seeds": args.validation_seeds,
                   "evaluation_seeds": args.evaluation_seeds, "train_transitions": len(train.state),
                   "validation_transitions": len(validation.state), "evaluation_transitions": len(evaluation.state),
                   "epochs": args.epochs, "hidden_dim": args.hidden_dim, "device": args.device,
                   "training_regimes": ["A", "B", "D"], "heldout_regime": "C",
                   "selection_regimes": ["A", "B", "D"], "model_parameters": params}, handle, indent=2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in [*models, "privileged_factor_diagnostic"]:
        values = [r for r in histories if r["model"] == name]
        ax.plot([r["epoch"] for r in values], [r["train_nll"] for r in values], label=f"{name} train")
        ax.plot([r["epoch"] for r in values], [r["validation_nll"] for r in values], linestyle="--", label=f"{name} validation")
    ax.set(xlabel="Epoch", ylabel="Mixture NLL", title="RWT Stage 2A training curves (A/B/D only)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output_dir / "training_curves.png", dpi=160); plt.close(fig)
    print(f"train transitions={len(train.state)} validation={len(validation.state)} evaluation={len(evaluation.state)}")
    print(f"model parameters={params}")
    for row in summaries:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
