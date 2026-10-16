"""Run the RWT Stage 1.5 CPU identification sweep and export CSV/plots."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from rwt.agents import (
    BeliefActivePolicy,
    PassiveBayesPolicy,
    PureInformationGainPolicy,
    RandomProbePolicy,
    make_candidate_library,
)
from rwt.diagnostics import best_probe_actions, candidate_pair_rows, discrimination_summary
from rwt.dynamics import default_regimes
from rwt.environment import RecurrentDynamicsEnv
from rwt.evaluation import RegimeEvaluator
from rwt.schedule import make_schedule


def _construct_policy(
    approach: str,
    candidates,
    *,
    noise_std: float,
    seed: int,
    horizon: int,
    probing_budget: int,
    max_action_norm: float,
    hazard_rate: float,
):
    shared = dict(
        candidates=candidates,
        noise_std=noise_std,
        seed=seed,
        max_action_norm=max_action_norm,
        horizon=horizon,
        probing_budget=probing_budget,
        hazard_rate=hazard_rate,
    )
    if approach == "passive_bayes":
        return PassiveBayesPolicy(**shared)
    if approach == "random_probe":
        return RandomProbePolicy(**shared)
    if approach == "pure_information_gain":
        return PureInformationGainPolicy(**shared)
    if approach.startswith("belief_active_beta_"):
        beta = float(approach.removeprefix("belief_active_beta_"))
        return BeliefActivePolicy(**shared, information_weight=beta)
    raise ValueError(f"unknown approach {approach!r}")


def run_episode(
    seed: int,
    approach: str,
    *,
    split: str,
    noise_std: float,
    min_dwell: int,
    max_dwell: int,
    probing_budget: int,
    max_action_norm: float,
    hazard_rate: float,
    max_state_norm: float,
):
    regimes = default_regimes()
    schedule = make_schedule(split, seed, min_dwell=min_dwell, max_dwell=max_dwell)
    candidates = make_candidate_library()
    policy = _construct_policy(
        approach,
        candidates,
        noise_std=noise_std,
        seed=seed,
        horizon=schedule.horizon,
        probing_budget=probing_budget,
        max_action_norm=max_action_norm,
        hazard_rate=hazard_rate,
    )
    env = RecurrentDynamicsEnv(
        schedule,
        regimes=regimes,
        noise_std=noise_std,
        max_action_norm=max_action_norm,
        max_state_norm=max_state_norm,
    )
    state, _ = env.reset(seed=seed)
    policy.reset()
    rewards: list[float] = []
    posteriors = []
    trajectory: list[dict[str, object]] = []
    terminated_unstable = False
    change_steps = set(schedule.change_steps())
    change_index_by_step = {step: index + 1 for index, step in enumerate(schedule.change_steps())}
    for step in range(schedule.horizon):
        action = policy.act(state, env.target.copy(), step)
        if np.linalg.norm(action) > max_action_norm + 1e-10:
            raise AssertionError("policy violated the shared action-norm constraint")
        previous_state = state.copy()
        next_state, reward, terminated, truncated, info = env.step(action)
        if any(key in info for key in ("regime", "regime_id", "change_time", "env_change", "delta_change")):
            raise AssertionError("environment exposed hidden regime metadata")
        policy.observe(previous_state, action, next_state, reward)
        posterior = policy.filter.posterior.copy()
        posteriors.append(posterior)
        rewards.append(reward)
        truth_name = schedule.regime_name_at(step)
        truth = regimes[truth_name].parameters
        mean_kl, min_kl = discrimination_summary(
            truth, candidates, previous_state, action, noise_std
        )
        regime_probabilities = {
            name: float(posterior[candidates.index(regime.parameters)])
            for name, regime in regimes.items()
        }
        trajectory.append({
            "seed": seed,
            "approach": approach,
            "step": step,
            "regime": truth_name,
            "change_point": int(step in change_steps),
            "change_index": change_index_by_step.get(step, 0),
            "reward": float(reward),
            "cumulative_return": float(sum(rewards)),
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "action_norm": float(np.linalg.norm(action)),
            "is_probe": int(step in policy.probe_steps),
            "posterior_entropy": policy.filter.entropy,
            "identification_correct": int(candidates[int(np.argmax(posterior))] == truth),
            "true_candidate_mean_kl": mean_kl,
            "true_candidate_min_kl": min_kl,
            **{f"posterior_{name}": value for name, value in regime_probabilities.items()},
        })
        state = next_state
        if terminated:
            terminated_unstable = True
            break
        if truncated:
            break

    score = RegimeEvaluator(schedule, candidates).score(rewards, posteriors)
    result = {
        "seed": seed,
        "approach": approach,
        "split": split,
        "cumulative_return": score.cumulative_return,
        "identification_accuracy": score.identification_accuracy,
        "change_detection_delay": score.change_detection_delay,
        "mean_posterior_entropy": score.mean_posterior_entropy,
        "mean_action_norm": float(np.mean([row["action_norm"] for row in trajectory])),
        "mean_true_candidate_kl": float(np.mean([row["true_candidate_mean_kl"] for row in trajectory])),
        "min_true_candidate_kl": float(np.mean([row["true_candidate_min_kl"] for row in trajectory])),
        "probing_budget": min(probing_budget, schedule.horizon),
        "probe_actions_used": int(sum(row["is_probe"] for row in trajectory)),
        "steps": len(rewards),
        "terminated_unstable": int(terminated_unstable),
        "noise_std": noise_std,
        "hazard_rate": hazard_rate,
        "max_action_norm": max_action_norm,
    }
    for name in ("A", "B", "C", "D"):
        result[f"accuracy_{name}"] = score.accuracy_by_regime.get(name, float("nan"))
    return result, trajectory


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _make_plots(output_dir: Path, rows: list[dict[str, object]], trajectories: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    approaches = list(dict.fromkeys(str(row["approach"]) for row in rows))
    colors = plt.get_cmap("tab10")
    change_points_by_index: dict[int, list[int]] = {}
    for row in trajectories:
        change_index = int(row["change_index"])
        if change_index:
            change_points_by_index.setdefault(change_index, []).append(int(row["step"]))
    change_points = sorted(float(np.mean(values)) for values in change_points_by_index.values())

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    for index, approach in enumerate(approaches):
        selected = [row for row in trajectories if row["approach"] == approach]
        by_step: dict[int, list[dict[str, object]]] = {}
        for row in selected:
            by_step.setdefault(int(row["step"]), []).append(row)
        xs = sorted(by_step)
        accuracy = [np.mean([float(item["identification_correct"]) for item in by_step[xs0]]) for xs0 in xs]
        entropy = [np.mean([float(item["posterior_entropy"]) for item in by_step[xs0]]) for xs0 in xs]
        returns = [np.mean([float(item["cumulative_return"]) for item in by_step[xs0]]) for xs0 in xs]
        color = colors(index % 10)
        axes[0].plot(xs, accuracy, label=approach, color=color)
        axes[1].plot(xs, entropy, label=approach, color=color)
        axes[2].plot(xs, returns, label=approach, color=color)
    for axis in axes:
        for point in change_points:
            axis.axvline(point, color="black", linestyle="--", alpha=0.18, linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("Per-step identification accuracy")
    axes[1].set_ylabel("Posterior entropy (nats)")
    axes[2].set_ylabel("Cumulative return")
    axes[2].set_xlabel("Environment step (dashed lines show seed-level change points)")
    fig.suptitle("RWT Stage 1.5: identification and control over time")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_5_timeline.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    for index, approach in enumerate(approaches):
        selected = [row for row in rows if row["approach"] == approach]
        ret = np.asarray([float(row["cumulative_return"]) for row in selected])
        accuracy = np.asarray([float(row["identification_accuracy"]) for row in selected])
        axis.errorbar(
            accuracy.mean(), ret.mean(),
            xerr=accuracy.std(ddof=1) if len(accuracy) > 1 else 0.0,
            yerr=ret.std(ddof=1) if len(ret) > 1 else 0.0,
            marker="o", capsize=3, label=approach, color=colors(index % 10),
        )
    axis.set_xlabel("Regime identification accuracy")
    axis.set_ylabel("Cumulative return")
    axis.set_title("Identification–return trade-off (mean ± sample SD)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_5_tradeoff.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(42, 62)))
    parser.add_argument("--split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--min-dwell", type=int, default=12)
    parser.add_argument("--max-dwell", type=int, default=20)
    parser.add_argument("--probing-budget", type=int, default=24)
    parser.add_argument("--max-action-norm", type=float, default=0.75)
    parser.add_argument("--max-state-norm", type=float, default=75.0)
    parser.add_argument("--hazard-rate", type=float, default=0.05)
    parser.add_argument("--beta-sweep", default="0,0.25,1,4")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "results" / "rwt" / "stage1_5",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        betas = tuple(float(item.strip()) for item in args.beta_sweep.split(",") if item.strip())
    except ValueError as error:
        parser.error(f"invalid seed or beta list: {error}")
    if len(seeds) < 20:
        parser.error("Stage 1.5 confirmation requires at least 20 seeds")
    if not betas or any(beta < 0.0 for beta in betas):
        parser.error("beta-sweep must contain non-negative values")

    approaches = ["passive_bayes", "random_probe", "pure_information_gain"]
    approaches.extend(f"belief_active_beta_{beta:g}" for beta in betas)
    rows: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("results.csv", "trajectories.csv"):
        path = args.output_dir / filename
        if path.exists():
            path.unlink()
    for seed in seeds:
        for approach in approaches:
            result, trace = run_episode(
                seed,
                approach,
                split=args.split,
                noise_std=args.noise_std,
                min_dwell=args.min_dwell,
                max_dwell=args.max_dwell,
                probing_budget=args.probing_budget,
                max_action_norm=args.max_action_norm,
                hazard_rate=args.hazard_rate,
                max_state_norm=args.max_state_norm,
            )
            rows.append(result)
            trajectories.extend(trace)
            _append_csv(args.output_dir / "results.csv", [result])
            _append_csv(args.output_dir / "trajectories.csv", trace)
        print(f"completed seed {seed} ({len(rows)} / {len(seeds) * len(approaches)} runs)", flush=True)
    _write_csv(args.output_dir / "results.csv", rows)
    _write_csv(args.output_dir / "trajectories.csv", trajectories)

    candidates = make_candidate_library()
    pair_rows = candidate_pair_rows(
        candidates, noise_std=args.noise_std, max_action_norm=args.max_action_norm
    )
    _write_csv(args.output_dir / "pairwise_kl.csv", pair_rows)
    ranking_rows: list[dict[str, object]] = []
    for state_name, diagnostic_state in (
        ("rest", np.zeros(4)),
        ("moving", np.asarray([0.0, 0.0, 1.0, -0.5])),
    ):
        ranked = best_probe_actions(
            candidates, diagnostic_state, noise_std=args.noise_std,
            max_action_norm=args.max_action_norm,
        )
        ranking_rows.extend({
            "state_case": state_name,
            "rank": rank,
            "action_x": float(action[0]),
            "action_y": float(action[1]),
            "mean_pairwise_kl_nats": mean_kl,
            "weakest_pair_kl_nats": min_kl,
        } for rank, (action, mean_kl, min_kl) in enumerate(ranked, start=1))
    _write_csv(args.output_dir / "probe_action_rankings.csv", ranking_rows)

    summary_rows = []
    for approach in approaches:
        selected = [row for row in rows if row["approach"] == approach]
        summary: dict[str, object] = {"approach": approach, "seeds": len(selected)}
        for metric in (
            "cumulative_return", "identification_accuracy", "change_detection_delay",
            "mean_posterior_entropy", "mean_action_norm", "mean_true_candidate_kl",
            "min_true_candidate_kl",
        ):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        for regime in ("A", "B", "C", "D"):
            values = np.asarray([float(row[f"accuracy_{regime}"]) for row in selected], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"accuracy_{regime}_mean"] = float(values.mean()) if len(values) else float("nan")
            summary[f"accuracy_{regime}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    if not args.skip_plots:
        _make_plots(args.output_dir, rows, trajectories)

    print(f"wrote {len(rows)} seed/approach rows to {args.output_dir / 'results.csv'}")
    print(f"wrote {len(pair_rows)} pairwise candidate KL rows")
    for summary in summary_rows:
        print(
            f"{summary['approach']}: return {summary['cumulative_return_mean']:.3f}"
            f" +/- {summary['cumulative_return_std']:.3f}; identification"
            f" {summary['identification_accuracy_mean']:.3f}"
            f" +/- {summary['identification_accuracy_std']:.3f}; delay"
            f" {summary['change_detection_delay_mean']:.3f}"
            f" +/- {summary['change_detection_delay_std']:.3f}; entropy"
            f" {summary['mean_posterior_entropy_mean']:.3f}"
            f" +/- {summary['mean_posterior_entropy_std']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
