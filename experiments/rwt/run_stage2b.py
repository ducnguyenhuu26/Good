"""Stage 2B online structural-disambiguation benchmark (CPU-first)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.rwt.run_stage2a import collect
from rwt.dynamics import default_regimes
from rwt.environment import RecurrentDynamicsEnv
from rwt.schedule import RegimeSchedule, Segment
from rwt.stage2a import (
    IndependentExpertsDynamics, MonolithicDynamics, TransitionArrays,
    save_checkpoint, train_latent_model,
)
from rwt.stage2b import (
    OnlineStructuralAgent, PredictionSet, enumerate_completions,
    learned_compositional_prediction_set, learned_expert_prediction_set,
    learned_monolithic_prediction_set, oracle_prediction_set,
)


SEGMENTS = (Segment("A", 40), Segment("B", 40), Segment("D", 40),
            Segment("C", 48), Segment("A", 40), Segment("C", 48))
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32)
MAX_ACTION_NORM = 0.75
PROBING_BUDGET = 24
HAZARD = 0.05


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _arrays(rows: list[dict]) -> TransitionArrays:
    return TransitionArrays(
        np.asarray([r["state"] for r in rows], np.float32),
        np.asarray([r["action"] for r in rows], np.float32),
        np.asarray([r["next_state"] for r in rows], np.float32),
    )


def train_unlabeled_models(output: Path, *, epochs: int, hidden_dim: int, seed: int):
    """Fit two baseline families from the same blinded A/B/D arrays."""
    train, train_rows = collect("train", list(range(42, 54)))
    validation, val_rows = collect("train", [142, 143, 144])
    if not len(train.state) or any(r["regime"] == "C" for r in train_rows + val_rows):
        raise AssertionError("C must be absent from train and validation")
    # Labels are checked by the experiment harness and then discarded. Both
    # learners receive only arrays of (s,a,s').
    assert set(r["regime"] for r in train_rows) <= {"A", "B", "D"}
    assert set(r["regime"] for r in val_rows) <= {"A", "B", "D"}
    histories = []
    models = {}
    model_specs = (
        ("independent_experts", IndependentExpertsDynamics(hidden_dim=hidden_dim), "experts"),
        ("monolithic", MonolithicDynamics(hidden_dim=max(12, int(hidden_dim * np.sqrt(3)))), "monolithic"),
    )
    for index, (name, model, kind) in enumerate(model_specs):
        history = train_latent_model(
            model, train, validation, model_kind=kind, epochs=epochs,
            seed=seed + index, device="cpu", batch_size=256,
        )
        histories.extend({"model": name, **row} for row in history)
        config = ({"state_dim": 4, "action_dim": 2, "hidden_dim": hidden_dim}
                  if kind == "experts" else
                  {"state_dim": 4, "action_dim": 2, "hidden_dim": max(12, int(hidden_dim * np.sqrt(3)))})
        save_checkpoint(output / "checkpoints" / f"{name}.pt", model, model_kind=kind, config=config)
        model.eval()
        models[name] = model
    manifest = {
        "train_seeds": list(range(42, 54)), "validation_seeds": [142, 143, 144],
        "train_regimes": sorted({str(r["regime"]) for r in train_rows}),
        "validation_regimes": sorted({str(r["regime"]) for r in val_rows}),
        "C_excluded": True, "train_transitions": len(train.state),
        "validation_transitions": len(validation.state), "epochs": epochs,
        "hidden_dim_experts": hidden_dim,
        "hidden_dim_monolithic": max(12, int(hidden_dim * np.sqrt(3))),
        "selection_metric": "A/B/D validation marginal NLL",
        "agent_inputs": "transition arrays only: state, action, next_state",
    }
    return models, histories, manifest, train


def evaluator_alignment(model, train_data: TransitionArrays, mode: str) -> dict[str, int]:
    """Post-hoc, evaluator-only alignment for permutation-invariant scoring."""
    rng = np.random.default_rng(823_510)
    n = min(512, len(train_data.state))
    ids = rng.choice(len(train_data.state), n, replace=False)
    states = train_data.state[ids].astype(np.float64)
    actions = train_data.action[ids].astype(np.float64)
    true = default_regimes()
    names = ("A", "B", "D")
    # Mean velocity prediction for each unlabeled learned expert.
    seen = []
    for s, a in zip(states, actions):
        with torch.no_grad():
            st = torch.as_tensor(s[None], dtype=torch.float32)
            ac = torch.as_tensor(a[None], dtype=torch.float32)
            seen.append(model.predict_codes(st, ac)[0, :, 2:].cpu().numpy())
    seen = np.asarray(seen)  # [N, experts, velocity]
    truth = np.asarray([[true[n].parameters.predict_velocity(s[2:], a) for n in names]
                        for s, a in zip(states, actions)])  # [N, regimes, velocity]
    best_perm = min(
        itertools.permutations(range(3)),
        key=lambda perm: sum(float(np.mean((seen[:, perm[j]] - truth[:, j]) ** 2)) for j in range(3)),
    )
    mapping = {name: int(best_perm[i]) for i, name in enumerate(names)}
    if mode == "experts":
        return mapping
    # Completion index follows the evaluator-only A/B/D permutation: C is the
    # sum of B and D with the A anchor removed. This uses no C dynamics or C
    # transitions, and the mapping is never passed into the online agent.
    if mode == "composition":
        mapping["C"] = 3 + mapping["A"]
    return mapping


def oracle_alignment() -> dict[str, int]:
    return {"A": 0, "B": 1, "D": 2, "C": 3}


def get_specs(models: dict[str, object] | None, train_data: TransitionArrays | None,
              branches: tuple[str, ...] = ("diagnostic", "learned")):
    regimes = default_regimes()
    oracle_models = tuple(regimes[n].parameters for n in ("A", "B", "D"))
    specs = []
    alignments = {}
    if "diagnostic" in branches:
        for policy in ("passive", "random", "active"):
            specs.append({
                "branch": "diagnostic", "model": "oracle_composition", "policy": policy,
                "factory": lambda s, a, ms=oracle_models: oracle_prediction_set(ms, s, a, noise_std=0.025),
                "names": ("seen_0", "seen_1", "seen_2", "completion_0", "completion_1", "completion_2"),
                "known_count": 3, "completion_indices": (3, 4, 5), "alignment": oracle_alignment(),
                "memory": True,
            })
        specs.append({
            "branch": "diagnostic", "model": "oracle_composition", "policy": "active_no_memory",
            "factory": lambda s, a, ms=oracle_models: oracle_prediction_set(ms, s, a, noise_std=0.025),
            "names": ("seen_0", "seen_1", "seen_2", "completion_0", "completion_1", "completion_2"),
            "known_count": 3, "completion_indices": (3, 4, 5), "alignment": oracle_alignment(),
            "memory": False,
        })
    if "learned" in branches:
        if models is None or train_data is None:
            raise ValueError("learned branch requires A/B/D-only models and transitions")
        expert = models["independent_experts"]
        mono = models["monolithic"]
        comp_alignment = evaluator_alignment(expert, train_data, "composition")
        expert_alignment = evaluator_alignment(expert, train_data, "experts")
        alignments = {"learned_composition": comp_alignment, "independent_experts": expert_alignment}
        expert_names = ("expert_0", "expert_1", "expert_2")
        comp_names = (*expert_names, "completion_0", "completion_1", "completion_2")
        for policy in ("passive", "random", "active"):
            specs.append({
                "branch": "learned", "model": "learned_composition", "policy": policy,
                "factory": lambda s, a, m=expert: learned_compositional_prediction_set(m, s, a),
                "names": comp_names, "known_count": 3, "completion_indices": (3, 4, 5),
                "alignment": comp_alignment, "memory": True,
            })
            specs.append({
                "branch": "learned", "model": "independent_experts", "policy": policy,
                "factory": lambda s, a, m=expert: learned_expert_prediction_set(m, s, a),
                "names": expert_names, "known_count": 3, "completion_indices": (),
                "alignment": {**expert_alignment, "C": -1}, "memory": True,
            })
            specs.append({
                "branch": "learned", "model": "monolithic", "policy": policy,
                "factory": lambda s, a, m=mono: learned_monolithic_prediction_set(m, s, a),
                "names": ("monolithic",), "known_count": 1, "completion_indices": (),
                "alignment": {}, "memory": True,
            })
    return specs, alignments


def _segments_for(schedule: RegimeSchedule) -> list[dict]:
    rows = []
    offset = 0
    for segment in schedule.segments:
        rows.append({"regime": segment.regime, "start": offset, "end": offset + segment.duration,
                     "duration": segment.duration})
        offset += segment.duration
    return rows


def _posterior_accuracy(posterior: np.ndarray, regime: str, alignment: dict[str, int]) -> float | None:
    target = alignment.get(regime, -1)
    if target < 0:
        return None
    return float(int(np.argmax(posterior)) == target)


def run_episode(spec: dict, seed: int, *, max_state_norm: float = 75.0):
    schedule = RegimeSchedule(SEGMENTS)
    env = RecurrentDynamicsEnv(
        schedule, noise_std=0.025, target=(2.0, 2.0), max_action_norm=MAX_ACTION_NORM,
        max_state_norm=max_state_norm,
    )
    state, _ = env.reset(seed=seed + 50_000)
    agent = OnlineStructuralAgent(
        spec["factory"], names=spec["names"], known_count=spec["known_count"],
        completion_indices=spec["completion_indices"], policy=("active" if spec["policy"] == "active_no_memory" else spec["policy"]),
        seed=seed + 70_000, horizon=schedule.horizon, probing_budget=PROBING_BUDGET,
        max_action_norm=MAX_ACTION_NORM, hazard=HAZARD,
        use_memory=bool(spec["memory"]),
    )
    rewards, traces = [], []
    segments = _segments_for(schedule)
    c_segments = [x for x in segments if x["regime"] == "C"]
    c_reward = {x["start"]: 0.0 for x in c_segments}
    terminated_at = None
    for step in range(schedule.horizon):
        regime = schedule.regime_name_at(step)  # evaluator sidecar; never sent to the agent
        target_before = spec["alignment"].get(regime, -1)
        memory_had_target = int(target_before in agent.filter.remembered) if target_before >= 0 else 0
        action = agent.act(state, step)
        next_state, reward, terminated, truncated, _info = env.step(action)
        observation = agent.observe(state, action, next_state)
        segment = next(x for x in segments if x["start"] <= step < x["end"])
        offset = step - segment["start"]
        target = spec["alignment"].get(regime, -1)
        target_probability = float(observation.posterior[target]) if target >= 0 else np.nan
        prior_novel_mass = (float(observation.prior[list(agent.filter.completion_indices)].sum())
                            if agent.filter.completion_indices else 0.0)
        max_comp_prob = (float(observation.posterior[list(agent.filter.completion_indices)].max())
                         if agent.filter.completion_indices else np.nan)
        traces.append({
            "branch": spec["branch"], "model": spec["model"], "policy": spec["policy"],
            "seed": seed, "step": step, "regime_eval_only": regime,
            "segment_index": segments.index(segment), "segment_offset": offset,
            "change_point_eval_only": int(offset == 0 and step > 0),
            "probe_slot": int(agent.last_probe),
            "probe_action_used": int(agent.last_probe and agent.last_probe_cost > 1e-10),
            "probe_cost": agent.last_probe_cost, "information_gain": agent.last_information_gain,
            "mse_preupdate": observation.mse, "nll_preupdate": observation.nll,
            "posterior_entropy": observation.entropy,
            "prior_novel_mass": prior_novel_mass,
            "posterior_novel_mass": float(observation.posterior[list(agent.filter.completion_indices)].sum())
            if agent.filter.completion_indices else 0.0,
            "posterior_target_theory": target_probability,
            "prior_target_theory": float(observation.prior[target]) if target >= 0 else np.nan,
            "target_theory_correct": _posterior_accuracy(observation.posterior, regime, spec["alignment"]),
            "posterior_map_name": spec["names"][int(np.argmax(observation.posterior))],
            "memory_contains_target": memory_had_target,
            "memory_size": len(agent.filter.remembered),
            "action_norm": float(np.linalg.norm(action)), "reward": reward,
            "unstable_termination": int(terminated), "time_limit_truncated": int(truncated),
            "state_norm": float(np.linalg.norm(next_state)),
            "cumulative_return": float(sum(rewards) + reward),
        })
        rewards.append(float(reward))
        if segment["regime"] == "C":
            c_reward[segment["start"]] += float(reward)
        state = next_state
        if terminated:
            terminated_at = step
        if terminated or truncated:
            break
    env.close()
    truth_rows = [r for r in traces if r["regime_eval_only"] in {"A", "B", "D"}]
    c_rows = [r for r in traces if r["regime_eval_only"] == "C"]
    id_values = [r["target_theory_correct"] for r in truth_rows if r["target_theory_correct"] is not None]
    c_id_values = [r["target_theory_correct"] for r in c_rows if r["target_theory_correct"] is not None]
    c_delays = []
    for csegment in c_segments:
        rows = [r for r in traces if r["segment_index"] == segments.index(csegment)]
        if spec["alignment"].get("C", -1) < 0 or "C" not in spec["alignment"]:
            c_delays.append(np.nan)
            continue
        delay = csegment["duration"]
        for row in rows:
            if row["posterior_target_theory"] >= 0.8:
                delay = int(row["segment_offset"] + 1)
                break
        c_delays.append(float(delay))
    identified_compositions = [i for i in agent.filter.remembered if i in agent.filter.completion_indices]
    def trace_mean(segment_index: int, key: str):
        values = [r[key] for r in traces if r["segment_index"] == segment_index
                  and r[key] is not None and np.isfinite(float(r[key]))]
        return float(np.mean(values)) if values else np.nan

    false_alarm_rows = [r for r in traces if r["regime_eval_only"] in {"A", "B", "D"}]
    summary = {
        "branch": spec["branch"], "model": spec["model"], "policy": spec["policy"], "seed": seed,
        "cumulative_return": float(sum(rewards)), "n_transitions": len(traces),
        "known_regime_identification_accuracy": float(np.mean(id_values)) if id_values else np.nan,
        "novel_composition_identification_accuracy": float(np.mean(c_id_values)) if c_id_values else np.nan,
        "mean_prediction_mse": float(np.mean([r["mse_preupdate"] for r in traces])) if traces else np.nan,
        "mean_prediction_nll": float(np.mean([r["nll_preupdate"] for r in traces])) if traces else np.nan,
        "mean_posterior_entropy": float(np.mean([r["posterior_entropy"] for r in traces])) if traces else np.nan,
        "identification_delay_first_C": c_delays[0] if len(c_delays) > 0 else np.nan,
        "identification_delay_return_C": c_delays[1] if len(c_delays) > 1 else np.nan,
        "first_C_return": float(c_reward[c_segments[0]["start"]]),
        "return_C_return": float(c_reward[c_segments[1]["start"]]),
        "A_initial_prediction_mse": trace_mean(0, "mse_preupdate"),
        "A_recurrent_prediction_mse": trace_mean(4, "mse_preupdate"),
        "A_initial_identification_accuracy": trace_mean(0, "target_theory_correct"),
        "A_recurrent_identification_accuracy": trace_mean(4, "target_theory_correct"),
        "false_novel_alarm_rate": float(np.mean([r["posterior_novel_mass"] >= 0.8 for r in false_alarm_rows])) if false_alarm_rows and agent.filter.completion_indices else np.nan,
        "probe_slots": int(sum(r["probe_slot"] for r in traces)),
        "probe_actions_used": int(sum(r["probe_action_used"] for r in traces)),
        "probing_cost": float(sum(r["probe_cost"] for r in traces)),
        "mean_probe_information_gain": float(np.mean([r["information_gain"] for r in traces if r["probe_slot"]])) if any(r["probe_slot"] for r in traces) else 0.0,
        "remembered_composition_count": len(identified_compositions),
        "c_target_in_memory_at_return": int(any(
            r["segment_index"] == segments.index(c_segments[1]) and r["segment_offset"] == 0
            and r["memory_contains_target"] for r in traces
        )),
        "unstable_termination": int(terminated_at is not None),
        "termination_step": -1 if terminated_at is None else terminated_at,
    }
    for name in ("A", "B", "D", "C"):
        rows = [r for r in traces if r["regime_eval_only"] == name]
        summary[f"prediction_mse_{name}"] = float(np.mean([r["mse_preupdate"] for r in rows])) if rows else np.nan
        summary[f"prediction_nll_{name}"] = float(np.mean([r["nll_preupdate"] for r in rows])) if rows else np.nan
        correct = [r["target_theory_correct"] for r in rows if r["target_theory_correct"] is not None]
        summary[f"identification_accuracy_{name}"] = float(np.mean(correct)) if correct else np.nan
    for name, segment_index in (("first_C", 3), ("recurrence_C", 5)):
        rows = [r for r in traces if r["segment_index"] == segment_index]
        summary[f"{name}_prediction_mse"] = float(np.mean([r["mse_preupdate"] for r in rows])) if rows else np.nan
        summary[f"{name}_prediction_nll"] = float(np.mean([r["nll_preupdate"] for r in rows])) if rows else np.nan
    adaptation = []
    for c_index, csegment in enumerate(c_segments):
        rows = [r for r in traces if r["segment_index"] == segments.index(csegment)]
        # k=0 is the belief/prediction immediately before the first C transition.
        if rows:
            adaptation.append({
                "branch": spec["branch"], "model": spec["model"], "policy": spec["policy"],
                "seed": seed, "appearance": "first" if c_index == 0 else "recurrence",
                "transitions_seen": 0, "mse_next": rows[0]["mse_preupdate"],
                "nll_next": rows[0]["nll_preupdate"], "posterior_target_theory": rows[0]["prior_target_theory"],
                "posterior_novel_mass": rows[0]["prior_novel_mass"], "return_so_far": 0.0,
                "target_identified": int(rows[0]["prior_target_theory"] >= 0.8),
            })
        for k in CHECKPOINTS[1:]:
            if k >= csegment["duration"] or len(rows) <= k:
                continue
            # Row k predicts the transition after k C observations were assimilated.
            row = rows[k]
            adaptation.append({
                "branch": spec["branch"], "model": spec["model"], "policy": spec["policy"],
                "seed": seed, "appearance": "first" if c_index == 0 else "recurrence",
                "transitions_seen": k, "mse_next": row["mse_preupdate"],
                "nll_next": row["nll_preupdate"], "posterior_target_theory": row["prior_target_theory"],
                "posterior_novel_mass": row["prior_novel_mass"],
                "return_so_far": float(sum(x["reward"] for x in rows[:k])),
                "target_identified": int(row["prior_target_theory"] >= 0.8),
            })
    return summary, traces, adaptation


def _bootstrap_mean(values: np.ndarray, seed: int = 517):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = dict(zip(keys, group))
        record["n"] = len(values)
        for metric in metrics:
            mean, sd, low, high = _bootstrap_mean(np.asarray([r[metric] for r in values], np.float64),
                                                   seed=517 + sum(ord(c) for c in metric + str(group)))
            record.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                           f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        output.append(record)
    return output


def paired_comparisons(rows: list[dict], metrics: tuple[str, ...]):
    index = {(r["branch"], r["model"], r["policy"], int(r["seed"])): r for r in rows}
    output = []
    contrasts = (
        ("diagnostic", "oracle_composition", "active", "oracle_composition", "random"),
        ("diagnostic", "oracle_composition", "active", "oracle_composition", "passive"),
        ("diagnostic", "oracle_composition", "random", "oracle_composition", "passive"),
        ("diagnostic", "oracle_composition", "active", "oracle_composition", "active_no_memory"),
        ("learned", "learned_composition", "active", "learned_composition", "random"),
        ("learned", "learned_composition", "active", "learned_composition", "passive"),
        ("learned", "learned_composition", "random", "learned_composition", "passive"),
        ("learned", "learned_composition", "active", "independent_experts", "active"),
        ("learned", "learned_composition", "active", "monolithic", "active"),
        ("learned", "learned_composition", "active", "learned_composition", "active_no_memory"),
        ("learned", "independent_experts", "active", "independent_experts", "random"),
        ("learned", "independent_experts", "active", "independent_experts", "passive"),
        ("learned", "independent_experts", "random", "independent_experts", "passive"),
        ("learned", "monolithic", "random", "monolithic", "passive"),
    )
    # Main learned active composition has a no-cache recurrence ablation too;
    # add it lazily to preserve the identical online schedule.
    for contrast in contrasts:
        branch, m1, p1, m2, p2 = contrast
        seed_ids = sorted({int(r["seed"]) for r in rows if r["branch"] == branch})
        for metric in metrics:
            diffs = []
            for seed in seed_ids:
                a = index.get((branch, m1, p1, seed)); b = index.get((branch, m2, p2, seed))
                if a is not None and b is not None and np.isfinite(a[metric]) and np.isfinite(b[metric]):
                    diffs.append(float(a[metric] - b[metric]))
            if not diffs:
                continue
            mean, sd, low, high = _bootstrap_mean(np.asarray(diffs), 987 + sum(map(ord, str(contrast) + metric)))
            output.append({"contrast": f"{branch}:{m1}/{p1} - {m2}/{p2}", "metric": metric,
                           "paired_mean_difference": mean, "paired_sd": sd,
                           "ci95_low": low, "ci95_high": high, "n_pairs": len(diffs)})
    return output


def adaptation_summary(rows: list[dict]):
    return summarize(rows, ("branch", "model", "policy", "appearance", "transitions_seen"),
                     ("mse_next", "nll_next", "posterior_target_theory", "posterior_novel_mass",
                      "return_so_far", "target_identified"))


def plot_adaptation(rows: list[dict], path: Path):
    summary = adaptation_summary(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    styles = {"passive": "--", "random": ":", "active": "-", "active_no_memory": "-."}
    colors = {"diagnostic": "#1368aa", "learned": "#d1495b"}
    for ax, appearance, metric, title in (
        (axes[0, 0], "first", "mse_next_mean", "First C: one-step MSE"),
        (axes[0, 1], "recurrence", "mse_next_mean", "Returning C: one-step MSE"),
        (axes[1, 0], "first", "posterior_target_theory_mean", "First C: posterior on aligned theory"),
        (axes[1, 1], "recurrence", "posterior_target_theory_mean", "Returning C: posterior on aligned theory"),
    ):
        selected = []
        for row in summary:
            if row["appearance"] != appearance or row["policy"] not in styles:
                continue
            if row["branch"] == "learned" and row["model"] != "learned_composition":
                continue
            if row["branch"] == "diagnostic" and row["model"] != "oracle_composition":
                continue
            selected.append(row)
        group_map = defaultdict(list)
        for row in selected:
            group_map[(row["branch"], row["policy"])].append(row)
        for (branch, policy), values in group_map.items():
            values.sort(key=lambda r: int(r["transitions_seen"]))
            x = np.asarray([r["transitions_seen"] for r in values], dtype=int)
            y = np.asarray([r[metric] for r in values], dtype=float)
            ax.plot(x, y, marker="o", linestyle=styles[policy], color=colors[branch],
                    label=f"{branch} {policy}")
            low = np.asarray([r[f"{metric.removesuffix('_mean')}_ci95_low"] for r in values], dtype=float)
            high = np.asarray([r[f"{metric.removesuffix('_mean')}_ci95_high"] for r in values], dtype=float)
            ax.fill_between(x, low, high, color=colors[branch], alpha=.10)
        ax.axvline(0, color="black", linestyle="--", alpha=.45, linewidth=1)
        ax.text(.02, .97, "C change point", transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="#444444")
        ax.set_xlabel("C transitions assimilated")
        ax.set_title(title)
        ax.grid(alpha=.25)
    axes[0, 0].set_ylabel("MSE")
    axes[1, 0].set_ylabel("Posterior probability")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rwt/stage2b")
    parser.add_argument("--evaluation-seeds", default=",".join(str(x) for x in range(3200, 3220)))
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20261015)
    parser.add_argument("--max-state-norm", type=float, default=75.0)
    parser.add_argument("--phase", choices=("diagnostic", "learned", "all"), default="all")
    args = parser.parse_args()
    seeds = [int(x) for x in args.evaluation_seeds.split(",")]
    if len(seeds) < 20 or len(set(seeds)) != len(seeds):
        raise SystemExit("Stage 2B requires at least 20 unique evaluation seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in {"learned", "all"}:
        models, train_curves, manifest, train_data = train_unlabeled_models(
            args.output_dir, epochs=args.epochs, hidden_dim=args.hidden_dim, seed=args.train_seed,
        )
    else:
        models, train_curves, train_data = None, [], None
        manifest = {"C_excluded": True, "agent_models": "ground-truth A/B/D used only for diagnostic"}
    branches = ("diagnostic", "learned") if args.phase == "all" else (args.phase,)
    specs, alignments = get_specs(models, train_data, branches)
    all_summaries, all_traces, all_adaptation = [], [], []
    for si, seed in enumerate(seeds):
        for spec in specs:
            summary, traces, adaptation = run_episode(spec, seed, max_state_norm=args.max_state_norm)
            all_summaries.append(summary); all_traces.extend(traces); all_adaptation.extend(adaptation)
        if (si + 1) % 5 == 0:
            print(f"finished {si + 1}/{len(seeds)} paired evaluation seeds", flush=True)
    write_csv(args.output_dir / "training_curves.csv", train_curves)
    write_csv(args.output_dir / "episode_metrics.csv", all_summaries)
    write_csv(args.output_dir / "online_traces.csv", all_traces)
    write_csv(args.output_dir / "adaptation_curves.csv", all_adaptation)
    write_csv(args.output_dir / "adaptation_summary.csv", adaptation_summary(all_adaptation))
    write_csv(args.output_dir / "paired_comparisons.csv", paired_comparisons(all_summaries, (
        "cumulative_return", "known_regime_identification_accuracy",
        "novel_composition_identification_accuracy", "identification_delay_first_C",
        "identification_delay_return_C", "first_C_return", "return_C_return",
        "probe_actions_used", "probing_cost", "mean_prediction_mse", "mean_prediction_nll",
        "A_initial_prediction_mse", "A_recurrent_prediction_mse",
        "A_initial_identification_accuracy", "A_recurrent_identification_accuracy",
        "false_novel_alarm_rate", "unstable_termination",
        "prediction_mse_A", "prediction_mse_B", "prediction_mse_D", "prediction_mse_C",
        "identification_accuracy_A", "identification_accuracy_B", "identification_accuracy_D",
        "identification_accuracy_C", "first_C_prediction_mse", "recurrence_C_prediction_mse",
    )))
    write_csv(args.output_dir / "summary.csv", summarize(all_summaries,
        ("branch", "model", "policy"), (
            "cumulative_return", "known_regime_identification_accuracy",
            "novel_composition_identification_accuracy", "identification_delay_first_C",
            "identification_delay_return_C", "first_C_return", "return_C_return",
            "probe_actions_used", "probing_cost", "mean_prediction_mse", "mean_prediction_nll",
            "mean_posterior_entropy", "c_target_in_memory_at_return", "false_novel_alarm_rate",
            "A_initial_prediction_mse", "A_recurrent_prediction_mse",
            "A_initial_identification_accuracy", "A_recurrent_identification_accuracy",
            "prediction_mse_A", "prediction_mse_B", "prediction_mse_D", "prediction_mse_C",
            "identification_accuracy_A", "identification_accuracy_B", "identification_accuracy_D",
            "identification_accuracy_C", "first_C_prediction_mse", "recurrence_C_prediction_mse",
            "unstable_termination",
        )))
    manifest.update({
        "evaluation_seeds": seeds, "evaluation_schedule": [(s.regime, s.duration) for s in SEGMENTS],
        "probe_budget_ceiling": PROBING_BUDGET, "max_action_norm": MAX_ACTION_NORM,
        "noise_std": 0.025, "change_hazard": HAZARD,
        "adaptation_checkpoints": CHECKPOINTS,
        "training_model_selection_c_excluded": True,
        "models": ["oracle-seen diagnostic", "learned independent experts", "learned monolithic"],
        "completed_phase": args.phase,
    })
    (args.output_dir / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "evaluator_only_alignment.json").write_text(json.dumps(alignments, indent=2), encoding="utf-8")
    plot_adaptation(all_adaptation, args.output_dir / "stage2b_adaptation.png")
    print(f"Wrote Stage 2B results: {args.output_dir}")
    print(f"methods={len(specs)} paired_seeds={len(seeds)} transitions_per_episode={sum(x.duration for x in SEGMENTS)}")
    for row in summarize(all_summaries, ("branch", "model", "policy"), (
        "cumulative_return", "novel_composition_identification_accuracy",
        "identification_delay_first_C", "identification_delay_return_C",
        "probe_actions_used", "probing_cost", "mean_prediction_mse",
    )):
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
