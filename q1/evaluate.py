"""Held-out Q1 program inference, baseline, and rollout evaluation."""

from __future__ import annotations

import numpy as np
import torch

from q1.datasets import stage_pair
from q1.inference import rank_programs
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, apply_program, response


_CLASS_CACHE: dict[tuple, tuple[list[int], np.ndarray]] = {}


def _functional_classes_cached(params):
    key = (tuple(np.asarray(params.B).reshape(-1)), params.d, params.b, params.kappa)
    if key not in _CLASS_CACHE:
        from q1.oracle import functional_classes
        _CLASS_CACHE[key] = functional_classes(params)
    return _CLASS_CACHE[key][0]


def _support_tensor(states: np.ndarray, actions: np.ndarray, normalizer: TransitionNormalizer,
                    device: torch.device) -> torch.Tensor:
    dv = states[1:, 2:] - states[:-1, 2:]
    rows = np.concatenate((states[:-1], actions, dv), axis=1)
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
    return torch.cat((normalizer.normalize_x(tensor[:, :6]), normalizer.normalize_dv(tensor[:, 6:])), -1)[None]


def _public_transitions(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    dv = states[1:, 2:] - states[:-1, 2:]
    return np.concatenate((states[:-1], actions, dv), axis=1).astype(np.float32)


def _learned_program_mse(program, context, query, model, bank, normalizer, device):
    rows = _public_transitions(query.states, query.actions)
    x = torch.as_tensor(rows[:, :6], dtype=torch.float32, device=device)
    y = torch.as_tensor(rows[:, 6:], dtype=torch.float32, device=device)
    prediction = normalizer.denormalize_dv(model.executor(normalizer.normalize_x(x),
                                                          bank.apply_program(context, program)))
    return float(torch.mean((prediction - y) ** 2).cpu())


def _direct_context(context, adaptation: np.ndarray, model, normalizer, device,
                    *, predictive_variance: float = 1e-4,
                    steps: int = 40, regularization: float = 1e-3) -> torch.Tensor:
    x = torch.as_tensor(adaptation[:, :6], dtype=torch.float32, device=device)
    y = torch.as_tensor(adaptation[:, 6:], dtype=torch.float32, device=device)
    xn, yn = normalizer.normalize_x(x), normalizer.normalize_dv(y)
    adapted = context.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([adapted], lr=0.02)
    for _ in range(steps):
        prediction = normalizer.denormalize_dv(model.executor(xn, adapted))
        target = normalizer.denormalize_dv(yn)
        loss = torch.mean(0.5 * (np.log(2.0 * np.pi * predictive_variance)
                                 + (prediction - target) ** 2 / predictive_variance))
        loss = loss + regularization * torch.mean((adapted - context.detach()) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return adapted.detach()


@torch.no_grad()
def _rollout_rmse(query, context, program, model, bank, normalizer, device,
                  dt: float = 0.05, horizons=(5, 10, 20)):
    outputs = {}
    for horizon in horizons:
        state = query.states[0].copy()
        predictions = [state.copy()]
        for t in range(min(horizon, len(query.actions))):
            row = np.concatenate((state, query.actions[t])).astype(np.float32)
            xn = normalizer.normalize_x(torch.as_tensor(row[None], dtype=torch.float32, device=device))
            latent = bank.apply_program(context, program)
            dv = normalizer.denormalize_dv(model.executor(xn, latent))[0].cpu().numpy()
            velocity_next = state[2:] + dv
            state = np.concatenate((state[:2] + dt * velocity_next, velocity_next))
            predictions.append(state.copy())
        truth = query.states[:len(predictions)]
        outputs[str(horizon)] = float(np.sqrt(np.mean((np.asarray(predictions) - truth) ** 2)))
    return outputs


def _rank_true_program(before_params, adaptation: np.ndarray, dt: float):
    errors = []
    for program in PROGRAMS:
        params = apply_program(before_params, program)
        q, v = adaptation[:, :2], adaptation[:, 2:4]
        actions, observed = adaptation[:, 4:6], adaptation[:, 6:8]
        pred = np.tanh(actions @ params.B.T) - params.d * v - params.b * np.linalg.norm(v, axis=1, keepdims=True) * v - params.kappa * q
        errors.append(float(np.mean((dt * pred - observed) ** 2)))
    return PROGRAMS[int(np.argmin(errors))]


def choose_length_penalty(validation_cases, model, bank, normalizer, code_map, device,
                          candidates=(0.0, 0.1, 1.0, 10.0), predictive_variance: float = 1e-4) -> float:
    """Choose the length prior on validation backgrounds and permitted programs only."""
    from q1.oracle import functional_classes
    scores = {float(value): [] for value in candidates}
    inverse = {code: int(group) for group, code in code_map.items()}
    for before, old, adaptation, true_program in validation_cases:
        support = _support_tensor(old.states, old.actions, normalizer, device)
        context = model.encode(support)
        transitions = _public_transitions(adaptation.states, adaptation.actions)
        code_program = tuple(code_map[str(i)] for i in true_program)
        classes, _ = functional_classes(before)
        true_class = classes[PROGRAMS.index(tuple(true_program))]
        for value in candidates:
            ordered, _ = rank_programs(context, transitions, encoder=model.encoder,
                                       executor=model.executor, primitives=bank,
                                       normalizer=normalizer, length_penalty=float(value),
                                       predictive_variance=predictive_variance, device=device)
            predicted_physics = tuple(inverse[c] for c in ordered[0])
            scores[float(value)].append(
                classes[PROGRAMS.index(predicted_physics)] == true_class
            )
    return max(scores, key=lambda value: (float(np.mean(scores[value])), -value))


def evaluate_regime(cases, model: ContextDynamicsModel, bank: PrimitiveBank,
                    normalizer: TransitionNormalizer, group_to_code: dict[str, int],
                    length_penalty: float, device: torch.device,
                    *, direct_steps: int = 40, direct_regularization: float = 1e-3,
                    predictive_variance: float = 1e-4,
                    dt: float = 0.05,
                    regimes: tuple[str, ...] = ()) -> dict:
    inverse = {code: int(group) for group, code in group_to_code.items()}
    model.eval()
    bank.eval()
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    result: dict[str, dict] = {}
    for regime, before, old, adaptation, query, true_physics_program, n in cases:
        support = _support_tensor(old.states, old.actions, normalizer, device)
        context = model.encode(support)
        transitions = _public_transitions(adaptation.states[:n + 1], adaptation.actions[:n])
        ranked, scores = rank_programs(context, transitions, encoder=model.encoder,
                                       executor=model.executor, primitives=bank,
                                       normalizer=normalizer, length_penalty=length_penalty,
                                       predictive_variance=predictive_variance, device=device)
        predicted = ranked[0]
        true_code = tuple(group_to_code[str(code)] for code in true_physics_program)
        subprograms = [p for p in PROGRAMS if len(p) <= 1]
        # Same objective with composition candidates removed.
        transitions_t = torch.as_tensor(transitions, dtype=torch.float32, device=device)
        xn = normalizer.normalize_x(transitions_t[:, :6])
        yn = normalizer.normalize_dv(transitions_t[:, 6:])
        simple_costs = []
        for p in subprograms:
            pred = normalizer.denormalize_dv(model.executor(xn, bank.apply_program(context, p)))
            target = normalizer.denormalize_dv(yn)
            nll = 0.5 * (torch.log(torch.as_tensor(2.0 * torch.pi * predictive_variance,
                                                  dtype=pred.dtype, device=device))
                         + (pred - target) ** 2 / predictive_variance)
            simple_costs.append(torch.sum(nll) + length_penalty * len(p))
        single_pred = subprograms[int(torch.argmin(torch.stack(simple_costs)).item())]
        direct = _direct_context(context, transitions, model, normalizer, device,
                                 predictive_variance=predictive_variance,
                                 steps=direct_steps, regularization=direct_regularization)
        query_rows = _public_transitions(query.states, query.actions)
        query_t = torch.as_tensor(query_rows, dtype=torch.float32, device=device)
        query_pred = normalizer.denormalize_dv(model.executor(
            normalizer.normalize_x(query_t[:, :6]), bank.apply_program(context, predicted)))
        query_truth = query_t[:, 6:]
        m2_context_pred = normalizer.denormalize_dv(model.executor(
            normalizer.normalize_x(query_t[:, :6]), context))
        direct_pred = normalizer.denormalize_dv(model.executor(normalizer.normalize_x(query_t[:, :6]), direct))
        oracle_code_mse = _learned_program_mse(true_code, context, query, model, bank, normalizer, device)
        direct_mse = float(torch.mean((direct_pred - query_truth) ** 2).cpu())
        persistence_mse = float(torch.mean((m2_context_pred - query_truth) ** 2).cpu())
        actual_program_index = PROGRAMS.index(tuple(true_physics_program))
        class_ids = _functional_classes_cached(before)
        pred_physics = tuple(inverse[c] for c in predicted)
        accuracy = predicted == true_code
        class_accuracy = class_ids[PROGRAMS.index(pred_physics)] == class_ids[actual_program_index]
        true_oracle = _rank_true_program(before, transitions, 0.05)
        key = f"{regime}|N={n}"
        class_count = max(class_ids) + 1
        row = result.setdefault(key, {"count": 0, "identity_correct": 0, "class_correct": 0,
                                      "single_only_correct": 0, "oracle_identity_query_mse": [],
                                      "learned_search_query_mse": [], "direct_context_query_mse": [],
                                      "persistence_query_mse": [], "true_simulator_oracle_correct": 0,
                                      "rollout_rmse": {"5": [], "10": [], "20": []},
                                      "confusion_program_ids": np.zeros((len(PROGRAMS), len(PROGRAMS)), dtype=np.int64),
                                      "confusion_equivalence_classes": np.zeros((class_count, class_count), dtype=np.int64)})
        row["count"] += 1
        row["identity_correct"] += int(accuracy)
        row["class_correct"] += int(class_accuracy)
        row["single_only_correct"] += int(single_pred == true_code)
        row["confusion_program_ids"][PROGRAMS.index(true_code), PROGRAMS.index(predicted)] += 1
        row["confusion_equivalence_classes"][class_ids[actual_program_index],
                                              class_ids[PROGRAMS.index(pred_physics)]] += 1
        row["oracle_identity_query_mse"].append(oracle_code_mse)
        row["learned_search_query_mse"].append(float(torch.mean((query_pred - query_truth) ** 2).cpu()))
        row["direct_context_query_mse"].append(direct_mse)
        row["persistence_query_mse"].append(persistence_mse)
        row["true_simulator_oracle_correct"] += int(true_oracle == tuple(true_physics_program))
        if true_physics_program in ((0, 1), (1, 0)):
            code_01 = tuple(group_to_code[str(c)] for c in (0, 1))
            code_10 = tuple(group_to_code[str(c)] for c in (1, 0))
            probe_x = normalizer.normalize_x(torch.as_tensor(query_rows[:256, :6], dtype=torch.float32, device=device))
            response_01 = model.executor(probe_x, bank.apply_program(context, code_01))
            response_10 = model.executor(probe_x, bank.apply_program(context, code_10))
            row.setdefault("noncommuting_wrong_order_rmse", []).append(
                float(torch.sqrt(torch.mean((response_01 - response_10) ** 2)).cpu()))
        rollout = _rollout_rmse(query, context, predicted, model, bank, normalizer, device, dt=dt)
        for horizon, value in rollout.items():
            row["rollout_rmse"][horizon].append(value)
    for row in result.values():
        count = row["count"]
        for key in ("identity_correct", "class_correct", "single_only_correct", "true_simulator_oracle_correct"):
            row[key.replace("_correct", "_accuracy")] = row.pop(key) / count
        for key in ("oracle_identity_query_mse", "learned_search_query_mse", "direct_context_query_mse", "persistence_query_mse"):
            values = row[key]
            row[key] = float(np.mean(values))
        row["rollout_rmse"] = {h: float(np.mean(v)) for h, v in row["rollout_rmse"].items()}
        row["noncommuting_wrong_order_rmse"] = float(np.mean(row.get("noncommuting_wrong_order_rmse", [0.0])))
        row["confusion_program_ids"] = row["confusion_program_ids"].tolist()
        row["confusion_equivalence_classes"] = row["confusion_equivalence_classes"].tolist()
    return result
