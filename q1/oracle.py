"""Oracle-only program discrimination diagnostics for the M1 gate."""

from __future__ import annotations

import numpy as np

from q1.datasets import rollout
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, response, transition


def _batch_delta_v(states: np.ndarray, actions: np.ndarray, params: DynamicsParams, dt: float) -> np.ndarray:
    q, v = states[:-1, :2], states[:-1, 2:]
    acceleration = (np.tanh(actions @ params.B.T) - params.d * v
                    - params.b * np.linalg.norm(v, axis=1, keepdims=True) * v
                    - params.kappa * q)
    return dt * acceleration


def functional_classes(params: DynamicsParams | list[DynamicsParams], *, dt: float = 0.05,
                      tolerance: float = 1e-7, seed: int = 9182) -> tuple[list[int], np.ndarray]:
    """Partition programs by response on a fixed evaluation-only probe set."""
    rng = np.random.default_rng(seed)
    probes = np.column_stack((rng.uniform(-0.7, 0.7, (256, 2)),
                              rng.uniform(-0.5, 0.5, (256, 2)),
                              rng.uniform(-1, 1, (256, 2))))
    backgrounds = params if isinstance(params, list) else [params]
    signatures = [np.concatenate([response(apply_program(background, p), probes, dt=dt)
                                  for background in backgrounds]) for p in PROGRAMS]
    class_ids: list[int] = []
    representatives: list[np.ndarray] = []
    for signature in signatures:
        match = next((i for i, other in enumerate(representatives)
                      if np.max(np.abs(signature - other)) <= tolerance), None)
        if match is None:
            match = len(representatives)
            representatives.append(signature)
        class_ids.append(match)
    distances = np.zeros((len(PROGRAMS), len(PROGRAMS)), dtype=np.float64)
    for i, left in enumerate(signatures):
        for j, right in enumerate(signatures):
            distances[i, j] = np.sqrt(np.mean((left - right) ** 2))
    return class_ids, distances


def oracle_discrimination(params: DynamicsParams | list[DynamicsParams], *, seed: int, budgets=(5, 10, 20, 50, 100),
                         cases_per_program: int = 4, dt: float = 0.05,
                         sigma: float = 0.0, initial_q_std: float = 0.25,
                         initial_v_std: float = 0.25, action_limit: float = 1.0,
                         action_signal_scale: float = 0.85, action_jitter: float = 0.15) -> dict:
    """Score all 21 physical programs on the same trajectories at each budget."""
    programs = list(PROGRAMS)
    backgrounds = params if isinstance(params, list) else [params]
    classes, distances = functional_classes(backgrounds, dt=dt)
    n_classes = max(classes) + 1
    reports = {}
    for budget in budgets:
        matrix = np.zeros((len(programs), len(programs)), dtype=np.int64)
        margins = []
        order_correct = {"(U1,U2)": [], "(U2,U1)": []}
        for true_i, program in enumerate(programs):
            for case in range(cases_per_program):
                background = backgrounds[case % len(backgrounds)]
                trajectory = rollout(apply_program(background, program), n=budget,
                                     seed=seed + case, dt=dt, sigma=sigma,
                                     initial_q_std=initial_q_std, initial_v_std=initial_v_std,
                                     action_limit=action_limit, action_signal_scale=action_signal_scale,
                                     action_jitter=action_jitter)
                errors = []
                for candidate in programs:
                    candidate_params = apply_program(background, candidate)
                    pred = _batch_delta_v(trajectory.states, trajectory.actions, candidate_params, dt)
                    errors.append(float(np.mean((pred - trajectory.delta_v) ** 2)))
                order = np.argsort(errors)
                predicted = int(order[0])
                # If candidates are functionally equivalent, choose the true equivalence class.
                true_class = classes[true_i]
                best_class_candidate = next((int(i) for i in order if classes[int(i)] == true_class), predicted)
                matrix[true_i, predicted] += 1
                margins.append(float(errors[order[1]] - errors[order[0]]))
                if program == (0, 1):
                    order_correct["(U1,U2)"].append(classes[predicted] == true_class)
                elif program == (1, 0):
                    order_correct["(U2,U1)"].append(classes[predicted] == true_class)
        class_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        for i in range(len(programs)):
            for j in range(len(programs)):
                class_matrix[classes[i], classes[j]] += matrix[i, j]
        raw_total = int(matrix.sum())
        raw_correct = int(np.trace(matrix))
        class_total = int(class_matrix.sum())
        class_correct = int(np.trace(class_matrix))
        reports[str(budget)] = {
            "confusion_matrix_program_ids": matrix.tolist(),
            "confusion_matrix_equivalence_classes": class_matrix.tolist(),
            "identity_accuracy": raw_correct / raw_total,
            "functional_class_accuracy": class_correct / class_total,
            "mean_best_second_margin_mse": float(np.mean(margins)),
            "median_best_second_margin_mse": float(np.median(margins)),
            "noncommuting_order_class_accuracy": {
                key: float(np.mean(values)) if values else None
                for key, values in order_correct.items()
            },
        }
    return {
        "programs": [list(p) for p in programs],
        "functional_class_ids": classes,
        "equivalence_tolerance": 1e-7,
        "probe_count": 256,
        "pairwise_functional_rmse": distances.tolist(),
        "budgets": reports,
        "noncommuting_probe_rmse_u1_u2_vs_u2_u1": float(
            distances[PROGRAMS.index((0, 1)), PROGRAMS.index((1, 0))]
        ),
    }
