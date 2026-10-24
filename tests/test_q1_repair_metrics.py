import inspect
from types import SimpleNamespace

import numpy as np
import torch

from q1.datasets import stage_pair
from q1.evaluate import _rank_true_program, _score_program_prediction
from q1.model import TransitionNormalizer
from q1.oracle import functional_classes
from q1.primitives import map_intervention_program
from q1.simulator import PROGRAMS, DynamicsParams, apply_program
from q1.train_dynamics import _fixed_validation_query_indices
from q1.train_dynamics import Task, _fit_models, _transitions
from q1.datasets import rollout
from q1.train_primitives import _functional_distances


class _CodeBank:
    def apply_one(self, context, code):
        addition = torch.full_like(context, float(code + 1))
        return context + addition


def _known_executor(x, context):
    if context.ndim == 2 and x.ndim == 2 and context.shape[0] == 1:
        context = context.expand(x.shape[0], -1)
    return x[:, :2] * context[:, :2]


def test_functional_distances_normalize_probes_and_report_physical_units():
    normalizer = TransitionNormalizer(
        x_mean=np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        x_std=np.array([2, 3, 4, 5, 6, 7], dtype=np.float32),
        dv_mean=np.array([0.5, -0.25], dtype=np.float32),
        dv_std=np.array([2, 4], dtype=np.float32),
    )
    raw_common = np.array([[3, 5, 0, 0, 1, 2], [5, 8, 0, 0, -1, 0]], dtype=np.float32)
    natural_x = normalizer.normalize_x(torch.tensor(raw_common))
    prepared = [(SimpleNamespace(), torch.zeros(1, 2), natural_x,
                 torch.zeros(2, 2))]
    actual = _functional_distances(prepared, _CodeBank(), _known_executor, normalizer,
                                   seed=0, device=torch.device("cpu"), common_probes=raw_common)
    normalized = normalizer.normalize_x(torch.tensor(raw_common))
    manual_outputs = [normalizer.denormalize_dv(normalized[:, :2] * (code + 1))
                      for code in range(4)]
    manual_distance = torch.sqrt(torch.mean((manual_outputs[0] - manual_outputs[1]) ** 2)).item()
    assert np.isclose(actual["common_probe_rmse_delta_v"][0, 1], manual_distance)
    assert np.isclose(actual["validation_visited_rmse_delta_v"][0, 1], manual_distance)
    assert actual["common_probe_count"] == 2


def test_physical_oracle_identity_and_functional_class_scores_are_separate():
    params = DynamicsParams(np.array([[0.92, 0.06], [-0.04, 0.88]]), 0.34, 0.11, 0.23)
    classes, _ = functional_classes(params)
    for left, right in (((0, 2), (2, 0)), ((1, 3), (3, 1))):
        left_i, right_i = PROGRAMS.index(left), PROGRAMS.index(right)
        assert classes[left_i] == classes[right_i]
        score = _score_program_prediction(params, left, right)
        assert score == {"identity_correct": False, "functional_class_correct": True}


def test_physical_oracle_stage_pair_ranking_matches_m1_class_convention():
    params = DynamicsParams(np.array([[0.92, 0.06], [-0.04, 0.88]]), 0.34, 0.11, 0.23)
    for program in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)):
        _, adaptation, _ = stage_pair(params, program, seed=32, n_support=30,
                                      n_query=100, sigma=0)
        rows = np.concatenate((adaptation.states[:-1], adaptation.actions,
                               adaptation.delta_v), axis=1)
        predicted = _rank_true_program(params, rows, 0.05)
        score = _score_program_prediction(params, program, predicted)
        assert score["functional_class_correct"]


def test_declared_pair_split_and_opaque_code_mapping_are_consistent():
    import yaml
    split = yaml.safe_load(open("configs/q1_splits.yaml", encoding="utf-8"))
    training = {tuple(pair) for pair in split["composition_training_pairs"]}
    held_out = {tuple(pair) for pair in split["composition_held_out_pairs"]}
    assert (0, 1) in held_out and (1, 0) in held_out
    assert not (training & held_out)
    code_map = {"0": 2, "1": 0, "2": 3, "3": 1}
    assert map_intervention_program((0, 1, 3), code_map) == (2, 0, 1)
    assert len(set(code_map.values())) == 4


def test_inference_api_does_not_accept_physics_or_intervention_labels():
    from q1.inference import rank_programs
    signature = inspect.signature(rank_programs)
    assert not {"params", "before_params", "true_program", "group", "intervention"} & set(signature.parameters)


def test_m2_validation_query_indices_are_fixed_and_use_all_transitions():
    tasks = [SimpleNamespace(query=np.zeros((96, 8))), SimpleNamespace(query=np.zeros((73, 8)))]
    first = _fixed_validation_query_indices(tasks)
    second = _fixed_validation_query_indices(tasks)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert first[0].tolist() == list(range(96))
    assert first[1].tolist() == list(range(73))
    assert PROGRAMS.index((0, 1)) != PROGRAMS.index((1, 0))


def test_m2_fit_selects_best_epoch_on_full_fixed_validation_queries():
    params = DynamicsParams(np.array([[0.9, 0.05], [-0.03, 0.85]]), 0.34, 0.11, 0.23)
    train = []
    valid = []
    for idx in range(2):
        support = _transitions(rollout(params, n=12, seed=10 + idx))
        query = _transitions(rollout(params, n=7, seed=20 + idx))
        train.append(Task(support, query, np.zeros((8, 4)), np.zeros((7, 2)), params, f"train-{idx}"))
    for idx in range(2):
        support = _transitions(rollout(params, n=12, seed=30 + idx))
        query = _transitions(rollout(params, n=9, seed=40 + idx))
        valid.append(Task(support, query, np.zeros((10, 4)), np.zeros((9, 2)), params, f"val-{idx}"))
    normalizer = TransitionNormalizer.fit([t.support for t in train] + [t.query for t in train])
    config = {"model": {"context_dim": 4, "hidden_dim": 8, "depth": 1, "dropout": 0.0,
                        "predictive_variance": 1e-4},
              "training": {"dynamics_epochs": 2, "batch_size": 2,
                           "learning_rate": 1e-3}}
    _, _, history, selected = _fit_models(train, valid, normalizer, config, 7, torch.device("cpu"))
    expected = min(history, key=lambda row: row["validation_mse_delta_v"])
    assert selected["validation_task_count"] == 2
    assert selected["validation_transition_count"] == 18
    assert selected["context_best_epoch"] == expected["epoch"]
    assert selected["final_epoch_metrics"]["validation_mse_delta_v"] == history[-1]["validation_mse_delta_v"]
