import inspect

import numpy as np
import torch

from q1.datasets import rollout
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.simulator import DynamicsParams
from q1.train_dynamics import _transitions


def test_neural_transition_tensor_contains_only_public_values():
    params = DynamicsParams(np.array([[0.9, 0.1], [-0.1, 0.8]]), 0.35, 0.1, 0.22)
    values = _transitions(rollout(params, n=8, seed=4))
    assert values.shape == (8, 8)
    assert np.isfinite(values).all()


def test_learned_primitive_source_has_no_simulator_dependency():
    import q1.primitives as module
    source = inspect.getsource(module)
    assert "q1.simulator" not in source
    assert "intervention(" not in source


def test_all_21_candidates_rank_from_public_adaptation_only():
    model = ContextDynamicsModel(context_dim=8, hidden_dim=16)
    bank = PrimitiveBank(context_dim=8, hidden_dim=16)
    values = np.random.default_rng(2).normal(size=(6, 8)).astype(np.float32)
    normalizer = TransitionNormalizer(np.zeros(6), np.ones(6), np.zeros(2), np.ones(2))
    context = torch.zeros(1, 8)
    from q1.inference import rank_programs
    ordered, costs = rank_programs(context, values, encoder=model.encoder,
                                   executor=model.executor, primitives=bank,
                                   normalizer=normalizer, predictive_variance=1e-4)
    assert len(ordered) == 21
    assert costs.shape == (21,)
    assert torch.isfinite(costs).all()
