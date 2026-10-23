import pytest
import torch

from q1.model import ContextDynamicsModel, DynamicsExecutor
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS


def test_encoder_executor_and_primitive_shapes():
    model = ContextDynamicsModel(context_dim=12, hidden_dim=32)
    support = torch.zeros(3, 16, 8)
    x = torch.zeros(3, 6)
    context = model.encode(support)
    assert context.shape == (3, 12)
    assert model.predict(x, context).shape == (3, 2)
    primitive = PrimitiveBank(context_dim=12)
    transformed = primitive.apply_program(context, (0, 1))
    assert transformed.shape == context.shape
    assert len(PROGRAMS) == 21


def test_executor_rejects_wrong_feature_shape():
    executor = DynamicsExecutor(context_dim=4, hidden_dim=8)
    with pytest.raises(ValueError):
        executor(torch.zeros(2, 5), torch.zeros(2, 4))
