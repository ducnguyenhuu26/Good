import numpy as np
import torch

from rwt.clsm.belief import LawFilter
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.evolution import LawEvolution
from rwt.clsm.memory import TheoryMemory
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.train import approximate_estep


def test_public_sequence_estep_event_masses_and_timing():
    schema = EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2),
                       np.ones(2), -np.ones(2), np.ones(2), .01)
    model = ExecutableLaw(schema, hidden_dim=8)
    filt = LawFilter(model, LawEvolution(), TheoryMemory(), budget=128)
    obs = torch.zeros((2, 3), dtype=torch.float64)
    action = torch.ones((2, 2), dtype=torch.float64)
    next_obs = torch.zeros((2, 3), dtype=torch.float64)
    result = approximate_estep(filt, obs, action, next_obs)
    assert result.log_predictive.shape == (2,)
    assert torch.isfinite(result.log_predictive).all()
    assert len(result.ancestry) == 1
    assert len(result.gamma) == 2
    assert all(abs(float(x.sum()) - 1) < 1e-10 for x in result.gamma)
    assert abs(sum(float(x.sum()) for x in result.smoothed.event_xi[0].values()) - 1) < 1e-10
    assert filt.time == 1 and filt.needs_evolution
    assert result.max_dropped_mass < 1e-3
