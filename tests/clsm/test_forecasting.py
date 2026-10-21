import numpy as np
import torch

from rwt.clsm.belief import LawFilter
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.evolution import LawEvolution
from rwt.clsm.forecast import LawForecaster
from rwt.clsm.memory import TheoryMemory
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.scoring import energy_score


def setup():
    schema = EnvSchema("toy",3,2,(1,2),(0,1),np.eye(2),np.ones(2),-np.ones(2),np.ones(2),.01)
    law = ExecutableLaw(schema,hidden_dim=8)
    evolution = LawEvolution()
    memory = TheoryMemory()
    return law,evolution,memory,LawFilter(law,evolution,memory)


def test_forecast_h_zero_and_frozen_comparator():
    law,evolution,memory,filt = setup()
    forecast = LawForecaster(law,evolution,memory,samples=128)
    seed = 32
    draws = forecast.sample_laws(filt.predict_current(),[0,1,8],np.random.default_rng(seed),frozen=True)
    assert torch.equal(draws[0],draws[1]) and torch.equal(draws[1],draws[8])
    assert len(set(draws[0][:,0].tolist())) > 1  # Distribution, not point mean.


def test_forecast_rng_isolation():
    law,evolution,memory,filt = setup()
    forecast = LawForecaster(law,evolution,memory,samples=16)
    before = filt.state_dict()
    probes = (torch.zeros((2,3),dtype=torch.float64),torch.ones((2,2),dtype=torch.float64))
    result = forecast.sample_signatures(filt.predict_current(),probes,[1,8],np.random.default_rng(3))
    after = filt.state_dict()
    assert result[8].shape == (16,6)
    assert before["time"]==after["time"]
    for c,d in zip(before["components"],after["components"]):
        assert torch.equal(c["log_weight"],d["log_weight"])
        assert torch.equal(c["mean"],d["mean"])


def test_energy_score_unbiased_pair_term():
    samples = torch.tensor([[0.],[2.]],dtype=torch.float64)
    assert torch.allclose(energy_score(samples,torch.tensor([0.],dtype=torch.float64)),torch.tensor(0.,dtype=torch.float64))
