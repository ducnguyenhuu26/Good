from pathlib import Path

import numpy as np

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import EnvSchema


def test_spec_hyperparameters_resolved_and_applied():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "experiments/rwt/clsm/configs/cpu_smoke.yaml")
    assert config["network"]["hidden_dim"] == 256
    assert config["filter"]["components"] == 128
    assert config["training"]["outer_gem_iterations"] == 1
    assert config["forecast"]["horizons"] == [1,8,24,64]
    schema = EnvSchema("toy",3,2,(1,2),(0,1),np.eye(2),np.ones(2),-np.ones(2),np.ones(2),.01)
    model = build_clsm(schema,config)
    assert model.law.mechanics.delta_min == config["mechanisms"]["delta_min"]
    assert model.law.net[0].out_features == 256
    assert model.filter.budget == 128
    assert model.forecaster.samples == 256
