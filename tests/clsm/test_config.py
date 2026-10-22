from pathlib import Path

import numpy as np
import pytest

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import EnvSchema


def test_revision2_hyperparameters_are_resolved_and_applied():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "experiments/rwt/clsm/configs/cpu_smoke.yaml")
    assert config["architecture_revision"] == 2
    assert config["filter"]["age_representation"] == "exact_log_dp"
    assert config["filter"]["max_world_length"] == 500
    assert config["forecast"]["horizons"] == [1, 8, 24, 64]
    schema = EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2), np.ones(2),
                       -np.ones(2), np.ones(2), .01)
    model = build_clsm(schema, config)
    assert model.law.net[0].out_features == 256
    assert model.filter.log_code_age.shape == (8, 501)
    assert model.forecaster.samples == 256


def test_configuration_rejects_unversioned_legacy_config(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("method: clsm\ntrack: comp\nseed: 41\n", encoding="utf-8")
    with pytest.raises(ValueError, match="architecture_revision: 2"):
        load_config(path)


def test_configuration_rejects_unknown_revision2_hyperparameter(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("method: clsm\narchitecture_revision: 2\nnetwork: {made_up: 1}\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="unknown CLSM-R2 hyperparameter"):
        load_config(path)
