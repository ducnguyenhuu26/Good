import numpy as np
import torch

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.inference_dp import AgeLawFilter
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.support import DisjointPartition
from experiments.rwt.clsm.run_train import validate_r2_checkpoint


def _schema():
    return EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2), np.ones(2),
                     -np.ones(2), np.ones(2), .01)


def test_train_only_profile_scores_all_finite_support_candidates():
    path = "experiments/rwt/clsm/configs/cpu_smoke.yaml"
    config = load_config(path)
    model = build_clsm(_schema(), config).law
    obs = torch.tensor([[0.1, .2, -.1], [.2, .1, -.2]], dtype=torch.float64)
    action = torch.tensor([[.5, -.3], [.4, .2]], dtype=torch.float64)
    worlds = [(obs, action, obs + .01), (obs * .7, action.flip(0), obs * .7 + .02)]
    candidates = DisjointPartition.enumerate(2)
    assert len(candidates) == 1
    score = DisjointPartition.profile_score(candidates[0], worlds, model, gh_nodes=12)
    assert torch.isfinite(score)


def test_filter_observation_then_evolution_preserves_exact_mass():
    config = load_config("experiments/rwt/clsm/configs/cpu_smoke.yaml")
    model = build_clsm(_schema(), config)
    filt = model.filter
    obs = torch.zeros(3, dtype=torch.float64)
    action = torch.tensor([.1, -.2], dtype=torch.float64)
    nxt = torch.full((3,), .01, dtype=torch.float64)
    result = filt.observe(obs, action, nxt)
    assert np.isfinite(float(result["predictive_log_likelihood"]))
    assert torch.isclose(filt.log_code_age.exp().sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12)
    filt.evolve()
    assert torch.isclose(filt.log_code_age.exp().sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12)
    assert filt.time == 1


def test_old_checkpoint_revision_is_explicitly_rejected():
    config = load_config("experiments/rwt/clsm/configs/cpu_smoke.yaml")
    assert config["architecture_revision"] == 2
    import pytest
    with pytest.raises(ValueError, match="architecture_revision mismatch"):
        validate_r2_checkpoint({"architecture_revision": 1, "law": {}}, config)


def test_oracle_diagnostic_checkpoint_rejected_by_deployable_loader():
    import pytest
    with pytest.raises(ValueError, match="oracle_diagnostic"):
        validate_r2_checkpoint({"architecture_revision": 2, "role": "oracle_diagnostic"})
