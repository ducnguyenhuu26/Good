import inspect

import numpy as np
import pytest

from experiments.rwt.run_stage2b import SEGMENTS
from rwt.stage2b import (
    OnlineStructuralAgent,
    OnlineTheoryFilter,
    PredictionSet,
    enumerate_completions,
    expected_information_gain,
    hypothesis_log_likelihoods,
)


def test_completion_enumeration_contains_all_three_anchor_choices():
    seen = np.asarray([[1.0, 0.0], [3.0, 1.0], [5.0, 2.0]])
    completions, _ = enumerate_completions(seen)
    np.testing.assert_allclose(completions, [[7.0, 3.0], [3.0, 1.0], [-1.0, -1.0]])


def test_bayesian_filter_updates_only_after_observation_and_caches_confident_theory():
    names = ("seen0", "seen1", "seen2", "comp0", "comp1", "comp2")
    belief = OnlineTheoryFilter(names, 3, (3, 4, 5), hazard=0.01, remember_threshold=0.8)
    before = belief.posterior.copy()
    predicted_prior = belief.predict_weights()
    np.testing.assert_array_equal(belief.posterior, before)
    likelihood = np.asarray([-20.0, -20.0, -20.0, 0.0, -25.0, -25.0])
    posterior = belief.update(likelihood)
    assert posterior[3] > 0.999
    assert 3 in belief.remembered
    assert belief.base_prior()[3] > 0
    # With cached theories the change-point prior restores more mass to comp0.
    cached_prior = belief.predict_weights()
    uncached = OnlineTheoryFilter(names, 3, (3, 4, 5), hazard=0.01, use_memory=False)
    uncached.posterior = belief.posterior.copy()
    assert cached_prior[3] > uncached.predict_weights()[3]
    assert predicted_prior.sum() == pytest.approx(1.0)


def test_eig_uses_candidate_predictive_distributions_and_is_zero_for_identical_models():
    separated = PredictionSet(
        names=("left", "right"), means=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        scales=np.full((2, 2), 0.1), known_count=2, completion_indices=(),
    )
    identical = PredictionSet(
        names=("left", "right"), means=np.zeros((2, 2)),
        scales=np.full((2, 2), 0.1), known_count=2, completion_indices=(),
    )
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(2, 64, 2))
    prior = np.asarray([0.5, 0.5])
    assert expected_information_gain(separated, prior, noise) > 0.5
    assert expected_information_gain(identical, prior, noise) == 0.0


def test_online_likelihood_normalizes_and_has_expected_velocity_dimension():
    predictions = PredictionSet(
        names=("a", "b"), means=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        scales=np.full((2, 2), 0.1), known_count=2, completion_indices=(),
    )
    likelihood = hypothesis_log_likelihoods(predictions, np.asarray([0.95, 0.02]))
    assert likelihood.shape == (2,)
    assert likelihood[1] > likelihood[0]


def test_agent_interface_has_no_regime_or_change_schedule_input():
    assert "regime" not in str(inspect.signature(OnlineStructuralAgent.__init__)).lower()
    assert "schedule" not in str(inspect.signature(OnlineStructuralAgent.__init__)).lower()
    assert "regime" not in str(inspect.signature(OnlineStructuralAgent.act)).lower()
    assert "regime" not in str(inspect.signature(OnlineStructuralAgent.observe)).lower()
    assert [segment.regime for segment in SEGMENTS] == ["A", "B", "D", "C", "A", "C"]


def test_random_probe_actions_are_paired_across_different_bank_sizes():
    small = PredictionSet(("x", "y"), np.asarray([[0.0, 0.0], [0.2, 0.0]]),
                          np.full((2, 2), 0.1), 2, ())
    large = PredictionSet(("x", "y", "z"), np.asarray([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]]),
                          np.full((3, 2), 0.1), 3, ())
    left = OnlineStructuralAgent(lambda s, a: small, names=small.names, known_count=2,
                                 completion_indices=(), policy="random", seed=99, horizon=64)
    right = OnlineStructuralAgent(lambda s, a: large, names=large.names, known_count=3,
                                  completion_indices=(), policy="random", seed=99, horizon=64)
    state = np.asarray([0.0, 0.0, 0.1, -0.1])
    for step in range(64):
        left_action = left.act(state, step)
        right_action = right.act(state, step)
        np.testing.assert_array_equal(left_action, right_action)
        left.observe(state, left_action, state)
        right.observe(state, right_action, state)
