import numpy as np
import pytest
import torch

from rwt.dynamics import default_regimes
from rwt.identifiability import (
    composition_closure,
    enumerate_hmm_log_evidence,
    fit_three_code_assignment,
    hmm_filter_step,
    hmm_log_evidence,
)
from rwt.stage2a import LinearStructuredDynamics, load_checkpoint, save_checkpoint


def test_ground_truth_dynamics_have_zero_additive_composition_closure():
    regimes = {name: regime.parameters for name, regime in default_regimes().items()}
    rng = np.random.default_rng(177)
    state = rng.normal(size=(50, 4))
    action = rng.normal(size=(50, 2)) * 0.75
    residual, relative = composition_closure(regimes, state, action)
    assert np.max(np.abs(residual)) < 1e-12
    assert np.max(relative) < 1e-12


def test_two_latent_assignments_fit_abd_exactly_but_predict_different_11():
    observed = {"A": np.asarray([0.0]), "B": np.asarray([1.0]), "D": np.asarray([3.0])}
    map_one = {"A": (0, 0), "B": (1, 0), "D": (0, 1)}
    map_two = {"B": (0, 0), "A": (1, 0), "D": (0, 1)}
    reconstructed1, code11_1 = fit_three_code_assignment(observed, map_one)
    reconstructed2, code11_2 = fit_three_code_assignment(observed, map_two)
    for name in observed:
        np.testing.assert_allclose(reconstructed1[name], observed[name])
        np.testing.assert_allclose(reconstructed2[name], observed[name])
    assert not np.allclose(code11_1, code11_2)
    np.testing.assert_allclose(code11_1, [4.0])
    np.testing.assert_allclose(code11_2, [2.0])


def test_hmm_forward_algorithm_matches_exact_path_enumeration():
    ll = torch.tensor([[-0.1, -1.5, -3.0], [-2.0, -0.2, -1.0], [-0.3, -1.0, -0.5]], dtype=torch.float64)
    actual = hmm_log_evidence(ll, hazard=0.08)
    exact = enumerate_hmm_log_evidence(ll, hazard=0.08)
    torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)


def test_hmm_forward_has_gradients_and_filter_is_normalized():
    ll = torch.randn(7, 3, requires_grad=True)
    loss = -hmm_log_evidence(ll, hazard=0.05)
    loss.backward()
    assert ll.grad is not None and torch.isfinite(ll.grad).all()
    posterior = hmm_filter_step(np.asarray([0.8, 0.1, 0.1]), np.asarray([-4.0, -0.1, -3.0]), 0.05)
    assert posterior.sum() == pytest.approx(1.0)
    assert posterior[1] > posterior[0]


def test_linear_structural_model_has_expected_composed_output_shapes():
    model = LinearStructuredDynamics()
    state, action = torch.randn(9, 4), torch.randn(9, 2)
    codes = model.predict_codes(state, action)
    assert codes.shape == (9, 4, 4)
    torch.testing.assert_close(codes[:, 3], codes[:, 1] + codes[:, 2] - codes[:, 0])
    loss = model.log_prob_codes(state, action, torch.randn(9, 4))[:, :3].mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_linear_structural_checkpoint_round_trip_preserves_predictions(tmp_path):
    model = LinearStructuredDynamics()
    rng = np.random.default_rng(12)
    model.fit_input_normalization(rng.normal(size=(20, 4)), rng.normal(size=(20, 2)))
    state, action = torch.randn(5, 4), torch.randn(5, 2)
    before = model.predict_codes(state, action)
    checkpoint = tmp_path / "linear_structured.pt"
    save_checkpoint(checkpoint, model, model_kind="linear", config={"state_dim": 4, "action_dim": 2})
    loaded, _ = load_checkpoint(checkpoint)
    after = loaded.predict_codes(state, action)
    torch.testing.assert_close(after, before)
