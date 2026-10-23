import numpy as np
import pytest

from q1.datasets import rollout, stage_pair
from q1.simulator import DynamicsParams, PROGRAMS, apply_program, intervention, transition


def params():
    return DynamicsParams(np.array([[0.9, 0.1], [-0.05, 0.8]]), 0.35, 0.1, 0.22)


def test_deterministic_transition_uses_new_velocity_for_position():
    p = params()
    state = np.array([0.2, -0.1, 0.3, -0.2])
    action = np.array([0.5, -0.3])
    nxt = transition(state, action, p, dt=0.05)
    q, v = state[:2], state[2:]
    u = np.tanh(p.B @ action) - p.d * v - p.b * np.linalg.norm(v) * v - p.kappa * q
    v_next = v + 0.05 * u
    np.testing.assert_allclose(nxt[2:], v_next)
    np.testing.assert_allclose(nxt[:2], q + 0.05 * v_next)


def test_seeded_noise_is_reproducible_and_position_tracks_noisy_velocity():
    p = params()
    state = np.zeros(4)
    action = np.array([0.2, 0.7])
    a = transition(state, action, p, sigma=0.02, rng=np.random.default_rng(3))
    b = transition(state, action, p, sigma=0.02, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:2], 0.05 * a[2:])


def test_intervention_composition_order_and_noncommutativity():
    p = params()
    actual_01 = apply_program(p, (0, 1)).B
    actual_10 = apply_program(p, (1, 0)).B
    rotation = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                         [np.sin(np.pi/6), np.cos(np.pi/6)]])
    diagonal = np.diag([1.5, 1.0])
    np.testing.assert_allclose(actual_01, diagonal @ rotation @ p.B)
    np.testing.assert_allclose(actual_10, rotation @ diagonal @ p.B)
    np.testing.assert_allclose(actual_01, intervention(intervention(p, 0), 1).B)
    assert np.linalg.norm(actual_01 - actual_10) > 1e-3
    assert len(PROGRAMS) == 21


def test_stage_pair_has_no_adaptation_reset_and_independent_query():
    old, adaptation, query = stage_pair(params(), (0,), seed=11, n_support=8, n_query=6)
    np.testing.assert_array_equal(adaptation.states[0], old.states[-1])
    assert not np.array_equal(query.states[0], adaptation.states[0])
    assert adaptation.states.shape == (7, 4)


def test_parameter_validation():
    with pytest.raises(ValueError):
        DynamicsParams(np.zeros((2, 2)), 0.3, 0.1, 0.2)
    with pytest.raises(ValueError):
        transition(np.zeros(3), np.zeros(2), params())
