"""CPU validation for RWT Stage 1.5 inference and probing contracts."""

import inspect

import numpy as np

from rwt.agents import (
    BayesianTheoryFilter,
    BeliefActivePolicy,
    PassiveBayesPolicy,
    PureInformationGainPolicy,
    RandomProbePolicy,
    evenly_spaced_probe_steps,
    make_candidate_library,
    probe_action_set,
)
from rwt.diagnostics import transition_kl
from rwt.dynamics import CandidateLibrary, default_regimes
from rwt.environment import RecurrentDynamicsEnv
from rwt.schedule import RegimeSchedule, Segment, make_schedule


def test_schedule_recurrence_and_held_out_composition():
    train = make_schedule("train", 7, min_dwell=2, max_dwell=5)
    evaluation = make_schedule("evaluation", 7, min_dwell=2, max_dwell=5)
    assert [segment.regime for segment in train.segments] == ["A", "B", "A", "D", "B"]
    assert all(segment.regime != "C" for segment in train.segments)
    assert [segment.regime for segment in evaluation.segments] == ["A", "B", "A", "D", "C", "B"]
    assert len({segment.duration for segment in evaluation.segments}) > 1
    assert make_schedule("evaluation", 7, min_dwell=2, max_dwell=5) == evaluation

    regimes = default_regimes()
    assert regimes["C"].parameters.friction == regimes["B"].parameters.friction
    assert regimes["C"].parameters.rotation == regimes["D"].parameters.rotation
    assert regimes["C"].parameters.gain == regimes["A"].parameters.gain


def test_all_four_dynamics_primitives_are_explicit():
    params = CandidateLibrary().parameters()
    assert any(item.friction != 0.12 for item in params)
    assert any(item.gain != 1.0 for item in params)
    assert any(item.rotation != 0.0 for item in params)
    assert any(item.external_force != (0.0, 0.0) for item in params)
    state = np.asarray([0.0, 0.0, 0.5, -0.25])
    action = np.asarray([0.4, -0.2])
    item = next(p for p in params if p.friction == 0.48 and p.gain == 1.35 and p.rotation != 0 and p.external_force != (0.0, 0.0))
    next_state = item.predict_state(state, action)
    assert np.allclose(next_state[2:], item.predict_velocity(state[2:], action))
    assert np.allclose(next_state[:2], state[:2] + next_state[2:])


def test_regime_changes_take_effect_on_the_named_transition():
    schedule = RegimeSchedule((Segment("A", 2), Segment("B", 3), Segment("A", 2)))
    env = RecurrentDynamicsEnv(schedule, noise_std=0.0, max_state_norm=1e6)
    state, _ = env.reset(seed=11, options={"initial_state": np.asarray([0.0, 0.0, 1.0, 0.0])})
    action = np.zeros(2, dtype=np.float64)
    velocities = []
    for _ in range(7):
        state, _, _, _, info = env.step(action)
        assert info == {}
        velocities.append(state[2:].copy())
    expected_x_velocity = [0.88**1, 0.88**2]
    expected_x_velocity.extend([0.88**2 * 0.52**1, 0.88**2 * 0.52**2, 0.88**2 * 0.52**3])
    expected_x_velocity.extend([expected_x_velocity[-1] * 0.88, expected_x_velocity[-1] * 0.88**2])
    assert np.allclose(np.asarray(velocities)[:, 0], expected_x_velocity)
    assert schedule.change_steps() == (2, 5)
    env.close()


def test_filter_applies_transition_prior_before_transition_likelihood():
    candidates = make_candidate_library()
    regimes = default_regimes()
    truth = regimes["B"].parameters
    belief = BayesianTheoryFilter(candidates, noise_std=0.01, hazard_rate=0.1)
    # Simulate a highly concentrated old posterior with zero mass on B.
    belief.posterior.fill(0.0)
    belief.posterior[candidates.index(regimes["A"].parameters)] = 1.0
    prior = belief.transition_prior
    b_index = candidates.index(truth)
    assert prior[b_index] > 0.0

    state = np.asarray([0.0, 0.0, 1.0, 0.0])
    action = np.zeros(2)
    next_velocity = truth.predict_velocity(state[2:], action)
    next_state = np.concatenate((state[:2] + next_velocity, next_velocity))
    posterior = belief.update(state, action, next_state)
    expected = belief._posterior_for_velocity(state, action, next_velocity, prior)
    assert np.allclose(posterior, expected)
    equivalent = np.asarray([
        np.allclose(candidate.predict_velocity(state[2:], action), next_velocity)
        for candidate in candidates
    ])
    assert posterior[equivalent].sum() > 0.999
    assert np.isclose(posterior.sum(), 1.0)

    no_hazard = BayesianTheoryFilter(candidates, noise_std=0.01, hazard_rate=0.0)
    no_hazard.posterior.fill(0.0)
    a_index = candidates.index(regimes["A"].parameters)
    no_hazard.posterior[a_index] = 1.0
    stuck = no_hazard.update(state, action, next_state)
    assert stuck[a_index] > 0.999


def test_expected_information_gain_is_finite_and_action_sensitive():
    candidates = make_candidate_library()
    belief = BayesianTheoryFilter(candidates, noise_std=0.025, hazard_rate=0.05)
    state_at_rest = np.zeros(4)
    common_noise = np.zeros((len(candidates), 8, 2))
    no_action = belief.expected_information_gain(state_at_rest, np.zeros(2), standard_noise=common_noise)
    x_action = belief.expected_information_gain(state_at_rest, np.asarray([0.75, 0.0]), standard_noise=common_noise)
    assert np.isfinite(no_action) and np.isfinite(x_action)
    assert x_action > no_action


def test_pairwise_kl_points_to_state_action_excitation():
    regimes = default_regimes()
    rest = np.zeros(4)
    moving = np.asarray([0.0, 0.0, 1.0, -0.5])
    zero = np.zeros(2)
    x_action = np.asarray([0.75, 0.0])
    # Friction cannot be identified at rest, but becomes visible under velocity.
    assert transition_kl(regimes["A"].parameters, regimes["B"].parameters, rest, zero, 0.025) == 0.0
    assert transition_kl(regimes["A"].parameters, regimes["B"].parameters, moving, zero, 0.025) > 0.0
    # Rotation is visible with non-zero action even from rest.
    assert transition_kl(regimes["A"].parameters, regimes["D"].parameters, rest, x_action, 0.025) > 0.0


def test_shared_probe_budget_and_action_magnitude_constraints():
    candidates = make_candidate_library()
    max_norm = 0.75
    actions = probe_action_set(max_norm)
    assert np.all(np.linalg.norm(actions, axis=1) <= max_norm + 1e-12)
    assert len(evenly_spaced_probe_steps(100, 20)) == 20
    assert len(evenly_spaced_probe_steps(5, 20)) == 5
    for policy_type in (RandomProbePolicy, PureInformationGainPolicy, BeliefActivePolicy):
        policy = policy_type(candidates, 0.025, seed=5, max_action_norm=max_norm, horizon=100, probing_budget=20)
        assert policy.probe_steps == evenly_spaced_probe_steps(100, 20)
        assert np.linalg.norm(policy.control_action(np.zeros(4), np.asarray([2.0, 2.0]))) <= max_norm + 1e-12


def test_probe_policies_choose_only_from_the_shared_feasible_action_set():
    candidates = make_candidate_library()
    state = np.asarray([-1.0, -0.5, 0.2, -0.1])
    target = np.asarray([2.0, 2.0])
    for policy_type in (RandomProbePolicy, PureInformationGainPolicy, BeliefActivePolicy):
        policy = policy_type(candidates, 0.025, seed=9, max_action_norm=0.75, horizon=20, probing_budget=5)
        feasible = policy.feasible_actions(state, target)
        selected = policy.act(state, target, min(policy.probe_steps))
        assert any(np.allclose(selected, action) for action in feasible)
        assert np.linalg.norm(selected) <= 0.75 + 1e-12


def test_environment_terminates_on_large_finite_state_without_reward_clipping():
    schedule = RegimeSchedule((Segment("A", 3),))
    env = RecurrentDynamicsEnv(schedule, noise_std=0.0, max_state_norm=10.0)
    state, _ = env.reset(seed=1, options={"initial_state": np.asarray([9.8, 0.0, 0.5, 0.0])})
    next_state, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert np.linalg.norm(next_state) > 10.0
    assert terminated and not truncated
    assert reward == -(np.square(next_state[:2] - env.target).sum() + 0.08 * np.square(next_state[2:]).sum())
    assert info == {}
    env.close()


def test_agent_interface_and_observation_do_not_expose_ground_truth():
    schedule = make_schedule("evaluation", 2, min_dwell=2, max_dwell=2)
    env = RecurrentDynamicsEnv(schedule)
    observation, info = env.reset(seed=11)
    assert observation.shape == (4,)
    assert info == {}
    action = np.zeros(2, dtype=np.float64)
    next_observation, reward, terminated, truncated, info = env.step(action)
    assert next_observation.shape == (4,)
    assert np.isscalar(reward)
    assert not terminated and not truncated
    assert info == {}
    assert not hasattr(env, "regimes")
    for privileged_name in ("env_change", "delta_change", "get_planning_env"):
        assert not hasattr(env, privileged_name)
    assert "regime" not in inspect.signature(PassiveBayesPolicy.act).parameters
    assert "schedule" not in inspect.signature(PassiveBayesPolicy.act).parameters
    assert "env" not in inspect.signature(PassiveBayesPolicy.observe).parameters
    env.close()


def test_seeded_replay_is_deterministic():
    schedule = make_schedule("evaluation", 13, min_dwell=2, max_dwell=3)
    traces = []
    for _ in range(2):
        env = RecurrentDynamicsEnv(schedule, noise_std=0.01)
        state, _ = env.reset(seed=123)
        rng = np.random.default_rng(123)
        trace = []
        for _ in range(5):
            action = rng.uniform(-0.5, 0.5, 2)
            state, reward, _, _, _ = env.step(action)
            trace.append((state, reward))
        traces.append(trace)
        env.close()
    for left, right in zip(*traces):
        assert np.array_equal(left[0], right[0])
        assert left[1] == right[1]


def test_gymnasium_checker_contract():
    from gymnasium.utils.env_checker import check_env

    env = RecurrentDynamicsEnv(make_schedule("train", 1, min_dwell=2, max_dwell=2))
    check_env(env, skip_render_check=True)
    env.close()
