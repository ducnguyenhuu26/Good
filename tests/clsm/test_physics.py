import numpy as np
import pytest
import torch

from experiments.rwt.clsm.oracle_probes import fixed_probe_bank, oracle_signature
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld, stationary_public_reward


def test_reward_stationary_public():
    o = np.arange(17, dtype=np.float64)
    a = np.ones(6, dtype=np.float64)
    assert stationary_public_reward(o, a) == stationary_public_reward(o.copy(), a.copy())


def test_physics_restoration_and_timing():
    env = HiddenPhysicsWorld("HalfCheetah-v5", [HiddenLaw(1, motor_1=.5), HiddenLaw(4, damping=1.5)])
    try:
        original_gain = env._base_gain.copy()
        original_damping = env._base_damping.copy()
        obs, info = env.reset(seed=41)
        assert info == {}
        assert env.law_at.code == 1
        assert env.unwrapped.model.actuator_gainprm[0,0] == .5 * original_gain[0,0]
        next_obs, reward, _, _, info = env.step(np.zeros(6))
        assert info == {} and np.isfinite(reward)
        assert env.last_transition_law.code == 1 and env.law_at.code == 4
        assert np.array_equal(env.unwrapped.model.actuator_gainprm, original_gain)
        assert np.allclose(env.unwrapped.model.dof_damping[3:], 1.5 * original_damping[3:])
        env.reset(seed=41)
        assert np.array_equal(env.unwrapped.model.dof_damping, original_damping)
        assert env.last_transition_law is None
    finally:
        env.close()
    assert np.array_equal(env.unwrapped.model.actuator_gainprm, original_gain)
    assert np.array_equal(env.unwrapped.model.dof_damping, original_damping)


def test_bad_physics_range():
    with pytest.raises(ValueError):
        HiddenLaw(1, motor_1=1.2)


def test_time_limit_truncation_does_not_create_a_law_switch():
    schedule = [HiddenLaw(0), HiddenLaw(0), HiddenLaw(0)]
    env = HiddenPhysicsWorld("Reacher-v5", schedule)
    try:
        # Force the Gymnasium TimeLimit boundary on the first physical step.
        env.env._max_episode_steps = 1
        obs, _ = env.reset(seed=10)
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
        assert truncated and not terminated
        assert env.position == 1
        assert env.last_transition_law.code == env.law_at.code == 0
    finally:
        env.close()


def test_terminal_transition_is_kept_before_advancing_the_law(monkeypatch):
    schedule = [HiddenLaw(1), HiddenLaw(2)]
    env = HiddenPhysicsWorld("Reacher-v5", schedule)
    try:
        obs, _ = env.reset(seed=11)
        terminal_obs = np.asarray(obs).copy() + .01

        def terminal_step(action):
            return terminal_obs.copy(), 0.0, True, False, {}

        monkeypatch.setattr(env.env, "step", terminal_step)
        next_obs, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert terminated and not truncated
        assert np.array_equal(next_obs, terminal_obs)
        assert info == {}
        assert env.last_transition_law.code == 1
        assert env.law_at.code == 2  # The next law applies only after this emitted transition.
    finally:
        env.close()


def test_fixed_probe_bank_and_oracle_repeat(tmp_path):
    bank, path = fixed_probe_bank("Reacher-v5", 2, tmp_path)
    second, second_path = fixed_probe_bank("Reacher-v5", 2, tmp_path)
    assert path == second_path and np.array_equal(bank.states, second.states)
    first_signature = oracle_signature("Reacher-v5", bank, HiddenLaw(0),
                                       torch.ones(10, dtype=torch.float64))
    second_signature = oracle_signature("Reacher-v5", bank, HiddenLaw(0),
                                        torch.ones(10, dtype=torch.float64))
    assert torch.equal(first_signature, second_signature)
