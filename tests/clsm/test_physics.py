import numpy as np
import pytest

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
