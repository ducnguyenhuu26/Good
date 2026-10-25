import inspect
import json
from pathlib import Path

import numpy as np

from q1.simulator import DynamicsParams, apply_program, response as simulator_response
from q1_v5.model import ProgramModel, response as model_response
from q1_v5.prechange_design import (
    ENERGY_TARGET, POLICIES, action_energy, action_sequence, estimate_multiplicative_ratio,
    functional_class, matched_population_b, matched_population_u4_effect,
    stage_corrected_jacobian,
)
from scripts.q1_v5_fixed_confirmatory import select_v3_candidate
from scripts.q1_v5_prechange_experiment import _draw_seed_groups


def _c():
    return {"weights": np.array([.2, .3, .5]), "slopes": np.array([.5, 1., 2.])}


def _theta():
    return np.array([.8, .1, -.05, .7, np.log(.35), np.log(.12), np.log(.23)])


def test_V3_candidate_argmin_not_first_unsorted():
    rows = [{"program": (), "marginal_score": 50.},
            {"program": (2,), "marginal_score": 3.},
            {"program": (0, 1), "marginal_score": 8.}]
    assert select_v3_candidate(rows)["program"] == (2,)


def test_ordered_action_programs_remain_distinct_and_commuting_swaps_collapse():
    assert functional_class((0, 1)) != functional_class((1, 0))
    assert functional_class((2, 0)) == functional_class((0, 2))
    assert functional_class((3, 2)) == functional_class((2, 3))
    assert functional_class((3,)) != functional_class((3, 3))


def test_finite_operator_composition_error_obeys_telescoping_bound():
    true = [np.array([[.8, .1], [0., .9]]), np.array([[1.1, 0.], [.2, .7]])]
    est = [true[0] + np.eye(2)*.01, true[1] - np.eye(2)*.02]
    error = np.linalg.norm(est[1] @ est[0] - true[1] @ true[0], ord=2)
    bound = (np.linalg.norm(est[1], 2)*np.linalg.norm(est[0]-true[0], 2) +
             np.linalg.norm(est[1]-true[1], 2)*np.linalg.norm(true[0], 2))
    assert error <= bound + 1e-12


def test_U4_post_identical_hypotheses_given_equal_boundary_state():
    b, r = .12, 1.5
    p1 = DynamicsParams(np.array([[.8, .1], [-.05, .7]]), .35, b, .23)
    p2 = DynamicsParams(p1.B, p1.d, b/r, p1.kappa)
    h1 = apply_program(p1, (3,))
    h2 = apply_program(p2, (3, 3))
    rng = np.random.default_rng(9)
    x = np.c_[rng.normal(size=(32, 2)), rng.normal(size=(32, 2)), rng.uniform(-1, 1, (32, 2))]
    np.testing.assert_allclose(simulator_response(h1, x), simulator_response(h2, x), atol=1e-14)


def test_U4_effect_exact_factorization():
    c = _c(); theta = _theta(); r = 1.5
    rng = np.random.default_rng(5)
    x = np.c_[rng.normal(size=(48, 2)), rng.normal(size=(48, 2)), rng.uniform(-1, 1, (48, 2))]
    after = theta.copy(); after[5] += np.log(r)
    delta = model_response(x, after, c) - model_response(x, theta, c)
    expected = -.05 * np.exp(theta[5]) * (r-1) * np.linalg.norm(x[:, 2:4], axis=1, keepdims=True) * x[:, 2:4]
    np.testing.assert_allclose(delta, expected, rtol=2e-12, atol=1e-14)


def test_action_composition_order_and_scalar_rate_commutation():
    angle = np.pi/6
    rmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dmat = np.diag([1.5, 1.])
    B = np.array([[.8, .1], [-.05, .7]])
    assert np.linalg.det(B) != 0
    assert not np.allclose(dmat @ rmat @ B, rmat @ dmat @ B)
    gamma = np.array([np.log(.35), np.log(.12), np.log(.23)])
    a = np.array([.2, 0., 0.]); b = np.array([0., .3, 0.])
    np.testing.assert_allclose((gamma+a)+b, (gamma+b)+a)


def test_stage_correct_Fisher_Jacobian_applies_post_chain_rule():
    c = _c(); theta = _theta()
    ops = {0: {"mechanism": "action", "L": np.array([[1., .2], [-.1, .9]])},
           1: {"mechanism": "quadratic_drag", "index": 1, "ratio": 1.5}}
    model = ProgramModel(c, ops, theta, np.ones(7))
    rng = np.random.default_rng(8)
    x0 = np.c_[rng.normal(size=(15, 2)), rng.normal(size=(15, 2)), rng.uniform(-1, 1, (15, 2))]
    x1 = np.c_[rng.normal(size=(13, 2)), rng.normal(size=(13, 2)), rng.uniform(-1, 1, (13, 2))]
    program = (0, 1); joint = stage_corrected_jacobian(x0, x1, theta, program, model)
    post_theta = model.transform(theta, program)
    from q1_v5.model import jacobian
    expected = jacobian(x1, post_theta, c) @ model.transform_jac(program)
    np.testing.assert_allclose(joint[2*len(x0):, :7], expected, rtol=1e-11, atol=1e-12)


def test_energy_matched_controls_are_bounded_and_have_declared_energy():
    for policy in POLICIES:
        actions = action_sequence(policy, seed=4, n=96)
        assert actions.shape == (96, 2)
        assert np.max(np.abs(actions)) <= 1.
        energy = action_energy(actions)["mean_squared_norm"]
        if policy not in ("ORIGINAL_SINUSOIDAL", "BLOCK_FULL_24"):
            assert abs(energy-ENERGY_TARGET) < 1e-8
    assert action_energy(action_sequence("BLOCK_FULL_24", seed=4))["mean_squared_norm"] == 2.


def test_block_schedule_has_four_quadrants_and_open_loop_api_has_no_oracle_inputs():
    actions = action_sequence("BLOCK_24_ENERGY_MATCHED", seed=17, n=96)
    assert len({tuple(x) for x in np.sign(actions)}) == 4
    np.testing.assert_array_equal(actions[:24], np.broadcast_to(actions[0], (24, 2)))
    assert not {"params", "program", "intervention", "true_theta"}.intersection(inspect.signature(action_sequence).parameters)


def test_ratio_ols_and_matched_population_use_train_only_arithmetic_b_mean():
    f = np.array([-2., -1., 1., 2.]); r = 1.5
    assert abs(estimate_multiplicative_ratio(f, r*f)-r) < 1e-12
    cal = matched_population_b(np.array([.1, .2, .3]))
    assert cal["n_train"] == 3 and abs(cal["b_mean"]-.2) < 1e-12
    c = _c(); theta = _theta()
    model = ProgramModel(c, {0: {"mechanism":"quadratic_drag","index":1,"ratio":1.5}}, theta, np.ones(7))
    x = np.array([[0., 0., .3, -.4, .5, -.2]])
    delta = matched_population_u4_effect(x, theta, lambda xx,tt:model_response(xx,tt,c),
                                         model.transform, 0, cal)
    expected = -.05 * cal["b_mean"] * .5 * np.linalg.norm(x[:,2:4],axis=1,keepdims=True) * x[:,2:4]
    np.testing.assert_allclose(delta, expected, rtol=1e-12, atol=1e-14)


def test_seed_groups_are_disjoint_from_existing_development_seeds():
    forbidden = {3, 5, 7}
    groups = _draw_seed_groups(123, forbidden, {"train": 8, "development": 12})
    values = [x for group in groups.values() for x in group]
    assert len(values) == len(set(values))
    assert not set(values).intersection(forbidden)


def test_original_fixed_and_postchange_verdicts_remain_immutable():
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root/"results/q1/SUMMARY.json").read_text(encoding="utf-8-sig"))
    assert summary["fixed_policy_verdict"] == "FAIL"
    assert summary["active_excitation_verdict"] == "FAIL"
