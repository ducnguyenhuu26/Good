import numpy as np

from q1_v5.active_design import FisherActionPolicy, full_response_jacobian


def test_greedy_fisher_action_is_bounded_deterministic_and_observation_only():
    c={"weights":[.1,.85,.05],"slopes":[.6,1.,1.8]}
    theta=np.r_[[.9,.1,-.05,1.1],np.log([.35,.12,.23])]
    history=np.random.default_rng(4).normal(0,.15,(12,6))
    state=np.array([.2,-.1,.4,-.3])
    policy=FisherActionPolicy("efficient_logb",horizon=2)
    action=policy.choose(state,history,theta,c)
    assert action.shape==(2,) and np.max(np.abs(action))<=1
    assert policy.candidates_evaluated==9
    assert np.isfinite([v for _,v in policy.last_scores]).all()
    assert np.array_equal(action,policy.choose(state,history,theta,c))


def test_d_optimal_objective_and_cached_observable_information():
    c={"weights":[.1,.85,.05],"slopes":[.6,1.,1.8]}
    theta=np.r_[[.9,.1,-.05,1.1],np.log([.35,.12,.23])]
    history=np.random.default_rng(7).normal(0,.2,(10,6))
    info=full_response_jacobian(history,theta,c).T@full_response_jacobian(history,theta,c)/(.0005**2)
    policy=FisherActionPolicy("d_optimal",horizon=2)
    action=policy.choose(np.zeros(4),history,theta,c,history_information=info)
    assert np.max(np.abs(action))<=1
    assert len(policy.last_scores)==9
