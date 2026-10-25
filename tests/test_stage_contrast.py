import numpy as np
import pytest
from q1_v4.stage_contrast import (
    MECHANISMS, action_features, fit_old, fit_pair, fit_joint_velocity,
    learn_opaque_codes, predict_velocity_effect, robust_empirical_bayes,
)


def observed(rng, n, params, *, dt=.05, noise=.0005):
    x = np.column_stack((rng.normal(0,.3,(n,2)),
                         rng.normal(0,.45,(n,2)), rng.uniform(-1,1,(n,2))))
    a = x[:,4:6]
    u = a@params['B'].T
    # An unknown odd polynomial response, deliberately not a simulator formula.
    force = (u - .2*u**3 + .03*u**5 - params['d']*x[:,2:4]
             - params['b']*np.linalg.norm(x[:,2:4],axis=1,keepdims=True)*x[:,2:4]
             - params['k']*x[:,:2])
    y = dt*force + rng.normal(0,noise,(n,2))
    return x,y


def test_odd_features_are_signed_and_finite():
    a=np.asarray([[.25,-.6],[-.25,.6]])
    v=action_features(a,5)
    assert v.shape==(2,12)
    np.testing.assert_allclose(v[0],-v[1])
    assert np.isfinite(v).all()


def test_exact_observation_model_recovers_beta():
    rng=np.random.default_rng(2)
    p=dict(B=np.array([[.9,.1],[-.1,1.1]]),d=.38,b=.2,k=.26)
    x,y=observed(rng,1500,p,noise=0)
    beta,_=fit_old(x,y,action_ridge=1e-9)
    np.testing.assert_allclose(beta,[.38,.2,.26],atol=4e-4)


def test_stage_contrast_learns_opacity_and_effects():
    rng=np.random.default_rng(3)
    groups={17:[],23:[],31:[]}
    for _ in range(8):
        p=dict(B=np.array([[.8,.1],[-.12,1.1]]),d=rng.uniform(.28,.46),
               b=rng.uniform(.13,.22),k=.22)
        old=observed(rng,600,p,noise=.00025)
        next_b={**p,'b':1.5*p['b']}
        groups[17].append((*old,*observed(rng,600,next_b,noise=.00025)))
        next_d={**p,'d':1.4*p['d']}
        groups[23].append((*old,*observed(rng,600,next_d,noise=.00025)))
        next_action={**p,'B':np.array([[.88,.30],[-.24,.90]])}
        groups[31].append((*old,*observed(rng,600,next_action,noise=.00025)))
    codes=learn_opaque_codes(groups,sigma=.00025,action_ridge=.005)
    assert codes[17].mechanism=='quadratic_drag',codes[17]
    assert codes[23].mechanism=='linear_drag',codes[23]
    assert codes[31].mechanism=='action',codes[31]
    assert abs(codes[17].ratio-1.5)<.08,codes[17]
    assert abs(codes[23].ratio-1.4)<.04,codes[23]


def test_joint_fit_and_effect_on_unseen_samples():
    rng=np.random.default_rng(20)
    p=dict(B=np.array([[.92,.1],[-.11,.94]]),d=.41,b=.19,k=.28)
    before=observed(rng,96,p,noise=.0005)
    after=observed(rng,100,{**p,'b':1.5*p['b']},noise=.0005)
    beta,cond=fit_joint_velocity(*before,*after,mechanism='quadratic_drag',ratio=1.5,action_ridge=.015)
    assert np.isfinite(cond)
    assert beta.shape==(3,)
    q,_=observed(rng,200,p,noise=0)
    eff=predict_velocity_effect(q,mechanism='quadratic_drag',before_coefficient=beta[1],ratio=1.5)
    expected=-.05*.5*p['b']*np.linalg.norm(q[:,2:4],axis=1,keepdims=True)*q[:,2:4]
    assert eff.shape==(200,2)
    assert np.mean((eff-expected)**2)<np.mean(expected**2)


def test_invalid_inputs_do_not_silently_pass():
    x=np.zeros((10,6));y=np.zeros((10,2))
    with pytest.raises(ValueError):fit_pair(x,y,x,y,'bad')
    with pytest.raises(ValueError):fit_joint_velocity(x,y,x,y,mechanism='action',ratio=1)
    with pytest.raises(ValueError):predict_velocity_effect(x,mechanism='action',before_coefficient=1,ratio=1)


def test_robust_empirical_bayes_shrinks_in_distribution_and_adapts_ood():
    mu=.11;sd=.02
    near,weight_near=robust_empirical_bayes(.13,.025,mu,sd)
    far,weight_far=robust_empirical_bayes(.31,.025,mu,sd)
    assert .11<near<.13
    assert far>.24
    assert weight_far>weight_near
