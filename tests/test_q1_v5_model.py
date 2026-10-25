import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from q1_v5.model import psi,dpsi,response,jacobian,ProgramModel,fit_theta,joint_map,normal_mixture_shrink

def mixture():
    return {'slopes':[.6,1.0,1.8],'weights':[.1,.85,.05]}

def test_odd_monotone_and_bounded():
    c=mixture();z=np.linspace(-12,12,2501)
    assert np.allclose(psi(-z,c),-psi(z,c),atol=1e-13)
    assert np.min(np.diff(psi(z,c)))>0
    assert max(abs(psi(z,c)))<3
    assert np.all(dpsi(z,c)>0)
    assert abs(dpsi(np.array([0]),c)[0]-1)<1e-14

def test_analytical_jacobian_matches_finite_difference():
    rng=np.random.default_rng(3);x=rng.normal(size=(12,6))*.2
    t=np.r_[np.array([1.,.1,-.05,.95]),np.log([.3,.11,.25])];c=mixture()
    J=jacobian(x,t,c)
    eps=1e-5
    numeric=np.stack([((response(x,t+np.eye(7)[j]*eps,c)-response(x,t-np.eye(7)[j]*eps,c))/(2*eps)).ravel() for j in range(7)],axis=-1)
    assert np.allclose(J,numeric,atol=2e-9,rtol=2e-5)

def test_operator_exact_composition_and_noncommutativity():
    c=mixture()
    o={0:{'mechanism':'action','L':np.array([[.85,-.5],[.5,.85]])},
       1:{'mechanism':'action','L':np.array([[1.5,0],[0,1]])},
       2:{'mechanism':'linear_drag','index':0,'ratio':1.4},
       3:{'mechanism':'quadratic_drag','index':1,'ratio':1.5}}
    m=ProgramModel(c,o)
    t=np.r_[np.array([1.,.1,-.05,.95]),np.log([.3,.11,.25])]
    for a in range(4):
        for b in range(4):
            assert np.allclose(m.transform(t,(a,b)),m.transform(m.transform(t,(a,)),(b,)))
    assert np.linalg.norm(m.transform(t,(0,1))-m.transform(t,(1,0)))>.02
    assert np.allclose(m.transform(t,(2,3)),m.transform(t,(3,2)))

def test_task_fit_noiseless_and_marginal_scores_finite():
    rng=np.random.default_rng(2);x=rng.normal(size=(105,6))*.2
    theta=np.r_[np.array([.9,.08,-.04,1.1]),np.log([.38,.15,.23])];c=mixture()
    y=response(x,theta,c)
    est,cov,_=fit_theta(x[:80],y[:80],c)
    assert np.linalg.norm(est-theta)<.08
    assert np.min(np.linalg.eigvalsh(cov))>0
    m=ProgramModel(c,{0:{'mechanism':'quadratic_drag','index':1,'ratio':1.5}})
    rank=m.score_candidates(x[80:],y[80:],est,cov,[(),(0,),(0,0)],[5,10,20])
    assert len(rank[20])==3 and all(np.isfinite(row[0]) for row in rank[20])

def test_joint_map_uses_observed_data_only():
    rng=np.random.default_rng(81);x=rng.normal(size=(150,6))*.3;c=mixture()
    t=np.r_[np.array([1.,.02,.01,.9]),np.log([.34,.12,.22])]
    m=ProgramModel(c,{0:{'mechanism':'quadratic_drag','ratio':1.5,'index':1}})
    x0=x[:80];x1=x[80:];y0=response(x0,t,c);y1=response(x1,m.transform(t,(0,)),c)
    jt,cov,loss=joint_map(m,x0,y0,x1,y1,(0,))
    assert np.linalg.norm(jt-t)<.05 and loss<1e-4

def test_robust_normal_mixture_shrinkage():
    middle,post=normal_mixture_shrink(.115,.03,.11,.015)
    assert .10<middle<.13 and 0<post<1
