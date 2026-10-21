import itertools

import torch

from rwt.clsm.belief import StepAncestry
from rwt.clsm.belief import LawComponent
from rwt.clsm.smoothing import backward_component_weights, backward_gaussian_moments


def test_discrete_backward_weights_match_enumeration():
    initial = torch.tensor([.6,.4],dtype=torch.float64)
    transition = torch.tensor([[.8,.2],[.3,.7]],dtype=torch.float64)
    emission = torch.tensor([[.9,.2],[.3,.7],[.8,.4]],dtype=torch.float64)
    filtered=[]; steps=[]; prior=initial
    for t in range(3):
        posterior=prior*emission[t]; posterior=posterior/posterior.sum()
        filtered.append(posterior)
        if t < 2:
            predicted=posterior@transition
            steps.append(StepAncestry(posterior,predicted,transition,0.))
            prior=predicted
    result=backward_component_weights(filtered,steps)
    paths=list(itertools.product(range(2),repeat=3))
    masses=[]
    for path in paths:
        value=initial[path[0]]*emission[0,path[0]]
        for t in range(1,3):
            value=value*transition[path[t-1],path[t]]*emission[t,path[t]]
        masses.append(value)
    masses=torch.stack(masses); masses=masses/masses.sum()
    for t in range(3):
        exact=torch.stack([sum(masses[i] for i,p in enumerate(paths) if p[t]==state) for state in range(2)])
        assert torch.allclose(result.gamma[t],exact,atol=1e-12)
    for t in range(2):
        exact=torch.stack([torch.stack([sum(masses[i] for i,p in enumerate(paths) if p[t]==a and p[t+1]==b)
                                        for b in range(2)]) for a in range(2)])
        assert torch.allclose(result.xi[t],exact,atol=1e-12)


def test_linear_gaussian_drift_backward_conditional():
    prior = LawComponent(1,0,torch.zeros(3,dtype=torch.float64),
                         torch.diag(torch.tensor([1.,0.,0.],dtype=torch.float64)),
                         torch.tensor(0.,dtype=torch.float64))
    child = LawComponent(1,1,torch.tensor([1.,0.,0.],dtype=torch.float64),
                         torch.diag(torch.tensor([.2,0.,0.],dtype=torch.float64)),
                         torch.tensor(0.,dtype=torch.float64))
    one = torch.ones(1,dtype=torch.float64)
    matrix = torch.ones((1,1),dtype=torch.float64)
    step = StepAncestry(one,one,matrix,0.,{"stay_drift":matrix,
                         "stay_reset":torch.zeros_like(matrix),"switch":torch.zeros_like(matrix)})
    discrete = backward_component_weights([one,one],[step])
    out = backward_gaussian_moments([[prior],[child]],[step],discrete,
                                    torch.tensor([.1,.1,.1],dtype=torch.float64))
    gain = 1 / 1.1
    assert abs(float(out.mean[0][0][0]) - gain) < 1e-12
    assert abs(float(out.cov[0][0][0,0]) - (1 + gain**2*(.2-1.1))) < 1e-12
    assert torch.equal(out.mean[0][0][1:],torch.zeros(2,dtype=torch.float64))
