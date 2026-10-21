import itertools
import math

import torch

from rwt.clsm.evolution import LawEvolution, forward_log_evidence_discrete


def test_weibull_survival_hazard():
    e = LawEvolution()
    ages = torch.arange(10)
    codes = torch.zeros(10, dtype=torch.long)
    h = e.hazard(codes, ages)
    k = e.log_kappa0.exp()
    expected = 1 - torch.exp(-(((ages+1)/k)**e.nu - (ages/k)**e.nu))
    assert torch.allclose(h, expected)
    with torch.no_grad():
        e.nu_raw.copy_(torch.log(torch.expm1(torch.tensor(1., dtype=torch.float64))))
    assert torch.allclose(e.hazard(codes, ages), torch.full_like(h, float(e.hazard(codes[:1], ages[:1])[0].detach())))


def test_structural_kernel():
    a = LawEvolution().structural_log_transition().exp()
    assert torch.equal(a.diag(), torch.zeros(8, dtype=torch.float64))
    assert torch.allclose(a.sum(1), torch.ones(8, dtype=torch.float64))
    assert bool((a[~torch.eye(8, dtype=torch.bool)] > 0).all())


def test_discrete_exact_forward():
    emission = torch.tensor([[.9,.2,.1],[.3,.8,.2],[.1,.4,.7]], dtype=torch.float64)
    initial = torch.tensor([.2,.5,.3], dtype=torch.float64)
    matrix = torch.tensor([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7]], dtype=torch.float64)
    exact = 0.
    for path in itertools.product(range(3), repeat=3):
        probability = initial[path[0]]
        for t, state in enumerate(path):
            probability = probability * emission[t,state]
            if t < 2:
                probability = probability * matrix[state,path[t+1]]
        exact += probability
    actual = forward_log_evidence_discrete(emission.log(), initial, matrix)
    assert abs(float(actual) - math.log(float(exact))) < 1e-10
