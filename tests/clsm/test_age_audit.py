import math

import torch

from experiments.rwt.clsm.audit_discrete_age import _compress, _step, audit
from rwt.clsm.belief import LawComponent, compress_components
from rwt.clsm.evolution import LawEvolution


def test_long_age_mass_exact_vs_filter_compressor():
    transition = LawEvolution().structural_log_transition().exp().detach().numpy()
    ages = torch.arange(17, dtype=torch.float64)
    hazard = -torch.expm1(-(((ages + 1) / 20) ** 1.4 - (ages / 20) ** 1.4))
    exact = {(code, 0, hypothesis): 1 / 16
             for code in range(8) for hypothesis in (0, 1)}
    for _ in range(16):
        exact = _step(exact, transition, hazard.numpy())
    reference, reference_dropped = _compress(exact, 128)
    zero = torch.zeros(3, dtype=torch.float64)
    cov = torch.zeros((3, 3), dtype=torch.float64)
    components = [LawComponent(code, age, zero, cov,
                               torch.tensor(math.log(weight), dtype=torch.float64), hypothesis)
                  for (code, age, hypothesis), weight in exact.items()]
    actual, actual_dropped = compress_components(components, 128)
    observed = {(item.code, item.age, item.scratch_or_reuse): float(item.log_weight.exp())
                for item in actual}
    assert abs(actual_dropped - reference_dropped) < 1e-12
    assert observed.keys() == reference.keys()
    assert all(abs(observed[key] - reference[key]) < 1e-12 for key in reference)
    assert actual_dropped > .001  # This registered numerical gate fails here.


def test_age_budget_audit_detects_repeated_loss():
    result = audit(64, (128, 1024))
    small, large = result["results"]
    assert result["exact_support"] == 1040
    assert small["steps_above_0_1pct_gate"] > 0
    assert small["tv_to_exact_at_horizon"] > .5
    assert large["steps_above_0_1pct_gate"] == 0
    assert large["tv_to_exact_at_horizon"] < .001
