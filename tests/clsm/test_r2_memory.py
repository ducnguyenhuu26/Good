import pytest
import torch

from rwt.clsm.memory import PopulationHierarchy


def test_world_memory_matches_weighted_conjugate_update_and_processes_boundary_once():
    population = PopulationHierarchy()
    memory = population.new_world()
    prior_mean, prior_cov = memory.entry_prior(1)
    visit_mean = torch.tensor([.8, 9., 9.], dtype=torch.float64)
    visit_cov = torch.diag(torch.tensor([.2, 0., 0.], dtype=torch.float64))
    responsibility = .4
    omega = population.within_var[1, 0]
    prior_var = memory.variance[1, 0]
    expected_precision = 1. / prior_var + responsibility / (omega + visit_cov[0, 0])
    expected_var = 1. / expected_precision
    expected_mean = expected_var * (prior_mean[0] / prior_var +
                                    responsibility * visit_mean[0] / (omega + visit_cov[0, 0]))

    assert memory.record_completed_visit(17, 1, visit_mean, visit_cov, responsibility)
    posterior_mean, posterior_cov = memory.entry_prior(1)
    assert torch.allclose(posterior_mean[0], expected_mean, atol=1e-12, rtol=0)
    assert torch.allclose(posterior_cov[0, 0], expected_var + omega, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="already processed"):
        memory.record_completed_visit(17, 1, visit_mean, visit_cov, responsibility)


def test_scratch_prior_is_population_and_new_world_does_not_inherit_prototype():
    population = PopulationHierarchy()
    memory = population.new_world()
    visit_mean = torch.tensor([1.2, 0., 0.], dtype=torch.float64)
    visit_cov = torch.diag(torch.tensor([.01, 0., 0.], dtype=torch.float64))
    memory.record_completed_visit(3, 1, visit_mean, visit_cov, 1.)
    reuse_mean, _ = memory.entry_prior(1)
    scratch_mean, scratch_cov = memory.entry_prior(1, scratch=True)
    next_world = memory.new_world()
    next_mean, next_cov = next_world.entry_prior(1)
    assert not torch.equal(reuse_mean, scratch_mean)
    assert torch.equal(next_mean, scratch_mean)
    assert torch.equal(next_cov, scratch_cov)


def test_code_zero_has_no_physical_prototype_or_visit_precision():
    memory = PopulationHierarchy().new_world()
    prior_mean, prior_cov = memory.entry_prior(0)
    assert torch.equal(prior_mean, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(prior_cov, torch.zeros((3, 3), dtype=torch.float64))
    accepted = memory.record_completed_visit(4, 0, torch.ones(3, dtype=torch.float64),
                                             torch.eye(3, dtype=torch.float64), 1.)
    assert not accepted
    assert not memory.visits
