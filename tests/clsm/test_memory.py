import torch

from rwt.clsm.memory import TheoryMemory, VisitSummary


def test_memory_recurrence_theta_and_scratch():
    memory = TheoryMemory()
    scratch_before = memory.prior_for_entry(1,0)
    reuse_before = memory.prior_for_entry(1,1)
    memory.update_from_completed_visit(VisitSummary(1,torch.tensor([-1.,0.,0.],dtype=torch.float64),
                                                  torch.diag(torch.tensor([.1,0.,0.],dtype=torch.float64))))
    scratch_after = memory.prior_for_entry(1,0)
    reuse_after = memory.prior_for_entry(1,1)
    assert torch.equal(scratch_before[0],scratch_after[0])
    assert torch.equal(scratch_before[1],scratch_after[1])
    assert not torch.equal(reuse_before[0],reuse_after[0])
    assert reuse_after[1][0,0] > 0  # A new visit can differ from its prototype.
    assert torch.equal(reuse_after[0][1:],torch.zeros(2,dtype=torch.float64))
    assert memory.completed_visits == 1


def test_prequential_evidence_once():
    memory = TheoryMemory(scratch_weight=.5)
    memory.update_prequential_evidence(-1.,-2.)
    expected = torch.softmax(torch.tensor([-1.,-2.],dtype=torch.float64),0)
    assert torch.allclose(memory.log_model_weights.exp(),expected)


def test_no_future_memory_input():
    memory = TheoryMemory()
    _ = memory.prior_for_entry(7,1)
    before = memory.primitive_mean.clone()
    assert torch.equal(memory.primitive_mean,before)


def test_soft_visit_count_uses_responsibility_not_update_calls():
    memory = TheoryMemory()
    summary = VisitSummary(1, torch.tensor([-1., 0., 0.], dtype=torch.float64),
                           torch.diag(torch.tensor([.1, 0., 0.], dtype=torch.float64)), .2)
    memory.update_from_completed_visit(summary)
    memory.update_from_completed_visit(summary)
    assert abs(memory.completed_visits - .4) < 1e-12
