import numpy as np
import pytest

from rwt.stage2b import PredictionSet
from rwt.stage2c import OnlineTheoryRevision, RLSResidual, StreamingTheoryAdapters, residual_features


def _bank(state, action):
    del state, action
    return PredictionSet(
        names=("known", "completion"),
        means=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        scales=np.full((2, 2), 0.1), known_count=1, completion_indices=(1,),
    )


def test_rls_adapter_learns_only_observed_residual_and_has_small_budget():
    adapter = RLSResidual(ridge=1.0)
    state = np.zeros(4); action = np.asarray([0.2, -0.1])
    for _ in range(40):
        adapter.update(state, action, np.asarray([0.5, -0.25]))
    assert adapter.parameter_count == 10
    np.testing.assert_allclose(adapter.predict(state, action), [0.5, -0.25], atol=0.015)
    assert adapter.updates == 40


def test_rls_predictive_uncertainty_shrinks_after_observed_transitions():
    adapter=RLSResidual(ridge=1.0); state=np.zeros(4); action=np.zeros(2)
    before=adapter.predictive_variance(state,action,.03)
    for _ in range(10): adapter.update(state,action,np.asarray([.2,-.1]))
    assert adapter.predictive_variance(state,action,.03) < before


def test_revision_requires_persistent_predictive_mismatch():
    agent = OnlineTheoryRevision(_bank, ("known", "completion"),
                                 completion_indices=(1,), mismatch_threshold=2.0)
    state = np.zeros(4); action = np.zeros(2); novel = np.asarray([0., 0., 2., 2.])
    created = []
    for _ in range(3):
        agent.predict(state, action)
        step = agent.observe(state, action, novel)
        created.append(step.created)
    assert created == [False, False, True]
    assert agent.active is not None and agent.active.adapter.updates == 1


def test_recurrence_memory_is_retrieved_without_regime_input():
    agent = OnlineTheoryRevision(_bank, ("known", "completion"),
                                 completion_indices=(1,), mismatch_threshold=2.0,
                                 memory=True)
    state = np.zeros(4); action = np.zeros(2)
    # Discover and adapt a novel theory anchored to the best completion.
    for _ in range(12):
        agent.predict(state, action)
        agent.observe(state, action, np.asarray([0., 0., 2., 2.]))
    assert agent.active is not None
    # A known-theory run supplies evidence to archive the revised theory.
    for _ in range(3):
        agent.predict(state, action)
        agent.observe(state, action, np.asarray([0., 0., 0., 0.]))
    assert len(agent.memory) >= 1
    # The same unlabelled observations must retrieve that cached predictor.
    hits = 0
    for _ in range(2):
        agent.predict(state, action)
        outcome = agent.observe(state, action, np.asarray([0., 0., 2., 2.]))
        hits += int(outcome.cache_hit)
    assert hits >= 1


def test_revision_interface_has_no_schedule_or_regime_argument():
    import inspect
    signature = inspect.signature(OnlineTheoryRevision.observe)
    assert "regime" not in signature.parameters
    assert "change_time" not in signature.parameters
    assert residual_features(np.zeros(4), np.zeros(2)).shape == (5,)


def test_observe_requires_prior_prediction():
    agent = OnlineTheoryRevision(_bank, ("known", "completion"),
                                 completion_indices=(1,), mismatch_threshold=2.0)
    with pytest.raises(RuntimeError):
        agent.observe(np.zeros(4), np.zeros(2), np.zeros(4))


def test_streaming_residual_bank_updates_after_transition_and_reuses_candidate_adapter():
    agent=StreamingTheoryAdapters(_bank, ("known","completion"), memory=True, ridge=1.0)
    state=np.zeros(4); action=np.zeros(2); novel=np.asarray([0.,0.,2.,2.])
    base,means,scales,prior,pred,_,_=agent.predict(state,action)
    np.testing.assert_allclose(means[1],[1.,1.])
    agent.observe(novel)
    assert agent.adapters[1].updates==1
    assert agent.signatures[1].shape==(10,)
    _,means2,_,_,_,_,_=agent.predict(state,action)
    assert np.all(means2[1] > means[1])
    first=agent.observe(novel)
    assert first.cache_hit


def test_streaming_no_memory_resets_adapter_on_selected_theory_change():
    agent=StreamingTheoryAdapters(_bank, ("known","completion"), memory=False, ridge=1.0)
    state=np.zeros(4); action=np.zeros(2)
    for _ in range(5):
        agent.predict(state,action); agent.observe(np.asarray([0.,0.,2.,2.]))
    assert agent.active_index==1 and agent.active_adapter.updates>1
    # A sequence exactly predicted by the known theory should eventually
    # select it, replacing the previous transient correction.
    for _ in range(20):
        agent.predict(state,action); agent.observe(np.zeros(4))
    assert agent.active_index==0
    assert agent.active_adapter.updates>0
