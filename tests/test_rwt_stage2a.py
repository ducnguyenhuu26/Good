import numpy as np
import pytest
import torch

from rwt.dynamics import default_regimes
from rwt.schedule import make_schedule
from rwt.stage2a import (
    ALL_THEORY_CODES,
    CompositionalDynamics,
    ContinuousExecutorAdapter,
    TheoryBelief,
    TransitionArrays,
    load_checkpoint,
    save_checkpoint,
)


def test_composition_is_sum_of_learned_base_and_primitive_outputs():
    torch.manual_seed(4)
    model = CompositionalDynamics(hidden_dim=12)
    state, action = torch.randn(5, 4), torch.randn(5, 2)
    predictions = model.predict_codes(state, action)
    torch.testing.assert_close(predictions[:, 3], predictions[:, 1] + predictions[:, 2] - predictions[:, 0])
    assert predictions.shape == (5, 4, 4)
    assert ALL_THEORY_CODES[-1] == (1, 1)


def test_gradient_flows_through_base_and_each_primitive():
    model = CompositionalDynamics(hidden_dim=10)
    state, action = torch.randn(8, 4), torch.randn(8, 2)
    loss = model.predict_codes(state, action)[:, :3].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.base.parameters())
    for primitive in model.primitives:
        assert all(parameter.grad is not None for parameter in primitive.parameters())


def test_training_and_model_selection_schedules_exclude_heldout_C():
    regimes = default_regimes()
    train_schedule = make_schedule("train", 11)
    validation_schedule = make_schedule("train", 12)
    evaluation_schedule = make_schedule("evaluation", 13)
    assert "C" not in {train_schedule.regime_name_at(i) for i in range(train_schedule.horizon)}
    assert "C" not in {validation_schedule.regime_name_at(i) for i in range(validation_schedule.horizon)}
    assert "C" in {evaluation_schedule.regime_name_at(i) for i in range(evaluation_schedule.horizon)}
    params = {name: regimes[name].parameters for name in ("A", "B", "C", "D")}
    assert params["C"].friction == params["B"].friction
    assert params["C"].rotation == params["D"].rotation
    assert params["C"] != params["A"] and params["C"] != params["B"] and params["C"] != params["D"]


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    torch.manual_seed(10)
    model = CompositionalDynamics(hidden_dim=12).eval()
    state, action = torch.randn(3, 4), torch.randn(3, 2)
    before = model.predict_codes(state, action)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, model_kind="compositional", config={"hidden_dim": 12})
    loaded, metadata = load_checkpoint(path)
    after = loaded.predict_codes(state, action)
    torch.testing.assert_close(before, after)
    assert metadata["model_kind"] == "compositional"


@pytest.mark.parametrize("bad_state,bad_action", [
    (torch.zeros(3, 3), torch.zeros(3, 2)),
    (torch.zeros(3, 4), torch.zeros(3, 1)),
])
def test_prediction_rejects_bad_shapes(bad_state, bad_action):
    model = CompositionalDynamics(hidden_dim=8)
    with pytest.raises(ValueError):
        model.predict_codes(bad_state, bad_action)


def test_continuous_adapter_reuses_upstream_executor_and_distribution_shapes():
    from models.program_execution import ProgramExecutor

    adapter = ContinuousExecutorAdapter(state_dim=4, action_dim=2, hidden_dim=12)
    assert isinstance(adapter.executor, ProgramExecutor)
    state, action = torch.randn(6, 4), torch.randn(6, 2)
    delta = adapter(state, action)
    assert delta.shape == state.shape
    model = CompositionalDynamics(hidden_dim=12)
    mean, scale = model.predict(state, action, (1, 1))
    assert mean.shape == scale.shape == state.shape
    assert torch.all(scale > 0)


def test_online_belief_updates_only_from_transition_likelihood():
    belief = TheoryBelief(ALL_THEORY_CODES, hazard=0.0)
    posterior = belief.update(np.asarray([-10.0, -10.0, -0.1, -10.0]))
    assert posterior[2] > 0.99
    assert posterior.sum() == pytest.approx(1.0)


def test_transition_dataset_contract_has_no_privileged_fields():
    dataset = TransitionArrays(np.zeros((2, 4)), np.zeros((2, 2)), np.ones((2, 4)))
    assert not hasattr(dataset, "regime")
    assert not hasattr(dataset, "primitive_id")
    assert not hasattr(dataset, "physics_parameters")
