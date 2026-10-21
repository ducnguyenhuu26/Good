import torch

from rwt.clsm.baselines import PooledGaussianMLP


def test_pooled_public_input_gaussian_and_gradients():
    model = PooledGaussianMLP(3, 2, hidden_dim=8)
    obs = torch.zeros((4, 3), dtype=torch.float64)
    action = torch.ones((4, 2), dtype=torch.float64)
    next_obs = obs + .1
    model.fit_normalization(obs, action, next_obs)
    mean, variance = model(obs, action)
    assert mean.shape == variance.shape == obs.shape
    loss = -model.log_prob(obs, action, next_obs).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.net[0].weight.grad).all()
    assert model.net[0].in_features == 5
