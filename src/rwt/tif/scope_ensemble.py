"""Public-calibration convex stacking of compositional and pooled dynamics."""
import numpy as np
import torch


def calibration_weight(anchored, pooled, target, scale):
    """Exact empirical squared-risk minimizer over the convex segment.

    Since both endpoints are feasible, calibration risk cannot exceed either
    endpoint's risk. This is a finite-data algebraic fact, not a test-risk bound.
    """
    direction = ((anchored-pooled)/scale).ravel()
    response = ((target-pooled)/scale).ravel()
    denominator = float(direction@direction)
    return float(np.clip(direction@response/denominator, 0, 1)) if denominator>1e-12 else .5


class ValidatedScriptEnsemble(torch.nn.Module):
    def __init__(self, anchored, pooled, weight=0.5):
        super().__init__()
        self.anchored, self.pooled = anchored, pooled
        self.state_dim, self.action_dim = anchored.state_dim, anchored.action_dim
        self.register_buffer('mixture_weight', torch.tensor(float(weight)))
        self.register_buffer('likelihood_scale', torch.ones(self.state_dim))

    @property
    def delta_scale(self): return self.anchored.delta_scale

    @property
    def state_mean(self): return self.anchored.state_mean

    @property
    def state_scale(self): return self.anchored.state_scale

    def forward(self, states, actions, code, gains=None):
        return (self.mixture_weight*self.anchored(states,actions,code,gains)
                +(1-self.mixture_weight)*self.pooled(states,actions,0))
