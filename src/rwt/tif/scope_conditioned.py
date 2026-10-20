"""State policy plus a context residual that starts at exactly zero."""
from torch import nn


class ZeroStartContextNetwork(nn.Module):
    """Keep the initial policy/value independent of an untrained context path.

    The state path and the context residual are trained jointly. A zero final
    residual layer gives a precise initial equivalence to the state path;
    this does not guarantee that later policy updates improve return.
    """
    def __init__(self, input_dim: int, context_dim: int, output_dim: int, width: int):
        super().__init__()
        if not 0 < context_dim < input_dim:
            raise ValueError('context must be a nonempty suffix of the input')
        self.core_dim = input_dim - context_dim
        self.base = nn.Sequential(nn.Linear(self.core_dim, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU(), nn.Linear(width, output_dim))
        self.residual = nn.Sequential(nn.Linear(input_dim, max(width//2, 16)), nn.ReLU(),
                                      nn.Linear(max(width//2, 16), output_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, value):
        return self.base(value[..., :self.core_dim]) + self.residual(value)
