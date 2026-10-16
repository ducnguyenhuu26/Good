"""CPU-first benchmark for recurrent, compositional world theories."""

from rwt.dynamics import DynamicsParameters, Regime, default_regimes
from rwt.environment import RecurrentDynamicsEnv

__all__ = ["DynamicsParameters", "Regime", "RecurrentDynamicsEnv", "default_regimes"]
