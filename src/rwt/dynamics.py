"""Hidden-regime dynamics and oracle candidate models for the RWT benchmark.

The candidate library is deliberately an oracle diagnostic: it enumerates
known primitive values and their compositions. It is not a learned RWT model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DynamicsParameters:
    """Parameters in v'=(1-mu)v + gain R(phi)a + force + noise."""

    friction: float = 0.12
    gain: float = 1.0
    rotation: float = 0.0
    external_force: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.friction < 1.0:
            raise ValueError("friction must lie in [0, 1)")
        if self.gain <= 0.0:
            raise ValueError("gain must be positive")
        if len(self.external_force) != 2:
            raise ValueError("external_force must have two components")

    @property
    def action_matrix(self) -> FloatArray:
        cosine, sine = np.cos(self.rotation), np.sin(self.rotation)
        return self.gain * np.asarray([[cosine, -sine], [sine, cosine]])

    def predict_velocity(self, velocity: FloatArray, action: FloatArray) -> FloatArray:
        return (
            (1.0 - self.friction) * velocity
            + self.action_matrix @ action
            + np.asarray(self.external_force, dtype=np.float64)
        )

    def predict_state(self, state: FloatArray, action: FloatArray) -> FloatArray:
        next_velocity = self.predict_velocity(state[2:], action)
        return np.concatenate((state[:2] + next_velocity, next_velocity))


@dataclass(frozen=True, slots=True)
class Regime:
    """A named hidden regime; names are only available to evaluation code."""

    name: str
    parameters: DynamicsParameters


def default_regimes() -> dict[str, Regime]:
    """Return A/B/D and held-out composition C.

    B changes friction, D rotates the action effect, and C composes those two
    already-seen mechanisms. Gain and external force remain explicit model
    dimensions and can be varied through :class:`CandidateLibrary`.
    """

    baseline = DynamicsParameters()
    friction = DynamicsParameters(friction=0.48)
    rotation = DynamicsParameters(rotation=np.pi / 3.0)
    composed = DynamicsParameters(friction=friction.friction, rotation=rotation.rotation)
    return {
        "A": Regime("A", baseline),
        "B": Regime("B", friction),
        "D": Regime("D", rotation),
        "C": Regime("C", composed),
    }


@dataclass(frozen=True, slots=True)
class CandidateLibrary:
    """Finite oracle library formed by composing primitive parameter values."""

    frictions: tuple[float, ...] = (0.12, 0.48)
    gains: tuple[float, ...] = (1.0, 1.35)
    rotations: tuple[float, ...] = (0.0, np.pi / 3.0)
    external_forces: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.18, -0.12))

    def parameters(self) -> tuple[DynamicsParameters, ...]:
        values = {
            DynamicsParameters(mu, gain, angle, force)
            for mu, gain, angle, force in product(
                self.frictions, self.gains, self.rotations, self.external_forces
            )
        }
        return tuple(sorted(values, key=lambda p: (p.friction, p.gain, p.rotation, p.external_force)))

    def index_of(self, parameters: DynamicsParameters) -> int:
        try:
            return self.parameters().index(parameters)
        except ValueError as error:
            raise ValueError("regime parameters are absent from candidate library") from error
