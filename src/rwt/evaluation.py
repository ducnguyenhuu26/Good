"""Evaluator-only scoring of hidden regimes, entropy and change delay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rwt.dynamics import DynamicsParameters, default_regimes
from rwt.schedule import RegimeSchedule


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    cumulative_return: float
    identification_accuracy: float
    change_detection_delay: float
    mean_posterior_entropy: float
    accuracy_by_regime: dict[str, float]


class RegimeEvaluator:
    """Consumes private schedule metadata outside the agent-facing interface."""

    def __init__(
        self,
        schedule: RegimeSchedule,
        candidates: tuple[DynamicsParameters, ...],
        threshold: float = 0.5,
    ):
        self.schedule = schedule
        self.candidates = candidates
        self.threshold = threshold

    def score(self, rewards: list[float], posteriors: list[FloatArray]) -> EpisodeScore:
        if len(rewards) != len(posteriors) or not 1 <= len(rewards) <= self.schedule.horizon:
            raise ValueError("evaluation traces must have equal non-zero lengths within the horizon")
        regimes = default_regimes()
        correct_by_regime: dict[str, list[bool]] = {}
        for step, posterior in enumerate(posteriors):
            truth_name = self.schedule.regime_name_at(step)
            truth = regimes[truth_name].parameters
            best = self.candidates[int(np.argmax(posterior))]
            correct_by_regime.setdefault(truth_name, []).append(best == truth)
        accuracies = {name: float(np.mean(values)) for name, values in correct_by_regime.items()}
        total_correct = sum(sum(values) for values in correct_by_regime.values())

        delays: list[int] = []
        observed_steps = len(rewards)
        changes = tuple(step for step in self.schedule.change_steps() if step < observed_steps)
        full_changes = self.schedule.change_steps()
        for change_index, change_step in enumerate(changes):
            truth = regimes[self.schedule.regime_name_at(change_step)].parameters
            index = next((i for i, candidate in enumerate(self.candidates) if candidate == truth), None)
            if index is None:
                raise ValueError("true regime is absent from evaluator candidate library")
            next_change = next((step for step in full_changes if step > change_step), self.schedule.horizon)
            segment_end = min(next_change, observed_steps)
            delay = segment_end - change_step
            for step in range(change_step, segment_end):
                if posteriors[step][index] >= self.threshold:
                    delay = step - change_step
                    break
            delays.append(delay)
        return EpisodeScore(
            cumulative_return=float(sum(rewards)),
            identification_accuracy=total_correct / len(rewards),
            change_detection_delay=float(np.mean(delays)) if delays else 0.0,
            mean_posterior_entropy=float(np.mean([-np.sum(p[p > 0] * np.log(p[p > 0])) for p in posteriors])),
            accuracy_by_regime=accuracies,
        )
