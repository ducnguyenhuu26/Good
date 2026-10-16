"""Seeded recurrent schedules, kept outside agent observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rwt.dynamics import Regime, default_regimes


@dataclass(frozen=True, slots=True)
class Segment:
    regime: str
    duration: int

    def __post_init__(self) -> None:
        if self.duration < 1:
            raise ValueError("segment duration must be positive")


@dataclass(frozen=True, slots=True)
class RegimeSchedule:
    segments: tuple[Segment, ...]

    @property
    def horizon(self) -> int:
        return sum(segment.duration for segment in self.segments)

    def regime_name_at(self, step: int) -> str:
        if not 0 <= step < self.horizon:
            raise IndexError("step is outside the schedule")
        offset = step
        for segment in self.segments:
            if offset < segment.duration:
                return segment.regime
            offset -= segment.duration
        raise AssertionError("validated schedule failed to cover step")

    def change_steps(self) -> tuple[int, ...]:
        result: list[int] = []
        offset = 0
        previous: str | None = None
        for segment in self.segments:
            if previous is not None and segment.regime != previous:
                result.append(offset)
            offset += segment.duration
            previous = segment.regime
        return tuple(result)

    def validate(self, regimes: dict[str, Regime]) -> None:
        if not self.segments:
            raise ValueError("schedule must contain at least one segment")
        unknown = sorted({item.regime for item in self.segments} - regimes.keys())
        if unknown:
            raise ValueError(f"schedule references unknown regimes: {unknown}")


def make_schedule(
    split: str,
    seed: int,
    *,
    min_dwell: int = 12,
    max_dwell: int = 20,
) -> RegimeSchedule:
    """Create reproducible irregular train/evaluation schedules.

    Training omits C entirely. Evaluation includes the recurrent, held-out
    composition in A -> B -> A -> D -> C -> B order.
    """

    if split not in {"train", "evaluation"}:
        raise ValueError("split must be 'train' or 'evaluation'")
    if min_dwell < 1 or max_dwell < min_dwell:
        raise ValueError("dwell bounds must satisfy 1 <= min_dwell <= max_dwell")
    names = ("A", "B", "A", "D", "B") if split == "train" else ("A", "B", "A", "D", "C", "B")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x525754]))
    segments = tuple(
        Segment(name, int(rng.integers(min_dwell, max_dwell + 1))) for name in names
    )
    result = RegimeSchedule(segments)
    result.validate(default_regimes())
    return result
