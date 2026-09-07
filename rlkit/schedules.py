"""Hyper-parameter schedules for exploration rates and step sizes.

Real training runs almost never hold ``epsilon`` and ``alpha`` fixed: you
explore aggressively early, then anneal towards exploitation, and you decay
the step size to let the estimates settle. Every agent in :mod:`rlkit`
accepts either a plain float or one of these schedules wherever it takes
``alpha`` or ``epsilon``; the value is re-read once per training episode.

>>> sched = LinearSchedule(1.0, 0.05, decay_steps=100)
>>> sched(0), round(sched(50), 3), sched(1000)
(1.0, 0.525, 0.05)
"""
from __future__ import annotations

from .validation import (
    check_finite,
    check_non_negative_int,
    check_positive_int,
    check_range,
)

__all__ = [
    "Schedule",
    "ConstantSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "as_schedule",
]


class Schedule:
    """Base class: a callable mapping a step index to a scalar value."""

    def value(self, step: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, step: int) -> float:
        return self.value(step)

    @staticmethod
    def _check_step(step) -> int:
        return check_non_negative_int("step", step)


class ConstantSchedule(Schedule):
    """Always returns the same value (the default for a plain float)."""

    def __init__(self, value: float):
        self.start = self.end = check_finite("value", value)

    def value(self, step: int) -> float:
        self._check_step(step)
        return self.start

    def __repr__(self) -> str:
        return f"ConstantSchedule({self.start})"


class LinearSchedule(Schedule):
    """Interpolate from ``start`` to ``end`` over ``decay_steps``, then hold.

    ``decay_steps=0`` means "jump to ``end`` immediately", which is the
    natural limit of the interpolation and avoids a division by zero.
    """

    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = check_finite("start", start)
        self.end = check_finite("end", end)
        self.decay_steps = check_non_negative_int("decay_steps", decay_steps)

    def value(self, step: int) -> float:
        step = self._check_step(step)
        if self.decay_steps == 0:
            return self.end
        frac = min(1.0, step / self.decay_steps)
        return self.start + frac * (self.end - self.start)

    def __repr__(self) -> str:
        return f"LinearSchedule({self.start} -> {self.end} over {self.decay_steps})"


class ExponentialSchedule(Schedule):
    """Geometric decay ``start * rate ** (step // every)``, floored at ``end``.

    ``rate=1.0`` degenerates to a constant; ``end`` must not exceed ``start``
    since the value only ever decreases.
    """

    def __init__(self, start: float, decay_rate: float, end: float = 0.0,
                 decay_every: int = 1):
        self.start = check_finite("start", start)
        self.decay_rate = check_range("decay_rate", decay_rate, 0.0, 1.0,
                                      low_inclusive=False)
        self.end = check_finite("end", end)
        if self.end > self.start:
            raise ValueError(
                f"end ({self.end}) must be <= start ({self.start}); this schedule decays"
            )
        self.decay_every = check_positive_int("decay_every", decay_every)

    def value(self, step: int) -> float:
        step = self._check_step(step)
        return max(self.end, self.start * self.decay_rate ** (step // self.decay_every))

    def __repr__(self) -> str:
        return (f"ExponentialSchedule({self.start} * {self.decay_rate}^n, "
                f"floor={self.end})")


def as_schedule(value) -> Schedule:
    """Coerce a float (or an existing :class:`Schedule`) into a schedule."""
    if isinstance(value, Schedule):
        return value
    return ConstantSchedule(value)
