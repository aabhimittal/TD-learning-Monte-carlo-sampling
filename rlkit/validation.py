"""Defensive argument checks shared by agents, buffers and environments.

A teaching library is still a library: a misconfigured agent should fail
loudly at construction time with a message naming the offending argument,
rather than silently producing ``nan`` after an hour of training. Every
check here returns the coerced value so call sites can write
``self.gamma = check_range("gamma", gamma, 0.0, 1.0)``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "check_positive_int",
    "check_non_negative_int",
    "check_finite",
    "check_range",
    "check_probability",
]

_INT_TYPES = (int, np.integer)


def _as_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, _INT_TYPES):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def check_positive_int(name: str, value) -> int:
    """Require a strictly positive integer (sizes, capacities, counts)."""
    value = _as_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def check_non_negative_int(name: str, value) -> int:
    """Require a non-negative integer (horizons, decay steps, offsets)."""
    value = _as_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def check_finite(name: str, value) -> float:
    """Require a real, finite float -- rejects ``nan``, ``inf`` and strings."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def check_range(name: str, value, low: float, high: float,
                *, low_inclusive: bool = True, high_inclusive: bool = True) -> float:
    """Require a finite float inside ``[low, high]`` (bounds configurable)."""
    value = check_finite(name, value)
    too_low = value < low if low_inclusive else value <= low
    too_high = value > high if high_inclusive else value >= high
    if too_low or too_high:
        lo = "[" if low_inclusive else "("
        hi = "]" if high_inclusive else ")"
        raise ValueError(f"{name} must lie in {lo}{low}, {high}{hi}, got {value}")
    return value


def check_probability(name: str, value) -> float:
    """Require a finite float in ``[0, 1]``."""
    return check_range(name, value, 0.0, 1.0)
