"""
Reusable mathematical helpers for the simulation models.

Only lightweight utilities are placed here to avoid re-implementing the same
logic across multiple modules. Functions are written to be NumPy-agnostic so
they can operate on scalars or array-like values.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the inclusive range [low, high]."""
    return max(low, min(high, value))


def first_order_response(current: float, target: float, dt: float, tau: float) -> float:
    """Advance a first-order lag towards *target* using time constant *tau*."""
    if tau <= 0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + alpha * (target - current)


def moving_average(values: Iterable[float]) -> float:
    """Return the arithmetic mean of *values* (empty iterable -> 0.0)."""
    vals: List[float] = list(values)
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def random_variation(scale: float) -> float:
    """Generate a bounded random variation in [-scale, +scale]."""
    return random.uniform(-scale, scale)
