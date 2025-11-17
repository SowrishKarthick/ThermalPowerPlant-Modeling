"""
Reliability and availability modelling helpers.

Provides utilities to simulate stochastic failure/repair processes and to
summarise resulting metrics such as MTBF, MTTR, uptime percentage, and a
composite reliability index.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config.simulation_config import CONFIG


@dataclass
class ReliabilityEvent:
    """Container describing a single failure/repair episode."""

    start_time_hours: float
    duration_hours: float
    component: str = "plant"

    def as_dict(self) -> Dict[str, float | str]:
        return asdict(self)


def simulate_availability_profile(
    duration_hours: float,
    dt_seconds: float,
    mtbf_hours: float,
    mttr_hours: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[ReliabilityEvent]]:
    """
    Simulate on/off availability samples for the plant using a renewal process.

    Failures follow an exponential distribution with mean *mtbf_hours* and
    repairs follow an exponential distribution with mean *mttr_hours*.
    """
    total_steps = int(round((duration_hours * 3600.0) / dt_seconds))
    availability = np.ones(total_steps, dtype=float)
    events: List[ReliabilityEvent] = []

    current_time_h = 0.0
    next_failure_h = rng.exponential(mtbf_hours)

    while next_failure_h < duration_hours:
        downtime_h = rng.exponential(mttr_hours)
        start_idx = int(round(next_failure_h * 3600.0 / dt_seconds))
        end_time_h = min(next_failure_h + downtime_h, duration_hours)
        end_idx = int(round(end_time_h * 3600.0 / dt_seconds))
        availability[start_idx:end_idx] = 0.0
        events.append(ReliabilityEvent(start_time_hours=next_failure_h, duration_hours=downtime_h))

        current_time_h = end_time_h
        next_failure_h = current_time_h + rng.exponential(mtbf_hours)

    return availability, events


def compute_reliability_metrics(
    availability: Sequence[float],
    dt_seconds: float,
    events: Sequence[ReliabilityEvent],
    performance_ratio: Sequence[float],
) -> Dict[str, float | int | List[Dict[str, float | str]]]:
    """
    Summarise reliability metrics from availability samples.

    The composite reliability index is computed from uptime and the mean power
    delivery ratio using weights specified in the configuration.
    """
    samples = np.asarray(availability, dtype=float)
    perf = np.asarray(performance_ratio, dtype=float)
    total_time_h = samples.size * dt_seconds / 3600.0

    uptime_hours = samples.sum() * dt_seconds / 3600.0
    downtime_hours = max(total_time_h - uptime_hours, 0.0)
    uptime_percent = (uptime_hours / total_time_h) if total_time_h > 0 else 0.0
    mtbf_est = (uptime_hours / max(len(events), 1)) if events else CONFIG.MTBF_HOURS
    mttr_est = (downtime_hours / max(len(events), 1)) if events else CONFIG.MTTR_HOURS

    mean_performance = float(np.clip(np.nanmean(perf), 0.0, 1.2)) if perf.size else 1.0
    reliability_index = (
        CONFIG.RELIABILITY_WEIGHTING * uptime_percent
        + (1.0 - CONFIG.RELIABILITY_WEIGHTING) * (1.0 - CONFIG.RELIABILITY_DEGRADATION_RATE * (1.0 - mean_performance))
    )
    reliability_index = float(np.clip(reliability_index, 0.0, 1.0))

    return {
        "uptime_percent": float(uptime_percent * 100.0),
        "downtime_hours": float(downtime_hours),
        "mtbf_hours": float(mtbf_est),
        "mttr_hours": float(mttr_est),
        "event_count": int(len(events)),
        "reliability_index": reliability_index * 100.0,
        "events": [event.as_dict() for event in events],
    }
