"""
Higher-level analytics inspired by the Forsmark asset-management thesis.

The goal is to provide drop-in utilities that compute:
* Risk measures (Value-at-Risk, Conditional VaR) for financial outcomes.
* Project valuation metrics (NPV, Benefit-Investment Ratio).
* Failure-rate estimation with Weibull-style fitting heuristics.
* Three-point estimation (PERT) for uncertain cost/benefit inputs.
* Simple option-style deferral valuation to capture managerial flexibility.

All functions are dependency-light so they can run inside the existing
simulation workflow without pulling heavy solver stacks. They can later be
upgraded to full optimisation models or stochastic-programming layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Risk metrics                                                                #
# --------------------------------------------------------------------------- #


def value_at_risk(samples: Sequence[float], alpha: float = 0.95) -> float:
    """
    Compute Value-at-Risk at confidence level *alpha*.

    Parameters
    ----------
    samples:
        Iterable of simulated profit/loss values.
    alpha:
        Confidence level (0 < alpha < 1).
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.0
    clipped_alpha = min(max(alpha, 0.0), 0.9999)
    return float(np.quantile(arr, 1.0 - clipped_alpha))


def conditional_value_at_risk(samples: Sequence[float], alpha: float = 0.95) -> float:
    """Return Conditional VaR (expected shortfall) for the lower tail."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.0
    clipped_alpha = min(max(alpha, 0.0), 0.9999)
    threshold = np.quantile(arr, 1.0 - clipped_alpha)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(np.mean(tail))


# --------------------------------------------------------------------------- #
# Three-point (PERT) estimation                                               #
# --------------------------------------------------------------------------- #


def pert_estimate(optimistic: float, most_likely: float, pessimistic: float) -> Tuple[float, float]:
    """
    Compute the PERT expected value and variance from three-point estimates.
    """
    expected = (optimistic + 4.0 * most_likely + pessimistic) / 6.0
    variance = ((pessimistic - optimistic) / 6.0) ** 2
    return expected, variance


def sample_triangular(
    rng: np.random.Generator,
    pessimistic: float,
    most_likely: float,
    optimistic: float,
    size: int,
) -> np.ndarray:
    """Sample from a triangular distribution matching the three-point inputs."""
    return rng.triangular(pessimistic, most_likely, optimistic, size=size)


# --------------------------------------------------------------------------- #
# Project valuation metrics                                                   #
# --------------------------------------------------------------------------- #


def net_present_value(cash_flows: Sequence[float], discount_rate: float) -> float:
    """Discount a series of cash flows (period 0 included)."""
    npv = 0.0
    for idx, cf in enumerate(cash_flows):
        npv += cf / ((1.0 + discount_rate) ** idx)
    return float(npv)


def benefit_investment_ratio(benefits: float, investment: float) -> float:
    """Simple ratio metric; guard against division by zero."""
    if investment == 0:
        return 0.0
    return float(benefits / investment)


# --------------------------------------------------------------------------- #
# Failure-rate heuristics                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class WeibullEstimate:
    shape: float
    scale: float


def estimate_weibull_from_mtbf(mtbf_hours: float, variability_factor: float = 1.5) -> WeibullEstimate:
    """
    Heuristic Weibull parameters constructed from MTBF and variability factor.

    Without raw samples we approximate shape via variability; scale is then
    derived so that the mean matches MTBF.
    """
    shape = max(0.5, min(5.0, variability_factor))
    # Mean of Weibull = scale * Gamma(1 + 1/shape); invert to get scale.
    gamma_term = math.gamma(1.0 + 1.0 / shape)
    scale = mtbf_hours / gamma_term if gamma_term else mtbf_hours
    return WeibullEstimate(shape=shape, scale=scale)


@dataclass
class FailureRatePosterior:
    alpha: float
    beta: float
    mean_rate: float


def bayesian_failure_rate(
    prior_alpha: float,
    prior_beta: float,
    observed_failures: int,
    exposure_hours: float,
) -> FailureRatePosterior:
    """
    Conjugate Gamma posterior for exponential failure rates.

    prior_alpha/prior_beta encode the baseline belief over the rate parameter.
    exposure_hours is the total observed uptime span.
    """
    posterior_alpha = max(prior_alpha + observed_failures, 1e-9)
    posterior_beta = max(prior_beta + exposure_hours, 1e-9)
    mean_rate = posterior_alpha / posterior_beta
    return FailureRatePosterior(alpha=posterior_alpha, beta=posterior_beta, mean_rate=mean_rate)


# --------------------------------------------------------------------------- #
# Simple real-option proxy                                                    #
# --------------------------------------------------------------------------- #


def real_option_deferral_value(npv_now: float, npv_later: float, option_cost: float) -> float:
    """
    Proxy for valuing the option to defer a project.

    Positive value indicates it is preferable to wait (after subtracting
    option_cost).
    """
    return float(max(npv_later - option_cost - npv_now, 0.0))


# --------------------------------------------------------------------------- #
# Monte Carlo bookkeeping                                                     #
# --------------------------------------------------------------------------- #


def summarise_trials(samples: Sequence[float]) -> dict:
    """Return mean/std/min/max for Monte Carlo arrays."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }
