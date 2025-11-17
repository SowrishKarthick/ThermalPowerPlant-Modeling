"""
Economic and KPI calculations based on simulation outputs.

Functions in this module convert raw time-series data into aggregate financial
metrics for the executive summary dashboard.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from config.simulation_config import CONFIG, SimulationConfig


def compute_energy_outputs(power_w: np.ndarray, dt_seconds: float) -> Dict[str, float]:
    """Calculate cumulative electrical energy production."""
    energy_joules = float(np.trapezoid(power_w, dx=dt_seconds))
    energy_mwh = energy_joules / 3.6e9
    return {
        "total_energy_mwh": energy_mwh,
        "total_energy_j": energy_joules,
    }


def compute_fuel_usage(coal_flow: np.ndarray, dt_seconds: float) -> Dict[str, float]:
    """Integrate coal mass flow to obtain consumed tonnage."""
    mass_kg = float(np.trapezoid(coal_flow, dx=dt_seconds))
    return {
        "coal_consumed_kg": mass_kg,
        "coal_consumed_tons": mass_kg / 1000.0,
    }


def compute_financials(
    energy: Dict[str, float],
    fuel_usage: Dict[str, float],
    uptime_hours: float,
    config: SimulationConfig | None = None,
) -> Dict[str, float]:
    """Derive revenue, cost, and profitability metrics."""
    cfg = config or CONFIG
    energy_mwh = energy["total_energy_mwh"]
    coal_tons = fuel_usage["coal_consumed_tons"]

    coal_cost = coal_tons * cfg.COAL_PRICE_PER_TON
    variable_om = energy_mwh * cfg.VARIABLE_OM_COST_PER_MWH
    fixed_om = uptime_hours * cfg.OM_COST_PER_HOUR
    om_cost = variable_om + fixed_om

    revenue = (
        energy_mwh
        * cfg.ELECTRICITY_PRICE_PER_MWH
        * cfg.MARKET_PRICE_MULTIPLIER
        * cfg.INFLATION_FACTOR
    )

    total_cost = coal_cost + om_cost
    net_profit = revenue - total_cost
    profit_margin = (net_profit / revenue) if revenue > 0 else 0.0
    cost_per_mwh = (total_cost / energy_mwh) if energy_mwh > 0 else 0.0

    carbon_emissions = (
        energy_mwh * cfg.EMISSIONS_FACTOR_TONCO2_PER_MWH
    )

    return {
        "revenue": revenue,
        "coal_cost": coal_cost,
        "om_cost": om_cost,
        "total_cost": total_cost,
        "net_profit": net_profit,
        "profit_margin": profit_margin * 100.0,
        "cost_per_MWh": cost_per_mwh,
        "carbon_emissions_tons": carbon_emissions,
    }
