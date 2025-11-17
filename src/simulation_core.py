"""
Main simulation entrypoint: orchestrates process dynamics, reliability, IO.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running this module directly via `python src/simulation_core.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.simulation_config import CONFIG, SimulationConfig
from src.analytics.enhancements import (
    bayesian_failure_rate,
    benefit_investment_ratio,
    conditional_value_at_risk,
    estimate_weibull_from_mtbf,
    net_present_value,
    pert_estimate,
    real_option_deferral_value,
    sample_triangular,
    summarise_trials,
    value_at_risk,
)
from src.economic_models import compute_energy_outputs, compute_financials, compute_fuel_usage
from src.reliability_models import (
    compute_reliability_metrics,
    simulate_availability_profile,
)
from src.utils import io as io_utils
from src.utils.math_tools import clamp, first_order_response, random_variation


def _baseline_values(cfg: SimulationConfig) -> Dict[str, float]:
    """Create baseline steady-state values for reference."""
    baseline_clean_flow = cfg.COAL_FLOW_BASE_KG_S * cfg.CLEANING_EFFICIENCY
    baseline_fuel_energy_mw = baseline_clean_flow * cfg.COAL_LHV_MJ_PER_KG
    steam_specific_energy_j = cfg.STEAM_SPECIFIC_ENERGY_KJ_PER_KG * 1e3
    baseline_steam_flow = (
        baseline_fuel_energy_mw * 1e6 * cfg.BOILER_EFFICIENCY / steam_specific_energy_j
    )
    baseline_power_w = (
        baseline_steam_flow
        * steam_specific_energy_j
        * cfg.TURBINE_ISENTROPIC_EFFICIENCY
        * cfg.GENERATOR_EFFICIENCY
    )
    return {
        "clean_flow": baseline_clean_flow,
        "fuel_energy_mw": baseline_fuel_energy_mw,
        "steam_flow": baseline_steam_flow,
        "power_w": baseline_power_w,
    }


def simulate_process(
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float | int | List[Dict[str, float | str]]]]:
    """Simulate plant dynamics and return the time-series dataframe and reliability info."""
    total_steps = int(round(cfg.SIM_HOURS * 3600.0 / cfg.DT_SECONDS))
    time_s = np.arange(total_steps) * cfg.DT_SECONDS

    results = {
        "t_arr": time_s,
        "coal_flow": np.zeros(total_steps),
        "inventory": np.zeros(total_steps),
        "clean_flow": np.zeros(total_steps),
        "fuel_energy": np.zeros(total_steps),
        "steam_flow": np.zeros(total_steps),
        "boiler_gas_temp": np.zeros(total_steps),
        "air_temp": np.zeros(total_steps),
        "fw_temp_eco": np.zeros(total_steps),
        "steam_temp_sig": np.zeros(total_steps),
        "power_W": np.zeros(total_steps),
        "exhaust_temp": np.zeros(total_steps),
        "cond_temp": np.zeros(total_steps),
        "cw_temp": np.zeros(total_steps),
        "fw_meas": np.zeros(total_steps),
        "fw_temp_fwh": np.zeros(total_steps),
        "water_flow": np.zeros(total_steps),
        "water_temp": np.zeros(total_steps),
        "reliability_index": np.zeros(total_steps),
    }

    baseline = _baseline_values(cfg)
    availability, events = simulate_availability_profile(
        cfg.SIM_HOURS, cfg.DT_SECONDS, cfg.MTBF_HOURS, cfg.MTTR_HOURS, rng
    )

    # States -----------------------------------------------------------------
    inventory = cfg.INITIAL_INVENTORY_KG
    coal_flow = cfg.COAL_FLOW_BASE_KG_S
    clean_flow = baseline["clean_flow"]
    steam_flow = baseline["steam_flow"]
    boiler_temp = cfg.BOILER_GAS_TEMP_SET_C
    air_temp = cfg.AIR_PREHEATER_SET_C
    eco_temp = cfg.ECONOMIZER_OUTLET_SET_C
    steam_temp = cfg.STEAM_TEMP_SET_C
    exhaust_temp = cfg.EXHAUST_TEMP_BASE_C
    cond_temp = cfg.CONDENSER_TEMP_SET_C
    cw_temp = cfg.COOLING_WATER_OUTLET_SET_C
    fw_flow = cfg.FEEDWATER_FLOW_SETPOINT_KG_S
    fw_temp = cfg.FEEDWATER_HEATER_SET_C
    water_flow = cfg.MAKEUP_WATER_FLOW_BASE_KG_S
    water_temp = cfg.MAKEUP_WATER_TEMP_SET_C

    steam_specific_energy_j = cfg.STEAM_SPECIFIC_ENERGY_KJ_PER_KG * 1e3

    performance_ratio: List[float] = []

    for idx in range(total_steps):
        t_hours = time_s[idx] / 3600.0
        demand_wave = np.sin(
            2.0 * np.pi * t_hours / max(cfg.COAL_COMMAND_WAVE_PERIOD_H, 1e-3)
        )
        coal_cmd = cfg.COAL_FLOW_BASE_KG_S + demand_wave * cfg.COAL_FLOW_VARIATION_KG_S
        coal_cmd += random_variation(
            cfg.COAL_FLOW_VARIATION_KG_S * cfg.COAL_COMMAND_NOISE_RATIO
        )
        coal_cmd = clamp(coal_cmd, 0.0, cfg.COAL_FLOW_MAX_KG_S)

        availability_state = float(availability[idx])
        adjusted_cmd = coal_cmd * availability_state
        coal_flow = first_order_response(
            coal_flow, adjusted_cmd, cfg.DT_SECONDS, cfg.COAL_FLOW_RESPONSE_TAU_S
        )

        max_by_inventory = inventory / max(cfg.DT_SECONDS, 1.0)
        coal_flow = clamp(coal_flow, 0.0, min(cfg.COAL_FLOW_MAX_KG_S, max_by_inventory))

        inventory = max(
            inventory
            + (cfg.COAL_DELIVERY_RATE_KG_S - coal_flow) * cfg.DT_SECONDS,
            0.0,
        )

        clean_flow = first_order_response(
            clean_flow,
            coal_flow * cfg.CLEANING_EFFICIENCY,
            cfg.DT_SECONDS,
            cfg.HANDLING_RESPONSE_TAU_S,
        )
        fuel_energy_mw = clean_flow * cfg.COAL_LHV_MJ_PER_KG
        steam_target = (
            fuel_energy_mw * 1e6 * cfg.BOILER_EFFICIENCY / steam_specific_energy_j
        )
        steam_flow = first_order_response(
            steam_flow,
            steam_target * availability_state,
            cfg.DT_SECONDS,
            cfg.FUEL_TO_STEAM_LAG_S,
        )

        power_target = (
            steam_flow
            * steam_specific_energy_j
            * cfg.TURBINE_ISENTROPIC_EFFICIENCY
            * cfg.GENERATOR_EFFICIENCY
        )
        power_w = power_target * availability_state

        # Temperatures follow first-order dynamics towards setpoints with load coupling
        boiler_temp_target = (
            cfg.BOILER_GAS_TEMP_SET_C
            + cfg.BOILER_TEMP_GAIN * (fuel_energy_mw - baseline["fuel_energy_mw"])
        )
        boiler_temp = first_order_response(
            boiler_temp, boiler_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        air_temp_target = cfg.AIR_PREHEATER_SET_C + cfg.AIR_TEMP_GAIN * demand_wave
        air_temp = first_order_response(
            air_temp, air_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        eco_temp_target = (
            cfg.ECONOMIZER_OUTLET_SET_C
            + cfg.ECO_TEMP_GAIN * (steam_flow - baseline["steam_flow"])
        )
        eco_temp = first_order_response(
            eco_temp, eco_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        steam_temp_target = (
            cfg.STEAM_TEMP_SET_C
            + cfg.STEAM_TEMP_GAIN
            * (power_w / max(baseline["power_w"], 1.0) - 1.0)
            * cfg.STEAM_TEMP_RATIO_SCALE
        )
        steam_temp = first_order_response(
            steam_temp, steam_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        exhaust_temp_target = (
            cfg.EXHAUST_TEMP_BASE_C
            - cfg.EXHAUST_TEMP_OUTAGE_DROP_C * (1.0 - availability_state)
            + cfg.EXHAUST_TEMP_AMPLITUDE_C * demand_wave
        )
        exhaust_temp = first_order_response(
            exhaust_temp, exhaust_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        cond_temp_target = (
            cfg.CONDENSER_TEMP_SET_C
            + cfg.COND_TEMP_OUTAGE_RISE_C * (1.0 - availability_state)
            + cfg.COND_TEMP_AMPLITUDE_C * demand_wave
        )
        cond_temp = first_order_response(
            cond_temp, cond_temp_target, cfg.DT_SECONDS, cfg.COOLING_RESPONSE_TAU_S
        )

        cw_temp_target = cfg.COOLING_WATER_OUTLET_SET_C + 0.5 * (cond_temp - cfg.CONDENSER_TEMP_SET_C)
        cw_temp = first_order_response(
            cw_temp, cw_temp_target, cfg.DT_SECONDS, cfg.COOLING_RESPONSE_TAU_S
        )

        fw_flow_target = (
            cfg.FEEDWATER_FLOW_SETPOINT_KG_S
            + cfg.FW_FLOW_GAIN * (steam_flow - baseline["steam_flow"])
        )
        fw_flow = first_order_response(
            fw_flow, fw_flow_target, cfg.DT_SECONDS, cfg.FW_FLOW_RESPONSE_TAU_S
        )

        fw_temp_target = (
            cfg.FEEDWATER_HEATER_SET_C
            + cfg.FW_TEMP_GAIN * (cw_temp - cfg.COOLING_WATER_OUTLET_SET_C)
        )
        fw_temp = first_order_response(
            fw_temp, fw_temp_target, cfg.DT_SECONDS, cfg.TEMP_RESPONSE_TAU_S
        )

        water_flow_target = (
            cfg.MAKEUP_WATER_FLOW_BASE_KG_S
            + cfg.WATER_FLOW_GAIN
            * abs(fw_flow_target - cfg.FEEDWATER_FLOW_SETPOINT_KG_S)
        )
        water_flow = first_order_response(
            water_flow,
            water_flow_target * availability_state,
            cfg.DT_SECONDS,
            cfg.WATER_FLOW_RESPONSE_TAU_S,
        )

        water_temp_target = (
            cfg.MAKEUP_WATER_TEMP_SET_C
            + cfg.WATER_TEMP_GAIN * (cfg.AMBIENT_TEMP_C - cfg.MAKEUP_WATER_TEMP_SET_C)
        )
        water_temp = first_order_response(
            water_temp, water_temp_target, cfg.DT_SECONDS, cfg.COOLING_RESPONSE_TAU_S
        )

        performance = power_w / max(baseline["power_w"], 1e-3)
        performance_ratio.append(performance)

        results["coal_flow"][idx] = coal_flow
        results["inventory"][idx] = inventory
        results["clean_flow"][idx] = clean_flow
        results["fuel_energy"][idx] = fuel_energy_mw
        results["steam_flow"][idx] = steam_flow
        results["boiler_gas_temp"][idx] = boiler_temp
        results["air_temp"][idx] = air_temp
        results["fw_temp_eco"][idx] = eco_temp
        results["steam_temp_sig"][idx] = steam_temp
        results["power_W"][idx] = power_w
        results["exhaust_temp"][idx] = exhaust_temp
        results["cond_temp"][idx] = cond_temp
        results["cw_temp"][idx] = cw_temp
        results["fw_meas"][idx] = fw_flow
        results["fw_temp_fwh"][idx] = fw_temp
        results["water_flow"][idx] = water_flow
        results["water_temp"][idx] = water_temp
        results["reliability_index"][idx] = availability_state * 100.0

    kpis_reliability = compute_reliability_metrics(
        availability,
        cfg.DT_SECONDS,
        events,
        performance_ratio,
    )
    results_df = pd.DataFrame(results)
    return results_df, kpis_reliability


def run_monte_carlo_reliability(
    cfg: SimulationConfig,
    base_financials: Dict[str, float],
) -> Dict[str, float]:
    """Run Monte Carlo trials to estimate reliability and financial dispersion."""
    rng = np.random.default_rng(cfg.RANDOM_SEED + 999)
    uptime_samples: List[float] = []
    downtime_samples: List[float] = []

    for _ in range(cfg.MC_TRIALS):
        availability, events = simulate_availability_profile(
            cfg.SIM_HOURS, cfg.DT_SECONDS, cfg.MTBF_HOURS, cfg.MTTR_HOURS, rng
        )
        metrics = compute_reliability_metrics(
            availability,
            cfg.DT_SECONDS,
            events,
            performance_ratio=np.ones_like(availability),
        )
        uptime_samples.append(metrics["uptime_percent"])
        downtime_samples.append(metrics["downtime_hours"])

    profit_low, profit_mode, profit_high = cfg.PROFIT_PERT_MULTIPLIERS
    cost_low, cost_mode, cost_high = cfg.COST_PERT_MULTIPLIERS
    if base_financials["net_profit"] == 0.0:
        profit_samples = np.zeros(cfg.MC_TRIALS)
    else:
        profit_samples = base_financials["net_profit"] * sample_triangular(
            rng,
            profit_low,
            profit_mode,
            profit_high,
            size=cfg.MC_TRIALS,
        )
    cost_samples = base_financials["total_cost"] * sample_triangular(
        rng,
        cost_low,
        cost_mode,
        cost_high,
        size=cfg.MC_TRIALS,
    )

    profit_summary = summarise_trials(profit_samples)
    var_profit = value_at_risk(profit_samples, cfg.RISK_CONFIDENCE)
    cvar_profit = conditional_value_at_risk(profit_samples, cfg.RISK_CONFIDENCE)
    profit_summary["value_at_risk"] = var_profit
    profit_summary["conditional_value_at_risk"] = cvar_profit
    profit_summary["confidence"] = cfg.RISK_CONFIDENCE

    cost_summary = summarise_trials(cost_samples)

    return {
        "uptime_percent_mean": float(np.mean(uptime_samples)) if uptime_samples else 0.0,
        "uptime_percent_std": float(np.std(uptime_samples)) if uptime_samples else 0.0,
        "downtime_hours_mean": float(np.mean(downtime_samples)) if downtime_samples else 0.0,
        "downtime_hours_std": float(np.std(downtime_samples)) if downtime_samples else 0.0,
        "trial_count": cfg.MC_TRIALS,
        "financial": {
            "profit": profit_summary,
            "cost": cost_summary,
        },
    }


def run_simulation(output_dir: Path | None = None) -> Path:
    """Execute a full simulation run and persist outputs to disk."""
    cfg = CONFIG
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    results_df, reliability_metrics = simulate_process(cfg, rng)

    energy = compute_energy_outputs(results_df["power_W"].to_numpy(), cfg.DT_SECONDS)
    fuel_usage = compute_fuel_usage(results_df["coal_flow"].to_numpy(), cfg.DT_SECONDS)
    uptime_hours = (reliability_metrics["uptime_percent"] / 100.0) * cfg.SIM_HOURS
    financials = compute_financials(energy, fuel_usage, uptime_hours, cfg)
    availability_stats = run_monte_carlo_reliability(cfg, financials)
    profit_low, profit_mode, profit_high = cfg.PROFIT_PERT_MULTIPLIERS
    cost_low, cost_mode, cost_high = cfg.COST_PERT_MULTIPLIERS
    pessimistic_profit = financials["net_profit"] * profit_low
    optimistic_profit = financials["net_profit"] * profit_high
    pert_profit_mean, pert_profit_var = pert_estimate(
        optimistic_profit,
        financials["net_profit"] * profit_mode,
        pessimistic_profit,
    )
    pert_profit_std = math.sqrt(max(pert_profit_var, 0.0))

    pessimistic_cost = financials["total_cost"] * cost_low
    optimistic_cost = financials["total_cost"] * cost_high
    pert_cost_mean, _ = pert_estimate(
        optimistic_cost,
        financials["total_cost"] * cost_mode,
        pessimistic_cost,
    )

    cash_flows = [-cfg.CAPITAL_EXPENDITURE]
    yearly_profit = financials["net_profit"]
    for _ in range(max(cfg.PROJECT_YEARS, 0)):
        cash_flows.append(yearly_profit)
    project_npv = net_present_value(cash_flows, cfg.DISCOUNT_RATE)
    total_benefits = yearly_profit * max(cfg.PROJECT_YEARS, 0)
    benefit_ratio = benefit_investment_ratio(total_benefits, cfg.CAPITAL_EXPENDITURE)
    deferred_cash_flows = [-cfg.CAPITAL_EXPENDITURE, 0.0]
    for _ in range(max(cfg.PROJECT_YEARS - 1, 0)):
        deferred_cash_flows.append(yearly_profit)
    npv_deferred = net_present_value(deferred_cash_flows, cfg.DISCOUNT_RATE)
    option_value = real_option_deferral_value(project_npv, npv_deferred, cfg.OPTION_DEFER_COST)

    weibull = estimate_weibull_from_mtbf(cfg.MTBF_HOURS, variability_factor=1.0 + cfg.RELIABILITY_DEGRADATION_RATE)
    failure_rate_posterior = bayesian_failure_rate(
        prior_alpha=1.0,
        prior_beta=cfg.MTBF_HOURS,
        observed_failures=int(reliability_metrics["event_count"]),
        exposure_hours=cfg.SIM_HOURS,
    )
    bayes_mtbf = (1.0 / failure_rate_posterior.mean_rate) if failure_rate_posterior.mean_rate > 0 else 0.0

    profit_risk = availability_stats.get("financial", {}).get("profit", {})
    profit_var = profit_risk.get("value_at_risk")
    profit_cvar = profit_risk.get("conditional_value_at_risk")
    capacity_factor = 0.0
    rated_energy = cfg.RATED_POWER_MW * cfg.SIM_HOURS
    if rated_energy > 0:
        capacity_factor = (energy["total_energy_mwh"] / rated_energy) * 100.0

    kpis_payload = {
        **energy,
        **financials,
        "uptime_percent": reliability_metrics["uptime_percent"],
        "availability": reliability_metrics["uptime_percent"] / 100.0,
        "profit_margin": financials["profit_margin"],
        "cost_per_MWh": financials["cost_per_MWh"],
        "coal_consumed_tons": fuel_usage["coal_consumed_tons"],
        "capacity_factor_percent": capacity_factor,
        "pert_expected_profit": pert_profit_mean,
        "pert_profit_std": pert_profit_std,
        "pert_expected_cost": pert_cost_mean,
        "npv": project_npv,
        "benefit_investment_ratio": benefit_ratio,
        "real_option_value": option_value,
        "profit_value_at_risk": profit_var,
        "profit_cvar": profit_cvar,
        "monte_carlo_trials": cfg.MC_TRIALS,
        "monte_carlo_confidence": cfg.RISK_CONFIDENCE,
        "discount_rate": cfg.DISCOUNT_RATE,
        "capital_expenditure": cfg.CAPITAL_EXPENDITURE,
    }

    reliability_payload = {
        **reliability_metrics,
        "monte_carlo": availability_stats,
        "weibull_shape": weibull.shape,
        "weibull_scale": weibull.scale,
        "bayesian_failure_rate": {
            "alpha": failure_rate_posterior.alpha,
            "beta": failure_rate_posterior.beta,
            "mean_rate": failure_rate_posterior.mean_rate,
        },
        "bayesian_mtbf_hours": bayes_mtbf,
    }

    run_path = output_dir or io_utils.create_run_directory(runtime_hours=cfg.SIM_HOURS)
    io_utils.clear_load_caches()
    io_utils.save_dataframe(run_path / "results.csv", results_df)
    io_utils.save_json(run_path / "reliability.json", reliability_payload)
    io_utils.save_json(run_path / "KPIs.json", kpis_payload)

    assumptions = asdict(cfg)
    assumptions["SIM_HOURS"] = cfg.SIM_HOURS
    io_utils.save_json(run_path / "assumptions.json", assumptions)

    return run_path


if __name__ == "__main__":
    run_dir = run_simulation()
    print(f"Simulation complete. Results stored in: {run_dir}")
