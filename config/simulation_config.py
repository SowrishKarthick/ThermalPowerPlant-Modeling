"""
Central configuration for the dynamic plant simulation and dashboard.

All user-modifiable parameters that influence process dynamics, reliability,
and economics are defined here to ensure a single source of truth. The rest of
the codebase imports values from this module so that adjustments only require
editing this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """Container for all simulation and economic assumptions."""

    # Core execution controls -------------------------------------------------
    SIM_DAYS: float = 120
    DT_SECONDS: float = 60.0
    MC_TRIALS: int = 5000
    RANDOM_SEED: int = 1234
    DISCOUNT_RATE: float = 0.08
    PROJECT_YEARS: int = 10
    CAPITAL_EXPENDITURE: float = 20_000_000.0
    OPTION_DEFER_COST: float = 1_500_000.0
    PROFIT_PERT_MULTIPLIERS: Tuple[float, float, float] = (0.85, 1.0, 1.2)
    COST_PERT_MULTIPLIERS: Tuple[float, float, float] = (0.9, 1.0, 1.15)
    RISK_CONFIDENCE: float = 0.95

    # Process & thermodynamic parameters -------------------------------------
    INITIAL_INVENTORY_KG: float = 2.5e6  # Stored coal at t=0
    COAL_DELIVERY_RATE_KG_S: float = 60.0  # Replenishment from rail/yard
    COAL_FLOW_BASE_KG_S: float = 55.0  # Nominal coal feed command
    COAL_FLOW_VARIATION_KG_S: float = 8.0  # Amplitude of demand fluctuation
    COAL_FLOW_MAX_KG_S: float = 80.0  # Physical conveyor limit
    CLEANING_EFFICIENCY: float = 0.97  # Mass fraction after handling losses
    COAL_LHV_MJ_PER_KG: float = 24.0  # Lower heating value
    BOILER_EFFICIENCY: float = 0.88
    STEAM_SPECIFIC_ENERGY_KJ_PER_KG: float = 2800.0
    TURBINE_ISENTROPIC_EFFICIENCY: float = 0.44
    GENERATOR_EFFICIENCY: float = 0.97
    RATED_POWER_MW: float = 550.0
    MAKEUP_WATER_FLOW_BASE_KG_S: float = 45.0
    FEEDWATER_FLOW_SETPOINT_KG_S: float = 55.0
    FUEL_TO_STEAM_LAG_S: float = 180.0
    TEMP_RESPONSE_TAU_S: float = 240.0
    COOLING_RESPONSE_TAU_S: float = 360.0
    COAL_FLOW_RESPONSE_TAU_S: float = 45.0
    HANDLING_RESPONSE_TAU_S: float = 30.0
    WATER_FLOW_RESPONSE_TAU_S: float = 90.0
    FW_FLOW_RESPONSE_TAU_S: float = 60.0
    COAL_COMMAND_WAVE_PERIOD_H: float = 8.0
    COAL_COMMAND_NOISE_RATIO: float = 0.3
    STEAM_TEMP_RATIO_SCALE: float = 50.0

    # Gain coefficients for coupling terms -----------------------------------
    BOILER_TEMP_GAIN: float = 0.08
    AIR_TEMP_GAIN: float = 5.0
    ECO_TEMP_GAIN: float = 0.04
    STEAM_TEMP_GAIN: float = 0.06
    EXHAUST_TEMP_BASE_C: float = 420.0
    EXHAUST_TEMP_OUTAGE_DROP_C: float = 45.0
    EXHAUST_TEMP_AMPLITUDE_C: float = 3.0
    COND_TEMP_OUTAGE_RISE_C: float = 2.0
    COND_TEMP_AMPLITUDE_C: float = 0.2
    CW_TEMP_GAIN: float = 0.5
    FW_FLOW_GAIN: float = 0.2
    FW_TEMP_GAIN: float = 0.1
    WATER_FLOW_GAIN: float = 0.1
    WATER_TEMP_GAIN: float = 0.05

    # Ambient & setpoints ----------------------------------------------------
    AMBIENT_TEMP_C: float = 25.0
    BOILER_GAS_TEMP_SET_C: float = 980.0
    AIR_PREHEATER_SET_C: float = 380.0
    ECONOMIZER_OUTLET_SET_C: float = 260.0
    STEAM_TEMP_SET_C: float = 540.0
    CONDENSER_TEMP_SET_C: float = 42.0
    COOLING_WATER_OUTLET_SET_C: float = 32.0
    FEEDWATER_HEATER_SET_C: float = 220.0
    MAKEUP_WATER_TEMP_SET_C: float = 30.0

    # Reliability assumptions -------------------------------------------------
    MTBF_HOURS: float = 72.0
    MTTR_HOURS: float = 3.5
    RELIABILITY_WEIGHTING: float = 0.6  # Weight for uptime vs. performance
    RELIABILITY_DEGRADATION_RATE: float = 0.15  # Impact of derated operation

    # Economic assumptions ----------------------------------------------------
    COAL_PRICE_PER_TON: float = 95.0
    ELECTRICITY_PRICE_PER_MWH: float = 68.0
    OM_COST_PER_HOUR: float = 3500.0
    MARKET_PRICE_MULTIPLIER: float = 1.05
    INFLATION_FACTOR: float = 1.02
    EMISSIONS_FACTOR_TONCO2_PER_MWH: float = 0.95
    VARIABLE_OM_COST_PER_MWH: float = 4.5

    def as_dict(self) -> Dict[str, float]:
        """Return configuration values as a mutable dictionary copy."""
        return dict(asdict(self))

    @property
    def SIM_HOURS(self) -> float:
        """Computed simulation duration in hours."""
        return self.SIM_DAYS * 24.0


CONFIG = SimulationConfig()


def get_config() -> SimulationConfig:
    """Convenience accessor for importing modules."""
    return CONFIG
