# Dynamic Plant Simulation & Executive Dashboard

## 1. Overview
- **Purpose:** Model a coal-fired power plant with coupled thermodynamics, reliability, and economics, then surface KPIs in a production-grade dashboard.
- **Scope:** Fuel handling, boiler/steam generation, turbine & condenser, feedwater & treatment loops, stochastic reliability, and fully costed financials.
- **What’s included:** Modular simulation engine, timestamped results archive, two-tab Dash dashboard (Executive Summary + Plant Components), and complete configuration in `config/simulation_config.py`.
- **Limitation:** The simulation assumes first-order dynamics for all plant subsystems. A more detailed representation (e.g., second-order models for each component) is required to achieve higher accuracy.

## 2. Repository Structure


project_root/
src/...
config/...
sim_out/...
dashboard/...
README.md

- `src/simulation_core.py` – orchestrates the timestep loop, reliability sampling, and I/O per run.
- `src/reliability_models.py` – exponential MTBF/MTTR processes plus composite reliability scoring.
- `src/economic_models.py` – converts energy/fuel usage into revenue, costs, margin, and emissions metrics.
- `src/plotting/` – pure Plotly figure builders for the dashboard tabs.
- `src/utils/io.py` – run discovery, CSV/JSON load/save helpers, cache reset utilities.
- `src/utils/math_tools.py` – common response filters, clamps, and random variation utilities.
- `config/simulation_config.py` – single source of truth for execution and economic assumptions.
- `dashboard/app.py` – Dash/Plotly application with Executive Summary and Plant Components tabs.
- `sim_out/` – auto-generated archive of simulation runs (`sim_run_YYYY-MM-DD_HH-MM/`).
- `README.md` – this documentation.

## 3. Quick Start
### Prerequisites
- Python 3.9+
- Packages: numpy, pandas, plotly, dash

### Installation
```bash
pip install -r requirements.txt

Configure

Edit config/simulation_config.py to set:

SIM_DAYS, DT_SECONDS, MC_TRIALS

Economic assumptions: coal price, electricity price, O&M, multipliers, inflation, etc.

Run Simulation
python -m src.simulation_core


Outputs will appear under sim_out/sim_run_YYYY-MM-DD_HH-MM/.

Launch Dashboard
python dashboard/app.py


Use the dropdown to select a simulation run.
```

## 4. Data Schema
`results.csv`

Columns:

`t_arr (s)`, `coal_flow (kg/s)`, `inventory (kg)`, `clean_flow (kg/s)`, `fuel_energy (MWth)`,

`steam_flow (kg/s)`, `boiler_gas_temp (°C)`, `air_temp (°C)`, `fw_temp_eco (°C)`, `steam_temp_sig (°C)`,

`power_W (W)`, `exhaust_temp (°C)`, `cond_temp (°C)`, `cw_temp (°C)`,

`fw_meas (kg/s)`, `fw_temp_fwh (°C)`, `water_flow (kg/s)`, `water_temp (°C)`, `reliability_index (%)`

`reliability.json`

Example fields: `uptime_percent`, `mtbf_hours`, `mttr_hours`, `downtime_hours`, `reliability_index`, `event_count`, `events` (list of `{start_time_hours, duration_hours, component}`), `monte_carlo` (mean/std of uptime & downtime).

`KPIs.json`

Example fields: `total_energy_mwh`, `cost_per_MWh`, `revenue`, `coal_cost`, `om_cost`, `total_cost`, `net_profit`, `profit_margin`, `capacity_factor_percent`, `uptime_percent`, `availability`, `coal_consumed_tons`, `carbon_emissions_tons`.

Auxiliary:

- `assumptions.json` mirrors `config/simulation_config.py` so each run records the drivers used.

## 5. Configuration

All modifiable parameters live in `config/simulation_config.py`.

Key controls:

- `SIM_DAYS`, `DT_SECONDS`, `MC_TRIALS`, `RANDOM_SEED` – execution horizon (days), resolution, Monte Carlo trials, reproducibility.
- Process: `INITIAL_INVENTORY_KG`, `COAL_DELIVERY_RATE_KG_S`, `COAL_FLOW_BASE_KG_S`, `COAL_FLOW_VARIATION_KG_S`, `COAL_FLOW_MAX_KG_S`, `CLEANING_EFFICIENCY`, `COAL_LHV_MJ_PER_KG`, `BOILER_EFFICIENCY`, `STEAM_SPECIFIC_ENERGY_KJ_PER_KG`, `TURBINE_ISENTROPIC_EFFICIENCY`, `GENERATOR_EFFICIENCY`, `RATED_POWER_MW`, time constants (`FUEL_TO_STEAM_LAG_S`, `TEMP_RESPONSE_TAU_S`, `COAL_FLOW_RESPONSE_TAU_S`, `HANDLING_RESPONSE_TAU_S`, `COOLING_RESPONSE_TAU_S`, `FW_FLOW_RESPONSE_TAU_S`, `WATER_FLOW_RESPONSE_TAU_S`) and gains (`BOILER_TEMP_GAIN`, `AIR_TEMP_GAIN`, `ECO_TEMP_GAIN`, etc.).
- Reliability: `MTBF_HOURS`, `MTTR_HOURS`, `RELIABILITY_WEIGHTING`, `RELIABILITY_DEGRADATION_RATE`.
- Economics: `COAL_PRICE_PER_TON`, `ELECTRICITY_PRICE_PER_MWH`, `OM_COST_PER_HOUR`, `MARKET_PRICE_MULTIPLIER`, `INFLATION_FACTOR`, `EMISSIONS_FACTOR_TONCO2_PER_MWH`, `VARIABLE_OM_COST_PER_MWH`.

These parameters drive the simulator: dynamics consume flow/efficiency values, reliability sampling uses MTBF/MTTR, and KPI calculations read prices/multipliers to compute revenue, cost per MWh, profit margin, and emissions impact.

## 6. Mathematical Models

### 6.1 Fuel & Energy

- Inventory balance: `inventory_{k+1} = inventory_k + (COAL_DELIVERY_RATE - coal_flow_k) * Δt`.
- Coal feed lag: `coal_flow_{k+1} = coal_flow_k + (coal_cmd - coal_flow_k) * (1 - e^{-Δt / COAL_FLOW_RESPONSE_TAU_S})`.
- Cleaning lag: `clean_flow_{k+1} = clean_flow_k + (CLEANING_EFFICIENCY * coal_flow_k - clean_flow_k) * (1 - e^{-Δt / HANDLING_RESPONSE_TAU_S})`.
- Fuel energy rate: `fuel_energy_MW = clean_flow * COAL_LHV_MJ_PER_KG`.

### 6.2 Boiler / Steam Path

- Steam generation: `steam_flow_target = (fuel_energy_MW * 10^6 * BOILER_EFFICIENCY) / (STEAM_SPECIFIC_ENERGY_KJ_PER_KG * 10^3)`.
- Steam flow lag: `steam_flow_{k+1} = steam_flow_k + (steam_flow_target - steam_flow_k) * (1 - e^{-Δt / FUEL_TO_STEAM_LAG_S})`.
- Boiler gas temperature: `T_boiler = T_boiler + (T_set + BOILER_TEMP_GAIN * Δfuel - T_boiler) * (1 - e^{-Δt / TEMP_RESPONSE_TAU_S})`.
- Air preheater: `T_air = T_air + (AIR_TEMP_GAIN * demand_wave + T_set - T_air) * (1 - e^{-Δt / TEMP_RESPONSE_TAU_S})`.
- Economizer & steam signal temperatures follow similar first-order updates using `ECO_TEMP_GAIN` and `STEAM_TEMP_GAIN * (power_ratio) * STEAM_TEMP_RATIO_SCALE`.

### 6.3 Turbine & Generator

- Power conversion: `power_W = steam_flow * STEAM_SPECIFIC_ENERGY_J * TURBINE_ISENTROPIC_EFFICIENCY * GENERATOR_EFFICIENCY * availability`.
- Exhaust temperature: `T_exhaust = EXHAUST_TEMP_BASE_C - EXHAUST_TEMP_OUTAGE_DROP_C * (1 - availability) + EXHAUST_TEMP_AMPLITUDE_C * demand_wave`, filtered with `TEMP_RESPONSE_TAU_S`.

### 6.4 Condenser & Cooling

- Condenser temp: `T_cond = T_cond + (CONDENSER_TEMP_SET_C + COND_TEMP_OUTAGE_RISE_C * (1 - availability) + COND_TEMP_AMPLITUDE_C * demand_wave - T_cond) * (1 - e^{-Δt / COOLING_RESPONSE_TAU_S})`.
- Cooling water outlet: `T_cw = T_cw + (COOLING_WATER_OUTLET_SET_C + CW_TEMP_GAIN * (T_cond - set) - T_cw) * (1 - e^{-Δt / COOLING_RESPONSE_TAU_S})`.

### 6.5 Feedwater & Treatment

- Feedwater flow: `fw_meas = fw_meas + (FEEDWATER_FLOW_SETPOINT + FW_FLOW_GAIN * (steam_flow - baseline) - fw_meas) * (1 - e^{-Δt / FW_FLOW_RESPONSE_TAU_S})`.
- Feedwater heater temperature: `fw_temp = fw_temp + (FEEDWATER_HEATER_SET_C + FW_TEMP_GAIN * (T_cw - set) - fw_temp) * (1 - e^{-Δt / TEMP_RESPONSE_TAU_S})`.
- Makeup water flow: `water_flow = water_flow + (MAKEUP_WATER_FLOW_BASE + WATER_FLOW_GAIN * |fw_target - set| - water_flow) * (1 - e^{-Δt / WATER_FLOW_RESPONSE_TAU_S}) * availability`.
- Makeup water temperature: `water_temp = water_temp + (MAKEUP_WATER_TEMP_SET_C + WATER_TEMP_GAIN * (AMBIENT_TEMP_C - set) - water_temp) * (1 - e^{-Δt / COOLING_RESPONSE_TAU_S})`.

### 6.6 Reliability

- Failure times drawn from `Exp(1 / MTBF_HOURS)`; repair durations from `Exp(1 / MTTR_HOURS)`. Availability is toggled to zero during outages.
- Metrics: `uptime_percent = (Σ availability * Δt) / total_time`, `downtime_hours = total_time - uptime_time`, `MTBF_est = uptime_time / max(event_count, 1)`, `MTTR_est = downtime_time / max(event_count, 1)`.
- Composite reliability index: `RI = RELIABILITY_WEIGHTING * uptime + (1 - RELIABILITY_WEIGHTING) * (1 - RELIABILITY_DEGRADATION_RATE * (1 - mean_performance))`, reported in percent.

### 6.7 Economics

- Energy integration: `E_MWh = (∫ power_W dt) / 3.6e9`.
- Coal usage: `m_coal = ∫ coal_flow dt`.
- Revenue: `revenue = E_MWh * ELECTRICITY_PRICE_PER_MWH * MARKET_PRICE_MULTIPLIER * INFLATION_FACTOR`.
- Capacity factor: `capacity_factor_percent = (E_MWh / (RATED_POWER_MW * SIM_DAYS * 24)) * 100`.
- Coal cost: `coal_cost = (m_coal / 1000) * COAL_PRICE_PER_TON`.
- O&M: `om_cost = uptime_hours * OM_COST_PER_HOUR + E_MWh * VARIABLE_OM_COST_PER_MWH`.
- Profit: `net_profit = revenue - (coal_cost + om_cost)`, `cost_per_MWh = total_cost / E_MWh`, `profit_margin = net_profit / revenue`.

## 7. Dashboard

**Executive Summary**
- KPI grid covering energy, financials (cost per MWh, revenue, O&M, net profit, profit margin), and reliability stats (uptime, MTBF, downtime).
- Stacked cost bar, revenue vs cost vs profit comparison, and reliability gauge.
- Assumption panel lists active economic drivers loaded from configuration.

**Plant Components**
- Dropdown to pick any archived run in `sim_out/`.
- Single 4×3 subplot figure linking subsystem dynamics:
  - Fuel path: coal flow & inventory, clean flow & fuel energy.
  - Boiler/steam: steam flow with boiler gas temperature, air preheater, economizer, superheat.
  - Turbine & cooling: power vs exhaust, condenser vs cooling water, reliability trend.
  - Feedwater/treatment: feedwater flow, heater temperature, makeup flow vs temperature.
- All traces load directly from `results.csv`; no in-memory mock data.

## 8. Why This Structure (vs. Layer-by-Layer)

- Separation of concerns keeps simulation, economics, reliability, plotting, and configuration decoupled—facilitating targeted tests and safe refactors.
- Scalability improves: adding a new subsystem or cost center touches only the relevant module, not a monolithic script.
- Clear data contract (`results.csv`, `reliability.json`, `KPIs.json`) enables the dashboard to stay in sync with the simulator without hidden dependencies.
- Ready for future ML, Monte Carlo, or distributed execution because the simulation core emits self-contained artefacts consumable by external services.

## 9. Testing & Validation

- Recommended unit tests: verify `economic_models.compute_financials` under different price assumptions, and `reliability_models.simulate_availability_profile` for expected uptime statistics.
- Sanity checks: energy balance (fuel energy vs electrical export), inventory non-negativity, KPI ranges (cost per MWh positive), and dashboard loading for each new run.

## 10. Roadmap
- Improve The modeling accuracy of the Subsystems by choosing appropriate plant dynamics.
- Add carbon capture subsystem with associated dynamics and cost streams.
- Introduce optimisation/RL controllers for fuel and feedwater scheduling.
- Export Parquet archives with run indices to accelerate analytics on large batches.



