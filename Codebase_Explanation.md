# Meta-System Modeling and Simulation Framework — Codebase Explanation

The repository models a thermal power plant as a meta-system that combines process dynamics, availability risk, economic accounting, and interactive visualisation. This explainer walks through the codebase component-by-component so a review panel can trace the full simulation lifecycle.

---

## 1. Overall Simulation Flow

1. **Configuration bootstrap**  
   - `config/simulation_config.py` instantiates a frozen `SimulationConfig` dataclass (`CONFIG`) that holds all execution controls, thermodynamic coefficients, reliability assumptions, and economic prices. Modules import this singleton to guarantee consistent parameters.

2. **Simulation entrypoint**  
   - `src/simulation_core.py` exposes `run_simulation()`, the CLI and programmatic entrypoint. When invoked (either by `python -m src.simulation_core` or from other Python code), it:
     1. Seeds a NumPy random generator using `CONFIG.RANDOM_SEED`.
     2. Calls `simulate_process(CONFIG, rng)` to execute the timestep loop.
     3. Consumes the returned `pandas.DataFrame` and reliability metrics to compute energy, fuel, and finance KPIs.
     4. Persists artefacts (`results.csv`, `reliability.json`, `KPIs.json`, `assumptions.json`) via `src.utils.io`.

3. **Timestep loop (`simulate_process`)**  
   - Calculates steady-state baselines (`_baseline_values`) for power, steam, and fuel.  
   - Preallocates NumPy arrays for all tracked signals (inventory, flows, temperatures, reliability indices).  
   - Generates an availability timeline and event log through `simulate_availability_profile` (renewal process with exponential MTBF/MTTR).  
   - Iterates over `total_steps = SIM_HOURS * 3600 / DT_SECONDS`:
     - Builds a sinusoidal coal demand command plus bounded random noise.
     - Applies outage gating, first-order lag responses (`first_order_response`) and clamps to update coal flow, inventory, clean coal, steam flow, and electrical output.
     - Evolves temperature states and auxiliary loops (air preheater, economiser, condenser, cooling water, feedwater, makeup water) with first-order dynamics tied to load or outages.
     - Records every variable into the preallocated arrays.
   - After the loop, compiles the results into a `DataFrame` and calculates reliability KPIs using recorded availability, performance ratios, and events.

4. **Post-processing & storage**  
   - `compute_energy_outputs`, `compute_fuel_usage`, and `compute_financials` integrate time-series data into KPI aggregates.  
   - Monte Carlo reliability statistics are computed by `run_monte_carlo_reliability`, which reruns the outage generator `MC_TRIALS` times.  
   - `src.utils.io.create_run_directory` makes a timestamped folder under `sim_out/`.  
   - Results are written with `save_dataframe` (CSV) and `save_json` (JSON). Caches in `io` are refreshed so the dashboard retrieves new runs without restarting.

---

## 2. System Behavior and Models

The simulator decomposes the plant into loosely coupled subsystems, each advanced by algebraic updates or first-order difference equations.

- **Fuel Handling & Inventory** (`simulate_process`)
  - Coal demand = nominal command + sinusoidal disturbance + random noise. Availability outages zero the effective command.
  - `first_order_response` mimics conveyor inertia; `clamp` enforces mechanical limits and prevents overdrawing inventory.
  - Inventory integrates coal delivery minus actual outflow each timestep.

- **Coal Cleaning & Boiler Fuel Energy**
  - A first-order lag models cleaning efficiency. Fuel energy (MW thermal) is the clean flow scaled by lower heating value.
  - Steam production target is proportional to fuel energy times boiler efficiency, divided by steam specific enthalpy.

- **Steam Generation & Turbine-Generator**
  - Steam flow tracks the fuel-driven target with combustion-to-steam lag (`FUEL_TO_STEAM_LAG_S`) and is gated by availability.
  - Electrical power is the steam enthalpy rate multiplied by turbine isentropic efficiency and generator efficiency.

- **Thermal Loops (Boiler, Air Preheater, Economiser, Superheater)**
  - Temperatures approach set points plus load-coupled offsets via first-order responses with `TEMP_RESPONSE_TAU_S`. Gains tie deviations to coal demand or steam/power deviations.

- **Exhaust & Cooling Circuit**
  - Exhaust gas temperature decreases during outages (`EXHAUST_TEMP_OUTAGE_DROP_C`) and oscillates with the demand wave.
  - Condenser and cooling water temperatures adjust through `COOLING_RESPONSE_TAU_S`, capturing thermal mass and cooling-tower feedback.

- **Feedwater & Makeup Water**
  - Feedwater flow responds to steam flow changes (`FW_FLOW_GAIN`). Heater outlet temperature tracks cooling water.
  - Makeup water flow depends on feedwater deviation; its temperature nudges toward ambient using `WATER_TEMP_GAIN`.

- **Reliability Layer** (`src/reliability_models.py`)
  - `simulate_availability_profile` draws exponential failure and repair times, writes zero availability during outages, and logs each event.
  - `compute_reliability_metrics` integrates uptime, downtime, MTBF, MTTR, and synthesises a composite reliability index weighted by uptime and performance degradation.

- **Economics & KPIs** (`src/economic_models.py`)
  - Integrates power to energy and coal flow to mass using trapezoidal integration.
  - Applies configuration prices to derive revenue, coal cost, O&M (fixed + variable), carbon cost, total cost, net profit, profit margin, and cost per MWh.

- **Monte Carlo Sampling**
  - `run_monte_carlo_reliability` repeats the outage sequence with unity performance ratios to quantify dispersion in uptime and downtime metrics.

---

## 3. Data and Parameter Handling

- **Parameter management**
  - `SimulationConfig` uses type-annotated fields with defaults for every control, thermodynamic constant, reliability assumption, and economic coefficient.
  - `as_dict()` and the `SIM_HOURS` property provide convenient exports used by the dashboard and archive metadata.
  - Because the dataclass is frozen, mutability is avoided; modules reference either `CONFIG` or accept an injected config instance for testing.

- **Runtime propagation**
  - Simulation modules receive the config explicitly (`simulate_process(cfg, rng)`, `compute_financials(..., config=cfg)`) which aids dependency injection.
  - Output metadata writes the complete assumption map into `assumptions.json` ensuring traceability per run.

- **Randomness and uncertainty**
  - NumPy's `default_rng` seeds deterministic sequences.  
  - Reliability outages rely on `rng.exponential` draws for both failure and repair durations.  
  - Demand noise uses `random_variation` (Python `random.uniform`) to inject bounded variability around the sinusoidal command.
  - Monte Carlo reliability reuses the same configuration but offsets the seed (`RANDOM_SEED + 999`) for independent sampling.

---

## 4. Dashboard and Visualization Logic

- **Framework**  
  - `dashboard/app.py` builds a Plotly Dash application with two tabs: “Executive Summary” and “Plant Components”.

- **Data ingestion**  
  - `load_run(run_name)` reads `results.csv`, `reliability.json`, `KPIs.json`, and `assumptions.json` for the selected archive.  
  - `io.list_run_directories()` populates dropdown options with available run folders.

- **Executive Summary tab**
  - Displays KPI cards (run duration, energy, capacity factor, cost per MWh, revenue, O&M, carbon, net profit, profit margin, uptime, MTBF, downtime).
  - Renders graphs via `src.plotting.executive_summary`:
    - `build_financial_outcome` (grouped bar: revenue, total cost, net profit).
    - `build_cost_breakdown` (stacked bar: coal, O&M, carbon costs).
    - `build_reliability_indicator` (gauge for uptime percent).
  - Shows Monte Carlo statistics and the parameter table (selected economic assumptions).

- **Plant Components tab**
  - Uses `build_components_figure` (Plotly subplot grid) to overlay twelve traces covering fuel flow/inventory, boiler/steam temps, exhaust vs power, condenser vs cooling water, reliability trend, feedwater, and makeup water signals.
  - `results.csv` columns map directly to subplot traces, ensuring fidelity between simulation outputs and visual diagnostics.

- **Interactivity & caching**
  - Dash `@app.callback` responds to dropdown/tab selections, retrieves run data, and supplies it to the tab builders.
  - `functools.lru_cache` memoises run loading, reducing repeated disk reads when toggling tabs.

---

## 5. File and Folder Structure

| Path | Purpose |
| --- | --- |
| `config/` | Central configuration (`simulation_config.py`) declaring all parameters. |
| `src/` | Core Python package with simulation engine, reliability/economic models, utilities, and plotting helpers. |
| `src/utils/` | I/O abstraction (`io.py`) and reusable math utilities (`math_tools.py`). |
| `src/plotting/` | Pure Plotly figure constructors for dashboard tabs. |
| `dashboard/` | Dash web application that surfaces archived results. |
| `sim_out/` | Timestamped run archives (`sim_run_YYYY-MM-DD_HH-MM_<duration>h/`) containing CSV, JSON results, and assumptions. |
| `Project_Presentation.md`, `Codebase_Explanation.md` | Generated documentation tailored for presentations and reviews. |
| `README.md` | High-level overview, quick start, model descriptions, and roadmap. |
| `pyproject.toml`, `requirements.txt` | Package metadata and runtime dependencies (NumPy, pandas, Plotly, Dash). |

Logs are not generated explicitly; instead, reproducible artefacts are captured in `sim_out/`. Visualisations are rendered dynamically in the dashboard rather than stored as static files.

---

## 6. Equations and Computations

Key relationships are implemented with simple algebra or first-order difference equations. In LaTeX-style notation:

- **First-order lag (shared helper `first_order_response`)**
  \[
  x_{k+1} = x_k + \left(1 - e^{-\Delta t / \tau}\right) \left(x^\star - x_k\right)
  \]
  Used for coal flow, clean flow, steam flow, temperatures, and water/flow dynamics.

- **Coal inventory balance**
  \[
  M_{k+1} = \max\!\left(0,\; M_k + \left(\dot{m}_{\text{delivery}} - \dot{m}_{\text{coal}}\right)\Delta t \right)
  \]

- **Fuel energy conversion**
  \[
  \dot{Q}_{\text{fuel}} = \dot{m}_{\text{clean}} \cdot \text{LHV} \quad;\quad
  \dot{m}_{\text{steam}}^{\star} = \frac{\dot{Q}_{\text{fuel}}\cdot\eta_{\text{boiler}}}{h_{\text{steam}}}
  \]

- **Electrical power output**
  \[
  P = \dot{m}_{\text{steam}} \cdot h_{\text{steam}} \cdot \eta_{\text{turbine}} \cdot \eta_{\text{generator}} \cdot A
  \]
  where \(A \in \{0,1\}\) is availability.

- **Reliability metrics** (`compute_reliability_metrics`)
  \[
  \begin{aligned}
  \text{Uptime}_{\%} &= 100 \cdot \frac{\sum A_k \Delta t}{T} \\
  \text{Downtime}_{h} &= T - \sum A_k \Delta t \\
  \text{MTBF} &= \frac{\sum A_k \Delta t}{N_{\text{events}}} \\
  \text{MTTR} &= \frac{T - \sum A_k \Delta t}{N_{\text{events}}} \\
  \text{RI} &= 100 \cdot \text{clip}\left( w \cdot \frac{\sum A_k \Delta t}{T} + (1-w) \left[1 - d \left(1 - \bar{p}\right)\right] \right)
  \end{aligned}
  \]
  with weighting \(w = \text{RELIABILITY\_WEIGHTING}\), degradation rate \(d\), and \(\bar{p}\) the mean performance ratio.

- **Energy & fuel integration** (`numpy.trapezoid`)
  \[
  E_{\text{MWh}} = \frac{1}{3.6\times 10^9} \int P(t) \, dt,\quad
  m_{\text{coal}} = \int \dot{m}_{\text{coal}}(t) \, dt
  \]

- **Economic stack**
  \[
  \begin{aligned}
  \text{Revenue} &= E_{\text{MWh}} \cdot \text{Price}_{\text{elec}} \cdot \text{Multiplier} \cdot \text{Inflation} \\
  \text{Coal Cost} &= m_{\text{coal}}^{\text{ton}} \cdot \text{Price}_{\text{coal}} \\
  \text{O\&M} &= E_{\text{MWh}} \cdot c_{\text{var}} + t_{\text{uptime}} \cdot c_{\text{fixed}} \\
  \text{Carbon Cost} &= E_{\text{MWh}} \cdot f_{\text{CO}_2} \cdot c_{\text{CO}_2} \\
  \text{Net Profit} &= \text{Revenue} - (\text{Coal} + \text{O\&M} + \text{Carbon}) \\
  \text{Profit Margin} &= 100 \cdot \frac{\text{Net Profit}}{\text{Revenue}}
  \end{aligned}
  \]

---

## 7. Extensibility

- **Adding new subsystems**  
  - Introduce new state variables within `simulate_process`; preallocate arrays similar to existing ones; update the timestep loop with the subsystem dynamics and append results to the `results` dictionary.  
  - Use `math_tools.first_order_response` or custom integrators as needed.  
  - Update `results.csv` consumers (dashboard plots) to visualise the added channels.

- **Enhanced reliability or risk layers**  
  - Extend `ReliabilityEvent` with component identifiers, swap in alternative distributions, or link reliability penalties directly to performance ratios.  
  - Monte Carlo logic is isolated, so additional sampling strategies (scenario analysis, parameter sweeps) can be slotted into `run_monte_carlo_reliability` or companion functions.

- **Economic/market modules**  
  - `compute_financials` accepts an optional config; new cost terms can be composed without affecting the simulation loop.  
  - Additional KPIs only require updating JSON payloads and exposing them in dashboard cards or charts.

- **Dashboard integrations**  
  - Plot builders consume pure data (`kpis`, `reliability`, `results`) and return Plotly figures, making it straightforward to add tabs, figures, or export features.  
  - The I/O layer already handles run discovery, so new visualisations automatically inherit the archive indexing.

- **Dependency management**  
  - `pyproject.toml` + `requirements.txt` capture runtime requirements (NumPy, pandas, Plotly, Dash). Subpackages are registered via setuptools with `package-dir = {"": "src"}`.  
  - Modules are decoupled and interact through explicit function arguments, maintaining a clear API surface for unit testing or third-party integration.

---

This documentation reflects the repository state at generation time, aligning the plant simulation code, reliability and economic layers, and Dash dashboard into a coherent narrative for viva or panel review.
