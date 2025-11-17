"""
Dash application providing Executive Summary and Plant Components views.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running this module directly via `python dashboard/app.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from config.simulation_config import CONFIG
from src.plotting.executive_summary import (
    build_cost_breakdown,
    build_financial_outcome,
    build_reliability_indicator,
)
from src.plotting.plant_components import build_components_figure
from src.utils import io as io_utils


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"\u20B9{value:,.0f}"


def format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def _dropdown_options() -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for path in io_utils.list_run_directories():
        options.append({"label": path.name, "value": path.name})
    return options


@lru_cache(maxsize=32)
def load_run(run_name: str) -> Dict[str, Any]:
    run_path = Path("sim_out") / run_name
    if not run_path.exists():
        return {}
    results = io_utils.load_csv(run_path / "results.csv")
    reliability = io_utils.load_json(run_path / "reliability.json")
    kpis = io_utils.load_json(run_path / "KPIs.json")
    assumptions_raw = io_utils.load_json(run_path / "assumptions.json")
    assumptions = assumptions_raw or CONFIG.as_dict()
    if "SIM_HOURS" not in assumptions:
        sim_days = assumptions.get("SIM_DAYS")
        if sim_days is not None:
            assumptions["SIM_HOURS"] = float(sim_days) * 24.0
        else:
            assumptions["SIM_HOURS"] = CONFIG.SIM_HOURS
    run_hours: Optional[float] = None
    if not results.empty and "t_arr" in results.columns:
        run_hours = float(results["t_arr"].max() / 3600.0)
    if run_hours is None and assumptions.get("SIM_HOURS") is not None:
        run_hours = float(assumptions.get("SIM_HOURS"))
    return {
        "path": run_path,
        "results": results,
        "reliability": reliability,
        "kpis": kpis,
        "assumptions": assumptions or CONFIG.as_dict(),
        "run_hours": run_hours,
    }


def _kpi_card(title: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "0.85rem", "color": "#6c757d"}),
            html.Div(value, style={"fontSize": "1.2rem", "fontWeight": "600"}),
        ],
        style={
            "padding": "12px 16px",
            "border": "1px solid #dee2e6",
            "borderRadius": "8px",
            "backgroundColor": "#fff",
            "minWidth": "160px",
        },
    )


def _assumptions_panel(assumptions: Dict[str, Any]) -> html.Div:
    fields = [
        ("SIM_DAYS", "Simulation duration (days)"),
        ("COAL_PRICE_PER_TON", "Coal price (per ton)"),
        ("ELECTRICITY_PRICE_PER_MWH", "Electricity price (per MWh)"),
        ("OM_COST_PER_HOUR", "O&M cost (per hour)"),
        ("MARKET_PRICE_MULTIPLIER", "Market multiplier"),
        ("INFLATION_FACTOR", "Inflation factor"),
        ("MC_TRIALS", "Monte Carlo trials"),
        ("DISCOUNT_RATE", "Discount rate"),
        ("PROJECT_YEARS", "Project horizon (years)"),
        ("CAPITAL_EXPENDITURE", "Capital expenditure"),
    ]
    rows = []
    for key, label in fields:
        value = assumptions.get(key, getattr(CONFIG, key))
        display_value: str
        if key == "CAPITAL_EXPENDITURE":
            display_value = format_currency(float(value))
        elif key in {"MC_TRIALS", "PROJECT_YEARS"}:
            display_value = format_number(float(value), 0)
        elif key == "DISCOUNT_RATE":
            display_value = f"{format_number(float(value) * 100.0, 2)} %"
        else:
            display_value = format_number(float(value), 2)
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(display_value),
                ]
            )
        )
    return html.Div(
        [
            html.H4("Active Assumptions", style={"marginTop": "0"}),
            html.Table(
                [
                    html.Tbody(rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            ),
        ],
        style={
            "border": "1px solid #dee2e6",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "backgroundColor": "#f8f9fa",
        },
    )


def _monte_carlo_panel(reliability: Dict[str, Any]) -> html.Div:
    mc = reliability.get("monte_carlo", {})
    if not mc:
        return html.Div(
            [
                html.H4("Monte Carlo Reliability", style={"marginTop": "0"}),
                html.Div("No Monte Carlo statistics available for this run.", style={"color": "#6c757d"}),
            ],
            style={
                "border": "1px solid #dee2e6",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "backgroundColor": "#fff",
            },
        )

    trial_count = mc.get("trial_count")
    heading = "Monte Carlo Reliability"
    if trial_count:
        heading = f"{heading} ({int(trial_count)} trials)"

    metrics = [
        ("Uptime mean (%)", format_number(mc.get("uptime_percent_mean"), 2)),
        ("Uptime std (%)", format_number(mc.get("uptime_percent_std"), 2)),
        ("Downtime mean (h)", format_number(mc.get("downtime_hours_mean"), 2)),
        ("Downtime std (h)", format_number(mc.get("downtime_hours_std"), 2)),
    ]

    cards = [
        _kpi_card(label, value)
        for label, value in metrics
    ]
    financial = mc.get("financial", {})
    fin_cards: List[html.Div] = []
    profit_stats = financial.get("profit", {})
    if profit_stats:
        fin_cards.extend(
            [
                _kpi_card("Profit mean", format_currency(profit_stats.get("mean"))),
                _kpi_card("Profit std", format_currency(profit_stats.get("std"))),
                _kpi_card(
                    f"Profit VaR ({format_number(profit_stats.get('confidence'), 2)})",
                    format_currency(profit_stats.get("value_at_risk")),
                ),
                _kpi_card(
                    f"Profit CVaR ({format_number(profit_stats.get('confidence'), 2)})",
                    format_currency(profit_stats.get("conditional_value_at_risk")),
                ),
            ]
        )
    cost_stats = financial.get("cost", {})
    if cost_stats:
        fin_cards.extend(
            [
                _kpi_card("Cost mean", format_currency(cost_stats.get("mean"))),
                _kpi_card("Cost std", format_currency(cost_stats.get("std"))),
            ]
        )

    children: List[Any] = [
        html.H4(heading, style={"margin": "0 0 8px 0"}),
        html.Div(
            cards,
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "12px"},
        ),
    ]
    if fin_cards:
        children.append(html.Hr(style={"borderColor": "#dee2e6"}))
        children.append(html.H4("Financial Risk Snapshots", style={"margin": "0 0 8px 0"}))
        children.append(
            html.Div(
                fin_cards,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                    "gap": "12px",
                },
            )
        )

    return html.Div(
        children,
        style={
            "border": "1px solid #dee2e6",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "backgroundColor": "#fdfdff",
        },
    )


def build_executive_tab(run_data: Dict[str, Any]) -> html.Div:
    kpis = run_data.get("kpis", {})
    reliability = run_data.get("reliability", {})
    run_hours = run_data.get("run_hours") or run_data.get("assumptions", {}).get("SIM_HOURS")
    run_days = (run_hours / 24.0) if run_hours is not None else None
    results = []
    results.append(
        html.Div(
            [
                _kpi_card("Run duration (h)", format_number(run_hours, 1)),
                _kpi_card("Run duration (days)", format_number(run_days, 2)),
                _kpi_card("Total energy (MWh)", format_number(kpis.get("total_energy_mwh"), 1)),
                _kpi_card("Capacity factor (%)", format_number(kpis.get("capacity_factor_percent"), 1)),
                _kpi_card("Cost per MWh", format_currency(kpis.get("cost_per_MWh"))),
                _kpi_card("Revenue", format_currency(kpis.get("revenue"))),
                _kpi_card("O&M cost", format_currency(kpis.get("om_cost"))),
                _kpi_card("Net profit", format_currency(kpis.get("net_profit"))),
                _kpi_card("Profit margin (%)", format_number(kpis.get("profit_margin"), 1)),
                _kpi_card("Uptime (%)", format_number(reliability.get("uptime_percent"), 1)),
                _kpi_card("MTBF (h)", format_number(reliability.get("mtbf_hours"), 1)),
                _kpi_card("Downtime (h)", format_number(reliability.get("downtime_hours"), 1)),
            ],
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"},
        )
    )

    risk_cards = [
        _kpi_card("NPV", format_currency(kpis.get("npv"))),
        _kpi_card("Benefit/Investment", format_number(kpis.get("benefit_investment_ratio"), 2)),
        _kpi_card("Real option value", format_currency(kpis.get("real_option_value"))),
        _kpi_card("Profit VaR", format_currency(kpis.get("profit_value_at_risk"))),
        _kpi_card("Profit CVaR", format_currency(kpis.get("profit_cvar"))),
        _kpi_card("PERT expected profit", format_currency(kpis.get("pert_expected_profit"))),
        _kpi_card("PERT profit σ", format_currency(kpis.get("pert_profit_std"))),
        _kpi_card("MC trials", format_number(kpis.get("monte_carlo_trials"), 0)),
    ]
    results.append(
        html.Div(
            risk_cards,
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"},
        )
    )

    reliability_cards = [
        _kpi_card("Weibull shape", format_number(reliability.get("weibull_shape"), 2)),
        _kpi_card("Weibull scale (h)", format_number(reliability.get("weibull_scale"), 1)),
        _kpi_card("Bayesian MTBF (h)", format_number(reliability.get("bayesian_mtbf_hours"), 1)),
    ]
    results.append(
        html.Div(
            reliability_cards,
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"},
        )
    )

    graphs = [
        dcc.Graph(figure=build_financial_outcome(kpis)),
        dcc.Graph(figure=build_cost_breakdown(kpis)),
        dcc.Graph(figure=build_reliability_indicator(reliability)),
    ]
    results.append(html.Div(graphs, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "16px"}))
    results.append(_monte_carlo_panel(reliability))
    results.append(_assumptions_panel(run_data.get("assumptions", {})))
    return html.Div(results, style={"display": "grid", "gap": "16px"})


def build_components_tab(run_data: Dict[str, Any]) -> html.Div:
    results_df: pd.DataFrame = run_data.get("results", pd.DataFrame())
    figure = build_components_figure(results_df)
    if figure.data:
        return html.Div([dcc.Graph(figure=figure)], style={"marginTop": "16px"})
    return html.Div("No component data available.", style={"color": "#6c757d"})


app = Dash(__name__)
app.title = "Plant Simulation Dashboard"

options = _dropdown_options()
default_value = options[0]["value"] if options else None

if not options:
    app.layout = html.Div(
        [
            html.H1("Plant Simulation Dashboard"),
            html.P("No simulation runs found. Execute `python -m src.simulation_core` to generate results.", style={"color": "#6c757d"}),
        ],
        style={"maxWidth": "960px", "margin": "0 auto", "padding": "24px"},
    )
else:
    app.layout = html.Div(
        [
            html.H1("Plant Simulation Dashboard"),
            html.Div(
                [
                    html.Span("Select run:", style={"fontWeight": "600"}),
                    dcc.Dropdown(id="run-selector", options=options, value=default_value, clearable=False, style={"minWidth": "280px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "16px"},
            ),
            dcc.Tabs(
                id="view-tabs",
                value="executive",
                children=[
                    dcc.Tab(label="Executive Summary", value="executive"),
                    dcc.Tab(label="Plant Components", value="components"),
                ],
            ),
            html.Div(id="tab-content", style={"marginTop": "16px"}),
        ],
        style={"maxWidth": "1280px", "margin": "0 auto", "padding": "24px"},
    )


@app.callback(Output("tab-content", "children"), Input("run-selector", "value"), Input("view-tabs", "value"))
def render_tab(run_name: Optional[str], tab_value: str) -> Any:
    if not run_name:
        return html.Div("Select a simulation run to display results.", style={"color": "#6c757d"})
    run_data = load_run(run_name)
    if not run_data:
        return html.Div("Run data could not be loaded.", style={"color": "red"})

    header_hours = run_data.get("run_hours")
    header_days = (header_hours / 24.0) if header_hours is not None else None
    header = html.Div(
        [
            html.Div(f"Run folder: {run_data['path'].name}", style={"fontWeight": "600"}),
            html.Div(
                f"Total run time: {format_number(header_hours, 2)} h"
                + (f" ({format_number(header_days, 2)} d)" if header_days is not None else ""),
                style={"color": "#495057"},
            )
        ],
        style={
            "backgroundColor": "#f1f3f5",
            "border": "1px solid #dee2e6",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "marginBottom": "16px",
        },
    )

    if tab_value == "components":
        content = build_components_tab(run_data)
    else:
        content = build_executive_tab(run_data)
    return html.Div([header, content])


if __name__ == "__main__":
    app.run(debug=False)
