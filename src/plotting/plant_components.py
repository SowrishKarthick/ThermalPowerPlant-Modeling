"""
Plotting helpers for the Plant Components dashboard tab.

All functions operate on objects supplied by the dashboard layer to keep the
plotting logic pure and side-effect free.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def format_time_axis(axis) -> None:
    """Apply common formatting to time axes."""
    axis.title = "Time [h]"


def build_components_figure(results: pd.DataFrame) -> go.Figure:
    """Create the 4x3 grouped subplot figure for plant components."""
    if results.empty:
        return go.Figure()

    t_hours = results["t_arr"].to_numpy() / 3600.0

    specs = [
        [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
        [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
        [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": True}],
    ]
    fig = make_subplots(
        rows=4,
        cols=3,
        specs=specs,
        shared_xaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
        subplot_titles=[
            "Coal Flow & Inventory",
            "Clean Flow & Fuel Energy",
            "Steam Flow & Boiler Gas Temp",
            "Boiler Temperatures",
            "Air Preheater Temperature",
            "Steam Temperature Signal",
            "Power & Exhaust Temperature",
            "Condenser & Cooling Water",
            "Reliability Trend",
            "Feedwater Flow",
            "Feedwater Heater Outlet Temp",
            "Water Treatment Flow & Temp",
        ],
    )

    # Row 1 ------------------------------------------------------------------
    inventory_kt = results["inventory"].to_numpy() / 1e3
    fig.add_trace(
        go.Scatter(x=t_hours, y=results["coal_flow"], name="coal_flow [kg/s]", line=dict(color="#1f77b4")),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=inventory_kt,
            name="inventory [kt]",
            line=dict(color="#ff7f0e", dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fuel_energy = results["fuel_energy"].to_numpy()
    fig.add_trace(
        go.Scatter(x=t_hours, y=results["clean_flow"], name="clean_flow [kg/s]", line=dict(color="#2ca02c")),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=fuel_energy,
            name="fuel_energy [MWth]",
            line=dict(color="#d62728", dash="dot"),
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    steam_flow = results["steam_flow"].to_numpy()
    fig.add_trace(
        go.Scatter(x=t_hours, y=steam_flow, name="steam_flow [kg/s]", line=dict(color="#17becf")),
        row=1,
        col=3,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["boiler_gas_temp"],
            name="boiler_gas_temp [°C]",
            line=dict(color="#bcbd22"),
        ),
        row=1,
        col=3,
        secondary_y=True,
    )

    # Row 2 ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["air_temp"],
            name="air_temp [°C]",
            line=dict(color="#9467bd"),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["fw_temp_eco"],
            name="fw_temp_eco [°C]",
            line=dict(color="#8c564b"),
        ),
        row=2,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["steam_temp_sig"],
            name="steam_temp_sig [°C]",
            line=dict(color="#e377c2"),
        ),
        row=2,
        col=3,
        secondary_y=False,
    )

    # Row 3 ------------------------------------------------------------------
    power_mw = results["power_W"].to_numpy() / 1e6
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=power_mw,
            name="power [MW]",
            line=dict(color="#1f77b4"),
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["exhaust_temp"],
            name="exhaust_temp [°C]",
            line=dict(color="#ff9896"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    reliability = results.get("reliability_index")
    if reliability is not None:
        power_scale = float(power_mw.max()) if power_mw.size else 1.0
        if power_scale <= 0:
            power_scale = 1.0
        fig.add_trace(
            go.Scatter(
                x=t_hours,
                y=np.asarray(reliability) * 0.01 * power_scale,
                name="reliability scaled",
                line=dict(color="#2ca02c", dash="dash"),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["cond_temp"],
            name="cond_temp [°C]",
            line=dict(color="#17becf"),
        ),
        row=3,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["cw_temp"],
            name="cw_temp [°C]",
            line=dict(color="#ff7f0e"),
        ),
        row=3,
        col=2,
        secondary_y=True,
    )

    reliability_line = results["reliability_index"].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=reliability_line,
            name="reliability_index [%]",
            line=dict(color="#2ca02c"),
        ),
        row=3,
        col=3,
        secondary_y=False,
    )

    # Row 4 ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["fw_meas"],
            name="fw_meas [kg/s]",
            line=dict(color="#7f7f7f"),
        ),
        row=4,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["fw_temp_fwh"],
            name="fw_temp_fwh [°C]",
            line=dict(color="#bcbd22"),
        ),
        row=4,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["water_flow"],
            name="water_flow [kg/s]",
            line=dict(color="#1f77b4"),
        ),
        row=4,
        col=3,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t_hours,
            y=results["water_temp"],
            name="water_temp [°C]",
            line=dict(color="#d62728"),
        ),
        row=4,
        col=3,
        secondary_y=True,
    )

    # Layout tweaks ----------------------------------------------------------
    for row in range(1, 5):
        for col in range(1, 4):
            axis_name = f"xaxis{(row-1)*3 + col if (row, col) != (1, 1) else ''}"
            fig.layout[axis_name].title = ""
            fig.layout[axis_name].showgrid = True
            fig.layout[axis_name].gridcolor = "rgba(0,0,0,0.1)"
    format_time_axis(fig.layout.xaxis10)
    format_time_axis(fig.layout.xaxis11)
    format_time_axis(fig.layout.xaxis12)

    fig.update_layout(
        height=900,
        legend=dict(orientation="h", yanchor="top", y=1.12, x=0.01),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig
