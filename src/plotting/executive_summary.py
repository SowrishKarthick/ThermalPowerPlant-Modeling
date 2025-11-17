"""
Plotting helpers for the Executive Summary dashboard tab.
"""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go


def build_cost_breakdown(kpis: Dict[str, float]) -> go.Figure:
    """Return a bar chart summarising major cost components."""
    coal_cost = kpis.get("coal_cost", 0.0)
    om_cost = kpis.get("om_cost", 0.0)

    fig = go.Figure(
        data=[
            go.Bar(name="Coal", x=["Costs"], y=[coal_cost], marker_color="#1f77b4"),
            go.Bar(name="O&M", x=["Costs"], y=[om_cost], marker_color="#ff7f0e"),
        ]
    )
    fig.update_layout(
        title="Cost Breakdown",
        barmode="stack",
        yaxis_title="Value [currency units]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_financial_outcome(kpis: Dict[str, float]) -> go.Figure:
    """Return a bar chart comparing revenue and net profit."""
    revenue = kpis.get("revenue", 0.0)
    cost = kpis.get("total_cost", 0.0)
    profit = kpis.get("net_profit", 0.0)

    fig = go.Figure(
        data=[
            go.Bar(name="Revenue", x=["Financials"], y=[revenue], marker_color="#17becf"),
            go.Bar(name="Total Cost", x=["Financials"], y=[cost], marker_color="#d62728"),
            go.Bar(name="Net Profit", x=["Financials"], y=[profit], marker_color="#9467bd"),
        ]
    )
    fig.update_layout(
        title="Revenue vs Cost vs Profit",
        barmode="group",
        yaxis_title="Value [currency units]",
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_reliability_indicator(reliability: Dict[str, float]) -> go.Figure:
    """Return a gauge indicator for uptime percentage."""
    uptime = reliability.get("uptime_percent", 0.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=uptime,
            number={"suffix": " %"},
            title={"text": "Uptime"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ca02c"},
                "steps": [
                    {"range": [0, 70], "color": "#ff9896"},
                    {"range": [70, 90], "color": "#f7b6d2"},
                    {"range": [90, 100], "color": "#98df8a"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=60, b=40))
    return fig
