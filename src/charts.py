"""Plotly chart builders shared by the dashboard pages.

Keeping the figures here rather than inline in ``app.py`` means the styling is
applied once and every page looks like part of the same product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import EFFICIENCY_COLORS, EFFICIENCY_ORDER, MODE_COLORS, SENSOR_ENVELOPES

GRID = "rgba(148,163,184,0.25)"
INK = "#0f172a"
MUTED = "#64748b"
ACCENT = "#1F6FB2"

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK),
    margin=dict(l=60, r=24, t=56, b=48),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(**_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=MUTED),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=280)


def efficiency_donut(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure()
    counts = df["Efficiency_Status"].value_counts().reindex(list(EFFICIENCY_ORDER)).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.to_numpy(), hole=0.62, sort=False,
        marker=dict(colors=[EFFICIENCY_COLORS[s] for s in counts.index],
                    line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(size=13),
        hovertemplate="%{label}: %{value:,} readings (%{percent})<extra></extra>",
    ))
    total = int(counts.sum())
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>readings</span>",
                       showarrow=False, font=dict(size=17, color=INK))
    fig.update_layout(showlegend=False)
    return _style(fig, "Efficiency status distribution", height=360)


def stacked_distribution(pct: pd.DataFrame, title: str, xlabel: str = "") -> go.Figure:
    if pct.empty:
        return empty_figure()
    fig = go.Figure()
    for status in EFFICIENCY_ORDER:
        if status not in pct:
            continue
        fig.add_bar(x=[str(i) for i in pct.index], y=pct[status], name=status,
                    marker_color=EFFICIENCY_COLORS[status],
                    hovertemplate=f"{status}: %{{y:.2f}}%<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title="% of readings", xaxis_title=xlabel)
    fig.update_yaxes(range=[0, 100])
    return _style(fig, title)


def sensor_timeseries(df: pd.DataFrame, column: str, freq: str = "D",
                      show_band: bool = True) -> go.Figure:
    """Resampled mean of a sensor with its inter-quartile band."""
    if df.empty or column not in df:
        return empty_figure()
    spec = SENSOR_ENVELOPES.get(column, {})
    series = df.set_index("DateTime")[column].resample(freq)
    agg = series.agg(["mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    agg.columns = ["mean", "q25", "q75"]
    agg = agg.dropna()
    if agg.empty:
        return empty_figure()

    fig = go.Figure()
    if show_band:
        fig.add_trace(go.Scatter(x=agg.index, y=agg["q75"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=agg.index, y=agg["q25"], fill="tonexty",
                                 fillcolor="rgba(31,111,178,0.16)", line=dict(width=0),
                                 name="inter-quartile range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg.index, y=agg["mean"], mode="lines",
                             line=dict(color=ACCENT, width=2), name="mean",
                             hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f}<extra></extra>"))
    if spec.get("warn") is not None:
        fig.add_hline(y=spec["warn"], line=dict(color="#C1382E", width=1.4, dash="dash"),
                      annotation_text="warning threshold",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#C1382E"))
    unit = f" ({spec['unit']})" if spec.get("unit") else ""
    fig.update_layout(yaxis_title=f"{spec.get('label', column)}{unit}", xaxis_title="")
    return _style(fig, f"{spec.get('label', column)} over time")


def machine_ranking(card: pd.DataFrame, column: str, label: str,
                    highlight: int | None = None, ascending: bool = True) -> go.Figure:
    """Horizontal ranking of all machines with 3-sigma control limits drawn."""
    if card.empty or column not in card:
        return empty_figure()
    data = card.sort_values(column, ascending=ascending)
    mean = float(data[column].mean())
    sd = float(data[column].std(ddof=1))
    colors = [
        "#C1382E" if abs(v - mean) > 3 * sd
        else ("#0F766E" if highlight is not None and m == highlight else "#94A3B8")
        for v, m in zip(data[column], data["Machine_ID"])
    ]
    fig = go.Figure(go.Bar(
        x=data[column], y=data["Machine_ID"].astype(str), orientation="h",
        marker_color=colors,
        hovertemplate="Machine %{y}<br>" + label + ": %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=mean, line=dict(color=INK, width=1.6),
                  annotation_text=f"fleet mean {mean:.2f}", annotation_position="top",
                  annotation_font=dict(size=11))
    for sign in (-3, 3):
        fig.add_vline(x=mean + sign * sd, line=dict(color="#C1382E", width=1, dash="dot"))
    span = max(sd * 4, 1e-6)
    fig.update_layout(xaxis_title=label, yaxis_title="Machine ID",
                      xaxis_range=[mean - span, mean + span])
    fig.update_yaxes(tickfont=dict(size=9))
    return _style(fig, f"{label}: every machine vs the fleet", height=760)


def scatter_relationship(binned: pd.DataFrame, x: str, y: str, overall: float,
                         title: str) -> go.Figure:
    """Binned mean with a 95% interval - the honest way to show a 100k-point cloud."""
    if binned.empty:
        return empty_figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=binned[x], y=binned[y], mode="markers+lines",
        error_y=dict(type="data", array=binned["ci95"], color="rgba(31,111,178,0.45)",
                     thickness=1.3, width=3),
        line=dict(color=ACCENT, width=1.8), marker=dict(size=7, color=ACCENT),
        name="binned mean",
        hovertemplate="%{x:.2f}<br>mean %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=overall, line=dict(color="#C1382E", width=1.5, dash="dash"),
                  annotation_text=f"overall mean {overall:.2f}",
                  annotation_position="top right", annotation_font=dict(size=11))
    xl = SENSOR_ENVELOPES.get(x, {}).get("label", x)
    yl = SENSOR_ENVELOPES.get(y, {}).get("label", y)
    fig.update_layout(xaxis_title=xl, yaxis_title=yl)
    return _style(fig, title)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    if corr.empty:
        return empty_figure()
    labels = [SENSOR_ENVELOPES.get(c, {}).get("label", c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-0.05, zmax=0.05,
        colorscale="RdBu_r", reversescale=False,
        text=corr.round(3).to_numpy(), texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Pearson r", thickness=14),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.4f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-35)
    return _style(fig, "Sensor correlation matrix (scale clipped to +/-0.05)", height=520)


def distribution_by_status(df: pd.DataFrame, column: str) -> go.Figure:
    """Box plot of a sensor split by efficiency class."""
    if df.empty or column not in df:
        return empty_figure()
    spec = SENSOR_ENVELOPES.get(column, {})
    fig = go.Figure()
    for status in EFFICIENCY_ORDER:
        sub = df.loc[df["Efficiency_Status"] == status, column]
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub, name=status, marker_color=EFFICIENCY_COLORS[status],
                             boxmean="sd", line=dict(width=1.4)))
    unit = f" ({spec['unit']})" if spec.get("unit") else ""
    fig.update_layout(yaxis_title=f"{spec.get('label', column)}{unit}", showlegend=False)
    return _style(fig, f"{spec.get('label', column)} by efficiency class")


def mode_comparison(df: pd.DataFrame, column: str) -> go.Figure:
    """Mean of a metric per operation mode with 95% confidence intervals."""
    if df.empty or column not in df:
        return empty_figure()
    grouped = df.groupby("Operation_Mode", observed=True)[column].agg(["mean", "std", "size"])
    grouped["ci95"] = 1.96 * grouped["std"] / np.sqrt(grouped["size"].clip(lower=1))
    grouped = grouped.dropna(subset=["mean"])
    if grouped.empty:
        return empty_figure()
    fig = go.Figure(go.Bar(
        x=[str(i) for i in grouped.index], y=grouped["mean"],
        error_y=dict(type="data", array=grouped["ci95"], thickness=1.5, width=6),
        marker_color=[MODE_COLORS.get(str(i), ACCENT) for i in grouped.index],
        hovertemplate="%{x}<br>mean %{y:.3f}<extra></extra>",
    ))
    spec = SENSOR_ENVELOPES.get(column, {})
    fig.update_layout(yaxis_title=spec.get("label", column), xaxis_title="")
    return _style(fig, f"{spec.get('label', column)} by operation mode", height=340)


def hourly_profile(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Normalised hour-of-day profile so several metrics share one axis."""
    if df.empty or not columns:
        return empty_figure()
    fig = go.Figure()
    palette = ["#1F6FB2", "#B5651D", "#2E8B57", "#7C3AED", "#C1382E"]
    for i, col in enumerate(columns):
        if col not in df:
            continue
        hourly = df.groupby("Hour", observed=True)[col].mean()
        if hourly.empty:
            continue
        spread = hourly.max() - hourly.min()
        centred = (hourly - hourly.mean()) / (spread if spread else 1) * 100
        label = SENSOR_ENVELOPES.get(col, {}).get("label", col)
        fig.add_trace(go.Scatter(
            x=hourly.index, y=centred, mode="lines+markers", name=label,
            line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=5),
            customdata=hourly.to_numpy(),
            hovertemplate=f"{label}<br>hour %{{x}}: %{{customdata:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(xaxis_title="hour of day",
                      yaxis_title="deviation from daily mean (% of range)")
    return _style(fig, "Hour-of-day profile")


def threshold_exposure_bars(exposure: pd.DataFrame) -> go.Figure:
    if exposure.empty:
        return empty_figure()
    data = exposure.sort_values("Share_%")
    fig = go.Figure(go.Bar(
        x=data["Share_%"], y=data["Sensor"], orientation="h",
        marker_color=["#C1382E" if v >= 30 else "#E0A800" if v >= 20 else "#94A3B8"
                      for v in data["Share_%"]],
        text=[f"{v:.1f}%" for v in data["Share_%"]], textposition="outside",
        hovertemplate="%{y}<br>%{x:.2f}% of readings near limit<extra></extra>",
    ))
    fig.update_layout(xaxis_title="% of readings in the worst quartile", yaxis_title="")
    fig.update_xaxes(range=[0, max(40, float(data["Share_%"].max()) * 1.25)])
    return _style(fig, "Time spent near sensor limits", height=420)


def label_rule_scatter(df: pd.DataFrame, sample: int = 9000) -> go.Figure:
    """The decisive plot: efficiency class in throughput-vs-error space."""
    if df.empty:
        return empty_figure()
    data = df.sample(min(sample, len(df)), random_state=42)
    fig = go.Figure()
    for status in EFFICIENCY_ORDER:
        sub = data[data["Efficiency_Status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=sub["Production_Speed_units_per_hr"], y=sub["Error_Rate_%"], mode="markers",
            name=status, marker=dict(size=4, color=EFFICIENCY_COLORS[status], opacity=0.55),
            hovertemplate=f"{status}<br>%{{x:.1f}} units/hr<br>%{{y:.2f}}% errors<extra></extra>",
        ))
    for x in (200, 400):
        fig.add_vline(x=x, line=dict(color=INK, width=1.2, dash="dash"))
    for y in (2, 5):
        fig.add_hline(y=y, line=dict(color=INK, width=1.2, dash="dash"))
    fig.update_layout(xaxis_title="Production speed (units/hr)", yaxis_title="Error rate (%)")
    return _style(fig, "Efficiency class is a rectangle rule on two columns", height=460)


def ablation_bars(ablation: pd.DataFrame) -> go.Figure:
    if ablation.empty:
        return empty_figure()
    data = ablation.iloc[::-1]
    colors = ["#94A3B8" if s == "reference" else "#2E8B57" if s == "yes" else "#C1382E"
              for s in data["Beats baseline"]]
    fig = go.Figure()
    fig.add_bar(x=data["Balanced accuracy"], y=data["Feature set"], orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in data["Balanced accuracy"]],
                textposition="outside",
                hovertemplate="%{y}<br>balanced accuracy %{x:.4f}<extra></extra>")
    fig.add_vline(x=1 / 3, line=dict(color=INK, width=1.5, dash="dash"),
                  annotation_text="random guessing (0.333)", annotation_position="top",
                  annotation_font=dict(size=11))
    fig.update_layout(xaxis_title="Balanced accuracy", yaxis_title="",
                      xaxis_range=[0, 1.15])
    return _style(fig, "Which columns actually predict efficiency?", height=420)
