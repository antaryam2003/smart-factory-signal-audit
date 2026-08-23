"""The five Key Performance Indicators defined in the project brief.

Each KPI is computed both at fleet level (:func:`factory_kpis`) and per machine
(:func:`machine_scorecard`) so the dashboard can rank machines against the
fleet rather than against an arbitrary target.

    Machine Health Index    composite of temperature, vibration and power
    Average Production Speed mean output rate per machine
    Defect Density Score    defect rate relative to production volume
    Error Frequency Index   rate of operational errors
    Efficiency Distribution High / Medium / Low spread
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EFFICIENCY_ORDER, NUMERIC_COLUMNS, SENSOR_ENVELOPES


def factory_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Fleet-wide headline numbers for the Factory Health Overview."""
    if df.empty:
        return {k: float("nan") for k in (
            "machine_health_index", "avg_production_speed", "defect_density",
            "error_frequency_index", "high_pct", "medium_pct", "low_pct",
            "good_units_per_hr", "machines", "readings", "breach_rate",
            "network_degraded_pct",
        )}
    status = df["Efficiency_Status"].value_counts(normalize=True) * 100
    return {
        "machine_health_index": float(df["Machine_Health_Index"].mean()),
        "avg_production_speed": float(df["Production_Speed_units_per_hr"].mean()),
        "defect_density": float(df["Defect_Density"].mean()),
        "error_frequency_index": float(df["Error_Rate_%"].mean()),
        "good_units_per_hr": float(df["Good_Units_per_hr"].mean()),
        "high_pct": float(status.get("High", 0.0)),
        "medium_pct": float(status.get("Medium", 0.0)),
        "low_pct": float(status.get("Low", 0.0)),
        "machines": int(df["Machine_ID"].nunique()),
        "readings": int(len(df)),
        "breach_rate": float((df["Breach_Count"] > 0).mean() * 100),
        "network_degraded_pct": float(df["Network_Degraded"].mean() * 100),
    }


def machine_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-machine KPI table, ranked worst-first on the health index."""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("Machine_ID", observed=True)
    card = grouped.agg(
        Readings=("Machine_Health_Index", "size"),
        Health_Index=("Machine_Health_Index", "mean"),
        Health_Index_P05=("Machine_Health_Index", lambda s: s.quantile(0.05)),
        Avg_Temperature=("Temperature_C", "mean"),
        Avg_Vibration=("Vibration_Hz", "mean"),
        Avg_Power=("Power_Consumption_kW", "mean"),
        Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
        Speed_Std=("Production_Speed_units_per_hr", "std"),
        Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
        Defect_Density=("Defect_Density", "mean"),
        Error_Frequency=("Error_Rate_%", "mean"),
        Maintenance_Score=("Predictive_Maintenance_Score", "mean"),
        Avg_Latency=("Network_Latency_ms", "mean"),
        Avg_Packet_Loss=("Packet_Loss_%", "mean"),
        Breach_Rate=("Breach_Count", lambda s: (s > 0).mean() * 100),
        Good_Units=("Good_Units_per_hr", "mean"),
    )

    eff = (
        df.pivot_table(index="Machine_ID", columns="Efficiency_Status",
                       values="Machine_Health_Index", aggfunc="size", observed=False)
        .reindex(columns=list(EFFICIENCY_ORDER))
        .fillna(0)
    )
    eff_pct = eff.div(eff.sum(axis=1).replace(0, np.nan), axis=0) * 100
    card["High_Pct"] = eff_pct["High"]
    card["Medium_Pct"] = eff_pct["Medium"]
    card["Low_Pct"] = eff_pct["Low"]

    # Coefficient of variation: how *consistent* a machine's output is, which
    # matters more operationally than its raw average.
    card["Speed_CV"] = card["Speed_Std"] / card["Avg_Production_Speed"] * 100
    card = card.reset_index().sort_values("Health_Index").reset_index(drop=True)
    return card.round(3)


def control_limits(series: pd.Series, sigma: float = 3.0) -> tuple[float, float, float]:
    """Classic Shewhart control limits (centre, lower, upper).

    Used to answer "is this machine genuinely different, or is it just noise?".
    """
    centre = float(series.mean())
    spread = float(series.std(ddof=1))
    return centre, centre - sigma * spread, centre + sigma * spread


def flag_outlier_machines(card: pd.DataFrame, column: str, sigma: float = 3.0) -> pd.DataFrame:
    """Return the machines whose KPI sits outside the fleet control limits.

    An empty result is a meaningful finding in itself: it says the fleet is
    statistically homogeneous on that metric and no machine warrants a
    targeted intervention.
    """
    if card.empty or column not in card:
        return pd.DataFrame()
    centre, lower, upper = control_limits(card[column], sigma)
    mask = (card[column] < lower) | (card[column] > upper)
    out = card.loc[mask, ["Machine_ID", column]].copy()
    out["Fleet_Mean"] = round(centre, 3)
    out["Lower_Limit"] = round(lower, 3)
    out["Upper_Limit"] = round(upper, 3)
    out["Deviation"] = (out[column] - centre).round(3)
    return out.sort_values("Deviation")


def threshold_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Share of readings sitting in the worst quartile of each sensor.

    Directly answers the brief's "identify machines operating near threshold
    limits" question at sensor granularity.
    """
    rows = []
    for col, spec in SENSOR_ENVELOPES.items():
        flag = f"Breach_{col}"
        if flag not in df:
            continue
        rows.append({
            "Sensor": spec["label"],
            "Column": col,
            "Warning_Threshold": spec["warn"],
            "Unit": spec["unit"],
            "Direction": "above" if spec["direction"] == "high" else "below",
            "Readings_Near_Limit": int(df[flag].sum()),
            "Share_%": round(float(df[flag].mean() * 100), 2),
            "Mean": round(float(df[col].mean()), 3),
            "P95": round(float(df[col].quantile(0.95)), 3),
        })
    return pd.DataFrame(rows).sort_values("Share_%", ascending=False).reset_index(drop=True)


def efficiency_distribution(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Percentage split of High/Medium/Low within each level of ``by``."""
    if df.empty or by not in df:
        return pd.DataFrame()
    counts = pd.crosstab(df[by], df["Efficiency_Status"])
    counts = counts.reindex(columns=list(EFFICIENCY_ORDER), fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return pct.round(2)


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Pearson correlation over the sensor columns."""
    cols = list(columns or NUMERIC_COLUMNS)
    cols = [c for c in cols if c in df]
    if df.empty or len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr().round(4)


def binned_relationship(df: pd.DataFrame, x: str, y: str, bins: int = 20) -> pd.DataFrame:
    """Mean of ``y`` within equal-width bins of ``x``, with a 95% interval.

    Scatter plots of 100k noisy points show nothing; binned means with error
    bars make it obvious whether a real trend exists or the line is flat.
    """
    if df.empty or x not in df or y not in df:
        return pd.DataFrame()
    work = df[[x, y]].dropna()
    if work.empty:
        return pd.DataFrame()
    work = work.assign(_bin=pd.cut(work[x], bins=bins))
    grouped = work.groupby("_bin", observed=True)[y]
    out = grouped.agg(["mean", "std", "size"]).reset_index()
    out[x] = out["_bin"].apply(lambda i: i.mid).astype(float)
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["size"].clip(lower=1))
    return out.drop(columns="_bin").rename(columns={"mean": y}).round(4)
