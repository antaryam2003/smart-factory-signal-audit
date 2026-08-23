"""Formal hypothesis tests backing the claims made in the research paper.

Every headline statement in ``reports/research_paper.md`` traces back to one of
these functions, so a reviewer can re-run the evidence rather than trust prose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import NUMERIC_COLUMNS


def uniformity_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Kolmogorov-Smirnov test of each sensor against a uniform distribution.

    A high p-value means we cannot reject "this sensor is uniform noise across
    its full range", which is not how a physical process behaves.
    """
    rows = []
    for col in NUMERIC_COLUMNS:
        if col not in df:
            continue
        x = df[col].dropna()
        if x.empty:
            continue
        lo, hi = float(x.min()), float(x.max())
        result = stats.kstest(x, "uniform", args=(lo, hi - lo))
        rows.append({
            "Sensor": col,
            "Min": round(lo, 3),
            "Max": round(hi, 3),
            "Mean": round(float(x.mean()), 3),
            "Std": round(float(x.std()), 3),
            "Skew": round(float(stats.skew(x)), 4),
            "Kurtosis": round(float(stats.kurtosis(x)), 4),
            "KS_D": round(float(result.statistic), 5),
            "p_value": round(float(result.pvalue), 4),
            "Uniform_at_5pct": bool(result.pvalue > 0.05),
        })
    return pd.DataFrame(rows)


def independence_tests(df: pd.DataFrame, target: str = "Efficiency_Status") -> pd.DataFrame:
    """Chi-square tests of association between categoricals and the target.

    Cramer's V is reported alongside chi-square because with 100k rows even a
    trivial association reaches significance; V measures whether it *matters*.
    """
    rows = []
    for col in ("Machine_ID", "Operation_Mode", "Shift", "DayOfWeek"):
        if col not in df or col == target:
            continue
        table = pd.crosstab(df[col], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
        rows.append({
            "Factor": col,
            "Chi2": round(float(chi2), 3),
            "dof": int(dof),
            "p_value": round(float(p), 4),
            "Cramers_V": round(cramers_v, 4),
            "Association": _effect_label(cramers_v, p),
        })
    return pd.DataFrame(rows)


def _effect_label(v: float, p: float) -> str:
    if p > 0.05:
        return "none (not significant)"
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "weak"
    if v < 0.5:
        return "moderate"
    return "strong"


def sensor_target_anova(df: pd.DataFrame, target: str = "Efficiency_Status") -> pd.DataFrame:
    """One-way ANOVA of each sensor across the target's classes, with eta^2.

    Eta-squared is the share of a sensor's variance explained by the efficiency
    label - the practical question behind "does this sensor matter?".
    """
    rows = []
    groups_index = df.groupby(target, observed=True).indices
    for col in NUMERIC_COLUMNS:
        if col not in df:
            continue
        samples = [df[col].to_numpy()[idx] for idx in groups_index.values()]
        samples = [s[~np.isnan(s)] for s in samples]
        if len(samples) < 2 or any(len(s) < 2 for s in samples):
            continue
        f_stat, p = stats.f_oneway(*samples)
        grand = np.concatenate(samples)
        ss_total = float(((grand - grand.mean()) ** 2).sum())
        ss_between = float(sum(len(s) * (s.mean() - grand.mean()) ** 2 for s in samples))
        eta_sq = ss_between / ss_total if ss_total else 0.0
        rows.append({
            "Sensor": col,
            "F": round(float(f_stat), 3),
            "p_value": float(f"{p:.3e}"),
            "Eta_Squared": round(eta_sq, 5),
            "Variance_Explained_%": round(eta_sq * 100, 3),
        })
    return pd.DataFrame(rows).sort_values("Eta_Squared", ascending=False).reset_index(drop=True)


def binomial_homogeneity(df: pd.DataFrame, status: str = "High") -> dict[str, float]:
    """Compare per-machine label rates against pure binomial sampling noise.

    If the observed spread across machines matches the spread you would get by
    dealing the same cards randomly, the machines are not actually different.
    """
    per_machine = df.groupby("Machine_ID", observed=True)["Efficiency_Status"].agg(
        n="size", rate=lambda s: (s == status).mean()
    )
    overall = float((df["Efficiency_Status"] == status).mean())
    mean_n = float(per_machine["n"].mean())
    expected_sd = float(np.sqrt(overall * (1 - overall) / mean_n) * 100)
    observed_sd = float((per_machine["rate"] * 100).std())
    return {
        "status": status,
        "overall_rate_pct": overall * 100,
        "observed_sd_pp": observed_sd,
        "binomial_expected_sd_pp": expected_sd,
        "ratio": observed_sd / expected_sd if expected_sd else float("nan"),
        "min_pct": float(per_machine["rate"].min() * 100),
        "max_pct": float(per_machine["rate"].max() * 100),
    }


def correlation_significance(df: pd.DataFrame) -> pd.DataFrame:
    """All pairwise Pearson correlations with p-values, strongest first."""
    cols = [c for c in NUMERIC_COLUMNS if c in df]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = df[[a, b]].dropna()
            if len(sub) < 3:
                continue
            r, p = stats.pearsonr(sub[a], sub[b])
            rows.append({
                "Sensor_A": a, "Sensor_B": b,
                "Pearson_r": round(float(r), 5),
                "Abs_r": round(abs(float(r)), 5),
                "p_value": round(float(p), 4),
                "R_Squared_%": round(float(r) ** 2 * 100, 4),
            })
    return pd.DataFrame(rows).sort_values("Abs_r", ascending=False).reset_index(drop=True)
