"""Reproducible end-to-end EDA.

Run from the repository root::

    python analysis/run_eda.py

Writes every figure used in the research paper to ``reports/figures/`` and a
machine-readable summary of every statistic quoted in the prose to
``reports/findings.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    EFFICIENCY_COLORS, EFFICIENCY_ORDER, FIGURES_DIR, HEALTH_SENSORS,
    MODE_COLORS, NUMERIC_COLUMNS, REPORTS_DIR, SENSOR_ENVELOPES,
)
from src.data_loader import load_data  # noqa: E402
from src.kpis import (  # noqa: E402
    binned_relationship, efficiency_distribution, factory_kpis,
    flag_outlier_machines, machine_scorecard, threshold_exposure,
)
from src.modeling import (  # noqa: E402
    ablation_study, mutual_information, permutation_sanity_check, recover_label_rule,
)
from src.statistics_tests import (  # noqa: E402
    binomial_homogeneity, correlation_significance, independence_tests,
    sensor_target_anova, uniformity_tests,
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
ORDER = list(EFFICIENCY_ORDER)


def _save(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(ROOT)}")
    return str(path.relative_to(ROOT)).replace("\\", "/")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_sensor_distributions(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), NUMERIC_COLUMNS):
        spec = SENSOR_ENVELOPES[col]
        ax.hist(df[col], bins=60, color="#1F6FB2", alpha=0.85, edgecolor="none")
        ax.axhline(len(df) / 60, color="#C1382E", ls="--", lw=1.4,
                   label="uniform expectation")
        ax.set_title(f"{spec['label']} ({spec['unit']})".strip())
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Every sensor is uniformly distributed across its full range",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_sensor_distributions")


def fig_correlation_heatmap(df: pd.DataFrame) -> str:
    corr = df[list(NUMERIC_COLUMNS)].corr()
    labels = [SENSOR_ENVELOPES[c]["label"] for c in NUMERIC_COLUMNS]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-0.05, vmax=0.05, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson r"}, xticklabels=labels,
                yticklabels=labels, ax=ax, annot_kws={"size": 8})
    ax.set_title("No sensor pair correlates beyond |r| = 0.01\n"
                 "(colour scale deliberately clipped to +/-0.05)")
    fig.tight_layout()
    return _save(fig, "02_correlation_heatmap")


def fig_efficiency_overview(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    counts = df["Efficiency_Status"].value_counts().reindex(ORDER)
    axes[0].bar(counts.index, counts.values,
                color=[EFFICIENCY_COLORS[s] for s in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center",
                     va="bottom", fontsize=9)
    axes[0].set_title("Efficiency status distribution")
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].set_ylabel("readings")

    mode_pct = efficiency_distribution(df, "Operation_Mode")
    mode_pct.plot(kind="bar", stacked=True, ax=axes[1],
                  color=[EFFICIENCY_COLORS[c] for c in mode_pct.columns], width=0.65)
    axes[1].set_title("Split is identical in every operation mode")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("% of readings")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].tick_params(axis="x", rotation=0)

    shift_pct = efficiency_distribution(df, "Shift")
    shift_pct.plot(kind="bar", stacked=True, ax=axes[2],
                   color=[EFFICIENCY_COLORS[c] for c in shift_pct.columns], width=0.65)
    axes[2].set_title("...and in every shift")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("% of readings")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save(fig, "03_efficiency_overview")


def fig_label_rule(df: pd.DataFrame) -> str:
    sample = df.sample(min(14000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for status in ORDER:
        sub = sample[sample["Efficiency_Status"] == status]
        axes[0].scatter(sub["Production_Speed_units_per_hr"], sub["Error_Rate_%"],
                        s=3, alpha=0.45, c=EFFICIENCY_COLORS[status], label=status)
    axes[0].axvline(400, color="black", ls="--", lw=1.2)
    axes[0].axvline(200, color="black", ls=":", lw=1.2)
    axes[0].axhline(2, color="black", ls="--", lw=1.2)
    axes[0].axhline(5, color="black", ls=":", lw=1.2)
    axes[0].set_xlabel("Production speed (units/hr)")
    axes[0].set_ylabel("Error rate (%)")
    axes[0].set_title("The label is a rectangle rule on two columns")
    leg = axes[0].legend(markerscale=4, fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1)

    for status in ORDER:
        sub = sample[sample["Efficiency_Status"] == status]
        axes[1].scatter(sub["Temperature_C"], sub["Vibration_Hz"], s=3, alpha=0.45,
                        c=EFFICIENCY_COLORS[status], label=status)
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_ylabel("Vibration (Hz)")
    axes[1].set_title("The same labels against the physical sensors: no structure")
    fig.tight_layout()
    return _save(fig, "04_label_rule")


def fig_machine_scorecard(card: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    mean, sd = card["Health_Index"].mean(), card["Health_Index"].std()
    # Red is reserved for a genuine control-limit breach; everything inside the
    # limits is shown in one neutral colour so ordinary ranking noise does not
    # read as an alarm.
    colors = ["#C1382E" if abs(v - mean) > 3 * sd else "#7A8FA6"
              for v in card["Health_Index"]]
    axes[0].barh(card["Machine_ID"].astype(str), card["Health_Index"], color=colors)
    axes[0].axvline(mean, color="black", ls="-", lw=1.4, label=f"fleet mean {mean:.2f}")
    for k, ls in ((3, "--"), (-3, "--")):
        axes[0].axvline(mean + k * sd, color="#C1382E", ls=ls, lw=1.2,
                        label="3-sigma control limit" if k > 0 else None)
    axes[0].set_xlim(mean - 4 * sd, mean + 4 * sd)
    axes[0].set_xlabel("Machine Health Index (0-100)")
    axes[0].set_ylabel("Machine ID")
    axes[0].set_title("No machine breaches the 3-sigma control limits")
    axes[0].tick_params(axis="y", labelsize=6)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].scatter(card["Avg_Production_Speed"], card["Error_Frequency"],
                    s=70, c=card["Health_Index"], cmap="viridis", edgecolor="white")
    for _, r in card.iterrows():
        axes[1].annotate(int(r["Machine_ID"]),
                         (r["Avg_Production_Speed"], r["Error_Frequency"]),
                         fontsize=6, ha="center", va="center", color="white")
    axes[1].set_xlabel("Average production speed (units/hr)")
    axes[1].set_ylabel("Error frequency index (%)")
    axes[1].set_title("All 50 machines cluster in one tight blob")
    fig.colorbar(axes[1].collections[0], ax=axes[1], label="Health Index")
    fig.tight_layout()
    return _save(fig, "05_machine_scorecard")


def fig_cross_metric(df: pd.DataFrame) -> str:
    pairs = [
        ("Temperature_C", "Quality_Control_Defect_Rate_%", "Temperature vs defect rate"),
        ("Vibration_Hz", "Error_Rate_%", "Vibration vs error rate"),
        ("Power_Consumption_kW", "Production_Speed_units_per_hr", "Power vs throughput"),
        ("Network_Latency_ms", "Error_Rate_%", "6G latency vs error rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (x, y, title) in zip(axes.ravel(), pairs):
        b = binned_relationship(df, x, y, bins=25)
        ax.errorbar(b[x], b[y], yerr=b["ci95"], fmt="o-", color="#1F6FB2",
                    ecolor="#9CC3E5", capsize=3, ms=4, lw=1.3)
        overall = df[y].mean()
        ax.axhline(overall, color="#C1382E", ls="--", lw=1.4,
                   label=f"overall mean {overall:.2f}")
        r = df[[x, y]].corr().iloc[0, 1]
        ax.set_title(f"{title}   (r = {r:+.4f})")
        ax.set_xlabel(SENSOR_ENVELOPES[x]["label"])
        ax.set_ylabel(SENSOR_ENVELOPES[y]["label"])
        ax.legend(fontsize=8)
    fig.suptitle("Binned means with 95% intervals: every relationship is flat",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_cross_metric_diagnostics")


def fig_ablation(ablation: pd.DataFrame) -> str:
    d = ablation.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    base = float(ablation.loc[0, "Accuracy"])
    colors = ["#7A8FA6" if s == "reference" else "#2E8B57" if s == "yes" else "#C1382E"
              for s in d["Beats baseline"]]
    axes[0].barh(d["Feature set"], d["Accuracy"], color=colors)
    axes[0].axvline(base, color="black", ls="--", lw=1.5,
                    label=f"majority baseline {base:.3f}")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Raw accuracy is misleading")
    axes[0].legend(fontsize=8, loc="lower right")
    for i, v in enumerate(d["Accuracy"]):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    axes[1].barh(d["Feature set"], d["Balanced accuracy"], color=colors)
    axes[1].axvline(1 / 3, color="black", ls="--", lw=1.5, label="random guess 0.333")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Balanced accuracy")
    axes[1].set_title("Balanced accuracy exposes the truth")
    axes[1].legend(fontsize=8, loc="lower right")
    for i, v in enumerate(d["Balanced accuracy"]):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle("Feature ablation: only throughput and error rate carry signal",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_feature_ablation")


def fig_temporal(df: pd.DataFrame) -> str:
    daily = df.groupby("Date").agg(
        speed=("Production_Speed_units_per_hr", "mean"),
        health=("Machine_Health_Index", "mean"),
        defect=("Quality_Control_Defect_Rate_%", "mean"),
        high=("Efficiency_Status", lambda s: (s == "High").mean() * 100),
    )
    hourly = df.groupby("Hour").agg(
        speed=("Production_Speed_units_per_hr", "mean"),
        error=("Error_Rate_%", "mean"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5))
    axes[0, 0].plot(daily.index, daily["speed"], color="#1F6FB2", lw=1.2)
    axes[0, 0].axhline(daily["speed"].mean(), color="#C1382E", ls="--", lw=1.2)
    axes[0, 0].set_title("Daily mean production speed")
    axes[0, 0].set_ylabel("units/hr")

    axes[0, 1].plot(daily.index, daily["high"], color="#2E8B57", lw=1.2)
    axes[0, 1].axhline(daily["high"].mean(), color="#C1382E", ls="--", lw=1.2)
    axes[0, 1].set_title("Daily share of High-efficiency readings")
    axes[0, 1].set_ylabel("%")

    axes[1, 0].plot(hourly.index, hourly["speed"], "o-", color="#1F6FB2", ms=4)
    axes[1, 0].axhline(hourly["speed"].mean(), color="#C1382E", ls="--", lw=1.2)
    axes[1, 0].set_title("Production speed by hour of day")
    axes[1, 0].set_xlabel("hour")
    axes[1, 0].set_ylabel("units/hr")

    axes[1, 1].plot(hourly.index, hourly["error"], "o-", color="#B5651D", ms=4)
    axes[1, 1].axhline(hourly["error"].mean(), color="#C1382E", ls="--", lw=1.2)
    axes[1, 1].set_title("Error rate by hour of day")
    axes[1, 1].set_xlabel("hour")
    axes[1, 1].set_ylabel("%")
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    for lbl in axes[0, 0].get_xticklabels() + axes[0, 1].get_xticklabels():
        lbl.set_rotation(20)
    fig.suptitle("No trend, seasonality or shift effect anywhere in the timeline",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "08_temporal_patterns")


def fig_health_components(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].hist(df["Machine_Health_Index"], bins=60, color="#1F6FB2")
    axes[0].axvline(df["Machine_Health_Index"].mean(), color="#C1382E", ls="--",
                    label=f"mean {df['Machine_Health_Index'].mean():.1f}")
    axes[0].set_title("Machine Health Index distribution")
    axes[0].set_xlabel("MHI (0-100)")
    axes[0].legend(fontsize=8)

    stress = df[["Thermal_Stress", "Vibration_Stress", "Power_Stress"]].mean()
    axes[1].bar(["Thermal", "Vibration", "Power"], stress.values,
                color=["#C1382E", "#B5651D", "#1F6FB2"])
    for i, v in enumerate(stress.values):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Mean stress contribution by component")
    axes[1].set_ylabel("stress (0-1)")

    counts = df["Breach_Count"].value_counts().sort_index()
    axes[2].bar(counts.index, counts.values, color="#7A8FA6")
    axes[2].set_title("Sensors near their limit, per reading")
    axes[2].set_xlabel("number of sensors in worst quartile")
    axes[2].set_ylabel("readings")
    fig.tight_layout()
    return _save(fig, "09_health_components")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading and validating data...")
    df, report = load_data()
    print(f"  {report.raw_rows:,} raw rows -> {report.clean_rows:,} clean rows")

    print("Computing KPIs...")
    kpis = factory_kpis(df)
    card = machine_scorecard(df)
    exposure = threshold_exposure(df)

    print("Running statistical tests...")
    uniformity = uniformity_tests(df)
    independence = independence_tests(df)
    anova = sensor_target_anova(df)
    corr_sig = correlation_significance(df)
    binom_high = binomial_homogeneity(df, "High")
    binom_low = binomial_homogeneity(df, "Low")

    print("Running the feature-ablation study (this takes a minute)...")
    ablation = ablation_study(df)
    mi = mutual_information(df)
    rule = recover_label_rule(df)
    permutation = permutation_sanity_check(df)

    print("Rendering figures...")
    figures = {
        "sensor_distributions": fig_sensor_distributions(df),
        "correlation_heatmap": fig_correlation_heatmap(df),
        "efficiency_overview": fig_efficiency_overview(df),
        "label_rule": fig_label_rule(df),
        "machine_scorecard": fig_machine_scorecard(card),
        "cross_metric": fig_cross_metric(df),
        "ablation": fig_ablation(ablation),
        "temporal": fig_temporal(df),
        "health_components": fig_health_components(df),
    }

    outliers = {
        col: flag_outlier_machines(card, col).to_dict("records")
        for col in ("Health_Index", "Avg_Production_Speed", "Error_Frequency",
                    "Defect_Density")
    }

    findings = {
        "dataset": {
            "raw_rows": report.raw_rows,
            "clean_rows": report.clean_rows,
            "machines": int(df["Machine_ID"].nunique()),
            "operation_modes": sorted(df["Operation_Mode"].dropna().unique().tolist()),
            "start": str(df["DateTime"].min()),
            "end": str(df["DateTime"].max()),
            "capture_blocks": report.blocks,
            "overlapping_window": report.overlapping_window,
            "duplicate_timestamps": report.duplicate_timestamps,
            "duplicate_machine_timestamps": report.duplicate_machine_timestamps,
            "missing_values": report.missing_values,
            "out_of_range": report.out_of_range,
        },
        "kpis": kpis,
        "label_rule": {
            "agreement": report.label_rule_agreement,
            "exceptions": report.label_rule_exceptions,
            "tree_accuracy": rule["accuracy"],
            "tree_balanced_accuracy": rule["balanced_accuracy"],
            "tree_rules": rule["rules"],
        },
        "uniformity_tests": uniformity.to_dict("records"),
        "independence_tests": independence.to_dict("records"),
        "anova": anova.to_dict("records"),
        "top_correlations": corr_sig.head(10).to_dict("records"),
        "max_abs_correlation": float(corr_sig["Abs_r"].max()),
        "binomial_homogeneity": {"High": binom_high, "Low": binom_low},
        "ablation": ablation.to_dict("records"),
        "mutual_information": mi.to_dict("records"),
        "permutation_check": permutation,
        "threshold_exposure": exposure.to_dict("records"),
        "machine_scorecard_extremes": {
            "worst_health": card.head(5).to_dict("records"),
            "best_health": card.tail(5).to_dict("records"),
            "health_spread_pp": float(card["Health_Index"].max() - card["Health_Index"].min()),
        },
        "control_limit_outliers": outliers,
        "figures": figures,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "findings.json"
    out.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    card.to_csv(REPORTS_DIR / "machine_scorecard.csv", index=False)
    print(f"\nWrote {out.relative_to(ROOT)} and machine_scorecard.csv")

    print("\n" + "=" * 74)
    print("HEADLINE RESULTS")
    print("=" * 74)
    print(f"Max |correlation| between any sensor pair : {corr_sig['Abs_r'].max():.5f}")
    print(f"Label rule agreement                      : {report.label_rule_agreement:.4%}")
    print(f"Health Index spread across 50 machines    : "
          f"{findings['machine_scorecard_extremes']['health_spread_pp']:.2f} points")
    print(f"Machines outside 3-sigma control limits   : "
          f"{sum(len(v) for v in outliers.values())}")
    print("\nAblation:")
    print(ablation.to_string(index=False))
    print("\nRecovered label rule:")
    print(rule["rules"])
    print(f"\nPermutation check: real={permutation['real_label_accuracy']:.4f} "
          f"shuffled={permutation['shuffled_label_accuracy_mean']:.4f} "
          f"difference={permutation['difference']:+.4f}")


if __name__ == "__main__":
    main()
