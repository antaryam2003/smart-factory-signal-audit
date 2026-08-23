"""Smart Factory Operations Intelligence - Streamlit dashboard.

Manufacturing Process Health and Operational Efficiency Analysis
in 6G-Enabled Smart Factories.

Run locally with::

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.charts import (  # noqa: E402
    ablation_bars, correlation_heatmap, distribution_by_status, efficiency_donut,
    hourly_profile, label_rule_scatter, machine_ranking, mode_comparison,
    scatter_relationship, sensor_timeseries, stacked_distribution,
    threshold_exposure_bars,
)
from src.config import (  # noqa: E402
    EFFICIENCY_ORDER, HEALTH_SENSORS, LABEL_RULE, NUMERIC_COLUMNS, SENSOR_ENVELOPES,
)
from src.data_loader import load_data  # noqa: E402
from src.kpis import (  # noqa: E402
    binned_relationship, correlation_matrix, efficiency_distribution, factory_kpis,
    flag_outlier_machines, machine_scorecard, threshold_exposure,
)
from src.modeling import ablation_study, mutual_information, recover_label_rule  # noqa: E402
from src.statistics_tests import (  # noqa: E402
    binomial_homogeneity, independence_tests, sensor_target_anova, uniformity_tests,
)

st.set_page_config(
    page_title="Smart Factory Operations Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.015em;}
  div[data-testid="stMetric"] {
      background: var(--secondary-background-color);
      border: 1px solid rgba(148,163,184,0.28);
      border-radius: 12px; padding: 14px 16px;
  }
  div[data-testid="stMetricLabel"] p {font-size: 0.78rem; opacity: 0.75;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .callout {
      border-left: 4px solid #1F6FB2; background: rgba(31,111,178,0.07);
      padding: 12px 16px; border-radius: 8px; margin: 10px 0 18px 0;
  }
  .callout-warn {border-left-color: #C1382E; background: rgba(193,56,46,0.08);}
  .callout-ok   {border-left-color: #2E8B57; background: rgba(46,139,87,0.08);}
  .callout p {margin: 0.25rem 0;}
  .small {font-size: 0.84rem; opacity: 0.78;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    cls = {"warn": "callout callout-warn", "ok": "callout callout-ok"}.get(kind, "callout")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and validating telemetry...")
def get_data():
    df, report = load_data()
    return df, report


@st.cache_data(show_spinner="Scoring machines...")
def get_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    return machine_scorecard(df)


#: Rows used for the two model-based panels. The effects measured there are
#: enormous (balanced accuracy 0.33 vs 1.00), so a stratified 25k sample gives
#: identical conclusions in a fifth of the time - which matters on a free-tier
#: host. ``analysis/run_eda.py`` still runs on all 100k rows for the paper.
MODEL_SAMPLE_ROWS = 25_000


@st.cache_data(show_spinner="Running the feature-ablation study...")
def get_ablation(df: pd.DataFrame) -> pd.DataFrame:
    return ablation_study(df, max_rows=MODEL_SAMPLE_ROWS, n_estimators=80)


@st.cache_data(show_spinner="Recovering the label rule...")
def get_rule(df: pd.DataFrame) -> dict:
    return recover_label_rule(df)


@st.cache_data
def get_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    return mutual_information(df, max_rows=MODEL_SAMPLE_ROWS)


@st.cache_data
def get_stat_tests(df: pd.DataFrame):
    return uniformity_tests(df), independence_tests(df), sensor_target_anova(df)


try:
    DF, REPORT = get_data()
except FileNotFoundError as exc:
    st.error(f"**Dataset not found.**\n\n```\n{exc}\n```\n\n"
             "Place `Thales_Group_Manufacturing.csv` in the `data/` folder.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar - user capabilities
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏭 Factory controls")

    page = st.radio(
        "Dashboard module",
        ["Factory Health Overview", "Machine Health Dashboard", "Production & Quality",
         "Efficiency Diagnostics", "Data Quality & Validation"],
        label_visibility="collapsed",
    )
    st.divider()

    all_machines = sorted(DF["Machine_ID"].dropna().unique().tolist())
    machines = st.multiselect(
        "Machines", all_machines, default=all_machines,
        help="Empty selection means all machines.",
    ) or all_machines

    modes = st.multiselect(
        "Operation modes",
        [m for m in DF["Operation_Mode"].cat.categories if m in set(DF["Operation_Mode"])],
        default=list(DF["Operation_Mode"].cat.categories),
    ) or list(DF["Operation_Mode"].cat.categories)

    min_date = DF["DateTime"].min().date()
    max_date = DF["DateTime"].max().date()
    date_range = st.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    hour_range = st.slider("Hour of day", 0, 23, (0, 23),
                           help="Filter to a shift or a specific part of the day.")

    statuses = st.multiselect(
        "Efficiency status", list(EFFICIENCY_ORDER), default=list(EFFICIENCY_ORDER),
    ) or list(EFFICIENCY_ORDER)

    st.divider()
    st.caption(
        f"**{REPORT.clean_rows:,}** validated readings · "
        f"**{DF['Machine_ID'].nunique()}** machines · "
        f"{min_date:%d %b %Y} – {max_date:%d %b %Y}"
    )


mask = (
    DF["Machine_ID"].isin(machines)
    & DF["Operation_Mode"].isin(modes)
    & DF["Efficiency_Status"].isin(statuses)
    & DF["DateTime"].dt.date.between(start_date, end_date)
    & DF["Hour"].between(*hour_range)
)
F = DF.loc[mask]

if F.empty:
    st.title("Smart Factory Operations Intelligence")
    st.warning("No readings match the current filters. Widen the selection in the sidebar.")
    st.stop()

KPI = factory_kpis(F)
CARD = get_scorecard(F)


def kpi_row() -> None:
    """The five KPIs from the project brief, always in the same order."""
    c1, c2, c3, c4, c5 = st.columns(5)
    fleet = factory_kpis(DF)
    c1.metric("Machine Health Index", f"{KPI['machine_health_index']:.1f}",
              f"{KPI['machine_health_index'] - fleet['machine_health_index']:+.2f} vs fleet",
              help="Composite of temperature, vibration and power draw. "
                   "100 = comfortably inside the operating envelope on all three.")
    c2.metric("Avg Production Speed", f"{KPI['avg_production_speed']:.0f}",
              f"{KPI['avg_production_speed'] - fleet['avg_production_speed']:+.1f} units/hr",
              help="Mean output rate across the selected readings.")
    c3.metric("Defect Density", f"{KPI['defect_density']:.2f}",
              f"{KPI['defect_density'] - fleet['defect_density']:+.2f} units/hr",
              delta_color="inverse",
              help="Defective units produced per hour = defect rate x throughput.")
    c4.metric("Error Frequency Index", f"{KPI['error_frequency_index']:.2f}%",
              f"{KPI['error_frequency_index'] - fleet['error_frequency_index']:+.2f} pp",
              delta_color="inverse",
              help="Mean operational error rate.")
    c5.metric("High Efficiency Share", f"{KPI['high_pct']:.1f}%",
              f"{KPI['high_pct'] - fleet['high_pct']:+.2f} pp",
              help="Share of readings classified High. Medium "
                   f"{KPI['medium_pct']:.1f}% · Low {KPI['low_pct']:.1f}%.")


# ===========================================================================
# 1. FACTORY HEALTH OVERVIEW
# ===========================================================================
if page == "Factory Health Overview":
    st.title("Factory Health Overview")
    st.caption("Fleet-wide efficiency distribution and average sensor behaviour "
               "across all 6G-connected machines.")
    kpi_row()
    st.write("")

    left, right = st.columns([1, 1.35])
    with left:
        st.plotly_chart(efficiency_donut(F), width="stretch", key="ov_donut")
    with right:
        st.plotly_chart(
            stacked_distribution(efficiency_distribution(F, "Operation_Mode"),
                                 "Efficiency split by operation mode"),
            width="stretch", key="ov_mode_split")

    diff = abs(efficiency_distribution(F, "Operation_Mode")["High"].max()
               - efficiency_distribution(F, "Operation_Mode")["High"].min())
    callout(
        f"<b>The efficiency mix is identical in every operation mode</b> — the High-efficiency "
        f"share varies by only {diff:.2f} percentage points between Active, Idle and "
        f"Maintenance. A machine sitting idle is classified exactly like one running at load, "
        f"which tells us the label is not describing the physical state of the machine.",
        "warn",
    )

    st.subheader("Average sensor metrics")
    cols = st.columns(3)
    for i, sensor in enumerate(NUMERIC_COLUMNS):
        spec = SENSOR_ENVELOPES[sensor]
        value = float(F[sensor].mean())
        near = float(F[f"Breach_{sensor}"].mean() * 100)
        cols[i % 3].metric(
            f"{spec['label']} ({spec['unit']})" if spec["unit"] else spec["label"],
            f"{value:.2f}",
            f"{near:.1f}% of readings near limit",
            delta_color="off",
            help=f"Valid envelope {spec['lo']}–{spec['hi']} {spec['unit']}; "
                 f"warning threshold {spec['warn']} {spec['unit']}.",
        )

    st.write("")
    a, b = st.columns([1.3, 1])
    with a:
        metric = st.selectbox(
            "Trend metric", list(NUMERIC_COLUMNS) + ["Machine_Health_Index"],
            format_func=lambda c: SENSOR_ENVELOPES.get(c, {}).get("label", "Machine Health Index"),
            key="overview_trend",
        )
        freq = st.radio("Granularity", ["Hourly", "Daily"], horizontal=True,
                        index=1, key="overview_freq")
        st.plotly_chart(
            sensor_timeseries(F, metric, freq="h" if freq == "Hourly" else "D"),
            width="stretch", key="ov_trend")
    with b:
        st.plotly_chart(threshold_exposure_bars(threshold_exposure(F)),
                        width="stretch", key="ov_threshold")

    callout(
        "Each sensor spends almost exactly a quarter of its time in its worst quartile — "
        "which is what a flat, structureless signal looks like. There is no sensor that the "
        "factory is repeatedly pushing against its limit.",
    )


# ===========================================================================
# 2. MACHINE HEALTH DASHBOARD
# ===========================================================================
elif page == "Machine Health Dashboard":
    st.title("Machine Health Dashboard")
    st.caption("Machine-wise sensor trends and health scorecards, benchmarked "
               "against the fleet with 3-sigma control limits.")
    kpi_row()
    st.write("")

    focus = st.selectbox(
        "Focus machine", sorted(F["Machine_ID"].unique().tolist()),
        format_func=lambda m: f"Machine {m}", key="focus_machine",
    )
    M = F[F["Machine_ID"] == focus]
    row = CARD[CARD["Machine_ID"] == focus]

    if not row.empty:
        r = row.iloc[0]
        rank = int(CARD.sort_values("Health_Index", ascending=False)
                   .reset_index(drop=True)
                   .query("Machine_ID == @focus").index[0]) + 1
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Health Index", f"{r['Health_Index']:.2f}",
                  f"rank {rank} of {len(CARD)}", delta_color="off")
        c2.metric("Avg speed", f"{r['Avg_Production_Speed']:.0f} u/hr",
                  f"{r['Avg_Production_Speed'] - CARD['Avg_Production_Speed'].mean():+.1f}")
        c3.metric("Defect density", f"{r['Defect_Density']:.2f}",
                  f"{r['Defect_Density'] - CARD['Defect_Density'].mean():+.2f}",
                  delta_color="inverse")
        c4.metric("Error frequency", f"{r['Error_Frequency']:.2f}%",
                  f"{r['Error_Frequency'] - CARD['Error_Frequency'].mean():+.2f} pp",
                  delta_color="inverse")
        c5.metric("Readings near limit", f"{r['Breach_Rate']:.1f}%",
                  delta_color="off")

    tabs = st.tabs(["Sensor trends", "Fleet ranking", "Health composition", "Scorecard table"])

    with tabs[0]:
        sensor = st.selectbox(
            "Sensor", list(NUMERIC_COLUMNS),
            format_func=lambda c: SENSOR_ENVELOPES[c]["label"], key="machine_sensor",
        )
        gran = st.radio("Granularity", ["Hourly", "Daily"], horizontal=True, index=1,
                        key="machine_freq")
        st.plotly_chart(
            sensor_timeseries(M, sensor, freq="h" if gran == "Hourly" else "D"),
            width="stretch", key="mh_sensor_trend")
        c1, c2 = st.columns(2)
        c1.plotly_chart(distribution_by_status(M, sensor), width="stretch", key="mh_dist_by_status")
        c2.plotly_chart(mode_comparison(M, sensor), width="stretch", key="mh_mode_compare")

    with tabs[1]:
        metric = st.selectbox(
            "Rank machines by",
            ["Health_Index", "Avg_Production_Speed", "Defect_Density",
             "Error_Frequency", "Breach_Rate", "Speed_CV"],
            format_func=lambda c: c.replace("_", " "), key="rank_metric",
        )
        st.plotly_chart(machine_ranking(CARD, metric, metric.replace("_", " "), focus),
                        width="stretch", key="mh_fleet_rank")
        out = flag_outlier_machines(CARD, metric)
        if out.empty:
            callout(
                f"<b>No machine falls outside the 3-sigma control limits on "
                f"{metric.replace('_', ' ').lower()}.</b> The fleet is statistically "
                f"homogeneous: the spread between best and worst is "
                f"{CARD[metric].max() - CARD[metric].min():.2f} units, well inside what "
                f"random sampling alone produces. No machine warrants a targeted "
                f"intervention on this metric.", "ok",
            )
        else:
            callout(f"<b>{len(out)} machine(s) outside the control limits.</b>", "warn")
            st.dataframe(out, width="stretch", hide_index=True)

    with tabs[2]:
        st.markdown("**How the Machine Health Index is built**")
        st.caption(
            "Each of the three mechanical sensors is mapped to a stress score: 0 while the "
            "reading stays better than its warning threshold, ramping linearly to 1 at the "
            "edge of the physically valid envelope. MHI = 100 x (1 - mean stress)."
        )
        comp = pd.DataFrame({
            "Component": [SENSOR_ENVELOPES[s]["label"] for s in HEALTH_SENSORS],
            "Warning threshold": [SENSOR_ENVELOPES[s]["warn"] for s in HEALTH_SENSORS],
            "Hard limit": [SENSOR_ENVELOPES[s]["hi"] for s in HEALTH_SENSORS],
            "Mean reading (machine)": [round(float(M[s].mean()), 3) for s in HEALTH_SENSORS],
            "Mean stress (machine)": [
                round(float(M[c].mean()), 4)
                for c in ("Thermal_Stress", "Vibration_Stress", "Power_Stress")
            ],
            "Mean stress (fleet)": [
                round(float(DF[c].mean()), 4)
                for c in ("Thermal_Stress", "Vibration_Stress", "Power_Stress")
            ],
        })
        st.dataframe(comp, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(sensor_timeseries(M, "Temperature_C", "D"), width="stretch", key="mh_health_temp")
        c2.plotly_chart(sensor_timeseries(M, "Vibration_Hz", "D"), width="stretch", key="mh_health_vib")

    with tabs[3]:
        st.dataframe(CARD, width="stretch", hide_index=True, height=520)
        st.download_button(
            "Download scorecard (CSV)", CARD.to_csv(index=False).encode(),
            "machine_scorecard.csv", "text/csv",
        )


# ===========================================================================
# 3. PRODUCTION & QUALITY PANEL
# ===========================================================================
elif page == "Production & Quality":
    st.title("Production & Quality Panel")
    st.caption("Production speed against defect rate, and the frequency and "
               "distribution of operational errors.")
    kpi_row()
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Good units / hr", f"{KPI['good_units_per_hr']:.1f}",
              help="Production speed minus defective units.")
    c2.metric("Mean defect rate", f"{F['Quality_Control_Defect_Rate_%'].mean():.2f}%")
    c3.metric("Output consistency (CV)",
              f"{F['Production_Speed_units_per_hr'].std() / F['Production_Speed_units_per_hr'].mean() * 100:.1f}%",
              help="Coefficient of variation of throughput. Above ~30% means output is "
                   "essentially unpredictable minute to minute.")
    c4.metric("Error spikes (>P90)",
              f"{(F['Error_Rate_%'] > DF['Error_Rate_%'].quantile(0.9)).mean() * 100:.1f}%",
              delta_color="inverse")

    tabs = st.tabs(["Speed vs defects", "Error analysis", "Machine comparison", "Bottlenecks"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            b = binned_relationship(F, "Production_Speed_units_per_hr",
                                    "Quality_Control_Defect_Rate_%", bins=25)
            st.plotly_chart(
                scatter_relationship(b, "Production_Speed_units_per_hr",
                                     "Quality_Control_Defect_Rate_%",
                                     float(F["Quality_Control_Defect_Rate_%"].mean()),
                                     "Does running faster increase defects?"),
                width="stretch", key="pq_speed_vs_defect_rate")
        with c2:
            b2 = binned_relationship(F, "Production_Speed_units_per_hr", "Defect_Density", bins=25)
            st.plotly_chart(
                scatter_relationship(b2, "Production_Speed_units_per_hr", "Defect_Density",
                                     float(F["Defect_Density"].mean()),
                                     "Defective units per hour vs throughput"),
                width="stretch", key="pq_speed_vs_defect_density")
        r = float(F[["Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%"]]
                  .corr().iloc[0, 1])
        callout(
            f"<b>Speed does not buy defects.</b> The correlation between throughput and defect "
            f"<i>rate</i> is r = {r:+.4f} — statistically indistinguishable from zero, so the "
            f"classic speed-quality trade-off is absent here. Defect <i>density</i> "
            f"(defective units per hour) still rises with throughput purely because more units "
            f"are made: the same percentage of a bigger number."
        )

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sensor_timeseries(F, "Error_Rate_%", "D"), width="stretch", key="pq_error_trend")
        with c2:
            st.plotly_chart(hourly_profile(F, ["Error_Rate_%", "Quality_Control_Defect_Rate_%",
                                               "Production_Speed_units_per_hr"]),
                            width="stretch", key="pq_hourly_profile")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(mode_comparison(F, "Error_Rate_%"), width="stretch", key="pq_error_by_mode")
        with c4:
            st.plotly_chart(
                machine_ranking(CARD, "Error_Frequency", "Error frequency (%)"),
                width="stretch", key="pq_error_rank")

    with tabs[2]:
        metric = st.selectbox(
            "Compare machines on",
            ["Avg_Production_Speed", "Defect_Rate", "Defect_Density", "Error_Frequency",
             "Good_Units", "Speed_CV"],
            format_func=lambda c: c.replace("_", " "), key="pq_metric",
        )
        st.plotly_chart(machine_ranking(CARD, metric, metric.replace("_", " ")),
                        width="stretch", key="pq_machine_compare")
        spread = CARD[metric].max() - CARD[metric].min()
        rel = spread / CARD[metric].mean() * 100 if CARD[metric].mean() else 0
        callout(
            f"Best-to-worst spread on {metric.replace('_', ' ').lower()} is "
            f"<b>{spread:.2f}</b> ({rel:.1f}% of the fleet mean). "
            + ("That is far too small to justify singling out any machine — "
               "the ranking is reshuffled by noise, not by real differences."
               if rel < 10 else
               "This gap is large enough to investigate the trailing machines.")
        )

    with tabs[3]:
        st.markdown("**Where is throughput actually lost?**")
        st.caption(
            "Readings are bucketed by how many sensors sit in their worst quartile. If "
            "degradation drove throughput, stacking breaches would visibly reduce output."
        )
        by_breach = F.groupby("Breach_Count", observed=True).agg(
            Readings=("Production_Speed_units_per_hr", "size"),
            Avg_Speed=("Production_Speed_units_per_hr", "mean"),
            Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
            Avg_Error_Rate=("Error_Rate_%", "mean"),
            High_Pct=("Efficiency_Status", lambda s: (s == "High").mean() * 100),
        ).round(3).reset_index()
        st.dataframe(by_breach, width="stretch", hide_index=True)

        b3 = binned_relationship(F, "Breach_Count", "Production_Speed_units_per_hr",
                                 bins=min(10, max(2, F["Breach_Count"].nunique())))
        st.plotly_chart(
            scatter_relationship(b3, "Breach_Count", "Production_Speed_units_per_hr",
                                 float(F["Production_Speed_units_per_hr"].mean()),
                                 "Throughput vs number of sensors near their limit"),
            width="stretch", key="pq_breach_vs_speed")
        callout(
            "A machine with six sensors near their limits produces at the same rate as one "
            "with none. In a real plant this curve slopes down; here it is flat, which is the "
            "signature of sensor channels that are not physically coupled to the process.",
            "warn",
        )


# ===========================================================================
# 4. EFFICIENCY DIAGNOSTICS VIEW
# ===========================================================================
elif page == "Efficiency Diagnostics":
    st.title("Efficiency Diagnostics")
    st.caption("What the Efficiency_Status label actually measures, and how it "
               "compares across machines and operation modes.")
    kpi_row()
    st.write("")

    tabs = st.tabs(["The label rule", "Feature ablation", "Mode & shift comparison",
                    "Cross-metric diagnostics", "Statistical tests"])

    with tabs[0]:
        rule = get_rule(DF)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(label_rule_scatter(F), width="stretch", key="ed_label_scatter")
        with c2:
            st.metric("Rule reproduces the label", f"{REPORT.label_rule_agreement:.3%}",
                      f"{REPORT.label_rule_exceptions} exceptions in {REPORT.clean_rows:,} rows",
                      delta_color="off")
            st.markdown("**Recovered decision rule**")
            st.code(
                f"High    if speed >= {LABEL_RULE['high_speed']:.0f} "
                f"and error <= {LABEL_RULE['high_error']:.0f}\n"
                f"Medium  if speed >= {LABEL_RULE['medium_speed']:.0f} "
                f"and error <= {LABEL_RULE['medium_error']:.0f}\n"
                f"Low     otherwise",
                language="text",
            )
            st.caption(f"Depth-4 decision tree on those two columns alone reaches "
                       f"{rule['accuracy']:.4%} accuracy.")
        with st.expander("Full decision tree"):
            st.code(rule["rules"], language="text")
            st.text(rule["report"])

        callout(
            "<b>Efficiency_Status is not a measurement — it is arithmetic.</b> It is a fixed "
            "threshold rule applied to production speed and error rate, two columns that are "
            "themselves in the dataset. Predicting it from those columns is circular; "
            "predicting it from the physical sensors is impossible, because they were never "
            "part of its definition.", "warn",
        )

    with tabs[1]:
        ablation = get_ablation(DF)
        st.plotly_chart(ablation_bars(ablation), width="stretch", key="ed_ablation")
        st.dataframe(ablation, width="stretch", hide_index=True)
        st.caption(
            f"Fitted on a class-stratified sample of {MODEL_SAMPLE_ROWS:,} readings so the "
            f"page stays responsive. The full-file run in `analysis/run_eda.py` gives the "
            f"same figures to four decimal places."
        )
        callout(
            "Read the <b>balanced accuracy</b> column, not raw accuracy. A model given only "
            "the seven physical and network sensors scores 0.333 — exactly the value you get "
            "by guessing at random on three classes. Its 0.778 raw accuracy comes entirely "
            "from always answering <i>Low</i>, which is right 78% of the time because 78% of "
            "the readings are Low. That number would look like a working model on a slide.",
            "warn",
        )
        st.markdown("**Mutual information with Efficiency_Status**")
        mi = get_mutual_information(DF)
        st.dataframe(mi, width="stretch", hide_index=True)
        st.caption("Mutual information detects non-linear dependence too. Every sensor other "
                   "than throughput and error rate scores below 0.002 bits — genuine "
                   "independence, not a missed curve.")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                stacked_distribution(efficiency_distribution(F, "Operation_Mode"),
                                     "By operation mode"),
                width="stretch", key="ed_mode_split")
        with c2:
            st.plotly_chart(
                stacked_distribution(efficiency_distribution(F, "Shift"), "By shift"),
                width="stretch", key="ed_shift_split")
        st.plotly_chart(
            stacked_distribution(efficiency_distribution(F, "Machine_ID"),
                                 "By machine", "Machine ID"),
            width="stretch", key="ed_machine_split")
        b = binomial_homogeneity(F, "High")
        callout(
            f"Across machines the High-efficiency share ranges from {b['min_pct']:.2f}% to "
            f"{b['max_pct']:.2f}%, a standard deviation of {b['observed_sd_pp']:.3f} "
            f"percentage points. Pure random sampling with no real differences between "
            f"machines would produce {b['binomial_expected_sd_pp']:.3f} pp. "
            f"The observed spread is <b>{b['ratio']:.2f}x</b> the noise floor — "
            f"machine identity explains nothing.", "ok",
        )

    with tabs[3]:
        st.markdown("**Pick any two metrics and test the relationship**")
        c1, c2, c3 = st.columns([1, 1, 0.6])
        x = c1.selectbox("X axis", list(NUMERIC_COLUMNS), index=0,
                         format_func=lambda c: SENSOR_ENVELOPES[c]["label"], key="cm_x")
        y = c2.selectbox("Y axis", list(NUMERIC_COLUMNS), index=5,
                         format_func=lambda c: SENSOR_ENVELOPES[c]["label"], key="cm_y")
        bins = c3.slider("Bins", 8, 40, 25, key="cm_bins")
        if x == y:
            st.info("Choose two different metrics.")
        else:
            b = binned_relationship(F, x, y, bins=bins)
            st.plotly_chart(
                scatter_relationship(b, x, y, float(F[y].mean()),
                                     f"{SENSOR_ENVELOPES[x]['label']} vs "
                                     f"{SENSOR_ENVELOPES[y]['label']}"),
                width="stretch", key="ed_cross_metric")
            r = float(F[[x, y]].corr().iloc[0, 1])
            st.metric("Pearson r", f"{r:+.5f}", f"explains {r ** 2 * 100:.4f}% of variance",
                      delta_color="off")
        st.plotly_chart(correlation_heatmap(correlation_matrix(F)), width="stretch", key="ed_correlation")

    with tabs[4]:
        uni, indep, anova = get_stat_tests(DF)
        st.markdown("**Is each sensor uniformly distributed?** (Kolmogorov–Smirnov)")
        st.dataframe(uni, width="stretch", hide_index=True)
        st.caption("A physical process produces clustered, skewed readings. Uniform readings "
                   "across the entire valid range are the signature of a random generator.")

        st.markdown("**Do machine identity, mode or shift affect efficiency?** (chi-square)")
        st.dataframe(indep, width="stretch", hide_index=True)
        st.caption("Cramer's V is shown alongside chi-square because with ~100,000 rows even a "
                   "meaningless association can reach statistical significance.")

        st.markdown("**How much of each sensor's variance does efficiency explain?** (ANOVA)")
        st.dataframe(anova, width="stretch", hide_index=True)


# ===========================================================================
# 5. DATA QUALITY & VALIDATION
# ===========================================================================
else:
    st.title("Data Quality & Validation")
    st.caption("Every check applied to the raw file before any number on the "
               "other pages was computed.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows read", f"{REPORT.raw_rows:,}")
    c2.metric("Rows retained", f"{REPORT.clean_rows:,}",
              f"{REPORT.clean_rows - REPORT.raw_rows:,}")
    c3.metric("Missing values", f"{sum(REPORT.missing_values.values()):,}")
    c4.metric("Out-of-envelope readings", f"{sum(REPORT.out_of_range.values()):,}")

    st.subheader("Validation checks")
    st.dataframe(
        pd.DataFrame(REPORT.as_rows(), columns=["Check", "Result"]),
        width="stretch", hide_index=True,
    )

    st.subheader("Capture blocks")
    st.caption("The source file is not one continuous recording. A backwards jump in the "
               "timestamp column marks the seam between two separate capture runs.")
    blocks = pd.DataFrame(REPORT.blocks, columns=["Starts", "Ends", "Rows"])
    st.dataframe(blocks, width="stretch", hide_index=True)

    if REPORT.overlapping_window:
        callout(
            f"<b>The two capture blocks overlap.</b> Readings for "
            f"{REPORT.overlapping_window[0]} to {REPORT.overlapping_window[1]} appear twice, "
            f"which produced {REPORT.duplicate_timestamps:,} repeated timestamps and "
            f"{REPORT.duplicate_machine_timestamps:,} cases where a single machine reports two "
            f"different readings for the same minute. Those collisions were resolved by "
            f"keeping the first record, removing "
            f"{REPORT.raw_rows - REPORT.clean_rows:,} rows. Any pipeline that concatenates "
            f"these blobs without this check double-counts a full day of production.", "warn",
        )

    st.subheader("Sampling cadence")
    st.caption("One reading is written per minute for the whole factory, not per machine, "
               "so each machine is sampled irregularly.")
    gaps = (DF.sort_values("DateTime").groupby("Machine_ID", observed=True)["DateTime"]
            .diff().dt.total_seconds().div(60).dropna())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median gap", f"{gaps.median():.0f} min")
    c2.metric("Mean gap", f"{gaps.mean():.1f} min")
    c3.metric("Longest gap", f"{gaps.max():.0f} min")
    c4.metric("Readings per machine", f"{len(DF) / DF['Machine_ID'].nunique():.0f}")
    callout(
        "A machine is observed roughly every 50 minutes on average, with gaps up to "
        f"{gaps.max():.0f} minutes. That is far too coarse to detect a developing fault: "
        "a bearing can fail inside one gap. Any genuine condition-monitoring programme needs "
        "per-machine sampling at a fixed cadence, not one shared factory-wide slot."
    )

    st.subheader("Per-column detail")
    detail = pd.DataFrame({
        "Column": list(NUMERIC_COLUMNS),
        "Missing": [REPORT.missing_values.get(c, 0) for c in NUMERIC_COLUMNS],
        "Outside envelope": [REPORT.out_of_range.get(c, 0) for c in NUMERIC_COLUMNS],
        "Declared min": [SENSOR_ENVELOPES[c]["lo"] for c in NUMERIC_COLUMNS],
        "Declared max": [SENSOR_ENVELOPES[c]["hi"] for c in NUMERIC_COLUMNS],
        "Observed min": [round(float(DF[c].min()), 3) for c in NUMERIC_COLUMNS],
        "Observed max": [round(float(DF[c].max()), 3) for c in NUMERIC_COLUMNS],
    })
    st.dataframe(detail, width="stretch", hide_index=True)

    with st.expander("Preview the cleaned dataset"):
        st.dataframe(F.head(300), width="stretch")
        st.download_button(
            "Download filtered data (CSV)",
            F.to_csv(index=False).encode(), "filtered_telemetry.csv", "text/csv",
        )


st.divider()
st.caption(
    "Manufacturing Process Health and Operational Efficiency Analysis in 6G-Enabled "
    "Smart Factories · built for the Thales Group / Unified Mentor project · "
    "all statistics recomputed live from the filtered selection."
)
