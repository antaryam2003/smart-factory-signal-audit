"""Central configuration: file paths, sensor envelopes and KPI thresholds.

Every threshold used anywhere in the project is declared here so that the
research paper, the EDA pipeline and the Streamlit dashboard cannot drift
apart.  Values are documented as *operating envelopes* rather than hard-coded
magic numbers so a plant engineer can re-tune them for a real site.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATASET_NAME = "Thales_Group_Manufacturing.csv"

#: Candidate locations for the dataset, tried in order.
DATA_CANDIDATES = (DATA_DIR / DATASET_NAME, ROOT / DATASET_NAME)

RAW_DATE_FORMAT = "%d-%m-%Y"
RAW_DATETIME_FORMAT = "%d-%m-%Y %H:%M:%S"

# --------------------------------------------------------------------------
# Sensor operating envelopes
# --------------------------------------------------------------------------
# ``lo``/``hi``   physically valid range - readings outside are invalid.
# ``warn``        threshold above which the machine is "near its limit".
# ``direction``   'high' = larger is worse, 'low' = smaller is worse.
# The warn levels below sit at the 75th percentile of the observed
# distribution, i.e. a reading is flagged when it is in the worst quartile of
# everything the factory has ever reported for that sensor.
SENSOR_ENVELOPES: dict[str, dict] = {
    "Temperature_C": dict(lo=30.0, hi=90.0, warn=75.0, unit="°C", direction="high",
                          label="Temperature"),
    "Vibration_Hz": dict(lo=0.1, hi=5.0, warn=3.78, unit="Hz", direction="high",
                         label="Vibration"),
    "Power_Consumption_kW": dict(lo=1.5, hi=10.0, warn=7.86, unit="kW", direction="high",
                                 label="Power draw"),
    "Network_Latency_ms": dict(lo=1.0, hi=50.0, warn=37.8, unit="ms", direction="high",
                               label="6G latency"),
    "Packet_Loss_%": dict(lo=0.0, hi=5.0, warn=3.74, unit="%", direction="high",
                          label="Packet loss"),
    "Quality_Control_Defect_Rate_%": dict(lo=0.0, hi=10.0, warn=7.51, unit="%",
                                          direction="high", label="Defect rate"),
    "Production_Speed_units_per_hr": dict(lo=50.0, hi=500.0, warn=162.9, unit="units/hr",
                                          direction="low", label="Production speed"),
    "Predictive_Maintenance_Score": dict(lo=0.0, hi=1.0, warn=0.25, unit="", direction="low",
                                         label="Maintenance score"),
    "Error_Rate_%": dict(lo=0.0, hi=15.0, warn=11.27, unit="%", direction="high",
                         label="Error rate"),
}

#: The three sensors that compose the Machine Health Index.
HEALTH_SENSORS = ("Temperature_C", "Vibration_Hz", "Power_Consumption_kW")

#: Sensors carried over the 6G network link.
NETWORK_SENSORS = ("Network_Latency_ms", "Packet_Loss_%")

NUMERIC_COLUMNS = tuple(SENSOR_ENVELOPES)

CATEGORICAL_COLUMNS = ("Machine_ID", "Operation_Mode", "Efficiency_Status")

EFFICIENCY_ORDER = ("High", "Medium", "Low")
OPERATION_MODE_ORDER = ("Active", "Idle", "Maintenance")

# --------------------------------------------------------------------------
# Documented label rule (recovered empirically - see reports/research_paper.md)
# --------------------------------------------------------------------------
LABEL_RULE = dict(high_speed=400.0, high_error=2.0, medium_speed=200.0, medium_error=5.0)

#: Shift boundaries (start hour, inclusive) used for shift-level roll-ups.
SHIFTS = {"Night (00-08)": (0, 8), "Day (08-16)": (8, 16), "Evening (16-24)": (16, 24)}

# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
EFFICIENCY_COLORS = {"High": "#2E8B57", "Medium": "#D9A404", "Low": "#C1382E"}
MODE_COLORS = {"Active": "#1F6FB2", "Idle": "#7A8FA6", "Maintenance": "#B5651D"}
