"""Loading, validation and feature engineering for the manufacturing telemetry.

The public entry point is :func:`load_data`, which returns a fully cleaned and
enriched frame plus a :class:`ValidationReport` describing every problem found
in the raw file.  Nothing is silently dropped: the report is surfaced in the
dashboard's Data Quality panel so operators can see exactly what was corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_CANDIDATES,
    EFFICIENCY_ORDER,
    HEALTH_SENSORS,
    LABEL_RULE,
    NUMERIC_COLUMNS,
    OPERATION_MODE_ORDER,
    RAW_DATETIME_FORMAT,
    SENSOR_ENVELOPES,
    SHIFTS,
)


@dataclass
class ValidationReport:
    """Everything the cleaning stage discovered about the raw file."""

    raw_rows: int = 0
    clean_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    exact_duplicates: int = 0
    duplicate_timestamps: int = 0
    duplicate_machine_timestamps: int = 0
    unparseable_datetimes: int = 0
    overlapping_window: tuple[str, str] | None = None
    blocks: list[tuple[str, str, int]] = field(default_factory=list)
    label_rule_agreement: float = float("nan")
    label_rule_exceptions: int = 0

    def as_rows(self) -> list[tuple[str, str]]:
        """Flatten into (check, result) pairs for tabular display."""
        rows = [
            ("Rows read from source", f"{self.raw_rows:,}"),
            ("Rows after cleaning", f"{self.clean_rows:,}"),
            ("Missing values (all columns)", f"{sum(self.missing_values.values()):,}"),
            ("Readings outside sensor envelope", f"{sum(self.out_of_range.values()):,}"),
            ("Fully duplicated records", f"{self.exact_duplicates:,}"),
            ("Repeated timestamps", f"{self.duplicate_timestamps:,}"),
            ("Repeated machine+timestamp pairs", f"{self.duplicate_machine_timestamps:,}"),
            ("Unparseable date/time fields", f"{self.unparseable_datetimes:,}"),
            ("Label rule agreement", f"{self.label_rule_agreement:.3%}"),
            ("Label rule exceptions", f"{self.label_rule_exceptions:,}"),
        ]
        if self.overlapping_window:
            rows.append(("Overlapping capture window", " to ".join(self.overlapping_window)))
        return rows


def resolve_dataset_path(explicit: str | Path | None = None) -> Path:
    """Find the CSV, tolerating either the repo root or ``data/``."""
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Dataset not found at {path}")
    for candidate in DATA_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(c) for c in DATA_CANDIDATES)
    raise FileNotFoundError(f"Dataset not found. Looked in:\n  {searched}")


def _standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and title-case the categorical columns so stray casing collapses."""
    for col, order in (("Operation_Mode", OPERATION_MODE_ORDER),
                       ("Efficiency_Status", EFFICIENCY_ORDER)):
        cleaned = df[col].astype("string").str.strip().str.title()
        canonical = {v.title(): v for v in order}
        df[col] = cleaned.map(canonical).fillna(cleaned).astype("string")
    df["Machine_ID"] = pd.to_numeric(df["Machine_ID"], errors="coerce").astype("Int64")
    return df


def _detect_capture_blocks(dt: pd.Series) -> list[tuple[str, str, int]]:
    """Split the file where the timeline jumps backwards.

    The source file is a concatenation of separate capture runs; a negative
    step in the timestamp column marks the seam between two of them.
    """
    if len(dt) == 0:
        return []
    breaks = np.flatnonzero(dt.diff().to_numpy() < np.timedelta64(0, "ns"))
    bounds = [0, *breaks.tolist(), len(dt)]
    blocks = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end > start:
            chunk = dt.iloc[start:end]
            blocks.append((str(chunk.min()), str(chunk.max()), int(end - start)))
    return blocks


def apply_label_rule(df: pd.DataFrame) -> pd.Series:
    """Reproduce ``Efficiency_Status`` from throughput and error rate.

    Recovered empirically in the EDA: the published label is a deterministic
    two-variable threshold rule that carries no information from the physical
    sensors.  Kept in code so the dashboard can demonstrate the rule directly.
    """
    speed = df["Production_Speed_units_per_hr"]
    error = df["Error_Rate_%"]
    return pd.Series(
        np.where(
            (speed >= LABEL_RULE["high_speed"]) & (error <= LABEL_RULE["high_error"]),
            "High",
            np.where(
                (speed >= LABEL_RULE["medium_speed"]) & (error <= LABEL_RULE["medium_error"]),
                "Medium",
                "Low",
            ),
        ),
        index=df.index,
        dtype="string",
    )


def _stress(series: pd.Series, spec: dict) -> pd.Series:
    """Map a sensor reading onto 0 (comfortable) through 1 (at the hard limit).

    Readings better than the warning threshold score 0; the score then ramps
    linearly to 1 at the edge of the physically valid envelope.
    """
    if spec["direction"] == "high":
        span = spec["hi"] - spec["warn"]
        raw = (series - spec["warn"]) / span if span else series * 0.0
    else:
        span = spec["warn"] - spec["lo"]
        raw = (spec["warn"] - series) / span if span else series * 0.0
    return raw.clip(lower=0.0, upper=1.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the engineered columns every downstream module depends on."""
    df = df.copy()

    df["Date"] = df["DateTime"].dt.normalize()
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.day_name()
    df["Week"] = df["DateTime"].dt.isocalendar().week.astype(int)

    shift_index = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (start, end) in SHIFTS.items():
        shift_index = shift_index.mask(df["Hour"].between(start, end - 1), name)
    df["Shift"] = shift_index

    # --- Machine Health Index -------------------------------------------
    # Composite of the three mechanical sensors, expressed 0-100 where 100 is
    # a machine sitting comfortably inside its envelope on all three.
    stresses = [_stress(df[s], SENSOR_ENVELOPES[s]) for s in HEALTH_SENSORS]
    df["Thermal_Stress"], df["Vibration_Stress"], df["Power_Stress"] = stresses
    df["Machine_Health_Index"] = (100.0 * (1.0 - sum(stresses) / len(stresses))).round(3)

    # --- Defect density: defective units per hour, not just a percentage --
    df["Defect_Density"] = (
        df["Quality_Control_Defect_Rate_%"] * df["Production_Speed_units_per_hr"] / 100.0
    ).round(3)
    df["Good_Units_per_hr"] = (
        df["Production_Speed_units_per_hr"] - df["Defect_Density"]
    ).round(3)

    # --- Threshold breach flags -----------------------------------------
    for col, spec in SENSOR_ENVELOPES.items():
        if spec["direction"] == "high":
            flag = df[col] >= spec["warn"]
        else:
            flag = df[col] <= spec["warn"]
        df[f"Breach_{col}"] = flag
    breach_cols = [f"Breach_{c}" for c in SENSOR_ENVELOPES]
    df["Breach_Count"] = df[breach_cols].sum(axis=1).astype(int)

    # --- 6G network quality ----------------------------------------------
    df["Network_Degraded"] = df["Breach_Network_Latency_ms"] | df["Breach_Packet_Loss_%"]

    df["Efficiency_Status"] = pd.Categorical(
        df["Efficiency_Status"], categories=list(EFFICIENCY_ORDER), ordered=True
    )
    df["Operation_Mode"] = pd.Categorical(
        df["Operation_Mode"], categories=list(OPERATION_MODE_ORDER), ordered=True
    )
    return df


def load_data(path: str | Path | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Read, validate, clean and enrich the telemetry file."""
    csv_path = resolve_dataset_path(path)
    raw = pd.read_csv(csv_path)
    report = ValidationReport(raw_rows=len(raw))

    df = _standardise_categories(raw)

    # Numeric coercion plus envelope validation.
    for col, spec in SENSOR_ENVELOPES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        outside = ~df[col].between(spec["lo"], spec["hi"]) & df[col].notna()
        report.out_of_range[col] = int(outside.sum())
        df.loc[outside, col] = np.nan  # treat as sensor faults, imputed below

    report.missing_values = {c: int(df[c].isna().sum()) for c in NUMERIC_COLUMNS}

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Timestamp"].astype(str).str.strip(),
        format=RAW_DATETIME_FORMAT,
        errors="coerce",
    )
    report.unparseable_datetimes = int(df["DateTime"].isna().sum())
    df = df.dropna(subset=["DateTime"]).reset_index(drop=True)

    report.blocks = _detect_capture_blocks(df["DateTime"])
    if len(report.blocks) > 1:
        first_end = pd.Timestamp(report.blocks[0][1])
        second_start = pd.Timestamp(report.blocks[1][0])
        if second_start <= first_end:
            report.overlapping_window = (str(second_start), str(first_end))

    report.exact_duplicates = int(df.duplicated().sum())
    report.duplicate_timestamps = int(df.duplicated(subset=["DateTime"], keep=False).sum())
    report.duplicate_machine_timestamps = int(
        df.duplicated(subset=["DateTime", "Machine_ID"], keep=False).sum()
    )

    # A machine cannot report two different readings for the same minute, so
    # collapse those collisions; distinct machines sharing a minute are fine.
    df = df.drop_duplicates(subset=["DateTime", "Machine_ID"], keep="first")

    # Impute the handful of invalid readings with that machine's own median so
    # a single faulty sample never shifts a fleet-level statistic.
    for col in NUMERIC_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df.groupby("Machine_ID")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values(["DateTime", "Machine_ID"]).reset_index(drop=True)
    df = add_derived_features(df)

    rule = apply_label_rule(df)
    agreement = rule == df["Efficiency_Status"].astype("string")
    report.label_rule_agreement = float(agreement.mean())
    report.label_rule_exceptions = int((~agreement).sum())
    report.clean_rows = len(df)

    return df, report
