"""Ablation study: which columns actually carry the efficiency signal?

This is the decisive experiment of the study.  Rather than fitting one model
and quoting its accuracy, we fit the same model on nested feature sets and
compare each against the majority-class baseline.  A feature set that cannot
beat the baseline contains no usable signal, however impressive its raw
accuracy looks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

TARGET = "Efficiency_Status"

PHYSICAL_SENSORS = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
]
OUTCOME_FEATURES = ["Production_Speed_units_per_hr", "Error_Rate_%"]
ALL_FEATURES = PHYSICAL_SENSORS + OUTCOME_FEATURES

FEATURE_SETS: dict[str, list[str]] = {
    "All 9 numeric columns": ALL_FEATURES,
    "7 physical/network sensors only": PHYSICAL_SENSORS,
    "Throughput + error rate only": OUTCOME_FEATURES,
    "Temperature + vibration + power": ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"],
    "6G network quality only": ["Network_Latency_ms", "Packet_Loss_%"],
}


def subsample(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    """Class-stratified subsample, used to keep the dashboard responsive.

    The effects being measured here are enormous (balanced accuracy 0.33 vs
    1.00), so a 25k-row sample resolves them to more decimal places than anyone
    reports.  ``analysis/run_eda.py`` still runs on the full file for the paper.
    """
    if max_rows is None or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    sampled = df.groupby(TARGET, observed=True).sample(frac=frac, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed)


def _fit_eval(X: pd.DataFrame, y: pd.Series, seed: int = 42,
              n_estimators: int = 150) -> dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def ablation_study(df: pd.DataFrame, seed: int = 42, max_rows: int | None = None,
                   n_estimators: int = 150) -> pd.DataFrame:
    """Train one model per feature set and compare to the trivial baselines."""
    df = subsample(df, max_rows, seed)
    y = df[TARGET].astype(str)
    _, _, y_train, y_test = train_test_split(
        df[ALL_FEATURES], y, test_size=0.25, random_state=seed, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent").fit(df[ALL_FEATURES], y)
    baseline_pred = baseline.predict(df[ALL_FEATURES].iloc[: len(y_test)])
    n_classes = y.nunique()

    rows = [{
        "Feature set": "Majority-class baseline",
        "n_features": 0,
        "Accuracy": float((y == y.mode()[0]).mean()),
        "Balanced accuracy": 1.0 / n_classes,
        "Macro F1": float(f1_score(y_test, baseline_pred[: len(y_test)], average="macro",
                                   zero_division=0)),
        "Beats baseline": "reference",
    }]
    baseline_acc = rows[0]["Accuracy"]

    for name, feats in FEATURE_SETS.items():
        feats = [f for f in feats if f in df]
        if not feats:
            continue
        scores = _fit_eval(df[feats], y, seed=seed, n_estimators=n_estimators)
        lift = scores["accuracy"] - baseline_acc
        rows.append({
            "Feature set": name,
            "n_features": len(feats),
            "Accuracy": scores["accuracy"],
            "Balanced accuracy": scores["balanced_accuracy"],
            "Macro F1": scores["macro_f1"],
            "Beats baseline": "yes" if lift > 0.01 else "no",
        })

    out = pd.DataFrame(rows)
    for col in ("Accuracy", "Balanced accuracy", "Macro F1"):
        out[col] = out[col].round(4)
    return out


def mutual_information(df: pd.DataFrame, seed: int = 42,
                       max_rows: int | None = None) -> pd.DataFrame:
    """Mutual information between each numeric column and the target.

    Unlike correlation this catches non-linear dependence, so a near-zero value
    is strong evidence of genuine independence rather than a missed curve.
    """
    df = subsample(df, max_rows, seed)
    feats = [f for f in ALL_FEATURES if f in df]
    scores = mutual_info_classif(df[feats], df[TARGET].astype(str), random_state=seed)
    return (
        pd.DataFrame({"Feature": feats, "Mutual_Information_bits": scores.round(4)})
        .sort_values("Mutual_Information_bits", ascending=False)
        .reset_index(drop=True)
    )


def recover_label_rule(df: pd.DataFrame, seed: int = 42) -> dict[str, object]:
    """Fit a tiny decision tree on the two outcome columns and read the rule.

    Depth 4 is the shallowest tree that can express the full rule: the High
    class needs two thresholds on speed and two on error rate.  Reaching ~100%
    accuracy at that depth is proof the label is a deterministic threshold
    function, not a learned relationship.
    """
    feats = [f for f in OUTCOME_FEATURES if f in df]
    X, y = df[feats], df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_train, y_train)
    pred = tree.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "rules": export_text(tree, feature_names=feats).strip(),
        "report": classification_report(y_test, pred, zero_division=0),
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, pred, labels=sorted(y.unique())),
            index=[f"true {c}" for c in sorted(y.unique())],
            columns=[f"pred {c}" for c in sorted(y.unique())],
        ),
    }


def permutation_sanity_check(df: pd.DataFrame, seed: int = 42, n_repeats: int = 3) -> dict:
    """Fit on the 7 physical sensors against a *shuffled* target.

    If real labels score no better than shuffled ones, the model has learned
    nothing from those sensors.  This is the cleanest possible refutation of a
    spurious "our model is 78% accurate" claim.
    """
    feats = [f for f in PHYSICAL_SENSORS if f in df]
    y = df[TARGET].astype(str)
    real = _fit_eval(df[feats], y, seed=seed)["accuracy"]

    rng = np.random.default_rng(seed)
    shuffled_scores = []
    for _ in range(n_repeats):
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        shuffled_scores.append(_fit_eval(df[feats], y_shuf, seed=seed)["accuracy"])

    return {
        "real_label_accuracy": real,
        "shuffled_label_accuracy_mean": float(np.mean(shuffled_scores)),
        "shuffled_label_accuracy_std": float(np.std(shuffled_scores)),
        "difference": float(real - np.mean(shuffled_scores)),
    }
