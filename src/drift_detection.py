
import json
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency


# ==========================================================
# CONFIGURATION
# ==========================================================

NUMERICAL_FEATURES = [
    "age",
    "income",
    "difficulty",
    "duration",
    "discount_amount"
]

CATEGORICAL_FEATURES = [
    "gender",
    "offer_type",
    "channels"
]

OUTCOME_FEATURES = [
    "purchase"
]

ALL_MONITORED_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + OUTCOME_FEATURES
)


# ==========================================================
# POPULATION STABILITY INDEX
# ==========================================================

def calculate_psi(reference, current, bins=10):

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return None

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(
        reference, bins=edges
    )[0]

    cur_counts = np.histogram(
        current, bins=edges
    )[0]

    epsilon = 0.0001

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.maximum(ref_pct, epsilon)
    cur_pct = np.maximum(cur_pct, epsilon)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


# ==========================================================
# CATEGORICAL PSI
# ==========================================================

def calculate_categorical_psi(reference, current):

    reference = pd.Series(reference).fillna("Missing").astype(str)
    current = pd.Series(current).fillna("Missing").astype(str)

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    ref_pct = reference.value_counts(normalize=True).reindex(
        categories, fill_value=0
    )

    cur_pct = current.value_counts(normalize=True).reindex(
        categories, fill_value=0
    )

    epsilon = 0.0001

    ref_pct = np.maximum(ref_pct.to_numpy(), epsilon)
    cur_pct = np.maximum(cur_pct.to_numpy(), epsilon)

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


# ==========================================================
# NUMERICAL DRIFT
# ==========================================================

def detect_numerical_drift(reference, current):

    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna()

    if len(reference) < 2 or len(current) < 2:
        return {
            "status": "Insufficient Data",
            "psi": None,
            "p_value": None,
            "reference_mean": None,
            "current_mean": None
        }

    ks_result = ks_2samp(reference, current)

    psi = calculate_psi(reference, current)

    p_value = float(ks_result.pvalue)

    if psi >= 0.2 or p_value < 0.05:
        status = "Drift Detected"
    elif psi >= 0.1:
        status = "Warning"
    else:
        status = "Stable"

    return {
        "status": status,
        "psi": psi,
        "p_value": p_value,
        "reference_mean": float(reference.mean()),
        "current_mean": float(current.mean())
    }


# ==========================================================
# CATEGORICAL DRIFT
# ==========================================================

def detect_categorical_drift(reference, current):

    reference = pd.Series(reference).fillna("Missing").astype(str)
    current = pd.Series(current).fillna("Missing").astype(str)

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    ref_counts = reference.value_counts().reindex(
        categories, fill_value=0
    )

    cur_counts = current.value_counts().reindex(
        categories, fill_value=0
    )

    contingency = np.array([
        ref_counts.to_numpy(),
        cur_counts.to_numpy()
    ])

    if len(reference) == 0 or len(current) == 0:
        return {
            "status": "Insufficient Data",
            "psi": None,
            "p_value": None
        }

    if len(categories) < 2:
        p_value = 1.0
    else:
        _, p_value, _, _ = chi2_contingency(
            contingency
        )

    psi = calculate_categorical_psi(
        reference, current
    )

    if psi >= 0.2 or p_value < 0.05:
        status = "Drift Detected"
    elif psi >= 0.1:
        status = "Warning"
    else:
        status = "Stable"

    return {
        "status": status,
        "psi": psi,
        "p_value": float(p_value)
    }


# ==========================================================
# COMPLETE DRIFT DETECTION
# ==========================================================

def detect_data_drift(reference_df, current_df):

    results = {}

    missing_columns = [
        col for col in ALL_MONITORED_FEATURES
        if col not in reference_df.columns
        or col not in current_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for feature in NUMERICAL_FEATURES:

        results[feature] = detect_numerical_drift(
            reference_df[feature],
            current_df[feature]
        )

    for feature in CATEGORICAL_FEATURES:

        results[feature] = detect_categorical_drift(
            reference_df[feature],
            current_df[feature]
        )

    # Outcome is monitored separately.
    for feature in OUTCOME_FEATURES:

        results[feature] = detect_categorical_drift(
            reference_df[feature],
            current_df[feature]
        )

    drifted = sum(
        1 for item in results.values()
        if item["status"] == "Drift Detected"
    )

    warnings = sum(
        1 for item in results.values()
        if item["status"] == "Warning"
    )

    return {
        "total_features": len(results),
        "drifted_features": drifted,
        "warning_features": warnings,
        "overall_status": (
            "Drift Detected" if drifted > 0
            else "Warning" if warnings > 0
            else "Stable"
        ),
        "results": results
    }


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_drift_report(report, output_path):

    with open(output_path, "w") as file:
        json.dump(report, file, indent=4)

    print("Drift report saved:", output_path)
