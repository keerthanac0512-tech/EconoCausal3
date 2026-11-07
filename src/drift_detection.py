
import json
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency


# ==========================================================
# 1. CONFIGURATION
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

# Distribution drift thresholds
DRIFT_THRESHOLD = 0.2
WARNING_THRESHOLD = 0.1
P_VALUE_THRESHOLD = 0.05

# Missingness drift thresholds (absolute percentage-point change)
MISSING_DRIFT_THRESHOLD = 0.20
MISSING_WARNING_THRESHOLD = 0.10


# ==========================================================
# 2. MISSING VALUE MONITORING
# ==========================================================

def calculate_missingness(reference, current):

    reference = pd.Series(reference)
    current = pd.Series(current)

    reference_missing = float(reference.isna().mean())
    current_missing = float(current.isna().mean())

    difference = abs(current_missing - reference_missing)

    if difference >= MISSING_DRIFT_THRESHOLD:
        status = "Drift Detected"

    elif difference >= MISSING_WARNING_THRESHOLD:
        status = "Warning"

    else:
        status = "Stable"

    return {
        "reference_missing_rate": reference_missing,
        "current_missing_rate": current_missing,
        "missing_rate_difference": float(difference),
        "missingness_status": status
    }


# ==========================================================
# 3. POPULATION STABILITY INDEX - NUMERICAL
# ==========================================================

def calculate_psi(reference, current, bins=10):

    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if reference.empty or current.empty:
        return None

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    # Handle constant reference values.
    # This allows current values outside the constant
    # reference value to be detected as distribution drift.
    if len(edges) < 2:

        value = float(reference.iloc[0])

        edges = np.array([
            -np.inf,
            value,
            np.inf
        ])

    else:
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
# 4. CATEGORICAL PSI
# ==========================================================

def calculate_categorical_psi(reference, current):

    reference = pd.Series(reference).fillna(
        "Missing"
    ).astype(str)

    current = pd.Series(current).fillna(
        "Missing"
    ).astype(str)

    if reference.empty or current.empty:
        return None

    categories = sorted(
        set(reference.unique()) | set(current.unique())
    )

    ref_pct = reference.value_counts(
        normalize=True
    ).reindex(categories, fill_value=0)

    cur_pct = current.value_counts(
        normalize=True
    ).reindex(categories, fill_value=0)

    epsilon = 0.0001

    ref_pct = np.maximum(
        ref_pct.to_numpy(), epsilon
    )

    cur_pct = np.maximum(
        cur_pct.to_numpy(), epsilon
    )

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


# ==========================================================
# 5. COMBINE DISTRIBUTION AND MISSINGNESS STATUS
# ==========================================================

def combine_status(distribution_status, missingness_status):

    priority = {
        "Insufficient Data": 0,
        "Stable": 1,
        "Warning": 2,
        "Drift Detected": 3
    }

    if priority[distribution_status] >= priority[missingness_status]:
        return distribution_status

    return missingness_status


# ==========================================================
# 6. NUMERICAL DRIFT DETECTION
# ==========================================================

def detect_numerical_drift(reference, current):

    reference_original = pd.Series(reference)
    current_original = pd.Series(current)

    # Monitor missing values before dropping them.
    missingness = calculate_missingness(
        reference_original,
        current_original
    )

    reference = pd.to_numeric(
        reference_original, errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current_original, errors="coerce"
    ).dropna()

    if len(reference) < 2 or len(current) < 2:

        distribution_status = "Insufficient Data"

        return {
            "status": combine_status(
                distribution_status,
                missingness["missingness_status"]
            ),
            "psi": None,
            "p_value": None,
            "reference_mean": (
                float(reference.mean())
                if not reference.empty else None
            ),
            "current_mean": (
                float(current.mean())
                if not current.empty else None
            ),
            **missingness
        }

    ks_result = ks_2samp(
        reference, current
    )

    psi = calculate_psi(
        reference, current
    )

    p_value = float(ks_result.pvalue)

    if psi >= DRIFT_THRESHOLD or p_value < P_VALUE_THRESHOLD:
        distribution_status = "Drift Detected"

    elif psi >= WARNING_THRESHOLD:
        distribution_status = "Warning"

    else:
        distribution_status = "Stable"

    final_status = combine_status(
        distribution_status,
        missingness["missingness_status"]
    )

    return {
        "status": final_status,
        "psi": psi,
        "p_value": p_value,
        "reference_mean": float(reference.mean()),
        "current_mean": float(current.mean()),
        **missingness
    }


# ==========================================================
# 7. CATEGORICAL DRIFT DETECTION
# ==========================================================

def detect_categorical_drift(reference, current):

    reference_original = pd.Series(reference)
    current_original = pd.Series(current)

    missingness = calculate_missingness(
        reference_original,
        current_original
    )

    reference = reference_original.fillna(
        "Missing"
    ).astype(str)

    current = current_original.fillna(
        "Missing"
    ).astype(str)

    if len(reference) == 0 or len(current) == 0:

        distribution_status = "Insufficient Data"

        return {
            "status": combine_status(
                distribution_status,
                missingness["missingness_status"]
            ),
            "psi": None,
            "p_value": None,
            **missingness
        }

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

    psi = calculate_categorical_psi(
        reference, current
    )

    p_value = None

    # Chi-square is not useful with fewer than
    # two categories.
    if len(categories) >= 2:

        try:
            _, calculated_p_value, _, expected = (
                chi2_contingency(
                    contingency,
                    correction=False
                )
            )

            # Avoid using an unreliable p-value when
            # expected cell frequencies are too small.
            if np.all(expected >= 5):
                p_value = float(calculated_p_value)

        except ValueError:
            p_value = None

    if (
        psi is not None
        and (
            psi >= DRIFT_THRESHOLD
            or (
                p_value is not None
                and p_value < P_VALUE_THRESHOLD
            )
        )
    ):
        distribution_status = "Drift Detected"

    elif psi is not None and psi >= WARNING_THRESHOLD:
        distribution_status = "Warning"

    else:
        distribution_status = "Stable"

    final_status = combine_status(
        distribution_status,
        missingness["missingness_status"]
    )

    return {
        "status": final_status,
        "psi": psi,
        "p_value": p_value,
        **missingness
    }


# ==========================================================
# 8. COMPLETE DATA DRIFT DETECTION
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

    # Numerical features
    for feature in NUMERICAL_FEATURES:

        results[feature] = detect_numerical_drift(
            reference_df[feature],
            current_df[feature]
        )

    # Categorical features
    for feature in CATEGORICAL_FEATURES:

        results[feature] = detect_categorical_drift(
            reference_df[feature],
            current_df[feature]
        )

    # Outcome feature
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

    insufficient = sum(
        1 for item in results.values()
        if item["status"] == "Insufficient Data"
    )

    # Determine overall status.
    if drifted > 0:
        overall_status = "Drift Detected"

    elif warnings > 0:
        overall_status = "Warning"

    elif insufficient == len(results):
        overall_status = "Insufficient Data"

    else:
        overall_status = "Stable"

    return {
        "total_features": len(results),
        "drifted_features": drifted,
        "warning_features": warnings,
        "insufficient_features": insufficient,
        "overall_status": overall_status,
        "results": results
    }


# ==========================================================
# 9. SAVE DRIFT REPORT
# ==========================================================

def save_drift_report(report, output_path):

    with open(output_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False
        )

    print("Drift report saved:", output_path)
