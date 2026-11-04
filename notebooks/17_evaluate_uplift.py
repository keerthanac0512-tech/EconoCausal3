
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "dml_treatment_effects_v2.csv"
)

output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DML RESULTS
# ============================================================

df = pd.read_csv(input_path)

print("Dataset loaded:", df.shape)

# Your DML CSV uses actual_discount and purchase_outcome.
# Convert them into treatment and outcome columns.

required_source_columns = [
    "actual_discount",
    "purchase_outcome",
    "estimated_treatment_effect"
]

missing_columns = [
    col for col in required_source_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["actual_discount"] = pd.to_numeric(
    df["actual_discount"],
    errors="coerce"
)

df["purchase_outcome"] = pd.to_numeric(
    df["purchase_outcome"],
    errors="coerce"
)

df["estimated_treatment_effect"] = pd.to_numeric(
    df["estimated_treatment_effect"],
    errors="coerce"
)

# Treatment: 0 = 10% discount, 1 = 20% discount
df["treatment"] = (
    df["actual_discount"] == 20
).astype(int)

# Outcome: 0 = No purchase, 1 = Purchase
df["outcome"] = df["purchase_outcome"]

df = df.dropna(
    subset=[
        "treatment",
        "outcome",
        "estimated_treatment_effect"
    ]
).copy()

# ============================================================
# 3. VALIDATE DATA
# ============================================================

if not df["treatment"].isin([0, 1]).all():
    raise ValueError("Treatment must contain only 0 and 1.")

if not df["outcome"].isin([0, 1]).all():
    raise ValueError("Outcome must contain only 0 and 1.")

df["treatment"] = df["treatment"].astype(int)
df["outcome"] = df["outcome"].astype(int)

treated_count = (df["treatment"] == 1).sum()
control_count = (df["treatment"] == 0).sum()

if treated_count == 0 or control_count == 0:
    raise ValueError(
        "Both treatment and control groups are required."
    )

print("Valid records:", len(df))
print("Treated customers:", treated_count)
print("Control customers:", control_count)

# ============================================================
# 4. RANK CUSTOMERS BY ESTIMATED TREATMENT EFFECT
# ============================================================

df = df.sort_values(
    by="estimated_treatment_effect",
    ascending=False
).reset_index(drop=True)

df["customer_rank"] = np.arange(1, len(df) + 1)

df["fraction_targeted"] = (
    df["customer_rank"] / len(df)
)

# ============================================================
# 5. CALCULATE CUMULATIVE UPLIFT
# ============================================================

cumulative_treated = (
    df["treatment"] == 1
).cumsum()

cumulative_control = (
    df["treatment"] == 0
).cumsum()

cumulative_treated_purchases = (
    (df["treatment"] == 1) & (df["outcome"] == 1)
).cumsum()

cumulative_control_purchases = (
    (df["treatment"] == 0) & (df["outcome"] == 1)
).cumsum()

treated_rate = (
    cumulative_treated_purchases
    / cumulative_treated.replace(0, np.nan)
)

control_rate = (
    cumulative_control_purchases
    / cumulative_control.replace(0, np.nan)
)

df["cumulative_uplift"] = (
    treated_rate - control_rate
)

# ============================================================
# 6. CALCULATE QINI-STYLE CURVE
# ============================================================

df["qini_curve"] = (
    cumulative_treated_purchases
    - cumulative_treated * control_rate
)

# Replace undefined values where there are no control
# observations in the targeted prefix.
df["qini_curve"] = df["qini_curve"].fillna(0)

# Starting point for both curves
start_row = pd.DataFrame({
    "customer_rank": [0],
    "fraction_targeted": [0.0],
    "cumulative_uplift": [0.0],
    "qini_curve": [0.0]
})

curve_data = pd.concat(
    [
        start_row,
        df[
            [
                "customer_rank",
                "fraction_targeted",
                "cumulative_uplift",
                "qini_curve"
            ]
        ]
    ],
    ignore_index=True
)

# ============================================================
# 7. RANDOM TARGETING BASELINE
# ============================================================

final_qini = float(df["qini_curve"].iloc[-1])

curve_data["random_targeting"] = (
    curve_data["fraction_targeted"] * final_qini
)

# ============================================================
# 8. SAVE CHART DATA FOR REACT
# ============================================================

output_path = (
    output_dir / "uplift_evaluation_v2.csv"
)

curve_data.to_csv(
    output_path,
    index=False
)

print("\nChart data saved to:", output_path)

# ============================================================
# 9. CREATE QINI-STYLE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    curve_data["fraction_targeted"],
    curve_data["qini_curve"],
    label="DML-based targeting"
)

plt.plot(
    curve_data["fraction_targeted"],
    curve_data["random_targeting"],
    linestyle="--",
    label="Random targeting baseline"
)

plt.xlabel("Fraction of Customers Targeted")
plt.ylabel("Cumulative Qini-Style Score")
plt.title("EconoCausal: Qini-Style Targeting Curve")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

qini_path = output_dir / "qini_curve_v2.png"

plt.savefig(
    qini_path,
    dpi=150
)

plt.close()

print("Qini-style chart saved to:", qini_path)

# ============================================================
# 10. CREATE UPLIFT GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    curve_data["fraction_targeted"],
    curve_data["cumulative_uplift"],
    label="Cumulative purchase-rate difference"
)

plt.axhline(
    y=0,
    linestyle="--",
    label="Zero uplift"
)

plt.xlabel("Fraction of Customers Targeted")
plt.ylabel("Cumulative Purchase-Rate Difference")
plt.title("EconoCausal: Uplift Analysis")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

uplift_path = output_dir / "uplift_chart_v2.png"

plt.savefig(
    uplift_path,
    dpi=150
)

plt.close()

print("Uplift chart saved to:", uplift_path)

# ============================================================
# 11. SUMMARY
# ============================================================

print("\n========== UPLIFT EVALUATION COMPLETE ==========")

print("Total customers evaluated:", len(df))

print(
    "Mean estimated treatment effect:",
    round(
        df["estimated_treatment_effect"].mean(),
        4
    )
)

print(
    "Final Qini-style score:",
    round(final_qini, 4)
)

print("\nGenerated files:")
print(output_path)
print(qini_path)
print(uplift_path)
