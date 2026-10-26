import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DML RESULTS
# ============================================================

input_path = "data/processed/dml_treatment_effects_v2.csv"

df = pd.read_csv(input_path)

print("Dataset loaded:", df.shape)

required_columns = [
    "treatment",
    "outcome",
    "estimated_treatment_effect"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove records with missing required values
df = df.dropna(subset=required_columns).copy()

# Convert treatment and outcome to numeric values
df["treatment"] = pd.to_numeric(
    df["treatment"],
    errors="coerce"
)

df["outcome"] = pd.to_numeric(
    df["outcome"],
    errors="coerce"
)

df["estimated_treatment_effect"] = pd.to_numeric(
    df["estimated_treatment_effect"],
    errors="coerce"
)

df = df.dropna(subset=required_columns)

# ============================================================
# 2. VALIDATE DATA
# ============================================================

if not df["treatment"].isin([0, 1]).all():
    raise ValueError(
        "Treatment must contain only 0 and 1."
    )

if not df["outcome"].isin([0, 1]).all():
    raise ValueError(
        "Outcome must contain only 0 and 1."
    )

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
# 3. RANK CUSTOMERS BY ESTIMATED TREATMENT EFFECT
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
# 4. CALCULATE CUMULATIVE UPLIFT
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

# Cumulative purchase rates in each group
treated_rate = (
    cumulative_treated_purchases
    / cumulative_treated.replace(0, np.nan)
)

control_rate = (
    cumulative_control_purchases
    / cumulative_control.replace(0, np.nan)
)

# Difference in purchase rates.
# This is a descriptive uplift estimate, not a causal
# effect estimate for each individual customer.
df["cumulative_uplift"] = (
    treated_rate - control_rate
)

# ============================================================
# 5. CREATE QINI-STYLE CURVE DATA
# ============================================================

# The curve measures the cumulative excess purchases
# relative to the control purchase rate, scaled by the
# number of treated customers in the targeted prefix.
#
# This is a Qini-style curve; it is not a fully adjusted
# Qini estimator for unequal treatment probabilities.

df["qini_curve"] = (
    cumulative_treated_purchases
    - cumulative_treated * control_rate
)

# Set the starting point for the curves
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
# 6. CREATE RANDOM TARGETING BASELINE
# ============================================================

# The random-targeting reference is a straight line
# from zero to the final Qini-curve value.

final_qini = df["qini_curve"].iloc[-1]

curve_data["random_targeting"] = (
    curve_data["fraction_targeted"] * final_qini
)

# ============================================================
# 7. SAVE CHART DATA FOR REACT
# ============================================================

output_dir = "data/processed"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "uplift_evaluation_v2.csv"
)

curve_data.to_csv(
    output_path,
    index=False
)

print("\nChart data saved to:", output_path)

# ============================================================
# 8. CREATE QINI CURVE
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

qini_path = os.path.join(
    output_dir,
    "qini_curve_v2.png"
)

plt.savefig(
    qini_path,
    dpi=150
)

plt.close()

print("Qini-style chart saved to:", qini_path)

# ============================================================
# 9. CREATE UPLIFT CHART
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

uplift_path = os.path.join(
    output_dir,
    "uplift_chart_v2.png"
)

plt.savefig(
    uplift_path,
    dpi=150
)

plt.close()

print("Uplift chart saved to:", uplift_path)

# ============================================================
# 10. SUMMARY
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
    round(float(final_qini), 4)
)

print("\nGenerated files:")
print(output_path)
print(qini_path)
print(uplift_path)
