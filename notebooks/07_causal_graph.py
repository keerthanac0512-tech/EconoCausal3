from pathlib import Path
import warnings

import pandas as pd
from dowhy import CausalModel

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "causal_dataset.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
AUDIT_DIR = BASE_DIR / "reports" / "causal_audit"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# ==========================================
# 3. PREPARE DATA
# ==========================================

required_columns = [
    "offer_type",
    "offer_viewed",
    "offer_completed",
    "age",
    "income",
    "difficulty",
    "duration",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep BOGO and discount offers
df = df[df["offer_type"].isin(["bogo", "discount"])].copy()

# Convert required columns to numeric
numeric_columns = [
    "offer_viewed",
    "offer_completed",
    "age",
    "income",
    "difficulty",
    "duration",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert offer type to a binary variable
df["is_discount"] = (df["offer_type"] == "discount").astype(int)

model_columns = [
    "offer_viewed",
    "offer_completed",
    "age",
    "income",
    "difficulty",
    "duration",
    "is_discount",
]

df = df[model_columns].dropna()

# Remove infinite values
df = df[
    df[model_columns].apply(
        lambda col: col.map(lambda value: abs(value) != float("inf"))
    ).all(axis=1)
].copy()

# Validate binary variables
df = df[
    df["offer_viewed"].isin([0, 1])
    & df["offer_completed"].isin([0, 1])
].copy()

if df.empty:
    raise ValueError("No valid data remains after preprocessing.")

if df["offer_viewed"].nunique() != 2:
    raise ValueError("offer_viewed must contain both 0 and 1.")

print("Prepared dataset shape:", df.shape)

# ==========================================
# 4. CREATE CAUSAL MODEL
# ==========================================

print("\nCreating causal model...")

model = CausalModel(
    data=df,
    treatment="offer_viewed",
    outcome="offer_completed",
    common_causes=[
        "age",
        "income",
        "difficulty",
        "duration",
        "is_discount",
    ],
)

# ==========================================
# 5. IDENTIFY EFFECT
# ==========================================

print("\nIdentifying causal effect...")

estimand = model.identify_effect()
print(estimand)

# ==========================================
# 6. ESTIMATE EFFECT
# ==========================================

print("\nEstimating causal effect...")

estimate = model.estimate_effect(
    estimand,
    method_name="backdoor.linear_regression",
    control_value=0,
    treatment_value=1,
    target_units="ate",
)

ate = float(estimate.value)

print("\nEstimated Average Treatment Effect:", round(ate, 6))

# Save estimate
summary = pd.DataFrame([{
    "treatment": "offer_viewed",
    "outcome": "offer_completed",
    "estimated_ate": ate,
    "sample_size": len(df),
}])

summary.to_csv(
    AUDIT_DIR / "causal_effect_summary.csv",
    index=False,
)

# ==========================================
# 7. SAVE CAUSAL GRAPH
# ==========================================

print("\nGenerating causal graph...")

try:
    model.view_model(
        layout="dot",
        file_name=str(FIGURES_DIR / "causal_graph"),
    )
    print("Causal graph saved.")

except Exception as error:
    print("Graph generation failed:", error)

# ==========================================
# 8. SAVE DATA QUALITY SUMMARY
# ==========================================

quality_summary = pd.DataFrame({
    "metric": [
        "rows_used",
        "viewed_offers",
        "not_viewed_offers",
        "completed_offers",
        "not_completed_offers",
        "completed_without_viewing",
    ],
    "value": [
        len(df),
        int((df["offer_viewed"] == 1).sum()),
        int((df["offer_viewed"] == 0).sum()),
        int((df["offer_completed"] == 1).sum()),
        int((df["offer_completed"] == 0).sum()),
        int(
            (
                (df["offer_viewed"] == 0)
                & (df["offer_completed"] == 1)
            ).sum()
        ),
    ],
})

quality_summary.to_csv(
    AUDIT_DIR / "data_quality_summary.csv",
    index=False,
)

# ==========================================
# 9. FINISHED
# ==========================================

print("\n" + "=" * 45)
print("CAUSAL ANALYSIS COMPLETED")
print("=" * 45)

print("Rows analyzed:", len(df))
print("Estimated ATE:", round(ate, 6))

print("\nFiles saved:")
print(AUDIT_DIR / "causal_effect_summary.csv")
print(AUDIT_DIR / "data_quality_summary.csv")
print(FIGURES_DIR / "causal_graph")
