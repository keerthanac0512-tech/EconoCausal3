
import pandas as pd
import numpy as np

from pathlib import Path
from econml.dml import LinearDML
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_DIR / "causal_dataset_synthetic_v2.csv"
)

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 3. Keep 10% and 20% discounts
# --------------------------------------------------

df = df[
    df["discount_amount"].isin([10, 20])
].copy()

df = df.dropna(
    subset=["discount_amount", "purchase"]
)

# --------------------------------------------------
# 4. Treatment and outcome
# --------------------------------------------------

# 0 = 10% discount
# 1 = 20% discount

T = (df["discount_amount"] == 20).astype(int)
Y = df["purchase"].astype(int)

# --------------------------------------------------
# 5. Customer features
# --------------------------------------------------

feature_columns = [
    "age",
    "income",
    "difficulty",
    "duration"
]

X = df[feature_columns].copy()
X = X.fillna(X.median())

print("\nTreatment distribution:")
print(T.value_counts())

# --------------------------------------------------
# 6. Train-test split
# --------------------------------------------------

X_train, X_test, T_train, T_test, Y_train, Y_test = (
    train_test_split(
        X, T, Y,
        test_size=0.2,
        random_state=42,
        stratify=T
    )
)

# --------------------------------------------------
# 7. Train DML model
# --------------------------------------------------

model = LinearDML(
    model_y=RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=20,
        random_state=42
    ),
    model_t=RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=20,
        random_state=42
    ),
    discrete_treatment=True,
    random_state=42
)

model.fit(
    Y_train,
    T_train,
    X=X_train
)

print("\nDML model trained successfully!")

# --------------------------------------------------
# 8. Estimate treatment effects
# --------------------------------------------------

effects = model.effect(X_test)

print("\nTreatment effect summary:")
print("Mean:", np.mean(effects))
print("Minimum:", np.min(effects))
print("Maximum:", np.max(effects))

# --------------------------------------------------
# 9. Prepare customer-level results
# --------------------------------------------------

results = X_test.copy()

results["person"] = df.loc[X_test.index, "person"].values

results["actual_discount"] = (
    df.loc[X_test.index, "discount_amount"].values
)

results["purchase_outcome"] = (
    df.loc[X_test.index, "purchase"].values
)

results["estimated_treatment_effect"] = effects

# Positive effect means estimated benefit
# from upgrading from 10% to 20%.

results["incremental_purchase_probability"] = effects

# Keep recommendation based on estimated effect.
results["recommended_discount"] = np.where(
    effects > 0, 20, 10
)

# Rank customers by incremental estimated effect.
results["effect_rank"] = (
    results["estimated_treatment_effect"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

output_file = (
    DATA_DIR / "dml_treatment_effects_v2.csv"
)

results.to_csv(output_file, index=False)

print("\nTreatment effects saved successfully!")
print("Output:", output_file)

print("\nFinal output shape:", results.shape)

print("\nRecommended discount distribution:")
print(results["recommended_discount"].value_counts())

print("\nTop 10 customers:")
print(
    results[
        [
            "person",
            "estimated_treatment_effect",
            "recommended_discount",
            "effect_rank"
        ]
    ].head(10)
)
