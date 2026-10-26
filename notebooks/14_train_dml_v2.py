import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from econml.dml import LinearDML
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/processed/causal_dataset_synthetic_v2.csv"
)

print("Dataset loaded:", df.shape)


# ============================================================
# 2. DEFINE TREATMENT, OUTCOME, AND FEATURES
# ============================================================

# Treatment:
# 1 = $20 discount
# 0 = $10 discount

T = (df["discount_amount"] == 20).astype(int)

# Outcome:
# 1 = customer purchased
# 0 = customer did not purchase

Y = df["purchase"]

# Features used by the DML model

feature_columns = [
    "age",
    "income",
    "difficulty",
    "duration"
]

X = df[feature_columns]

# Known simulated treatment effects.
# Used only to validate model estimates on synthetic data.

true_effect = df["true_treatment_effect"]


# ============================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

(
    X_train,
    X_test,
    Y_train,
    Y_test,
    T_train,
    T_test,
    truth_train,
    truth_test
) = train_test_split(
    X,
    Y,
    T,
    true_effect,
    test_size=0.2,
    random_state=42,
    stratify=T
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. CREATE THE DOUBLE MACHINE LEARNING MODEL
# ============================================================

model = LinearDML(
    model_y=RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=50,
        random_state=42
    ),
    model_t=RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=50,
        random_state=42
    ),
    discrete_treatment=True,
    random_state=42
)


# ============================================================
# 5. TRAIN THE MODEL
# ============================================================

model.fit(
    Y_train,
    T_train,
    X=X_train
)

print("\nDML model trained successfully!")


# ============================================================
# 6. ESTIMATE INDIVIDUAL TREATMENT EFFECTS
# ============================================================

estimated_effect = np.asarray(
    model.effect(X_test)
).ravel()


# ============================================================
# 7. EVALUATE MODEL ESTIMATES
# ============================================================

true_effect_values = truth_test.to_numpy()

mse = mean_squared_error(
    true_effect_values,
    estimated_effect
)

correlation = np.corrcoef(
    true_effect_values,
    estimated_effect
)[0, 1]

print("\n========== DML MODEL EVALUATION ==========")

print(
    "Mean true treatment effect:",
    round(true_effect_values.mean(), 4)
)

print(
    "Mean estimated treatment effect:",
    round(estimated_effect.mean(), 4)
)

print(
    "Minimum estimated effect:",
    round(estimated_effect.min(), 4)
)

print(
    "Maximum estimated effect:",
    round(estimated_effect.max(), 4)
)

print(
    "Mean squared error:",
    round(mse, 4)
)

print(
    "Correlation with true effects:",
    round(correlation, 4)
)


# ============================================================
# 8. SAVE RESULTS FOR QINI CURVE AND UPLIFT ANALYSIS
# ============================================================

results = X_test.copy()

# Actual treatment assignment:
# 1 = $20 discount, 0 = $10 discount

results["treatment"] = T_test.to_numpy()

# Actual observed outcome:
# 1 = purchased, 0 = did not purchase

results["outcome"] = Y_test.to_numpy()

# Known simulated effect for validation

results["true_treatment_effect"] = true_effect_values

# Estimated effect from the trained DML model

results["estimated_treatment_effect"] = estimated_effect


# ============================================================
# 9. SAVE RESULTS TO CSV
# ============================================================

output_path = (
    "data/processed/dml_treatment_effects_v2.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\n========== RESULTS SAVED ==========")

print("Results saved to:", output_path)

print("Saved dataset shape:", results.shape)

print("\nSaved columns:")
print(results.columns.tolist())

print("\nFirst five rows:")
print(results.head())

print("\nDML training and evaluation completed successfully!")

