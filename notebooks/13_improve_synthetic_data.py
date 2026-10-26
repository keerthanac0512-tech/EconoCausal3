import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Load cleaned dataset
input_path = "data/processed/causal_dataset_synthetic_cleaned.csv"
df = pd.read_csv(input_path)

print("Loaded dataset:", df.shape)

# Ensure valid customer features
df = df.dropna(
    subset=["age", "income", "difficulty", "duration"]
).copy()

# Randomly assign discount amounts
df["discount_amount"] = np.random.choice(
    [10, 20], size=len(df)
)

# Treatment: 1 = $20 discount, 0 = $10 discount
T = (df["discount_amount"] == 20).astype(int)

# Customer characteristics
age = df["age"].clip(18, 100)
income = df["income"].clip(lower=0)
difficulty = df["difficulty"].clip(lower=0)

# Baseline purchase probability
baseline_logit = (
    -1.8
    + 0.015 * (age - 50)
    + 0.000015 * (income - 50000)
    - 0.04 * difficulty
)

baseline_probability = 1 / (
    1 + np.exp(-baseline_logit)
)

# Heterogeneous treatment effect:
# Customers with lower income and higher difficulty
# are assumed to respond more to the larger discount.
true_effect = (
    0.02
    + 0.06 * (income < 50000)
    + 0.04 * (difficulty >= 10)
)

# Keep probabilities within valid bounds
purchase_probability = np.where(
    T == 1,
    baseline_probability + true_effect,
    baseline_probability
)

purchase_probability = np.clip(
    purchase_probability, 0.01, 0.95
)

# Simulate purchase outcomes
df["purchase"] = np.random.binomial(
    1, purchase_probability
)

# Store known simulated effect for validation only.
# Do NOT use this column as a DML input feature.
df["true_treatment_effect"] = true_effect

# Save improved synthetic dataset
output_path = (
    "data/processed/causal_dataset_synthetic_v2.csv"
)

df.to_csv(output_path, index=False)

print("\nImproved synthetic dataset created!")
print("Dataset shape:", df.shape)
print("\nPurchase rate by discount:")
print(df.groupby("discount_amount")["purchase"].mean())

print("\nMean simulated treatment effect:",
      df["true_treatment_effect"].mean())

print("\nSaved to:", output_path)

