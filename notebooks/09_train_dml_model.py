import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import LinearDML

# Step 1: Load the synthetic dataset
df = pd.read_csv(
    "data/processed/causal_dataset_synthetic.csv"
)

print("Dataset loaded:", df.shape)

# Step 2: Define treatment and outcome
# T = 0 means $10 discount
# T = 1 means $20 discount
T = (df["discount_amount"] == 20).astype(int)

# Outcome: 0 = no purchase, 1 = purchase
Y = df["purchase"]

# Step 3: Select customer features
feature_columns = [
    "age",
    "income",
    "difficulty",
    "duration"
]

X = df[feature_columns].copy()

# Handle any missing values
X = X.fillna(X.median())

print("\nFeatures:", feature_columns)
print("Treatment distribution:")
print(T.value_counts())

# Step 4: Split the dataset
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y,
    test_size=0.2,
    random_state=42,
    stratify=T
)

# Step 5: Create the DML model
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

# Step 6: Train the model
model.fit(
    Y_train,
    T_train,
    X=X_train
)

print("\nDML model trained successfully!")

# Step 7: Estimate treatment effects
effects = model.effect(X_test)

print("\nEstimated treatment effects:")
print("Mean effect:", np.mean(effects))
print("Minimum effect:", np.min(effects))
print("Maximum effect:", np.max(effects))

# Step 8: Save customer-level estimates
results = X_test.copy()
results["estimated_treatment_effect"] = effects

results.to_csv(
    "data/processed/dml_treatment_effects.csv",
    index=False
)

print("\nTreatment effects saved successfully!")
print("File: data/processed/dml_treatment_effects.csv")