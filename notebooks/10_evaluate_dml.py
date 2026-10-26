import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load estimated treatment effects
df = pd.read_csv(
    "data/processed/dml_treatment_effects.csv"
)

effects = df["estimated_treatment_effect"]

# Summary statistics
print("Number of customers:", len(df))
print("\nTreatment Effect Summary:")
print(effects.describe())

# Count positive and negative effects
print("\nPositive estimated effects:", (effects > 0).sum())
print("Negative estimated effects:", (effects < 0).sum())
print("Zero estimated effects:", (effects == 0).sum())

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(effects, bins=30, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Estimated Treatment Effect")
plt.ylabel("Number of Customers")
plt.title("Distribution of Estimated Treatment Effects")
plt.tight_layout()
plt.savefig(
    "data/processed/treatment_effect_distribution.png",
    dpi=300
)
plt.show()

print("\nDistribution chart saved successfully!")