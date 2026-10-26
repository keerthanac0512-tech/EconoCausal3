import pandas as pd

# Load customer treatment effects
df = pd.read_csv(
    "data/processed/dml_treatment_effects.csv"
)

# Sort customers by estimated treatment effect
df = df.sort_values(
    by="estimated_treatment_effect",
    ascending=False
)

# Select customers with positive estimated effects
persuadable_customers = df[
    df["estimated_treatment_effect"] > 0
].copy()

# Save the results
persuadable_customers.to_csv(
    "data/processed/persuadable_customers.csv",
    index=False
)

print("Total customers evaluated:", len(df))
print("Customers with positive estimated effects:",
      len(persuadable_customers))

print("\nTop 10 customers by estimated treatment effect:")
print(persuadable_customers.head(10))

print("\nSaved to data/processed/persuadable_customers.csv")