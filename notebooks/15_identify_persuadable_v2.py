import pandas as pd

# Load DML treatment-effect estimates
input_path = "data/processed/dml_treatment_effects_v2.csv"
df = pd.read_csv(input_path)

# Sort customers by estimated treatment effect
df = df.sort_values(
    by="estimated_treatment_effect",
    ascending=False
).reset_index(drop=True)

# Identify customers with positive estimated effects
persuadable = df[
    df["estimated_treatment_effect"] > 0
].copy()

# Save all ranked customers
df.to_csv(
    "data/processed/ranked_customers_v2.csv",
    index=False
)

# Save customers with positive estimated effects
persuadable.to_csv(
    "data/processed/persuadable_customers_v2.csv",
    index=False
)

print("Total customers evaluated:", len(df))
print("Customers with positive estimated effects:", len(persuadable))

print("\nTop 10 persuadable customers:")
print(
    persuadable[
        [
            "age",
            "income",
            "difficulty",
            "duration",
            "true_treatment_effect",
            "estimated_treatment_effect"
        ]
    ].head(10).to_string(index=False)
)

print("\nFiles saved:")
print("data/processed/ranked_customers_v2.csv")
print("data/processed/persuadable_customers_v2.csv")
