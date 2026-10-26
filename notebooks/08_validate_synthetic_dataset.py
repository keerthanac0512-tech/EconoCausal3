import pandas as pd

# Load the synthetic dataset
df = pd.read_csv(
    "data/processed/causal_dataset_synthetic.csv"
)

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDiscount Amount Distribution:")
print(df["discount_amount"].value_counts())

print("\nPurchase Outcome Distribution:")
print(df["purchase"].value_counts())

print("\nPurchase Rate by Discount:")
print(
    df.groupby("discount_amount")["purchase"].agg(
        ["count", "mean"]
    )
)

# Validate the synthetic columns
assert df["discount_amount"].isin([10, 20]).all()
assert df["purchase"].isin([0, 1]).all()

print("\nBasic validation passed!")