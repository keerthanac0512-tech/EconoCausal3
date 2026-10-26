import pandas as pd

# Load the synthetic dataset
input_path = "data/processed/causal_dataset_synthetic.csv"
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Remove unrealistic ages above 100
df = df[df["age"] <= 100].copy()

# Save the cleaned dataset
output_path = "data/processed/causal_dataset_synthetic_cleaned.csv"
df.to_csv(output_path, index=False)

print("\nDataset cleaned successfully!")
print("Cleaned dataset shape:", df.shape)
print("Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())
print("Saved to:", output_path)

