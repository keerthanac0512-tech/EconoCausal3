import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned datasets
processed_path = "data/processed"
output_path = "reports/figures"

os.makedirs(output_path, exist_ok=True)

profile_df = pd.read_csv(
    os.path.join(processed_path, "profile_cleaned.csv")
)

portfolio_df = pd.read_csv(
    os.path.join(processed_path, "portfolio_cleaned.csv")
)

transcript_df = pd.read_csv(
    os.path.join(processed_path, "transcript_cleaned.csv")
)

# ----------------------------------
# 1. Dataset overview
# ----------------------------------

print("\nPROFILE DATASET")
print("Shape:", profile_df.shape)
print(profile_df.head())
print(profile_df.describe(include="all"))

print("\nPORTFOLIO DATASET")
print("Shape:", portfolio_df.shape)
print(portfolio_df.head())
print(portfolio_df.describe(include="all"))

print("\nTRANSCRIPT DATASET")
print("Shape:", transcript_df.shape)
print(transcript_df.head())
print(transcript_df.describe(include="all"))

# ----------------------------------
# 2. Event distribution
# ----------------------------------

print("\nEvent counts:")
print(transcript_df["event"].value_counts())

plt.figure(figsize=(8, 5))
transcript_df["event"].value_counts().plot(kind="bar")
plt.title("Distribution of Transcript Events")
plt.xlabel("Event")
plt.ylabel("Number of Records")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "event_distribution.png"))
plt.close()

# ----------------------------------
# 3. Customer demographics
# ----------------------------------

print("\nGender distribution:")
print(profile_df["gender"].value_counts(dropna=False))

plt.figure(figsize=(7, 5))
profile_df["gender"].fillna("Unknown").value_counts().plot(kind="bar")
plt.title("Customer Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "gender_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
profile_df["age"].dropna().plot(kind="hist", bins=30)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "age_distribution.png"))
plt.close()

# ----------------------------------
# 4. Offer types
# ----------------------------------

print("\nOffer type counts:")
print(portfolio_df["offer_type"].value_counts())

plt.figure(figsize=(7, 5))
portfolio_df["offer_type"].value_counts().plot(kind="bar")
plt.title("Distribution of Offer Types")
plt.xlabel("Offer Type")
plt.ylabel("Number of Offers")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "offer_type_distribution.png"))
plt.close()

# ----------------------------------
# 5. Transaction amounts
# ----------------------------------

transactions = transcript_df[
    transcript_df["event"] == "transaction"
]

print("\nTransaction amount summary:")
print(transactions["amount"].describe())

plt.figure(figsize=(8, 5))
transactions["amount"].dropna().plot(kind="hist", bins=50)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "transaction_amount_distribution.png"))
plt.close()

# ----------------------------------
# 6. Event-wise missing values
# ----------------------------------

print("\nMissing values by event:")
print(
    transcript_df.groupby("event")[
        ["offer_id", "amount", "reward"]
    ].apply(lambda group: group.isna().sum())
)

# ----------------------------------
# 7. Save summary
# ----------------------------------

print("\nEDA completed successfully.")
print("Charts saved in:", output_path)

