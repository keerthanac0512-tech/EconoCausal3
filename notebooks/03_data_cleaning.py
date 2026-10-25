import pandas as pd
import json
import os
import numpy as np

# ==========================================
# STEP 1: DEFINE PATHS
# ==========================================

raw_path = "data/raw"
processed_path = "data/processed"

os.makedirs(processed_path, exist_ok=True)


# ==========================================
# STEP 2: LOAD JSON FILES
# ==========================================

def load_json_file(filename):
    filepath = os.path.join(raw_path, filename)

    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        raise ValueError(f"{filename} is empty.")

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        records = []

        with open(filepath, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {filename}, "
                        f"line {line_number}"
                    ) from error

        return records


profile = load_json_file("profile.json")
portfolio = load_json_file("portfolio.json")
transcript = load_json_file("transcript.json")


# ==========================================
# STEP 3: CONVERT TO DATAFRAMES
# ==========================================

def convert_to_dataframe(data):
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "original_id"
        return df

    if isinstance(data, list):
        return pd.DataFrame(data)

    raise TypeError("Unsupported JSON format.")


profile_df = convert_to_dataframe(profile)
portfolio_df = convert_to_dataframe(portfolio)
transcript_df = convert_to_dataframe(transcript)


# ==========================================
# STEP 4: CLEAN CUSTOMER PROFILE
# ==========================================

print("\n--- CLEANING CUSTOMER PROFILE ---")

# Convert age to numeric
profile_df["age"] = pd.to_numeric(
    profile_df["age"], errors="coerce"
)

# Replace the age placeholder 118 with missing values
profile_df.loc[
    profile_df["age"] == 118, "age"
] = np.nan

# Convert income to numeric
profile_df["income"] = pd.to_numeric(
    profile_df["income"], errors="coerce"
)

# Standardize gender values
profile_df["gender"] = (
    profile_df["gender"]
    .replace({"None": np.nan, "": np.nan})
)

print("Missing ages:", profile_df["age"].isna().sum())
print("Missing genders:", profile_df["gender"].isna().sum())
print("Missing incomes:", profile_df["income"].isna().sum())


# ==========================================
# STEP 5: CLEAN MEMBERSHIP DATE
# ==========================================

profile_df["became_member_on"] = pd.to_datetime(
    profile_df["became_member_on"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

print(
    "Invalid membership dates:",
    profile_df["became_member_on"].isna().sum()
)


# ==========================================
# STEP 6: CLEAN PORTFOLIO
# ==========================================

print("\n--- CLEANING PORTFOLIO ---")

portfolio_df["reward"] = pd.to_numeric(
    portfolio_df["reward"], errors="coerce"
)

portfolio_df["difficulty"] = pd.to_numeric(
    portfolio_df["difficulty"], errors="coerce"
)

portfolio_df["duration"] = pd.to_numeric(
    portfolio_df["duration"], errors="coerce"
)

# Convert channels into readable text for CSV export
portfolio_df["channels"] = portfolio_df["channels"].apply(
    lambda x: json.dumps(x) if isinstance(x, list) else x
)

print("Offer types:")
print(portfolio_df["offer_type"].value_counts())


# ==========================================
# STEP 7: CLEAN TRANSCRIPT
# ==========================================

print("\n--- CLEANING TRANSCRIPT ---")

# Preserve all events.
# Do not drop duplicate events automatically.

# Extract offer ID from nested event values
def extract_offer_id(value):
    if isinstance(value, dict):
        return value.get("offer id")

    return None


transcript_df["offer_id"] = transcript_df["value"].apply(
    extract_offer_id
)

# Extract transaction amount where available
def extract_amount(value):
    if isinstance(value, dict):
        return value.get("amount")

    return None


transcript_df["amount"] = pd.to_numeric(
    transcript_df["value"].apply(extract_amount),
    errors="coerce"
)

# Extract reward where available
def extract_reward(value):
    if isinstance(value, dict):
        return value.get("reward")

    return None


transcript_df["reward"] = pd.to_numeric(
    transcript_df["value"].apply(extract_reward),
    errors="coerce"
)

# Preserve the original nested values as JSON text
transcript_df["value"] = transcript_df["value"].apply(
    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
)

print("Transcript event counts:")
print(transcript_df["event"].value_counts())


# ==========================================
# STEP 8: CHECK DUPLICATES
# ==========================================

print("\n--- DUPLICATE CHECK ---")

print("Profile duplicate rows:", profile_df.duplicated().sum())
print("Portfolio duplicate rows:", portfolio_df.duplicated().sum())
print("Transcript duplicate rows:", transcript_df.duplicated().sum())

print(
    "\nTranscript duplicate count is for review only; "
    "no transcript rows have been deleted."
)


# ==========================================
# STEP 9: SAVE CLEANED DATASETS
# ==========================================

profile_output = os.path.join(
    processed_path, "profile_cleaned.csv"
)

portfolio_output = os.path.join(
    processed_path, "portfolio_cleaned.csv"
)

transcript_output = os.path.join(
    processed_path, "transcript_cleaned.csv"
)

profile_df.to_csv(profile_output, index=False)
portfolio_df.to_csv(portfolio_output, index=False)
transcript_df.to_csv(transcript_output, index=False)


# ==========================================
# STEP 10: FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("FINAL CLEANING SUMMARY")
print("=" * 60)

print("Profile shape:", profile_df.shape)
print("Portfolio shape:", portfolio_df.shape)
print("Transcript shape:", transcript_df.shape)

print("\nProfile missing values:")
print(profile_df.isna().sum())

print("\nPortfolio missing values:")
print(portfolio_df.isna().sum())

print("\nTranscript missing values:")
print(transcript_df.isna().sum())

print("\nCleaned files saved in:", processed_path)

print("\nDATA CLEANING COMPLETED")
