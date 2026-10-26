import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Load cleaned datasets
profile = pd.read_csv(DATA_DIR / "profile_cleaned.csv")
portfolio = pd.read_csv(DATA_DIR / "portfolio_cleaned.csv")
transcript = pd.read_csv(DATA_DIR / "transcript_cleaned.csv")

# Display columns to verify the data
print("Profile columns:", profile.columns.tolist())
print("Portfolio columns:", portfolio.columns.tolist())
print("Transcript columns:", transcript.columns.tolist())

# Check required columns
required_profile = {"id", "age", "income"}
required_portfolio = {"id", "offer_type", "difficulty", "duration"}
required_transcript = {"person", "event", "time"}

if not required_profile.issubset(profile.columns):
    raise ValueError(
        f"Profile is missing: {required_profile - set(profile.columns)}"
    )

if not required_portfolio.issubset(portfolio.columns):
    raise ValueError(
        f"Portfolio is missing: {required_portfolio - set(portfolio.columns)}"
    )

if not required_transcript.issubset(transcript.columns):
    raise ValueError(
        f"Transcript is missing: {required_transcript - set(transcript.columns)}"
    )

# Remove duplicate customer and offer records if present
profile = profile.drop_duplicates(subset="id")
portfolio = portfolio.drop_duplicates(subset="id")

# Convert numeric columns
profile["age"] = pd.to_numeric(profile["age"], errors="coerce")
profile["income"] = pd.to_numeric(profile["income"], errors="coerce")

# Exclude invalid ages
profile.loc[profile["age"] == 118, "age"] = pd.NA

# Attach customer profile information to each event
events = transcript.merge(
    profile,
    left_on="person",
    right_on="id",
    how="left",
    validate="many_to_one"
)

# Attach offer details where an offer ID is available
if "offer_id" in events.columns:
    events = events.merge(
        portfolio,
        left_on="offer_id",
        right_on="id",
        how="left",
        suffixes=("", "_offer"),
        validate="many_to_one"
    )
else:
    print(
        "WARNING: 'offer_id' is missing. "
        "Offer details cannot yet be linked."
    )

# Save the event-level dataset
OUTPUT_FILE = DATA_DIR / "causal_events.csv"
events.to_csv(OUTPUT_FILE, index=False)

print("\nCausal event dataset created!")
print("Shape:", events.shape)
print("Saved to:", OUTPUT_FILE)

print("\nEvent counts:")
print(events["event"].value_counts())

print("\nMissing values in key columns:")
for column in ["person", "event", "age", "income", "offer_id"]:
    if column in events.columns:
        print(f"{column}: {events[column].isna().sum()}")

print("\nFirst five rows:")
print(events.head())