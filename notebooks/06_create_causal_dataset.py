import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# --------------------------------------------------
# 2. Load cleaned datasets
# --------------------------------------------------

events = pd.read_csv(DATA_DIR / "causal_events.csv")
profile = pd.read_csv(DATA_DIR / "profile_cleaned.csv")

print("Loaded event dataset:", events.shape)
print("Loaded profile dataset:", profile.shape)

print("\nOriginal event counts:")
print(events["event"].value_counts(dropna=False))

# --------------------------------------------------
# 3. Recover offer IDs from event values if necessary
# --------------------------------------------------

def extract_offer_id(value):
    """Extract an offer ID from a dictionary or JSON string."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(value, dict):
        return value.get("offer id") or value.get("offer_id")

    return None


# Recover missing offer IDs from the original value column.
# This is especially important for offer-completed events.
if "value" in events.columns:

    if "offer_id" not in events.columns:
        events["offer_id"] = None

    missing_offer_id = events["offer_id"].isna()

    events.loc[missing_offer_id, "offer_id"] = (
        events.loc[missing_offer_id, "value"]
        .apply(extract_offer_id)
    )

# --------------------------------------------------
# 4. Inspect offer IDs by event type
# --------------------------------------------------

print("\nOffer ID availability by event:")
print(
    events.assign(
        has_offer_id=events["offer_id"].notna()
    ).groupby("event")["has_offer_id"].agg(
        total="count",
        with_offer_id="sum"
    )
)

# --------------------------------------------------
# 5. Keep offer-related events
# --------------------------------------------------

offer_event_names = [
    "offer received",
    "offer viewed",
    "offer completed"
]

offer_events = events[
    events["event"].isin(offer_event_names)
].copy()

offer_events = offer_events.dropna(
    subset=["person", "offer_id"]
)

# Ensure offer IDs are strings
offer_events["offer_id"] = offer_events["offer_id"].astype(str)

# --------------------------------------------------
# 6. Build customer-offer records
# --------------------------------------------------

# Start with offers that customers actually received.
# This avoids treating a viewed or completed event alone
# as proof that the customer received that offer.

received = offer_events[
    offer_events["event"] == "offer received"
].copy()

if received.empty:
    raise ValueError(
        "No offer-received records with offer IDs were found. "
        "Check the offer ID extraction output above."
    )

# Each received event is a separate offer exposure.
# Keep event time so later events can be matched to the
# correct exposure.
if "time" not in received.columns:
    raise ValueError("The event dataset has no 'time' column.")

received["time"] = pd.to_numeric(
    received["time"], errors="coerce"
)

received = received.dropna(subset=["time"])

received = received.rename(
    columns={"time": "received_time"}
)

# Create a unique ID for each received exposure.
received = received.reset_index(drop=True)
received["exposure_id"] = received.index

# Keep the offer information available at receipt time.
offer_columns = [
    column for column in [
        "offer_type",
        "difficulty",
        "duration",
        "reward",
        "channels"
    ]
    if column in received.columns
]

causal_data = received[
    ["exposure_id", "person", "offer_id", "received_time"]
    + offer_columns
].copy()

# --------------------------------------------------
# 7. Match viewed events to received exposures
# --------------------------------------------------

viewed = offer_events[
    offer_events["event"] == "offer viewed"
].copy()

viewed["time"] = pd.to_numeric(
    viewed["time"], errors="coerce"
)

viewed = viewed.dropna(subset=["time"])

# Sort records so the closest previous received exposure
# can be found for each viewed event.
received_sorted = received[
    ["exposure_id", "person", "offer_id", "received_time"]
].sort_values("received_time")

viewed = viewed.sort_values("time")

if not viewed.empty:
    viewed = pd.merge_asof(
        viewed,
        received_sorted,
        left_on="time",
        right_on="received_time",
        by=["person", "offer_id"],
        direction="backward",
        allow_exact_matches=True
    )

    viewed = viewed.dropna(subset=["exposure_id"])

    viewed_ids = set(viewed["exposure_id"].astype(int))
else:
    viewed_ids = set()

causal_data["offer_viewed"] = (
    causal_data["exposure_id"].isin(viewed_ids).astype(int)
)

# --------------------------------------------------
# 8. Match completed events to received exposures
# --------------------------------------------------

completed = offer_events[
    offer_events["event"] == "offer completed"
].copy()

completed["time"] = pd.to_numeric(
    completed["time"], errors="coerce"
)

completed = completed.dropna(subset=["time"])
completed = completed.sort_values("time")

if not completed.empty:
    completed = pd.merge_asof(
        completed,
        received_sorted,
        left_on="time",
        right_on="received_time",
        by=["person", "offer_id"],
        direction="backward",
        allow_exact_matches=True
    )

    completed = completed.dropna(subset=["exposure_id"])

    completed_ids = set(
        completed["exposure_id"].astype(int)
    )
else:
    completed_ids = set()

causal_data["offer_completed"] = (
    causal_data["exposure_id"].isin(completed_ids).astype(int)
)

# --------------------------------------------------
# 9. Attach customer demographic information
# --------------------------------------------------

if "id" not in profile.columns:
    raise ValueError(
        "The profile dataset has no 'id' column. "
        "Check profile_cleaned.csv."
    )

profile_columns = [
    column for column in ["id", "age", "income", "gender"]
    if column in profile.columns
]

profile = profile[profile_columns].drop_duplicates(
    subset="id"
)

causal_data = causal_data.merge(
    profile,
    left_on="person",
    right_on="id",
    how="left",
    validate="many_to_one"
)

causal_data = causal_data.drop(columns=["id"], errors="ignore")

# --------------------------------------------------
# 10. Save the causal dataset
# --------------------------------------------------

output_file = DATA_DIR / "causal_dataset.csv"

causal_data.to_csv(output_file, index=False)

# --------------------------------------------------
# 11. Display validation results
# --------------------------------------------------

print("\nCausal dataset created.")
print("Shape:", causal_data.shape)

print("\nOffer types:")
print(causal_data["offer_type"].value_counts(dropna=False))

print("\nOffer viewed:")
print(causal_data["offer_viewed"].value_counts())

print("\nOffer completed:")
print(causal_data["offer_completed"].value_counts())

print("\nCompletion rate by offer type:")
print(
    causal_data.groupby("offer_type")["offer_completed"]
    .agg(["count", "sum", "mean"])
    .round(3)
)

print("\nMissing values:")
print(causal_data.isna().sum())

print("\nFirst five rows:")
print(causal_data.head())

print("\nSaved to:", output_file)
