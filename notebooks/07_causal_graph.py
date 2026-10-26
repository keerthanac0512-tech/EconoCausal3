import pandas as pd
from pathlib import Path
from dowhy import CausalModel

# --------------------------------------------------
# 1. Load the prepared dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "causal_dataset.csv"

data = pd.read_csv(DATA_FILE)

print("Dataset loaded:", data.shape)

# --------------------------------------------------
# 2. Select variables for causal analysis
# --------------------------------------------------

required_columns = [
    "person",
    "offer_id",
    "offer_type",
    "difficulty",
    "duration",
    "offer_viewed",
    "offer_completed",
    "age",
    "income"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# --------------------------------------------------
# 3. Prepare the analysis data
# --------------------------------------------------

analysis_data = data[
    [
        "offer_type",
        "difficulty",
        "duration",
        "offer_viewed",
        "offer_completed",
        "age",
        "income"
    ]
].copy()

# Exclude informational offers from this initial
# reward-based completion analysis.
analysis_data = analysis_data[
    analysis_data["offer_type"].isin(["bogo", "discount"])
].copy()

# Encode offer type as a binary variable:
# 1 = discount, 0 = BOGO
analysis_data["is_discount"] = (
    analysis_data["offer_type"] == "discount"
).astype(int)

# Remove rows missing required analysis values.
analysis_data = analysis_data.dropna(
    subset=[
        "difficulty",
        "duration",
        "offer_viewed",
        "offer_completed",
        "age",
        "income"
    ]
)

print("Analysis dataset:", analysis_data.shape)

# --------------------------------------------------
# 4. Define the causal graph
# --------------------------------------------------

causal_graph = """
digraph {
    age -> offer_viewed;
    income -> offer_viewed;
    difficulty -> offer_viewed;
    duration -> offer_viewed;
    is_discount -> offer_viewed;

    age -> offer_completed;
    income -> offer_completed;
    difficulty -> offer_completed;
    duration -> offer_completed;
    is_discount -> offer_completed;

    offer_viewed -> offer_completed;
}
"""

# --------------------------------------------------
# 5. Build the DoWhy causal model
# --------------------------------------------------

model = CausalModel(
    data=analysis_data,
    treatment="offer_viewed",
    outcome="offer_completed",
    graph=causal_graph
)

print("\nCausal model created successfully!")

# --------------------------------------------------
# 6. Identify the causal effect
# --------------------------------------------------

estimand = model.identify_effect()

print("\nIdentified estimand:")
print(estimand)

# --------------------------------------------------
# 7. Save a graph image if Graphviz is available
# --------------------------------------------------

try:
    model.view_model(
        layout="dot",
        file_name=str(BASE_DIR / "reports" / "figures" / "causal_graph")
    )
    print("\nCausal graph saved.")
except Exception as error:
    print("\nCould not render graph automatically:", error)
    print("The graph definition is still available in this script.")
# Step 3: Estimate the causal effect
causal_estimate = model.estimate_effect(
    estimand,
    method_name="backdoor.linear_regression"
)

print("\nEstimated Causal Effect:")
print(causal_estimate)