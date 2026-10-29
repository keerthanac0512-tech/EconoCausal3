from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ranked_customers_v2.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "targeting_strategy_comparison.csv"
)

BUDGET = 5000.0
EXTRA_DISCOUNT_COST = 10.0
RANDOM_SEED = 42


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "estimated_treatment_effect"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["estimated_treatment_effect"] = pd.to_numeric(
    df["estimated_treatment_effect"],
    errors="coerce"
)

df = df.dropna(
    subset=["estimated_treatment_effect"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid treatment-effect estimates found.")

if (df["estimated_treatment_effect"] < 0).any():
    print(
        "Warning: Some customers have negative estimated effects. "
        "Only positive effects will be selected for causal targeting."
    )


# --------------------------------------------------
# Strategy 1: Blanket targeting
# Give every customer the higher discount
# --------------------------------------------------

blanket_customers = len(df)

blanket_cost = (
    blanket_customers * EXTRA_DISCOUNT_COST
)

blanket_incremental_outcome = (
    df["estimated_treatment_effect"].sum()
)


# --------------------------------------------------
# Strategy 2: Budget-constrained causal targeting
# Select customers with the highest positive effects
# --------------------------------------------------

max_customers = int(
    BUDGET // EXTRA_DISCOUNT_COST
)

eligible = df[
    df["estimated_treatment_effect"] > 0
].sort_values(
    "estimated_treatment_effect",
    ascending=False
)

targeted = eligible.head(max_customers)

targeted_customers = len(targeted)

targeted_cost = (
    targeted_customers * EXTRA_DISCOUNT_COST
)

targeted_incremental_outcome = (
    targeted["estimated_treatment_effect"].sum()
)


# --------------------------------------------------
# Strategy 3: Budget-matched random targeting
# Randomly select the same number of customers
# --------------------------------------------------

random_customers = df.sample(
    n=targeted_customers,
    random_state=RANDOM_SEED
)

random_cost = (
    len(random_customers) * EXTRA_DISCOUNT_COST
)

random_incremental_outcome = (
    random_customers["estimated_treatment_effect"].sum()
)


# --------------------------------------------------
# Compare strategies
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "strategy": "Blanket targeting",
        "customers_selected": blanket_customers,
        "estimated_extra_cost": blanket_cost,
        "estimated_incremental_outcome":
            blanket_incremental_outcome,
        "within_budget": blanket_cost <= BUDGET
    },
    {
        "strategy": "Causal targeting",
        "customers_selected": targeted_customers,
        "estimated_extra_cost": targeted_cost,
        "estimated_incremental_outcome":
            targeted_incremental_outcome,
        "within_budget": targeted_cost <= BUDGET
    },
    {
        "strategy": "Random targeting",
        "customers_selected": len(random_customers),
        "estimated_extra_cost": random_cost,
        "estimated_incremental_outcome":
            random_incremental_outcome,
        "within_budget": random_cost <= BUDGET
    }
])

comparison["estimated_outcome_per_dollar"] = (
    comparison["estimated_incremental_outcome"]
    / comparison["estimated_extra_cost"]
)

comparison["estimated_cost_avoided_vs_blanket"] = (
    blanket_cost - comparison["estimated_extra_cost"]
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n========== TARGETING STRATEGY COMPARISON ==========")

print("Customers evaluated:", len(df))
print(f"Available additional-discount budget: ${BUDGET:,.2f}")
print(
    f"Additional cost per higher discount: "
    f"${EXTRA_DISCOUNT_COST:,.2f}"
)

print("\n", comparison.round(4).to_string(index=False))

print("\nEstimated cost avoided by causal targeting:")
print(f"${blanket_cost - targeted_cost:,.2f}")

print("\nCausal targeting vs budget-matched random targeting:")
print(
    "Difference in estimated incremental outcomes:",
    round(
        targeted_incremental_outcome
        - random_incremental_outcome,
        4
    )
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\n========== COMPARISON COMPLETE ==========")
