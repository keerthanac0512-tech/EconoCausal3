import pandas as pd
from scipy.optimize import linprog

# Load ranked customers
input_path = "data/processed/ranked_customers_v2.csv"
df = pd.read_csv(input_path)

# Budget assumptions
BUDGET = 5000
EXTRA_DISCOUNT_COST = 10

# Objective: maximize total estimated incremental purchases
# linprog minimizes, so use negative treatment effects.
c = -df["estimated_treatment_effect"].to_numpy()

# Each customer can receive at most one upgrade.
# x[i] = 1 means upgrade that customer's discount from $10 to $20.
# x[i] = 0 means keep the $10 discount.
budget_constraint = (
    EXTRA_DISCOUNT_COST
    * pd.Series(1.0, index=df.index).to_numpy()
)

result = linprog(
    c=c,
    A_ub=[budget_constraint],
    b_ub=[BUDGET],
    bounds=(0, 1),
    method="highs"
)

if not result.success:
    raise RuntimeError(
        f"Optimization failed: {result.message}"
    )

# Select customers receiving the higher discount
df["upgrade_decision"] = (
    result.x >= 0.5
).astype(int)

selected = df[df["upgrade_decision"] == 1].copy()

# Save decisions
output_path = "data/processed/budget_optimized_customers.csv"
df.to_csv(output_path, index=False)

# Summarize results
total_cost = len(selected) * EXTRA_DISCOUNT_COST
estimated_incremental_purchases = (
    selected["estimated_treatment_effect"].sum()
)

print("Budget optimization completed!")
print("Total customers evaluated:", len(df))
print("Customers selected for $20 discount:", len(selected))
print("Customers remaining at $10 discount:", len(df) - len(selected))
print("Budget:", f"${BUDGET:,.2f}")
print("Estimated budget used:", f"${total_cost:,.2f}")
print(
    "Estimated incremental purchases:",
    round(estimated_incremental_purchases, 2)
)
print("Unused budget:", f"${BUDGET - total_cost:,.2f}")
print("\nSaved to:", output_path)
