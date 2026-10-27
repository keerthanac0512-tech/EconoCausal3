from pathlib import Path
from io import BytesIO

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# --------------------------------------------------
# Application setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

REQUIRED_DML_COLUMNS = {
    "estimated_treatment_effect"
}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_csv(filename):
    """Load a processed CSV file."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")

    return pd.read_csv(file_path)


def records_from_dataframe(df, limit=100):
    """Convert a DataFrame to JSON-safe records."""
    safe_df = df.head(limit).astype(object).where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


def validate_budget(value):
    """Validate the supplied budget."""
    try:
        budget = float(value)
    except (TypeError, ValueError):
        raise ValueError("Budget must be a valid number.")

    if not pd.notna(budget) or budget < 0:
        raise ValueError("Budget must be a non-negative number.")

    if budget == float("inf") or budget == float("-inf"):
        raise ValueError("Budget must be finite.")

    return budget


# --------------------------------------------------
# Route 1: Health check
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "EconoCausal backend is running"
    })


# --------------------------------------------------
# Route 2: DML treatment effects
# --------------------------------------------------

@app.route("/api/treatment-effects", methods=["GET"])
def treatment_effects():
    try:
        df = load_csv("dml_treatment_effects_v2.csv")

        return jsonify({
            "status": "success",
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "data": records_from_dataframe(df)
        })

    except FileNotFoundError as error:
        return jsonify({"status": "error", "message": str(error)}), 404

    except (ValueError, pd.errors.ParserError) as error:
        return jsonify({"status": "error", "message": str(error)}), 500


# --------------------------------------------------
# Route 3: Existing budget optimization results
# --------------------------------------------------

@app.route("/api/budget-results", methods=["GET"])
def budget_results():
    try:
        df = load_csv("budget_optimized_customers.csv")

        return jsonify({
            "status": "success",
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "data": records_from_dataframe(df)
        })

    except FileNotFoundError as error:
        return jsonify({"status": "error", "message": str(error)}), 404

    except (ValueError, pd.errors.ParserError) as error:
        return jsonify({"status": "error", "message": str(error)}), 500


# --------------------------------------------------
# Route 4: Uplift evaluation results
# --------------------------------------------------

@app.route("/api/uplift-results", methods=["GET"])
def uplift_results():
    try:
        df = load_csv("uplift_evaluation_v2.csv")

        return jsonify({
            "status": "success",
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "data": records_from_dataframe(df)
        })

    except FileNotFoundError as error:
        return jsonify({"status": "error", "message": str(error)}), 404

    except (ValueError, pd.errors.ParserError) as error:
        return jsonify({"status": "error", "message": str(error)}), 500


# --------------------------------------------------
# Route 5: Upload and validate a campaign CSV
# --------------------------------------------------

@app.route("/api/upload-campaign", methods=["POST"])
def upload_campaign():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file provided. Upload a CSV using the 'file' field."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "status": "error",
            "message": "Please select a file."
        }), 400

    if Path(file.filename).suffix.lower() != ".csv":
        return jsonify({
            "status": "error",
            "message": "Only CSV files are supported."
        }), 400

    try:
        df = pd.read_csv(BytesIO(file.read()))

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV has no data rows."
            }), 400

        if len(df.columns) == 0:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV has no columns."
            }), 400

        if df.columns.duplicated().any():
            return jsonify({
                "status": "error",
                "message": "The CSV contains duplicate column names."
            }), 400

        return jsonify({
            "status": "success",
            "message": (
                "CSV uploaded and validated. "
                "No new treatment-effect predictions have been generated."
            ),
            "filename": Path(file.filename).name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": records_from_dataframe(df, limit=10)
        })

    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        return jsonify({
            "status": "error",
            "message": f"Could not read the CSV: {error}"
        }), 400


# --------------------------------------------------
# Route 6: Optimize budget using existing DML effects
# --------------------------------------------------

@app.route("/api/optimize-budget", methods=["POST"])
def optimize_budget():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "budget" not in payload:
        return jsonify({
            "status": "error",
            "message": "Send a JSON object containing 'budget'."
        }), 400

    try:
        budget = validate_budget(payload["budget"])

        # This prototype assumes that choosing the higher discount
        # costs an additional $10 per selected customer.
        extra_discount_cost = 10.0
        max_customers = int(budget // extra_discount_cost)

        df = load_csv("dml_treatment_effects_v2.csv")

        missing = REQUIRED_DML_COLUMNS - set(df.columns)
        if missing:
            return jsonify({
                "status": "error",
                "message": (
                    "DML dataset is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            }), 500

        df["estimated_treatment_effect"] = pd.to_numeric(
            df["estimated_treatment_effect"], errors="coerce"
        )

        df = df.dropna(subset=["estimated_treatment_effect"])

        # Select the largest positive estimated effects first.
        eligible = df[
            df["estimated_treatment_effect"] > 0
        ].sort_values(
            "estimated_treatment_effect", ascending=False
        )

        selected = eligible.head(max_customers).copy()
        selected["recommended_discount"] = 20
        selected["additional_discount_cost"] = extra_discount_cost

        selected_indices = set(selected.index)
        remaining = df.loc[~df.index.isin(selected_indices)].copy()
        remaining["recommended_discount"] = 10
        remaining["additional_discount_cost"] = 0.0

        allocation = pd.concat(
            [selected, remaining], ignore_index=True
        )

        spent = len(selected) * extra_discount_cost

        return jsonify({
            "status": "success",
            "budget": budget,
            "additional_discount_cost_per_customer": extra_discount_cost,
            "customers_selected_for_higher_discount": len(selected),
            "customers_remaining_at_lower_discount": len(remaining),
            "budget_spent": spent,
            "budget_remaining": budget - spent,
            "sum_selected_estimated_treatment_effects": float(
                selected["estimated_treatment_effect"].sum()
            ),
            "data": records_from_dataframe(
                allocation.sort_values(
                    "estimated_treatment_effect", ascending=False
                ),
                limit=100
            ),
            "note": (
                "Uses existing DML estimates. Uploaded campaign data "
                "is not scored by a trained model."
            )
        })

    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    except FileNotFoundError as error:
        return jsonify({"status": "error", "message": str(error)}), 404

    except (pd.errors.ParserError, KeyError) as error:
        return jsonify({"status": "error", "message": str(error)}), 500


# --------------------------------------------------
# Handle oversized uploads
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "status": "error",
        "message": "File too large. Maximum upload size is 10 MB."
    }), 413


# --------------------------------------------------
# Start development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
