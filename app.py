
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.drift_detection import detect_data_drift


# ==================================================
# 1. FLASK APPLICATION SETUP
# ==================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def load_csv(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


def validate_budget(value):
    try:
        budget = float(value)
    except (TypeError, ValueError):
        raise ValueError("Budget must be a valid number.")

    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    return budget


def records_from_dataframe(df, limit=100):
    clean_df = df.head(limit).copy()

    clean_df = clean_df.astype(object).where(
        pd.notna(clean_df),
        None
    )

    return clean_df.to_dict(orient="records")


def validate_csv_upload(file):

    if file is None:
        raise ValueError("Please select a CSV file.")

    if not file.filename:
        raise ValueError("Uploaded file has no filename.")

    if not file.filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    try:
        df = pd.read_csv(file)

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("The CSV contains duplicate column names.")

    return df


# ==================================================
# 3. HEALTH CHECK
# ==================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "EconoCausal API is running."
    }), 200


# ==================================================
# 4. TREATMENT EFFECTS
# ==================================================

@app.route("/api/treatment-effects", methods=["GET"])
def treatment_effects():

    try:

        df = load_csv("dml_treatment_effects_v2.csv")

        return jsonify({
            "status": "success",
            "total_records": len(df),
            "data": records_from_dataframe(df, 100)
        }), 200

    except FileNotFoundError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except Exception:

        app.logger.exception("Treatment effects error.")

        return jsonify({
            "status": "error",
            "message": "Unable to load treatment effects."
        }), 500


# ==================================================
# 5. BUDGET RESULTS
# ==================================================

@app.route("/api/budget-results", methods=["GET"])
def budget_results():

    try:

        df = load_csv("budget_optimized_customers.csv")

        return jsonify({
            "status": "success",
            "total_records": len(df),
            "data": records_from_dataframe(df, 100)
        }), 200

    except FileNotFoundError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except Exception:

        app.logger.exception("Budget results error.")

        return jsonify({
            "status": "error",
            "message": "Unable to load budget results."
        }), 500


# ==================================================
# 6. UPLIFT AND QINI RESULTS
# ==================================================

@app.route("/api/uplift-results", methods=["GET"])
def uplift_results():

    try:

        df = load_csv("uplift_evaluation_v2.csv")

        required_columns = [
            "customer_rank",
            "fraction_targeted",
            "cumulative_uplift",
            "qini_curve",
            "random_targeting"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:

            return jsonify({
                "status": "error",
                "message": f"Missing columns: {missing}"
            }), 500

        # Select chart columns
        df = df[required_columns].copy()

        # Replace NaN values with None
        df = df.astype(object).where(
            pd.notna(df),
            None
        )

        # Return ALL rows for Plotly charts
        return jsonify({

            "status": "success",

            "total_records": len(df),

            "data": df.to_dict(orient="records")

        }), 200

    except FileNotFoundError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except Exception:

        app.logger.exception("Uplift results error.")

        return jsonify({
            "status": "error",
            "message": "Unable to load uplift results."
        }), 500


# ==================================================
# 7. UPLOAD CAMPAIGN
# ==================================================

@app.route("/api/upload-campaign", methods=["POST"])
def upload_campaign():

    try:

        file = request.files.get("file")

        df = validate_csv_upload(file)

        return jsonify({

            "status": "success",

            "message": "Campaign CSV uploaded successfully.",

            "filename": file.filename,

            "rows": len(df),

            "columns": list(df.columns),

            "data": records_from_dataframe(df, 20)

        }), 200

    except ValueError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except Exception:

        app.logger.exception("Campaign upload error.")

        return jsonify({
            "status": "error",
            "message": "Unable to process campaign CSV."
        }), 500


# ==================================================
# 8. CAMPAIGN DATA QUALITY
# ==================================================

@app.route("/api/campaign-data-quality", methods=["POST"])
def campaign_data_quality():

    try:

        file = request.files.get("file")

        df = validate_csv_upload(file)

        total_cells = int(
            df.shape[0] * df.shape[1]
        )

        missing_values = int(
            df.isnull().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        missing_percentage = (
            round(missing_values / total_cells * 100, 2)
            if total_cells > 0 else 0
        )

        return jsonify({

            "status": "success",

            "filename": file.filename,

            "rows": int(len(df)),

            "columns": int(len(df.columns)),

            "missing_values": missing_values,

            "missing_percentage": missing_percentage,

            "duplicate_rows": duplicate_rows,

            "total_cells": total_cells,

            "column_names": list(df.columns)

        }), 200

    except ValueError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except Exception:

        app.logger.exception("Campaign data quality error.")

        return jsonify({
            "status": "error",
            "message": "Unable to analyze campaign data quality."
        }), 500


# ==================================================
# 9. BUDGET / DISCOUNT STRATEGY ANALYSIS
# ==================================================

@app.route("/api/optimize-budget", methods=["POST"])
def optimize_budget():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "budget" not in payload:
        return jsonify({
            "status": "error",
            "message": "Please provide a budget."
        }), 400

    try:
        budget = validate_budget(payload["budget"])

        # Load customer-level DML treatment effects.
        df = load_csv("dml_treatment_effects_v2.csv")

        required_columns = {
            "person",
            "estimated_treatment_effect"
        }

        missing = required_columns - set(df.columns)

        if missing:
            return jsonify({
                "status": "error",
                "message": (
                    "Missing DML columns: "
                    + ", ".join(sorted(missing))
                )
            }), 500

        df["estimated_treatment_effect"] = pd.to_numeric(
            df["estimated_treatment_effect"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["person", "estimated_treatment_effect"]
        ).copy()

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "No valid treatment effects found."
            }), 400

        # Keep one prediction per customer and rank by estimated ITE.
        df = (
            df.sort_values(
                "estimated_treatment_effect",
                ascending=False
            )
            .drop_duplicates(subset=["person"], keep="first")
            .reset_index(drop=True)
        )

        # Positive ITE means the 20% discount is estimated to perform
        # better than the 10% discount for that customer.
        df["recommended_discount"] = (
            df["estimated_treatment_effect"]
            .apply(lambda effect: 20 if effect > 0 else 10)
        )

        df["allocation_reason"] = df[
            "estimated_treatment_effect"
        ].apply(
            lambda effect: (
                "Positive estimated effect: 20% is preferred over 10%"
                if effect > 0
                else "No positive estimated benefit from upgrading to 20%"
            )
        )

        df["effect_rank"] = (
            df["estimated_treatment_effect"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        records_at_20 = int(
            (df["recommended_discount"] == 20).sum()
        )
        records_at_10 = int(
            (df["recommended_discount"] == 10).sum()
        )

        # The sum of positive ITEs is a model-based expected incremental
        # purchase count versus assigning those same customers 10%.
        estimated_incremental_purchases = float(
            df.loc[
                df["recommended_discount"] == 20,
                "estimated_treatment_effect"
            ].sum()
        )

        allocation_matrix = [
            {
                "discount": 10,
                "records_allocated": records_at_10,
                "mean_estimated_treatment_effect": (
                    float(
                        df.loc[
                            df["recommended_discount"] == 10,
                            "estimated_treatment_effect"
                        ].mean()
                    )
                    if records_at_10 else 0.0
                )
            },
            {
                "discount": 20,
                "records_allocated": records_at_20,
                "mean_estimated_treatment_effect": (
                    float(
                        df.loc[
                            df["recommended_discount"] == 20,
                            "estimated_treatment_effect"
                        ].mean()
                    )
                    if records_at_20 else 0.0
                )
            }
        ]

        # Save the recommended customer-level allocation.
        allocation_path = DATA_DIR / "budget_optimized_customers.csv"
        df.sort_values("effect_rank").to_csv(
            allocation_path,
            index=False
        )

        return jsonify({
            "status": "success",
            "message": "Discount strategy analysis completed.",
            "summary": {
                "total_records": int(len(df)),
                "records_at_10": records_at_10,
                "records_at_20": records_at_20,
                "budget_reference": budget,
                "estimated_incremental_purchases": (
                    estimated_incremental_purchases
                )
            },
            "allocation_matrix": allocation_matrix,
            "budget_note": (
                "The entered budget is retained as a planning reference. "
                "It is not used to cap customer allocation because this "
                "dataset has no order value, purchase amount, or verified "
                "discount-cost column."
            ),
            "effect_note": (
                "Estimated treatment effect is the model-estimated change "
                "in purchase probability when moving from a 10% discount "
                "to a 20% discount. The summed positive effects are an "
                "expected incremental outcome estimate, not guaranteed sales."
            ),
            "data": records_from_dataframe(
                df.sort_values("effect_rank"),
                limit=100
            )
        }), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except FileNotFoundError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except (pd.errors.ParserError, KeyError) as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

    except Exception:
        app.logger.exception("Discount strategy analysis failed.")
        return jsonify({
            "status": "error",
            "message": "Unexpected discount strategy analysis error."
        }), 500


# ==================================================
# 10. TARGETING STRATEGY COMPARISON
# ==================================================

@app.route("/api/targeting-comparison", methods=["GET"])
def targeting_comparison():

    try:

        df = load_csv(
            "targeting_strategy_comparison.csv"
        )

        return jsonify({

            "status": "success",

            "total_records": len(df),

            "data": records_from_dataframe(
                df,
                100
            )

        }), 200

    except FileNotFoundError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except Exception:

        app.logger.exception("Targeting comparison error.")

        return jsonify({
            "status": "error",
            "message": "Unable to load targeting comparison."
        }), 500


# ==================================================
# 11. DATA DRIFT DETECTION
# ==================================================

@app.route("/api/data-drift", methods=["POST"])
def data_drift():

    try:

        file = request.files.get("file")

        if file is None:

            return jsonify({
                "status": "error",
                "message": "Please upload a CSV file."
            }), 400

        if not file.filename.lower().endswith(".csv"):

            return jsonify({
                "status": "error",
                "message": "Only CSV files are supported."
            }), 400

        uploaded_df = pd.read_csv(file)

        if uploaded_df.empty:

            return jsonify({
                "status": "error",
                "message": "Uploaded dataset is empty."
            }), 400

        reference_df = load_csv(
            "causal_dataset_synthetic_v2.csv"
        )

        drift_results = detect_data_drift(
            reference_df,
            uploaded_df
        )

        if isinstance(drift_results, pd.DataFrame):

            drift_results = records_from_dataframe(
                drift_results,
                limit=100
            )

        elif hasattr(drift_results, "to_dict"):

            drift_results = drift_results.to_dict()

        return jsonify({

            "status": "success",

            "message": "Data drift analysis completed.",

            "results": drift_results

        }), 200

    except FileNotFoundError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 404

    except (ValueError, pd.errors.ParserError) as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except Exception:

        app.logger.exception("Data drift detection error.")

        return jsonify({
            "status": "error",
            "message": "Unable to perform data drift detection."
        }), 500


# ==================================================
# 12. FILE SIZE ERROR HANDLER
# ==================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "status": "error",

        "message": (
            "File too large. Maximum upload size is 10 MB."
        )

    }), 413


# ==================================================
# 13. RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
