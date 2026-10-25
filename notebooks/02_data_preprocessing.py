import pandas as pd
import json
import os

# ==========================================
# STEP 1: DEFINE FOLDER PATHS
# ==========================================

# Run this script from the project root: D:\EconoCausal
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
        raise ValueError(f"The file {filename} is empty.")

    # Try loading a standard JSON document first
    try:
        return json.loads(content)

    except json.JSONDecodeError:
        # If the file contains one JSON object per line,
        # load each line separately.
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
                        f"line {line_number}: {error}"
                    ) from error

        return records


profile = load_json_file("profile.json")
portfolio = load_json_file("portfolio.json")
transcript = load_json_file("transcript.json")


# ==========================================
# STEP 3: CONVERT JSON DATA INTO DATAFRAMES
# ==========================================

def convert_to_dataframe(data):
    if isinstance(data, dict):
        # Standard Starbucks profile and portfolio files
        # are dictionaries keyed by customer/offer ID.
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "original_id"
        return df

    if isinstance(data, list):
        return pd.DataFrame(data)

    raise TypeError(
        f"Unsupported JSON data format: {type(data).__name__}"
    )


profile_df = convert_to_dataframe(profile)
portfolio_df = convert_to_dataframe(portfolio)
transcript_df = convert_to_dataframe(transcript)


# ==========================================
# STEP 4: DISPLAY FIRST FIVE ROWS
# ==========================================

print("\n" + "=" * 60)
print("STEP 4: FIRST FIVE ROWS")
print("=" * 60)

print("\nPROFILE DATA")
print(profile_df.head())

print("\nPORTFOLIO DATA")
print(portfolio_df.head())

print("\nTRANSCRIPT DATA")
print(transcript_df.head())


# ==========================================
# STEP 5: CHECK DATASET DIMENSIONS
# ==========================================

print("\n" + "=" * 60)
print("STEP 5: DATASET SHAPES")
print("=" * 60)

print("Profile:", profile_df.shape)
print("Portfolio:", portfolio_df.shape)
print("Transcript:", transcript_df.shape)


# ==========================================
# STEP 6: CHECK COLUMN NAMES AND DATA TYPES
# ==========================================

print("\n" + "=" * 60)
print("STEP 6: DATASET INFORMATION")
print("=" * 60)

print("\nPROFILE INFORMATION")
profile_df.info()

print("\nPORTFOLIO INFORMATION")
portfolio_df.info()

print("\nTRANSCRIPT INFORMATION")
transcript_df.info()


# ==========================================
# STEP 7: CHECK MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("STEP 7: MISSING VALUES")
print("=" * 60)


def check_missing_values(df, name):
    print(f"\n{name} MISSING VALUES")

    missing_count = df.isnull().sum()
    missing_percentage = (
        df.isnull().sum() / len(df) * 100
        if len(df) > 0
        else pd.Series(0.0, index=df.columns)
    )

    result = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print(result)


check_missing_values(profile_df, "PROFILE")
check_missing_values(portfolio_df, "PORTFOLIO")
check_missing_values(transcript_df, "TRANSCRIPT")

# ==========================================
# STEP 8: CHECK DUPLICATE ROWS
# ==========================================

print("\n" + "=" * 60)
print("STEP 8: DUPLICATE RECORDS")
print("=" * 60)


def count_duplicates(df):
    # Convert list and dictionary values to strings
    # in a temporary copy for duplicate checking.
    temp_df = df.copy()

    for column in temp_df.columns:
        temp_df[column] = temp_df[column].apply(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, (list, dict))
            else value
        )

    return temp_df.duplicated().sum()


print("Profile duplicates:", count_duplicates(profile_df))
print("Portfolio duplicates:", count_duplicates(portfolio_df))
print("Transcript duplicates:", count_duplicates(transcript_df))

# ==========================================
# STEP 9: CHECK CUSTOMER AGE VALUES
# ==========================================

print("\n" + "=" * 60)
print("STEP 9: CUSTOMER AGE ANALYSIS")
print("=" * 60)

if "age" in profile_df.columns:

    age = pd.to_numeric(profile_df["age"], errors="coerce")

    print("\nAge summary:")
    print(age.describe())

    print("\nMissing or non-numeric age values:")
    print(age.isna().sum())

    print("\nCustomers below age 18:")
    print((age < 18).sum())

    print("\nCustomers above age 100:")
    print((age > 100).sum())

    print("\nCustomers with age 118:")
    print((age == 118).sum())

else:
    print("Age column not found.")
    print("Available columns:", profile_df.columns.tolist())


# ==========================================
# STEP 10: CHECK CATEGORICAL VALUE COUNTS
# ==========================================

print("\n" + "=" * 60)
print("STEP 10: CATEGORICAL VALUE COUNTS")
print("=" * 60)


def display_value_counts(df, column):

    if column not in df.columns:
        print(f"\nColumn '{column}' not found.")
        return

    print(f"\n{column.upper()} DISTRIBUTION:")

    # Convert lists and dictionaries into strings
    # to avoid the unhashable-type error.
    values = df[column].apply(
        lambda value: str(value)
        if isinstance(value, (list, dict))
        else value
    )

    print(values.value_counts(dropna=False))


display_value_counts(profile_df, "gender")
display_value_counts(transcript_df, "event")
display_value_counts(portfolio_df, "offer_type")


# ==========================================
# STEP 11: CHECK TRANSCRIPT EVENT STRUCTURE
# ==========================================

print("\n" + "=" * 60)
print("STEP 11: TRANSCRIPT EVENT STRUCTURE")
print("=" * 60)

if "value" in transcript_df.columns:

    print("\nFirst five event values:")

    for value in transcript_df["value"].head():
        print(value)

    print("\nValue types:")
    print(
        transcript_df["value"]
        .apply(lambda value: type(value).__name__)
        .value_counts()
    )

else:
    print("The 'value' column was not found.")


# ==========================================
# STEP 12: SAVE INITIAL DATASET COPIES
# ==========================================

print("\n" + "=" * 60)
print("STEP 12: SAVE DATASET COPIES")
print("=" * 60)

profile_df.to_csv(
    os.path.join(processed_path, "profile_inspected.csv"),
    index=True,
    index_label="original_id"
)

portfolio_df.to_csv(
    os.path.join(processed_path, "portfolio_inspected.csv"),
    index=True,
    index_label="original_id"
)

transcript_df.to_csv(
    os.path.join(processed_path, "transcript_inspected.csv"),
    index=False
)

print("Initial dataset copies saved successfully.")

print("\nFiles saved in:", processed_path)

print("\nDATA PREPROCESSING INSPECTION COMPLETED")

