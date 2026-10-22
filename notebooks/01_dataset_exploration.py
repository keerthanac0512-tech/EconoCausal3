import pandas as pd
import json
from pathlib import Path

# Locate the raw data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    try:
        # Try loading standard JSON
        return json.loads(content)

    except json.JSONDecodeError:
        # Handle multiple JSON objects
        decoder = json.JSONDecoder()
        objects = []
        position = 0

        while position < len(content):
            # Skip whitespace between objects
            while position < len(content) and content[position].isspace():
                position += 1

            if position >= len(content):
                break

            obj, position = decoder.raw_decode(content, position)
            objects.append(obj)

        return objects


# Load the three datasets
portfolio = pd.DataFrame(
    load_json_data(DATA_DIR / "portfolio.json")
)

profile = pd.DataFrame(
    load_json_data(DATA_DIR / "profile.json")
)

transcript = pd.DataFrame(
    load_json_data(DATA_DIR / "transcript.json")
)


# Store datasets in a dictionary
datasets = {
    "Portfolio": portfolio,
    "Profile": profile,
    "Transcript": transcript
}


# Explore each dataset
for name, df in datasets.items():
    print("\n" + "=" * 50)
    print(f"{name} Dataset")
    print("=" * 50)

    print("\nNumber of rows and columns:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

print("\nDataset exploration completed!")



