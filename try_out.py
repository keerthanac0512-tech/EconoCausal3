import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "causal_dataset_synthetic_v2.csv")

print("Dataset columns:")
print(df.columns.tolist())