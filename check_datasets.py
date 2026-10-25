
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

print("=" * 60)
print("MUSHROOM DATASET VERIFICATION")
print("=" * 60)

csv_file = BASE_DIR / "data" / "mushrooms.csv"
df = pd.read_csv(csv_file)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head().to_string(index=False))

print("\nData Types:")
print(df.dtypes.to_string())

print("\nMissing Values:")
print(df.isnull().sum().to_string())

print("\nPotential Target Column:")
for col in df.columns:
    if col.lower() in ["class", "target", "label"]:
        print(col, df[col].value_counts().to_dict())

print("\n" + "=" * 60)
print("JSON DATASET VERIFICATION")
print("=" * 60)

json_file = BASE_DIR / "data" / "exercise_text.json"

with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

print("\nJSON Root Type:", type(json_data).__name__)

if isinstance(json_data, dict):
    print("JSON Keys:", list(json_data.keys()))
    print("Preview:", str(json_data)[:800])
elif isinstance(json_data, list):
    print("Total Items:", len(json_data))
    print("First Item:", str(json_data[0])[:800] if json_data else "Empty")

print("\nDataset verification completed!")
