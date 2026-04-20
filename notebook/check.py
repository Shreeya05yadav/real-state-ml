import pandas as pd

df = pd.read_csv("notebook/data.csv")

categorical_cols = [
    "TYPE",
    "STATE",
    "ADMINISTRATIVE_AREA_LEVEL_2",
    "LOCALITY",
    "SUBLOCALITY",
    "STREET_NAME"
]

for col in categorical_cols:
    print(f"\nColumn: {col}")
    print("Unique count:", df[col].nunique())
    print("Unique values:", df[col].unique())