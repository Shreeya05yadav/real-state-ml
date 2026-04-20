import pandas as pd
import pickle
from sklearn.metrics import r2_score

# Load model + preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

# Load test data
df = pd.read_csv("artifacts/test.csv")

# -----------------------------
# Feature Engineering (MANDATORY)
# -----------------------------
def add_features(df):
    df = df.copy()

    df["AREA_PER_ROOM"] = df["PROPERTYSQFT"] / (df["BEDS"] + df["BATH"] + 1)
    df["BATH_PER_BED"] = df["BATH"] / (df["BEDS"] + 1)
    df["TOTAL_ROOMS"] = df["BEDS"] + df["BATH"]
    df["SQFT_PER_BED"] = df["PROPERTYSQFT"] / (df["BEDS"] + 1)
    df["BATH_ROOM_DIFF"] = df["BATH"] - df["BEDS"]

    df["SIZE_CATEGORY"] = pd.cut(
        df["PROPERTYSQFT"],
        bins=[0, 500, 1000, 2000, 5000, 10000],
        labels=["VerySmall", "Small", "Medium", "Large", "Luxury"]
    )

    # 🔥 IMPORTANT: recreate LOCATION
    df["LOCATION"] = df["STATE"].astype(str) + "_" + df["SUBLOCALITY"].astype(str)

    return df


# Apply features
df = add_features(df)

# Split
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# Drop raw columns (same as training)
X.drop(columns=["STATE", "SUBLOCALITY"], inplace=True, errors="ignore")

# Transform
X_scaled = preprocessor.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Score
score = r2_score(y, y_pred)

print("Model R2 Score:", score)