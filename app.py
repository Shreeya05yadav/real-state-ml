import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model + preprocessor
# -----------------------------
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

# -----------------------------
# OPTIONS API (NEW)
# -----------------------------
@app.route("/options")
def get_options():
    df = pd.read_csv("notebook/cleaned_data.csv")

    return jsonify({
        "TYPE": sorted(df["TYPE"].dropna().unique().tolist()),
        "STATE": sorted(df["STATE"].dropna().unique().tolist()),
        "SUBLOCALITY": sorted(df["SUBLOCALITY"].dropna().unique().tolist())
    })


# -----------------------------
# FEATURE ENGINEERING (IMPORTANT)
# -----------------------------
def prepare_input(data):
    df = pd.DataFrame([data])

    # Derived sqft (same logic as before)
    df["PROPERTYSQFT"] = (df["BEDS"] * 500) + (df["BATH"] * 200)

    # Features (must match training)
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

    df["LOCATION"] = df["STATE"].astype(str) + "_" + df["SUBLOCALITY"].astype(str)

    # Drop raw columns (same as training)
    df.drop(columns=["STATE", "SUBLOCALITY"], inplace=True, errors="ignore")

    return df


# -----------------------------
# PREDICTION API
# -----------------------------
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json

    input_df = prepare_input(data)

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]

    return jsonify({
        "price": round(prediction, 2),
        "range": [
            round(prediction * 0.9, 2),
            round(prediction * 1.1, 2)
        ]
    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)