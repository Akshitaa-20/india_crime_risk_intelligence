import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "rf_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "feature_engineered_data.csv")


def load_model():
    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(DATA_PATH)


def get_historical_data():
    df = load_data()
    return df[["Year", "Crime_Count"]]


def get_feature_data():
    return load_data()


def forecast_next_n_years(n=3):
    model = load_model()
    df = load_data()

    last_row = df.iloc[-1].copy()
    current_features = last_row.drop(["Year", "Crime_Count"])
    current_year = int(last_row["Year"])

    future = []

    for _ in range(n):
        prediction = model.predict([current_features])[0]

        future_year = current_year + 1
        future.append({
            "Year": future_year,
            "Predicted_Crime": round(prediction, 0)
        })

        # Update features
        current_features["lag_2"] = current_features["lag_1"]
        current_features["lag_1"] = prediction
        current_features["rolling_3yr_avg"] = (
            current_features["rolling_3yr_avg"] * 2 + prediction
        ) / 3

        current_year = future_year

    return pd.DataFrame(future)
