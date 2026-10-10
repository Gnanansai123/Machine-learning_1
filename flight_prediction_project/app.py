from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from ml_models import train_linear_model, train_logistic_model, load_model_info

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "flight_data.csv"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

df = pd.read_csv(DATA_PATH)
TARGET = 'weather_delay'
FEATURES = ['year', 'month', 'day', 'day_of_week', 'airline', 'flight_number', 'tail_number', 'origin_airport', 'destination_airport', 'scheduled_departure', 'departure_time', 'departure_delay', 'taxi_out', 'wheels_off', 'scheduled_time', 'elapsed_time', 'air_time', 'distance', 'wheels_on', 'taxi_in', 'scheduled_arrival', 'arrival_time', 'arrival_delay', 'diverted', 'cancelled', 'cancellation_reason', 'air_system_delay', 'security_delay', 'airline_delay', 'late_aircraft_delay']
FORM_FEATURES = ['airline', 'year', 'month', 'day', 'day_of_week', 'flight_number', 'tail_number', 'origin_airport']


@app.route("/")
def dashboard():
    info = load_model_info(MODEL_DIR)
    return render_template(
        "dashboard.html",
        rows=len(df),
        columns=len(df.columns),
        target=TARGET,
        features=FEATURES,
        model_info=info
    )


@app.route("/data")
def data_loading():
    preview = df.head(15).to_html(classes="data-table", index=False)
    return render_template(
        "data_loading.html",
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        preview=preview
    )


@app.route("/preprocessing")
def preprocessing():
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_data = [(idx, int(val)) for idx, val in missing.items()]
    return render_template(
        "preprocessing.html",
        dtypes=[(c, str(df[c].dtype)) for c in df.columns],
        missing=missing_data,
        numeric=list(df.select_dtypes(include="number").columns),
        categorical=list(df.select_dtypes(exclude="number").columns)
    )


@app.route("/linear", methods=["GET", "POST"])
def linear():
    result = None
    if request.method == "POST":
        result = train_linear_model(DATA_PATH, MODEL_DIR)
    return render_template("linear.html", result=result, target=TARGET)


@app.route("/logistic", methods=["GET", "POST"])
def logistic():
    result = None
    if request.method == "POST":
        result = train_logistic_model(DATA_PATH, MODEL_DIR)
    return render_template("logistic.html", result=result, target=TARGET)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    message = None
    prediction = None

    if request.method == "POST":
        model_path = MODEL_DIR / "linear_model.joblib"
        preprocessor_path = MODEL_DIR / "linear_preprocessor.joblib"

        if not model_path.exists() or not preprocessor_path.exists():
            train_linear_model(DATA_PATH, MODEL_DIR)

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        row = {}
        for feature in FEATURES:
            raw = request.form.get(feature, "")
            # Use numeric conversion where the original column is numeric.
            if pd.api.types.is_numeric_dtype(df[feature]):
                try:
                    row[feature] = float(raw)
                except (TypeError, ValueError):
                    row[feature] = float(df[feature].median())
            else:
                row[feature] = raw if raw else str(df[feature].mode().iloc[0])

        X = pd.DataFrame([row], columns=FEATURES)
        prediction = float(model.predict(preprocessor.transform(X))[0])
        message = "Prediction generated successfully."

    choices = {}
    for f in FORM_FEATURES:
        if f in df.columns and not pd.api.types.is_numeric_dtype(df[f]):
            choices[f] = sorted(df[f].dropna().astype(str).unique().tolist())[:100]

    return render_template(
        "predict.html",
        features=FEATURES,
        form_features=FORM_FEATURES,
        df=df,
        choices=choices,
        prediction=prediction,
        message=message,
        target=TARGET
    )


if __name__ == "__main__":
    app.run(debug=True)
