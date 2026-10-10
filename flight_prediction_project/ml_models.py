from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = 'weather_delay'
FEATURES = ['year', 'month', 'day', 'day_of_week', 'airline', 'flight_number', 'tail_number', 'origin_airport', 'destination_airport', 'scheduled_departure', 'departure_time', 'departure_delay', 'taxi_out', 'wheels_off', 'scheduled_time', 'elapsed_time', 'air_time', 'distance', 'wheels_on', 'taxi_in', 'scheduled_arrival', 'arrival_time', 'arrival_delay', 'diverted', 'cancelled', 'cancellation_reason', 'air_system_delay', 'security_delay', 'airline_delay', 'late_aircraft_delay']


def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])


def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    return data


def train_linear_model(data_path, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    data = load_data(data_path)
    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_t, y_train)
    pred = model.predict(X_test_t)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    metrics = {
        "model": "Linear Regression",
        "target": TARGET,
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(rmse, 4),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    joblib.dump(model, model_dir / "linear_model.joblib")
    joblib.dump(preprocessor, model_dir / "linear_preprocessor.joblib")
    (model_dir / "linear_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_logistic_model(data_path, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    data = load_data(data_path)
    X = data[FEATURES].copy()

    # Binary classification: above median target price/fare = 1.
    threshold = float(data[TARGET].median())
    y = (data[TARGET] > threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_t, y_train)
    pred = model.predict(X_test_t)

    metrics = {
        "model": "Logistic Regression",
        "target": TARGET,
        "threshold": round(threshold, 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    joblib.dump(model, model_dir / "logistic_model.joblib")
    joblib.dump(preprocessor, model_dir / "logistic_preprocessor.joblib")
    (model_dir / "logistic_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model_info(model_dir):
    model_dir = Path(model_dir)
    info = {}
    for name in ["linear_metrics.json", "logistic_metrics.json"]:
        path = model_dir / name
        if path.exists():
            info[name.replace("_metrics.json", "")] = json.loads(path.read_text())
    return info


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    print(train_linear_model(base / "data" / "flight_data.csv", base / "models"))
    print(train_logistic_model(base / "data" / "flight_data.csv", base / "models"))
