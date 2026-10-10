import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "departure_delay",
    "distance",
    "air_time",
    "temperature",
    "wind_speed",
]

CATEGORICAL_FEATURES = [
    "airline",
    "origin",
    "destination",
]


def load_data(path):
    """Load the FlightFore CSV dataset."""
    return pd.read_csv(path)


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


def preprocess_data(df):
    """Return useful preprocessing information for the UI."""
    missing = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())

    return {
        "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": duplicate_count,
        "numeric_columns": NUMERIC_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "target_for_linear": "arrival_delay",
        "target_for_logistic": "status",
        "status_counts": df["status"].value_counts().to_dict(),
        "missing_by_column": missing[missing > 0].to_dict(),
    }


def train_linear_regression(path, values):
    """Predict arrival delay in minutes."""
    df = load_data(path)

    features = NUMERIC_FEATURES
    X = df[features]
    y = df["arrival_delay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    input_df = pd.DataFrame([values])
    prediction = float(model.predict(input_df)[0])

    return {
        "prediction": round(max(prediction, 0), 2),
        "mae": round(mean_absolute_error(y_test, predictions), 2),
        "rmse": round(mean_squared_error(y_test, predictions) ** 0.5, 2),
        "r2": round(r2_score(y_test, predictions), 3),
        "error": None,
    }


def train_logistic_regression(path, values):
    """Classify a flight as On-Time, Delayed, or Cancelled."""
    df = load_data(path)

    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[features]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = make_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    input_df = pd.DataFrame([values])
    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.named_steps["model"].classes_

    probability_map = {
        str(label): round(float(prob) * 100, 2)
        for label, prob in zip(classes, probabilities)
    }

    return {
        "prediction": str(prediction),
        "accuracy": round(float(accuracy) * 100, 2),
        "probabilities": probability_map,
        "error": None,
    }


def get_dashboard_stats(df):
    total = len(df)
    delayed = int((df["status"] == "Delayed").sum())
    cancelled = int((df["status"] == "Cancelled").sum())
    on_time = int((df["status"] == "On-Time").sum())

    return {
        "total_flights": total,
        "delayed": delayed,
        "cancelled": cancelled,
        "on_time": on_time,
        "average_arrival_delay": round(float(df["arrival_delay"].mean()), 2),
        "delay_rate": round((delayed / total) * 100, 2),
        "cancellation_rate": round((cancelled / total) * 100, 2),
        "airlines": sorted(df["airline"].unique().tolist()),
        "origins": sorted(df["origin"].unique().tolist()),
        "destinations": sorted(df["destination"].unique().tolist()),
    }
