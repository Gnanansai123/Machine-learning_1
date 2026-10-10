"""
Linear Regression module for the Flight ML project.

Target:
    weather_delay

The module:
1. Loads the flight CSV.
2. Removes columns that are completely empty.
3. Separates numeric and categorical features.
4. Handles missing values.
5. One-hot encodes categorical columns.
6. Scales numeric columns.
7. Trains Linear Regression.
8. Calculates R2, MAE and RMSE.
9. Saves the trained model and preprocessing pipeline.
10. Creates an Actual vs Predicted graph.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "flight_data.csv"
MODEL_DIR = BASE_DIR / "models"
GRAPH_DIR = BASE_DIR / "static" / "linear_regression"

TARGET = "weather_delay"


def train_linear_regression():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Remove columns that contain no useful values at all.
    df = df.dropna(axis=1, how="all")

    # Make sure target is numeric.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Remove very high-cardinality ID-like text columns.
    # This keeps the model practical for a beginner project.
    drop_columns = []
    for col in X.select_dtypes(exclude=np.number).columns:
        unique_ratio = X[col].nunique(dropna=True) / max(len(X), 1)
        if unique_ratio > 0.50:
            drop_columns.append(col)

    X = X.drop(columns=drop_columns)

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Save model and preprocessing pipeline.
    joblib.dump(model, MODEL_DIR / "linear_regression_model.joblib")
    joblib.dump(
        preprocessor,
        MODEL_DIR / "linear_regression_preprocessor.joblib"
    )

    metrics = {
        "model": "Linear Regression",
        "target": TARGET,
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features_used": int(X.shape[1]),
        "numeric_features": int(len(numeric_columns)),
        "categorical_features": int(len(categorical_columns)),
        "removed_high_cardinality_columns": drop_columns,
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4)
    }

    (MODEL_DIR / "linear_regression_metrics.json").write_text(
        json.dumps(metrics, indent=4)
    )

    # Actual vs predicted graph.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_test, predictions, alpha=0.35)

    minimum = min(float(y_test.min()), float(predictions.min()))
    maximum = max(float(y_test.max()), float(predictions.max()))
    ax.plot([minimum, maximum], [minimum, maximum])

    ax.set_title("Linear Regression: Actual vs Predicted")
    ax.set_xlabel("Actual Weather Delay")
    ax.set_ylabel("Predicted Weather Delay")

    fig.tight_layout()
    graph_path = GRAPH_DIR / "actual_vs_predicted.png"
    fig.savefig(graph_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    result = train_linear_regression()
    print("\nLinear Regression completed successfully.")
    for key, value in result.items():
        print(f"{key}: {value}")
