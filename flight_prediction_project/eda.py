"""
EDA (Exploratory Data Analysis) module for the Flight ML project.

This module reads data/flight_data.csv and prepares:
- Dataset shape and column information
- Missing-value summary
- Descriptive statistics
- Numeric correlations
- Airline-wise analysis when airline exists
- Monthly analysis when month exists
- Delay analysis
- Plot images saved under static/eda/
"""

from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "flight_data.csv"
EDA_DIR = BASE_DIR / "static" / "eda"


def _save_plot(fig, filename):
    """Save a matplotlib figure and return a web-friendly path."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f"/static/eda/{filename}"


def _safe_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def _find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def run_eda():
    """Run the complete EDA and return data needed by the Flask template."""
    df = pd.read_csv(DATA_PATH)

    numeric_cols = _safe_numeric_columns(df)
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Basic information
    shape = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_rows = [
        {
            "column": str(col),
            "count": int(count),
            "percentage": round(float(count / len(df) * 100), 2)
        }
        for col, count in missing.items()
    ]

    # Descriptive statistics
    if numeric_cols:
        stats = (
            df[numeric_cols]
            .describe()
            .round(2)
            .transpose()
            .reset_index()
            .rename(columns={"index": "column"})
        )
        stats_html = stats.to_html(
            classes="eda-table",
            index=False,
            border=0
        )
    else:
        stats_html = "<p>No numeric columns available.</p>"

    # Duplicate count
    duplicate_count = int(df.duplicated().sum())

    # Correlation
    correlation_html = "<p>Not enough numeric columns for correlation analysis.</p>"
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        correlation_html = corr.to_html(
            classes="eda-table",
            border=0
        )

    charts = []

    # 1. Missing values chart
    if missing_rows:
        top_missing = missing.head(10).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_missing.index.astype(str), top_missing.values)
        ax.set_title("Top Missing-Value Columns")
        ax.set_xlabel("Number of Missing Values")
        charts.append(("Missing Values", _save_plot(fig, "missing_values.png")))

    # 2. Target/delay distribution
    delay_col = _find_column(
        df,
        ["weather_delay", "departure_delay", "arrival_delay", "delay"]
    )
    if delay_col:
        values = pd.to_numeric(df[delay_col], errors="coerce").dropna()
        if not values.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(values, bins=30)
            ax.set_title(f"Distribution of {delay_col}")
            ax.set_xlabel(delay_col)
            ax.set_ylabel("Frequency")
            charts.append(
                (f"{delay_col} Distribution",
                 _save_plot(fig, f"{delay_col}_distribution.png"))
            )

    # 3. Airline count
    airline_col = _find_column(df, ["airline"])
    if airline_col:
        counts = df[airline_col].astype(str).value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.index, counts.values)
        ax.set_title("Flights by Airline")
        ax.set_xlabel("Airline")
        ax.set_ylabel("Number of Flights")
        ax.tick_params(axis="x", rotation=45)
        charts.append(("Flights by Airline", _save_plot(fig, "airline_counts.png")))

    # 4. Average delay by airline
    if airline_col and delay_col:
        temp = df[[airline_col, delay_col]].copy()
        temp[delay_col] = pd.to_numeric(temp[delay_col], errors="coerce")
        avg = (
            temp.dropna()
            .groupby(airline_col)[delay_col]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        if not avg.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(avg.index.astype(str), avg.values)
            ax.set_title(f"Average {delay_col} by Airline")
            ax.set_xlabel("Airline")
            ax.set_ylabel(f"Average {delay_col}")
            ax.tick_params(axis="x", rotation=45)
            charts.append(
                ("Average Delay by Airline",
                 _save_plot(fig, "average_delay_airline.png"))
            )

    # 5. Flights by month
    month_col = _find_column(df, ["month"])
    if month_col:
        month = pd.to_numeric(df[month_col], errors="coerce").dropna()
        counts = month.astype(int).value_counts().sort_index()
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title("Flights by Month")
            ax.set_xlabel("Month")
            ax.set_ylabel("Number of Flights")
            charts.append(("Flights by Month", _save_plot(fig, "monthly_flights.png")))

    # 6. Numeric distributions (first 4 useful columns)
    for col in numeric_cols[:4]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(values, bins=30)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        filename = f"distribution_{col}.png"
        charts.append((f"Distribution: {col}", _save_plot(fig, filename)))

    # 7. Correlation heatmap without seaborn
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr.values, aspect="auto")
        ax.set_title("Numeric Feature Correlation Heatmap")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        charts.append(("Correlation Heatmap", _save_plot(fig, "correlation_heatmap.png")))

    # 8. Boxplots for first 4 numeric columns
    box_cols = numeric_cols[:4]
    if box_cols:
        values = [
            pd.to_numeric(df[c], errors="coerce").dropna().values
            for c in box_cols
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(values, labels=box_cols)
        ax.set_title("Outlier Analysis using Boxplots")
        ax.tick_params(axis="x", rotation=30)
        charts.append(("Outlier Boxplots", _save_plot(fig, "outlier_boxplots.png")))

    # High-level insights
    insights = []
    if duplicate_count:
        insights.append(f"{duplicate_count:,} duplicate rows were found.")
    else:
        insights.append("No duplicate rows were found.")

    if missing_rows:
        biggest = missing_rows[0]
        insights.append(
            f"The column '{biggest['column']}' has the most missing values "
            f"({biggest['percentage']}%)."
        )
    else:
        insights.append("No missing values were found.")

    if delay_col:
        delay_values = pd.to_numeric(df[delay_col], errors="coerce").dropna()
        if not delay_values.empty:
            insights.append(
                f"The average {delay_col} is {delay_values.mean():.2f}."
            )

    if airline_col:
        top_airline = df[airline_col].astype(str).value_counts().index[0]
        insights.append(f"The airline with the most records is {top_airline}.")

    return {
        "shape": shape,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_rows": missing_rows,
        "duplicate_count": duplicate_count,
        "stats_html": stats_html,
        "correlation_html": correlation_html,
        "charts": charts,
        "insights": insights,
    }


if __name__ == "__main__":
    result = run_eda()
    print("EDA completed.")
    print("Rows:", result["shape"]["rows"])
    print("Columns:", result["shape"]["columns"])
    print("Charts generated:", len(result["charts"]))
