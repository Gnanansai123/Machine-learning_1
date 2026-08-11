import os
import pandas as pd


# ================= DATASET PATH =================

DATA_PATH = r"C:\Users\hp\PycharmProjects\PythonProject\placementpredictor\placement_predict_50k Dataset.csv"


# ================= LOAD DATA =================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    return df


# ================= DATA SUMMARY =================

def get_data_summary(path: str = DATA_PATH) -> dict:

    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict(
            orient="records"
        )
    }

    return summary


# ================= TEST =================

if __name__ == "__main__":

    data = load_data()

    print("Dataset loaded successfully!")

    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nDataset Summary:")
    print(get_data_summary())