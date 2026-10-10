# Flight Prediction ML Project

PyCharm-ready Flask + Machine Learning project built from the uploaded flight dataset.

## Dataset
- Original workbook: `81b9b350-a469-496e-80bb-f7a7f00e9198.xlsx`
- Sheet used: `Sheet1`
- Rows: 22275
- Columns: 31
- Target selected for prediction: `weather_delay`

## Main features
- Dashboard
- Data Loading
- Preprocessing
- Linear Regression
- Logistic Regression
- Prediction page
- Dataset preview
- Model metrics

## Run in PyCharm

1. Open this folder in PyCharm.
2. Create/select a Python virtual environment.
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Run `app.py`.
5. Open:

`http://127.0.0.1:5000`

The application automatically prepares the models when required.

## Important
Linear Regression predicts the selected numeric target (`weather_delay`).

Logistic Regression converts the target into a binary class:
- 0 = below/equal median
- 1 = above median

This is for educational/project demonstration purposes.
