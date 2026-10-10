# FlightFore: AI-Based Flight Delay and Cancellation Prediction System

A beginner-friendly Flask + Machine Learning project for predicting:

1. **Arrival delay minutes** using Linear Regression.
2. **Flight status** — On-Time, Delayed, or Cancelled — using Logistic Regression.

## Project Structure

```text
FlightFore_AI_Project/
│
├── app.py
├── ml_utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── flights.csv
│
├── models/
│   └── README.txt
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── data_loading.html
    ├── preprocessing.html
    ├── linear_regression.html
    └── logistic_regression.html
```

## How to run in PyCharm

### 1. Open the project
Open the `FlightFore_AI_Project` folder in PyCharm.

### 2. Create a virtual environment
PyCharm normally offers to create one automatically.

### 3. Install packages

Open the PyCharm Terminal:

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the browser

```text
http://127.0.0.1:5000
```

## ML flow

```text
CSV Dataset
     ↓
Data Loading
     ↓
Preprocessing
     ↓
 ┌───────────────┬─────────────────────┐
 ↓               ↓
Linear           Logistic Regression
Regression       Classification
 ↓               ↓
Arrival Delay    On-Time / Delayed /
in minutes       Cancelled
```

## Important

The included CSV is a **demo dataset generated for this project**, so the application runs immediately.

For an academic project, you can later replace `data/flights.csv` with a larger real flight dataset, while keeping the same column names or updating the preprocessing code.
