from flask import Flask, render_template, request, jsonify
from ml_utils import (
    load_data,
    preprocess_data,
    train_linear_regression,
    train_logistic_regression,
    get_dashboard_stats,
)

app = Flask(__name__)

DATA_PATH = "data/flights.csv"


@app.route("/")
def dashboard():
    df = load_data(DATA_PATH)
    stats = get_dashboard_stats(df)
    return render_template("dashboard.html", stats=stats)


@app.route("/data-loading")
def data_loading():
    df = load_data(DATA_PATH)
    preview = df.head(15).to_dict(orient="records")
    columns = list(df.columns)
    return render_template(
        "data_loading.html",
        preview=preview,
        columns=columns,
        rows=len(df),
        cols=len(df.columns),
    )


@app.route("/preprocessing")
def preprocessing():
    df = load_data(DATA_PATH)
    info = preprocess_data(df)
    return render_template("preprocessing.html", info=info)


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    result = None

    if request.method == "POST":
        try:
            values = {
                "departure_delay": float(request.form["departure_delay"]),
                "distance": float(request.form["distance"]),
                "air_time": float(request.form["air_time"]),
                "temperature": float(request.form["temperature"]),
                "wind_speed": float(request.form["wind_speed"]),
            }

            result = train_linear_regression(DATA_PATH, values)
        except (ValueError, KeyError) as exc:
            result = {"error": f"Please enter valid values. {exc}"}

    return render_template("linear_regression.html", result=result)


@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():
    result = None

    if request.method == "POST":
        try:
            values = {
                "airline": request.form["airline"],
                "origin": request.form["origin"],
                "destination": request.form["destination"],
                "departure_delay": float(request.form["departure_delay"]),
                "distance": float(request.form["distance"]),
                "air_time": float(request.form["air_time"]),
                "temperature": float(request.form["temperature"]),
                "wind_speed": float(request.form["wind_speed"]),
            }

            result = train_logistic_regression(DATA_PATH, values)
        except (ValueError, KeyError) as exc:
            result = {"error": f"Please enter valid values. {exc}"}

    return render_template("logistic_regression.html", result=result)


@app.route("/api/dashboard")
def dashboard_api():
    df = load_data(DATA_PATH)
    return jsonify(get_dashboard_stats(df))


if __name__ == "__main__":
    app.run(debug=True)
