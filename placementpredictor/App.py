from flask import Flask, render_template

from load_data import get_data_summary


app = Flask(__name__)


# ================= HOME PAGE =================

@app.route("/")
def index():

    error = None
    summary = None

    try:
        # Load dataset summary
        summary = get_data_summary()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ================= DATA LOADING PAGE =================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:
        # Load dataset summary
        summary = get_data_summary()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ================= RUN APPLICATION =================

if __name__ == "__main__":
    app.run(debug=True)