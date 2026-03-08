# app.py

from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import warnings
from score import score

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------------------------------------------------
# Load Model Safely
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# HTML Template (Clean & Minimal)
# --------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Spam Classifier</title>
</head>
<body style="font-family: Arial; text-align: center; margin-top: 50px;">
    <h2>Spam Classifier</h2>
    <form method="post">
        <input type="text" name="text" placeholder="Enter message" required style="width:300px;">
        <br><br>
        <input type="submit" value="Check">
    </form>

    {% if result %}
        <h3>Prediction: {{ result.prediction }}</h3>
        <p>Propensity: {{ result.propensity }}</p>
    {% endif %}
</body>
</html>
"""


# --------------------------------------------------
# Homepage (UI)
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            return render_template_string(HTML_PAGE, result=None)

        prediction, propensity = score(text, model, threshold=0.5)

        result = {
            "prediction": "SPAM" if prediction else "HAM",
            "propensity": round(float(propensity), 4)
        }

        return render_template_string(HTML_PAGE, result=result)

    return render_template_string(HTML_PAGE, result=None)


# --------------------------------------------------
# API Endpoint (Assignment Requirement)
# --------------------------------------------------

@app.route("/score", methods=["POST"])
def score_endpoint():
    if request.is_json:
        data = request.get_json()
        text = data.get("text") if data else None
    else:
        text = request.form.get("text")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Valid 'text' field is required"}), 400

    prediction, propensity = score(text, model, threshold=0.5)

    return jsonify({
        "prediction": int(prediction),
        "propensity": float(propensity)
    }), 200


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)