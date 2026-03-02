# app.py

from flask import Flask, request, jsonify
import pickle
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.form

    if "text" not in data:
        return jsonify({"error": "Missing text parameter"}), 400

    text = data["text"]

    prediction, propensity = score(text, model, threshold=0.5)

    response = {
        "prediction": int(prediction),
        "propensity": float(propensity)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False)