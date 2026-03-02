# test.py

import pytest
import pickle
from score import score
from app import app


# -----------------------------
# Load trained model once
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))


# ==========================================================
# UNIT TESTS FOR score()
# ==========================================================

def test_smoke():
    """Smoke test: function runs and returns correct types"""
    prediction, propensity = score("Hello world", model, 0.5)
    assert isinstance(prediction, int)
    assert isinstance(propensity, float)


def test_prediction_binary():
    """Prediction must be 0 or 1"""
    prediction, _ = score("Test message", model, 0.6)
    assert prediction in [0, 1]


def test_propensity_range():
    """Propensity must be between 0 and 1"""
    _, propensity = score("Test message", model, 0.5)
    assert 0 <= propensity <= 1


def test_threshold_zero():
    """Threshold 0 should always predict 1"""
    prediction, _ = score("Any message", model, 0)
    assert prediction == 1


def test_threshold_one():
    """Threshold 1 should always predict 0"""
    prediction, _ = score("Any message", model, 1)
    assert prediction == 0


def test_obvious_spam():
    """Typical spam message"""
    text = "Congratulations! You have won $1000. Claim now."
    prediction, _ = score(text, model, 0.5)
    assert prediction == 1


def test_obvious_ham():
    """Typical non-spam message"""
    text = "Let's meet tomorrow at 10am."
    prediction, _ = score(text, model, 0.5)
    assert prediction == 0


# ==========================================================
# EDGE CASE TESTS FOR score()
# ==========================================================

def test_invalid_text_type():
    """Non-string text should raise TypeError"""
    with pytest.raises(TypeError):
        score(123, model, 0.5)


def test_invalid_threshold_type():
    """Invalid threshold value should raise ValueError"""
    with pytest.raises(ValueError):
        score("Hello", model, 1.5)


# ==========================================================
# INTEGRATION TESTS FOR FLASK API
# ==========================================================

def test_flask_success():
    """Flask endpoint returns valid JSON for correct input"""
    with app.test_client() as client:
        response = client.post("/score", data={"text": "Win money now!"})
        assert response.status_code == 200

        data = response.get_json()
        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["propensity"] <= 1


def test_flask_missing_text():
    """Flask endpoint should return 400 if text is missing"""
    with app.test_client() as client:
        response = client.post("/score", data={})
        assert response.status_code == 400