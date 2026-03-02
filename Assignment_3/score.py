# score.py

from typing import Tuple
import numpy as np

def score(text: str, model, threshold: float = 0.5) -> Tuple[int, float]:
    """
    Scores a trained model on a given text.

    Returns:
        prediction (int): 0 or 1
        propensity (float): probability of class 1
    """

    # -------- Input Validation --------
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    # -------- Model Scoring --------
    proba = model.predict_proba([text])[0][1]
    proba = float(proba)

    prediction = 1 if proba >= threshold else 0

    return prediction, proba