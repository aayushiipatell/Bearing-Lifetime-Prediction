"""
load_model.py
-----------------
Loads the trained Random Forest model.
"""

import joblib
from pathlib import Path


MODEL_PATH = Path("models/random_forest_model.pkl")


def load_trained_model():
    """
    Load the trained Random Forest model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)