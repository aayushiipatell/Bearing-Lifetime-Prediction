"""
predictor.py
-----------------
Prediction utilities for the Bearing Lifetime Prediction project.
"""

import pandas as pd


def predict_rul(model,
                mean,
                std,
                rms,
                kurtosis,
                skewness,
                peak_to_peak):
    """
    Predict Remaining Useful Life (RUL).
    """

    sample = pd.DataFrame({
        "Mean": [mean],
        "Std": [std],
        "RMS": [rms],
        "Kurtosis": [kurtosis],
        "Skewness": [skewness],
        "Peak_to_Peak": [peak_to_peak]
    })

    prediction = model.predict(sample)[0]

    return float(prediction)