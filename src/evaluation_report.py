import os
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


def export_metrics(model_path, dataset_path):

    model = joblib.load(model_path)

    df = pd.read_csv(dataset_path)

    X = df[
        [
            "Mean",
            "Std",
            "RMS",
            "Kurtosis",
            "Skewness",
            "Peak_to_Peak"
        ]
    ]

    y = df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2 Score"],
        "Value": [mae, rmse, r2]
    })

    os.makedirs("outputs", exist_ok=True)

    metrics.to_csv(
        "outputs/evaluation_metrics.csv",
        index=False
    )

    print(metrics)

    print("\nSaved:")
    print("outputs/evaluation_metrics.csv")