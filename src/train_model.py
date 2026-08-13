import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_random_forest(dataset_path):
    """
    Train Random Forest model for Remaining Useful Life prediction.
    """

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Features
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

    # Target
    y = df["RUL"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions,
        squared=False
    )
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("-" * 40)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    os.makedirs("models", exist_ok=True)

    model_path = "models/random_forest_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel Saved : {model_path}")

    return model