import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split


def visualize_predictions(model_path, dataset_path):
    """
    Compare Actual vs Predicted RUL
    """

    # Load model
    model = joblib.load(model_path)

    # Load dataset
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

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    print("\nFirst 20 Predictions\n")
    print(comparison.head(20))

    os.makedirs("outputs", exist_ok=True)

    comparison.to_csv(
        "outputs/prediction_results.csv",
        index=False
    )

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        'r--',
        linewidth=2
    )

    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs Predicted RUL")

    plt.tight_layout()

    plt.savefig(
        "outputs/actual_vs_predicted.png",
        dpi=300
    )

    plt.show()

    print("\nSaved:")
    print("outputs/prediction_results.csv")
    print("outputs/actual_vs_predicted.png")