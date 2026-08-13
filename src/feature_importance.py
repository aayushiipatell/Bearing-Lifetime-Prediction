import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model_path, dataset_path):
    """
    Plot Feature Importance of Random Forest Model
    """

    # Load trained model
    model = joblib.load(model_path)

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Feature names
    features = [
        "Mean",
        "Std",
        "RMS",
        "Kurtosis",
        "Skewness",
        "Peak_to_Peak"
    ]

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance\n")
    print(feature_df)

    os.makedirs("outputs", exist_ok=True)

    # Save CSV
    feature_df.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(8,5))

    plt.bar(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.title("Feature Importance")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png",
        dpi=300
    )

    plt.show()

    print("\nSaved:")
    print("outputs/feature_importance.csv")
    print("outputs/feature_importance.png")