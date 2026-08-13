import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_heatmap(dataset_path):
    """
    Generate Correlation Heatmap
    """

    df = pd.read_csv(dataset_path)

    columns = [
        "Mean",
        "Std",
        "RMS",
        "Kurtosis",
        "Skewness",
        "Peak_to_Peak",
        "RUL"
    ]

    corr = df[columns].corr()

    print("\nCorrelation Matrix\n")
    print(corr)

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        corr,
        cmap="coolwarm",
        interpolation="nearest"
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "outputs/correlation_heatmap.png",
        dpi=300
    )

    plt.show()

    corr.to_csv(
        "outputs/correlation_matrix.csv"
    )

    print("\nSaved:")
    print("outputs/correlation_heatmap.png")
    print("outputs/correlation_matrix.csv")