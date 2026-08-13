import os
import pandas as pd
import matplotlib.pyplot as plt


def compare_models():

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE"],
        "Initial": [270.16, 377.69],
        "Improved": [224.33, 349.57]
    })

    print("\nModel Performance Comparison\n")
    print(comparison)

    os.makedirs("outputs", exist_ok=True)

    comparison.to_csv(
        "outputs/model_comparison.csv",
        index=False
    )

    x = range(len(comparison))

    width = 0.35

    plt.figure(figsize=(7,5))

    plt.bar(
        [i-width/2 for i in x],
        comparison["Initial"],
        width=width,
        label="Initial Model"
    )

    plt.bar(
        [i+width/2 for i in x],
        comparison["Improved"],
        width=width,
        label="Improved Model"
    )

    plt.xticks(x, comparison["Metric"])

    plt.ylabel("Error")

    plt.title("Initial vs Improved Model Performance")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/model_comparison.png",
        dpi=300
    )

    plt.show()

    print("\nSaved:")
    print("outputs/model_comparison.csv")
    print("outputs/model_comparison.png")