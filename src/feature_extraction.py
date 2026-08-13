import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm


def extract_dataset_features(dataset_folder):
    """
    Extract statistical features from every vibration file
    in a dataset folder.
    """

    files = sorted(os.listdir(dataset_folder))

    dataset = []

    for file in tqdm(files, desc="Processing Files"):

        file_path = os.path.join(dataset_folder, file)

        if not os.path.isfile(file_path):
            continue

        signal = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None
        )

        # Bearing-1 Horizontal (Column 0)
        x = signal.iloc[:, 0].values

        features = {
            "File": file,
            "Mean": np.mean(x),
            "Std": np.std(x),
            "RMS": np.sqrt(np.mean(x ** 2)),
            "Kurtosis": kurtosis(x),
            "Skewness": skew(x),
            "Peak_to_Peak": np.ptp(x)
        }

        dataset.append(features)

    return pd.DataFrame(dataset)