import pandas as pd

def add_rul_labels(csv_file, dataset_name):
    """
    Add Remaining Useful Life (RUL) labels and dataset name.
    """

    df = pd.read_csv(csv_file)

    total = len(df)

    # Highest RUL at the beginning, 0 at the end
    df["RUL"] = list(range(total - 1, -1, -1))

    # Dataset identifier
    df["Dataset"] = dataset_name

    return df