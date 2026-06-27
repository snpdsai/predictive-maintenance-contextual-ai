import pandas as pd


def load_raw_data(filepath):
    """
    Load AI4I Predictive Maintenance dataset.
    """
    return pd.read_csv(filepath)


def sort_by_time(df):
    """
    Sort records by UDI to preserve temporal order.
    """
    df = df.sort_values("UDI")
    df = df.reset_index(drop=True)

    return df


def save_dataset(df, filepath):
    """
    Save dataframe to csv.
    """
    df.to_csv(filepath, index=False)

    print(f"Saved successfully -> {filepath}")