import pandas as pd

def load_data(filepath):
    """Loads the CSV file into a DataFrame."""
    return pd.read_csv(filepath)

def split_data(data, train_start, train_end, test_start, test_end):
    """
    Splits the data into training, testing, and evaluation sets.

    Args:
        data (pd.DataFrame): Complete dataset.
        train_start (int): Start index for training data.
        train_end (int): End index for training data.
        test_start (int): Start index for testing data.
        test_end (int): End index for testing data.

    Returns:
        tuple: (train_data, test_data, unseen_data)
    """
    train_data = data.iloc[train_start:train_end]
    test_data = data.iloc[test_start:test_end]
    unseen_data = data.iloc[train_end:]
    return train_data, test_data, unseen_data
