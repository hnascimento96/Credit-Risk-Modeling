
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import SPLIT_TRAIN, SPLIT_TEST, DATA_TRAIN, DATA_TEST

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_imputation_values(df):
    """
    Calculates and saves imputation values for the 'Income' and 'No.Dep'
    columns to a CSV file.

    The mean is used for 'Income', while the mode is used for 'No.Dep'.

    Parameters
    ----------
    df : pandas.DataFrame
        Training DataFrame used to calculate the imputation values.

    Returns
    -------
    None
    """
    imputation = {
        "Income": df["Income"].mean(),
        "No.Dep": df["No.Dep"].mode()[0],
    }

    pd.Series(imputation).to_csv(
        MODELS_DIR / "imputation_values.csv"
    )


def impute_missing_values(df):
    """
    Imputes missing values in the 'Income' and 'No.Dep' columns using
    previously calculated values stored in a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the 'Income' and 'No.Dep' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values replaced by their respective
        imputation values.
    """
    imputation = pd.read_csv(
        MODELS_DIR / "imputation_values.csv",
        index_col=0
    ).iloc[:, 0]

    df = df.copy()
    df["Income"] = df["Income"].fillna(imputation["Income"])
    df["No.Dep"] = df["No.Dep"].fillna(imputation["No.Dep"])

    return df


def set_features(df):
    """
    Creates additional features from existing columns.

    Features created
    ----------------
    AnyLate : int
        Indicates whether the customer has any late payments.
    TotalLate : numeric
        Total number of late payments across the three delinquency periods.
    LateScore : numeric
        Weighted score of late payments, assigning higher weights to
        longer delinquency periods.
    UtilAbove100 : int
        Indicates whether credit utilization exceeds 100.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the required input columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the additional features.
    """
    df = df.copy()

    df["AnyLate"] = (
        (df["Late30_59"] > 0)
        | (df["Late60_89"] > 0)
        | (df["Late90"] > 0)
    ).astype(int)

    df["TotalLate"] = (
        df["Late30_59"]
        + df["Late60_89"]
        + df["Late90"]
    )

    df["LateScore"] = (
        df["Late30_59"]
        + 2 * df["Late60_89"]
        + 3 * df["Late90"]
    )

    df["UtilAbove100"] = np.where(df["Util"] > 1, 1, 0)

    return df


if __name__ == "__main__":

    # Load the train and test sets
    train_data = pd.read_csv(SPLIT_TRAIN, index_col=0)
    test_data = pd.read_csv(SPLIT_TEST, index_col=0)

    # Impute missing values using statistics from the training set
    save_imputation_values(train_data)

    train_data = impute_missing_values(train_data)
    test_data = impute_missing_values(test_data)

    assert train_data[train_data.columns].isna().sum().sum() == 0
    assert test_data[test_data.columns].isna().sum().sum() == 0

    # Create features
    train_data = set_features(train_data)
    test_data = set_features(test_data)

    # Save processed datasets
    train_data.to_csv(DATA_TRAIN, index=False)
    test_data.to_csv(DATA_TEST, index=False)
