from src.config import(
    DATA_RAW, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL,
    X_TRAIN_DIR, X_VAL_DIR, X_TEST_DIR,
    Y_TEST_DIR, Y_TRAIN_DIR, Y_VAL_DIR
) 
from sklearn.model_selection import train_test_split
import pandas as pd

# ------------------ COLUMN DESCRIPTION ------------------

"""
ID
    Unique identifier for each borrower record

Default
    Loan default indicator (1 = default, 0 = no default)

Util
    Credit utilization ratio (revolving balance / credit limit)

Age
    Age of the borrower in years

Late30_59
    Number of times borrower was 30–59 days past due

Debt
    Debt-to-income ratio

Income
    Monthly income of the borrower

OpenCred
    Number of open credit lines and loans

Late90
    Number of times borrower was 90+ days past due

RealEstate
    Number of mortgage/real estate loans

Late60_89
    Number of times borrower was 60–89 days past due

No.Dep
    Number of dependents in the borrower's household
"""

def set_columns(credit_data):
    """
    Assigns column names to the credit dataset and sets 'ID' as the index.

    Parameters
    ----------
    credit_data : pandas.DataFrame
        Raw credit dataset containing the 12 columns in the expected order.

    Returns
    -------
    pandas.DataFrame
        Dataset with descriptive column names and 'ID' set as the index.

    Notes
    -----
    The input DataFrame is expected to have columns in the original dataset's
    predefined order. The function renames the columns and removes 'ID' from
    the regular columns by setting it as the index.
    """

    credit_data.columns = [
    'ID',
    'Default',
    'Util',
    'Age',
    'Late30_59',
    'Debt',
    'Income',
    'OpenCred',
    'Late90',
    'RealEstate',
    'Late60_89',
    'No.Dep'
    ]

    credit_data = credit_data.set_index('ID')
    return credit_data

def split_target_variable():
    """
    Separates the target variable from the features in the train, validation,
    and test datasets.

    Reads the previously generated split files, saves the 'Default' target
    variable to separate CSV files, and saves the remaining features to their
    respective CSV files.

    Returns
    -------
    None
        The function writes the target and feature datasets to CSV files.
        The 'ID' column is loaded as the index and preserved in the output
        files as the index.

    Notes
    -----
    The split files must already exist and contain the 'Default' column.
    The function processes the training, validation, and test datasets
    independently.
    """

    train_path = {
        "data_path" : SPLIT_TRAIN,
        "X_path" : X_TRAIN_DIR,
        "y_path" : Y_TRAIN_DIR
    }

    test_path = {
        "data_path" : SPLIT_TEST,
        "X_path" : X_TEST_DIR,
        "y_path" : Y_TEST_DIR
    }

    val_path = {
        "data_path" : SPLIT_VAL,
        "X_path" : X_VAL_DIR,
        "y_path" : Y_VAL_DIR
    }

    paths = [train_path, val_path, test_path]

    for path in paths:
        data = pd.read_csv(path["data_path"], index_col="ID")

        data["Default"].to_csv(path["y_path"], index=True)

        data = data.drop(columns=["Default"])
        data.to_csv(path["X_path"], index=True)

def split_data():
    """
    Splits the raw credit dataset into training, validation, and test sets.

    Loads the raw dataset, assigns descriptive column names, sets 'ID' as
    the index, and performs stratified splitting based on the 'Default'
    target variable. Saves the resulting datasets to separate CSV files
    and calls split_target_variable() to separate features and target labels.

    Returns
    -------
    None
        The function saves the split datasets and the corresponding feature
        and target files to disk.

    Notes
    -----
    The split proportions are 70% for training, 15% for validation, and
    15% for testing. Stratification preserves approximately the same class
    distribution of 'Default' across all three sets.

    The random_state parameter is set to 42 to ensure reproducibility.
    """
   
    credit_data = pd.read_csv(DATA_RAW)
    credit_data = set_columns(credit_data)

    # Split the dataset while preserving the class distribution
    # of the target variable 'Default'.
    train_df, temp_df = train_test_split(
        credit_data,
        test_size=0.3,
        random_state=42,
        stratify=credit_data["Default"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["Default"]
    )
    train_df.to_csv(SPLIT_TRAIN)
    test_df.to_csv(SPLIT_TEST)
    val_df.to_csv(SPLIT_VAL)

    split_target_variable()


if __name__ == "__main__":

    split_data()



