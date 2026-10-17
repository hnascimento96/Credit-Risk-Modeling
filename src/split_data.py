from src.config import DATA_RAW, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
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

def load_data(credit_data):

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

    credit_data = credit_data.drop(columns='ID')
    return credit_data


if __name__ == "__main__":
    credit_data = pd.read_csv(DATA_RAW)

    # Splits data into two datasets keeping the same proportion of column 'credit_data["Default"]' 
    # as in 'credit_data'

    credit_data = load_data(credit_data)

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

