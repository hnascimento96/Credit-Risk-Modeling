import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = plib.Path(__file__).resolve().parent

credit_data = pd.read_csv(ROOT / 'data' / 'cs-training.csv')

colunas_desc = """
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

#missing values treatment
credit_data['Income'] = credit_data['Income'].fillna(credit_data['Income'].mean())
credit_data['No.Dep'] = credit_data['No.Dep'].fillna(credit_data['No.Dep'].mode()[0])

#feature creation
credit_data['AnyLate'] = ((credit_data['Late30_59'] > 0) | (credit_data['Late60_89'] > 0) | (credit_data['Late90'] > 0)).astype(int)

credit_data['TotalLate'] = credit_data['Late30_59'] + credit_data['Late60_89'] + credit_data['Late90']

credit_data['LateScore'] = 1*credit_data['Late30_59'] + credit_data['Late60_89']*2 + credit_data['Late90']*3

credit_data['UtilAbove100'] = np.where(credit_data['Util'] > 100, 1, 0)

print(credit_data.head(10))

