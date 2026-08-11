import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp

import optuna
import joblib

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_RAW, DATA_CLEAN, DATA_FEATURES, DATA_TRAIN, DATA_TEST

credit_data = pd.read_csv(DATA_RAW)

# ------------------ COLUMN DESCRIPTION ------------------

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

# ------------------ MISSING VALUES TREATMENT --------------------
credit_data['Income'] = credit_data['Income'].fillna(credit_data['Income'].mean())
credit_data['No.Dep'] = credit_data['No.Dep'].fillna(credit_data['No.Dep'].mode()[0])

credit_data.to_csv(DATA_CLEAN)

# ------------------ FEATURE CREATION -----------------------------
credit_data['AnyLate'] = ((credit_data['Late30_59'] > 0) | (credit_data['Late60_89'] > 0) | (credit_data['Late90'] > 0)).astype(int)
credit_data['TotalLate'] = credit_data['Late30_59'] + credit_data['Late60_89'] + credit_data['Late90']
credit_data['LateScore'] = 1*credit_data['Late30_59'] + credit_data['Late60_89']*2 + credit_data['Late90']*3
credit_data['UtilAbove100'] = np.where(credit_data['Util'] > 100, 1, 0)

credit_data.to_csv(DATA_FEATURES)

# ------------------ TRAIN/TEST SPLIT -----------------------------
X = credit_data.drop(columns='Default')
y = credit_data['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv(DATA_TRAIN)
test.to_csv(DATA_TEST)



