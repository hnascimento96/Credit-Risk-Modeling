import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

#Splitting train/test

X = credit_data.drop(columns='Default')
y = credit_data['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#print(X_train.head(50))

print(f"y_train: {y_train.value_counts(normalize=True)}")
print(f"y_test: {y_test.value_counts(normalize=True)}")

################################################################
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################################################################
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:,1]

################################################################
auc = roc_auc_score(y_test, y_prob)
print(f"AUC:{auc}")

print(f"Confusion matrix:{confusion_matrix(y_test, y_pred)} ")

print(f"Report:\n {classification_report(y_test,y_pred)}")

#################################################################

print("-"*200)
print("Threshold\t Recall\t Precision\t F1-score")

thresholds = [0.05, 0.1, 0.25, 0.30, 0.4]

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)
    print(f"{threshold}\t {recall_score(y_test, y_pred)}\t {precision_score(y_test, y_pred)}\t {f1_score(y_test, y_pred)}")





