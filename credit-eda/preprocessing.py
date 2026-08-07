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

import optuna
import joblib

from lightgbm import LGBMClassifier

def optimize_lgbm(X_train, y_train):

    def objective_lgbm(trial):

        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        num_leaves = trial.suggest_int("num_leaves", 20, 200)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)


        model = LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)

        #Suffles and splits in 5 parts
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv = cv,
            scoring='roc_auc'
            ).mean()

        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lgbm, n_trials=50)

    return study.best_params, study.best_value


def model_evaluation(y_test, y_pred, y_prob, model_name):

    print("-"*50)
    print(f"Evaluation metrics for {model_name}\n")

    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC:{auc}")

    print(f"Confusion matrix\n:{confusion_matrix(y_test, y_pred)} ")

    print(f"Report:\n {classification_report(y_test,y_pred)}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Model {model_name}")
    plt.plot([0,1],[0,1], label='Random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()

    ks = np.abs(fpr - tpr)
    optimal_cut_index = np.argmax(ks)
    ks_max = np.max(ks)

    print(f"KS: {ks_max:.2f}, Best cut value: {thresholds[optimal_cut_index]:.2f}")

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
####### Scaling
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################################################################
####### Training

model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:,1]

################################################################

#model_evaluation(y_test, y_pred, y_prob, "Logistic Regression")


################################################################
#LightGBM + Optuna

lgbm_hyper_param, best_lgbm_auc = optimize_lgbm(X_train, y_train)
print(lgbm_hyper_param, best_lgbm_auc)

joblib.dump(lgbm_hyper_param, "best_lgbm_hyper_params.joblib")






