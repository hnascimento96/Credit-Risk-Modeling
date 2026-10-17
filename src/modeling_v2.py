import pandas as pd
import pathlib as plib
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ks_2samp

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def compute_metrics(name, y_test, y_prob, y_pred):
    auc  = roc_auc_score(y_test, y_prob)
    ks   = ks_2samp(y_prob[y_test==0], y_prob[y_test==1]).statistic
    gini = 2 * auc - 1
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"{name} → AUC: {auc:.4f} | KS: {ks:.4f} | Gini: {gini:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")


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

auc = roc_auc_score(y_test, y_prob)

print("-"*50)
print("Evaluation for Logistic Regression")
compute_metrics('Logistic Regression', y_test, y_prob, y_pred)

################################################################

#model_evaluation(y_test, y_pred, y_prob, "Logistic Regression")


################################################################
#LightGBM + Optuna

#lgbm_hyper_param, best_lgbm_auc = optimize_lgbm(X_train, y_train)
#print(lgbm_hyper_param, best_lgbm_auc)

#joblib.dump(lgbm_hyper_param, "best_lgbm_hyper_params.joblib")

lgbm_hyper_params = joblib.load("best_lgbm_hyper_params.joblib")
print("-"*50)

lgbm_hyper_params['scale_pos_weight'] = 13
lgbm_hyper_params['max_depth'] = -1
lgbm_hyper_params['random_state'] = 42
lgbm_hyper_params['n_jobs'] = -1

model = LGBMClassifier(
    **lgbm_hyper_params
)


model.fit(X_train, y_train)

# ------------ Evaluation --------------

y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

compute_metrics('LGBM', y_test, y_prob, y_pred)

# ------------ XGBoost Method -----------
# ----------- Training ----------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    scale_pos_weight=13,
    random_state=42
)

model.fit(X_train, y_train)

# ---------- Evaluation -------

y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_prob)

print("-"*50)
compute_metrics('XGBoost', y_test, y_prob, y_pred)


# ---------------- Random Forest --------------

# ------------- Training ----------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# -------------- Evaluating -----------------

yy_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_prob)

print("-"*50)
print("Evaluation for Random Forest")
compute_metrics('Random Forest', y_test, y_prob, y_pred)


################





















