"""
Training of the credit risk models.

Trains five binary classifiers of the probability of default (PD) and saves
each one to disk (paths in src.config):
  - Logistic Regression  -> preprocessor 'log_reg' (standardized features)
  - Random Forest, CatBoost, XGBoost, LightGBM -> preprocessor 'ensemble'
    (features not scaled)

The features come from the saved preprocessors, so run this first:
    python -m src.processing
then:
    python -m src.modeling

Class imbalance: the default rate is about 6.7%, so every model gives more
weight to the defaulters (class 1). Each library has its own parameter:
class_weight="balanced" (Logistic Regression, Random Forest, LightGBM),
auto_class_weights="Balanced" (CatBoost) and scale_pos_weight (XGBoost).
Because of this weighting, the predicted probabilities are inflated and are
NOT calibrated PDs (high Brier score); they are calibrated later in
src.calibration.

Hyperparameters are fixed (no tuning), so changes in the metrics between
versions come only from the preprocessing.
"""

import pandas as pd
import joblib

from src.config import (
    X_TRAIN_DIR, Y_TRAIN_DIR,
    MODEL_LOG_REG_DIR, MODEL_RANDOM_FOR_DIR, MODEL_CATBOOST_DIR,
    MODEL_XGBOOST_DIR, MODEL_LIGHTGBM_DIR
)

# Same preprocessing function used by evaluation, calibration and notebooks
from src.processing import process_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_data():
    """
    Loads the training features and target variable from CSV files.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features (10 original columns) indexed by customer ID.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted (1 = default).
    """

    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")

    # Converts DataFrame to 1D Series
    y_train = pd.read_csv(Y_TRAIN_DIR, index_col="ID").squeeze("columns")

    return X_train, y_train

def train_log_reg(X_train, y_train):
    """
    Trains a Logistic Regression classifier on the 'log_reg' preprocessed
    features and saves the trained model to disk.

    Classes are balanced with class_weight="balanced".

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted.

    Returns
    -------
    None
    """
    # Processing data
    X_train = process_data(model="log_reg", X=X_train)

    # Initializing classifier
    # L2 regularization (default penalty). C is the inverse of the
    # regularization strength: smaller C means stronger regularization.
    log_reg_model = LogisticRegression(C=1.0,
                                       max_iter=1000,
                                       random_state=42,
                                       class_weight="balanced")

    # Training
    log_reg_model.fit(X_train, y_train)

    # Saving model
    joblib.dump(log_reg_model, MODEL_LOG_REG_DIR)

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier on the 'ensemble' preprocessed
    features and saves the trained model to disk.

    Classes are balanced with class_weight="balanced".

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted.

    Returns
    -------
    None
    """

    # Processing input data
    X_train = process_data(model="ensemble", X=X_train)

    # Initializing classifier
    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    # Training model
    random_forest_model.fit(X_train, y_train)

    # Saving trained model
    joblib.dump(random_forest_model, MODEL_RANDOM_FOR_DIR)

def train_catboost(X_train, y_train):
    """
    Trains a CatBoost classifier on the 'ensemble' preprocessed features
    and saves the trained model to disk.

    Classes are balanced with auto_class_weights="Balanced".

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted.

    Returns
    -------
    None
    """

    # Processing input data
    X_train = process_data(model="ensemble", X=X_train)

    # Initializing classifier
    catboost_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    # Training
    catboost_model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(catboost_model, MODEL_CATBOOST_DIR)

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier on the 'ensemble' preprocessed features
    and saves the trained model to disk.

    Classes are balanced with scale_pos_weight = (number of non-defaulters)
    / (number of defaulters), computed from y_train (about 14 for this data).

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted.

    Returns
    -------
    None
    """
    # Process input data
    X_train = process_data(model="ensemble", X=X_train)

    # Initialize model
    xgboost_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        n_jobs=-1
    )

    # Train model
    xgboost_model.fit(X_train, y_train)

    # Save model
    joblib.dump(xgboost_model, MODEL_XGBOOST_DIR)

def train_lightgbm(X_train, y_train):
    """
    Trains a LightGBM classifier on the 'ensemble' preprocessed features
    and saves the trained model to disk.

    Classes are balanced with class_weight="balanced".

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series
        Training target indicating whether a customer defaulted.

    Returns
    -------
    None
    """
    # Process input data
    X_train = process_data(model="ensemble", X=X_train)

    # Initialize model
    lightgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    # Train model
    lightgbm_model.fit(X_train, y_train)

    # Save model
    joblib.dump(lightgbm_model, MODEL_LIGHTGBM_DIR)

if __name__ == "__main__":
    # Gets data from .csv files
    X_train, y_train = get_data()

    # Train models
    train_log_reg(X_train, y_train)
    train_random_forest(X_train, y_train)
    train_catboost(X_train, y_train)
    train_xgboost(X_train, y_train)
    train_lightgbm(X_train, y_train)