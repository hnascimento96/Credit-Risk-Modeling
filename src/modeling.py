import pandas as pd
import numpy as np
import joblib

from src.config import (
    X_TRAIN_DIR, X_VAL_DIR, Y_TRAIN_DIR, Y_VAL_DIR,
    PREPROCESSOR_LOG_REG, PREPROCESSOR_ENSEMBLE,
    MODEL_LOG_REG_DIR, MODEL_RANDOM_FOR_DIR, MODEL_CATBOOST_DIR,
    MODEL_XGBOOST_DIR
)

from src.processing import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def get_data():

    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")

    # Converts DataFrame to 1D Series
    y_train = pd.read_csv(Y_TRAIN_DIR, index_col="ID").squeeze("columns")

    return X_train, y_train

def train_log_reg(X_train, y_train):

    # Processing data
    log_reg_processor = joblib.load(PREPROCESSOR_LOG_REG)
    X_train = log_reg_processor.transform(X_train)

    # Initializing classifier
    # L2 regularization with lambda coefficient = 1/2C
    log_reg_model = LogisticRegression(C=1.0,
                                       max_iter=1000,
                                       random_state=42,
                                       class_weight="balanced")

    # Training
    log_reg_model.fit(X_train, y_train)

    # Saving model
    joblib.dump(log_reg_model, MODEL_LOG_REG_DIR)

def train_random_forest(X_train, y_train):
    # Processing input data
    ensemble_processor = joblib.load(PREPROCESSOR_ENSEMBLE)
    X_train = ensemble_processor.transform(X_train)

    # Initializating classifier
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
    # Processing input data
    ensemble_processor = joblib.load(PREPROCESSOR_ENSEMBLE)
    X_train = ensemble_processor.transform(X_train)

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
    Trains an XGBoost classifier using the saved preprocessing pipeline
    and saves the trained model to disk.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features containing the original input columns.
    y_train : pandas.Series or pandas.DataFrame
        Target variable indicating whether a customer defaulted.

    Returns
    -------
    None
    """
    # Process input data
    ensemble_processor = joblib.load(PREPROCESSOR_ENSEMBLE)
    X_train = ensemble_processor.transform(X_train)

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

if __name__ == "__main__":
    # Gets data from .csv files
    X_train, y_train = get_data()

    # Train models
    #train_log_reg(X_train, y_train)
    #train_random_forest(X_train, y_train)
    #train_catboost(X_train, y_train)

    train_xgboost(X_train, y_train)

