import pandas as pd
import numpy as np
import joblib

from src.config import (
    X_TRAIN_DIR, X_VAL_DIR, Y_TRAIN_DIR, Y_VAL_DIR,
    PREPROCESSOR_LOG_REG, PREPROCESSOR_ENSEMBLE,
    MODEL_LOG_REG_DIR, MODEL_RANDOM_FOR_DIR
)

from src.processing import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_data():

    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")

    # Converts DataFrame to 1D Series
    y_train = pd.read_csv(Y_TRAIN_DIR, index_col="ID").squeeze("columns")

    return X_train, y_train

def train_log_reg(X_train, y_train):

    # Processing data
    log_reg_processor = joblib.load(PREPROCESSOR_LOG_REG)
    X_train = log_reg_processor.transform(X_train)

    # Logistic Regression - Baseline model
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

    # Random Forest classifier
    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    random_forest_model.fit(X_train, y_train)

    joblib.dump(random_forest_model, MODEL_RANDOM_FOR_DIR)


if __name__ == "__main__":
    # Getting data from .csv files
    X_train, y_train = get_data()

    train_log_reg(X_train, y_train)
    train_random_forest(X_train, y_train)

