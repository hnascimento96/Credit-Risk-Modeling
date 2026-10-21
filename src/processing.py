
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import X_TRAIN_DIR, X_VAL_DIR, X_TEST_DIR
from src.feature_creation import FeatureCreation

import joblib
        

def preprocessing(model, X_train, X_test, X_val):

    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model: {model!r}."
            "Valid models: 'log_reg' or 'ensemble'"
        )

    median_features = [
        "Util",             # Col. No. 0
        "Age",              # Col. No. 1
        "Late30_59",        # Col. No. 2
        "Debt",             # Col. No. 3
        "Income",           # Col. No. 4
        "OpenCred",         # Col. No. 5
        "Late90",           # Col. No. 6
        "RealEstate",       # Col. No. 7
        "Late60_89",        # Col. No. 8
    ]

    # 'passthrough' indicates to 'Pipeline' that a step must be ignored
    scaler = "passthrough" if model == "ensemble" else StandardScaler()

    mode_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", scaler)
    ])

    median_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("feature_creater", FeatureCreation()),
            ("scaler", scaler)
    ])    
        
    preprocessor_pipeline = ColumnTransformer(
        [
            ("median_pipeline", median_pipeline, median_features),
            ("mode_pipeline", mode_pipeline, ["No.Dep"])
        ]
    )

    X_train = preprocessor_pipeline.fit_transform(X_train)

    X_val = preprocessor_pipeline.transform(X_val)
    X_test = preprocessor_pipeline.transform(X_test)

    joblib.dump(preprocessor_pipeline, f"models/preprocessor_{model}.joblib")


if __name__ == "__main__":

    X_dfs = []
    dirs = [X_TRAIN_DIR, X_VAL_DIR, X_TEST_DIR]

    for dir in dirs:
        df = pd.read_csv(dir, index_col="ID")
        X_dfs.append(df)

    preprocessing(model='log_reg',
                  X_train=X_dfs[0],
                  X_val=X_dfs[1],
                  X_test=X_dfs[2])

    preprocessing(model='ensemble',
                      X_train=X_dfs[0],
                      X_val=X_dfs[1],
                      X_test=X_dfs[2])
