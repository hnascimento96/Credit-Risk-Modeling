
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import X_TRAIN_DIR, X_VAL_DIR, X_TEST_DIR

import joblib


class FeatureCreation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)

        # We need to store something in a variable 'self.var_' in order to
        # pass the sklearn checking
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X, y=None):
        X = np.asarray(X)

        # Index for columns
        col_X = {
            "Util": 0,          # Col. No. 0
            "Age": 1,           # Col. No. 1
            "Late30_59": 2,     # Col. No. 2
            "Debt": 3,          # Col. No. 3
            "Income": 4,        # Col. No. 4
            "OpenCred": 5,      # Col. No. 5
            "Late90": 6,        # Col. No. 6
            "RealEstate": 7,    # Col. No. 7
            "Late60_89": 8,     # Col. No. 8
        }
        
        late30_59 = X[:, col_X["Late30_59"]]
        late60_89 = X[:, col_X["Late60_89"]]
        late90 = X[:, col_X["Late90"]]
        util = X[:, col_X["Util"]]

        any_late = (
            (late30_59 > 0)
            | (late60_89 > 0)
            | (late90 > 0)
        ).astype(int)

        total_late = late30_59 + late60_89 + late90

        late_score = (
            late30_59
            + 2 * late60_89
            + 3 * late90
        )

        util_above100 = (util > 1).astype(int)

        X = np.c_[
            X,
            any_late,
            total_late,
            late_score,
            util_above100
            ]

        return X
        

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

    joblib.dump(preprocessor_pipeline, MODELS_DIR / "preprocessor.joblib")



MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
