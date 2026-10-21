from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

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