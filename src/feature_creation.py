from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Late-payment columns (number of times the client was late)
LATE_COLS = ["Late30_59", "Late60_89", "Late90"]

# Codes 96/98 in the late-payment columns are not real counts (see EDA)
SPECIAL_CODES = [96, 98]

# Util above this value is treated as a data error (see EDA)
UTIL_LIMIT = 10


def check_dataframe(X, step_name):
    """
    Raises an error if X is not a DataFrame.

    The transformers below select columns by name, so they need a DataFrame.
    In the pipeline this is guaranteed by 'set_output(transform="pandas")'.

    Args:
        X: data received by the transformer.
        step_name: name of the transformer, used in the error message.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{step_name} expects a pandas DataFrame, got {type(X).__name__}. "
            "Use set_output(transform='pandas') in the pipeline."
        )


class DataCleaning(BaseEstimator, TransformerMixin):
    """
    Creates the flags decided in the EDA and fixes invalid values.

    It must run BEFORE the imputer and BEFORE FeatureCreation, so that:
      - missing Income is still NaN when IncomeMissing is created;
      - codes 96/98 do not enter AnyLate, TotalLate and LateScore;
      - invalid Util does not enter UtilAbove100.

    Treatments:
      - IncomeMissing: 1 if Income is NaN. Income itself stays NaN and is
        filled later by the median imputer.
      - SpecialCode: 1 if any late-payment column has code 96 or 98.
        The codes are replaced by NaN, so the median imputer fills them with
        the train median of each column (computed without the codes).
      - UtilInvalid: 1 if Util > UTIL_LIMIT. These values are replaced by the
        largest valid Util (<= UTIL_LIMIT) seen in the train set.

    Input: DataFrame containing at least Util, Income and the LATE_COLS.
    Output: the same DataFrame (cleaned) plus the columns IncomeMissing,
            SpecialCode and UtilInvalid.
    """

    FLAGS = ["IncomeMissing", "SpecialCode", "UtilInvalid"]

    def fit(self, X, y=None):
        """
        Learns the largest valid Util from the train set.

        Args:
            X: train DataFrame.
            y: ignored; kept for sklearn compatibility.

        Returns:
            self, with:
              - feature_names_in_: input column names;
              - n_features_in_: number of input columns;
              - util_max_: largest Util <= UTIL_LIMIT in the train set.
        """
        check_dataframe(X, "DataCleaning")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        util = X["Util"]
        self.util_max_ = util[util <= UTIL_LIMIT].max()

        return self

    def transform(self, X):
        """
        Applies the flags and replacements learned in 'fit'.

        Args:
            X: DataFrame to transform (same columns as in 'fit').

        Returns:
            Copy of X, cleaned, plus IncomeMissing, SpecialCode and UtilInvalid.
        """
        check_dataframe(X, "DataCleaning")

        # astype returns a copy, so the original data is never changed.
        # Float is needed because integer columns cannot hold NaN or the
        # Util cap (e.g. 9.5).
        X = X.astype(float)

        # IncomeMissing: created before any imputation
        X["IncomeMissing"] = X["Income"].isna().astype(int)

        # SpecialCode: flag rows with 96/98 in any late-payment column,
        # then replace the codes by NaN (imputed later by the median)
        is_code = X[LATE_COLS].isin(SPECIAL_CODES)
        X["SpecialCode"] = is_code.any(axis=1).astype(int)
        X[LATE_COLS] = X[LATE_COLS].mask(is_code)

        # UtilInvalid: flag Util > UTIL_LIMIT and cap it at the train maximum
        util_invalid = X["Util"] > UTIL_LIMIT
        X["UtilInvalid"] = util_invalid.astype(int)
        X.loc[util_invalid, "Util"] = self.util_max_

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output column names: input columns + the 3 flags.
        Needed so the pipeline can keep the column names.
        """
        return np.array(list(self.feature_names_in_) + self.FLAGS, dtype=object)


class FeatureCreation(BaseEstimator, TransformerMixin):
    """
    Creates the derived features. Runs AFTER DataCleaning and the imputer,
    so it receives late-payment counts without codes 96/98, valid Util
    and no missing values.

    New columns:
      - AnyLate: 1 if the client was late at least once (any late column > 0).
      - TotalLate: total number of late payments (sum of the 3 late columns).
      - LateScore: late payments weighted by severity
                   (Late30_59 + 2 * Late60_89 + 3 * Late90).
      - UtilAbove100: 1 if Util > 1 (credit use above 100% of the limit).

    Input: DataFrame containing at least Util and the LATE_COLS.
    Output: the same DataFrame plus the 4 new columns.
    """

    NEW_FEATURES = ["AnyLate", "TotalLate", "LateScore", "UtilAbove100"]

    def fit(self, X, y=None):
        """
        Nothing to learn; only stores the input column names.

        Args:
            X: train DataFrame.
            y: ignored; kept for sklearn compatibility.

        Returns:
            self, with feature_names_in_ and n_features_in_.
        """
        check_dataframe(X, "FeatureCreation")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Adds the derived features.

        Args:
            X: DataFrame to transform (same columns as in 'fit').

        Returns:
            Copy of X plus AnyLate, TotalLate, LateScore and UtilAbove100.
        """
        check_dataframe(X, "FeatureCreation")
        X = X.copy()

        X["AnyLate"] = (X[LATE_COLS] > 0).any(axis=1).astype(int)

        X["TotalLate"] = X["Late30_59"] + X["Late60_89"] + X["Late90"]

        X["LateScore"] = (
            X["Late30_59"]
            + 2 * X["Late60_89"]
            + 3 * X["Late90"]
        )

        X["UtilAbove100"] = (X["Util"] > 1).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output column names: input columns + the 4 new features.
        Needed so the pipeline can keep the column names.
        """
        return np.array(
            list(self.feature_names_in_) + self.NEW_FEATURES, dtype=object
        )