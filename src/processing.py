"""
Preprocessing of the model inputs.

Builds, fits and saves one preprocessor per model type, and applies the saved
preprocessor to new data.

Two model types share the same steps; they differ only in the scaling:
  - 'log_reg':  features are standardized (StandardScaler);
  - 'ensemble': features are not scaled (tree models do not need it).

Order of the steps (it matters):
  1. Original numeric features (median branch):
       DataCleaning   -> flags IncomeMissing, SpecialCode, UtilInvalid;
                         codes 96/98, Util > 10 and Debt without usable
                         income (Income NaN, 0 or 1) become NaN.
       SimpleImputer  -> fills NaN with the train median.
       FeatureCreation-> AnyLate, TotalLate, LateScore, UtilAbove100.
       Scaler         -> only for 'log_reg'.
     Cleaning comes first so that missing Income is still visible when its
     flag is created, and so that codes 96/98 and invalid Util never reach
     the derived features.
  2. No.Dep (mode branch): SimpleImputer with the most frequent value -> Scaler.

All statistics (medians, mode, scaler) are learned from the
train set only and then applied unchanged to validation and test data.

The output is a DataFrame with named columns and the original index.

Run this module to fit and save both preprocessors:
    python -m src.processing
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.config import(
    X_TRAIN_DIR,
    PREPROCESSOR_ENSEMBLE, PREPROCESSOR_LOG_REG
    )

from src.feature_creation import DataCleaning, FeatureCreation

import joblib


# Path of the saved preprocessor of each model type.
# Used both to save (preprocessing) and to load (process_data),
# so both always point to the same file.
PREPROCESSOR_DIR = {
    "log_reg": PREPROCESSOR_LOG_REG,
    "ensemble": PREPROCESSOR_ENSEMBLE,
}


def preprocessing(model, X_train):
    """
    Builds the preprocessor of one model type, fits it on the train set
    and saves it to disk.

    Parameters
    ----------
    model : str
        Model type, 'log_reg' or 'ensemble'. Defines whether the features
        are standardized ('log_reg') or not ('ensemble').
    X_train : pandas.DataFrame
        Train data with the 10 original columns (Util, Age, Late30_59, Debt,
        Income, OpenCred, Late90, RealEstate, Late60_89, No.Dep), indexed
        by ID.

    Returns
    -------
    None
        The fitted preprocessor is saved to PREPROCESSOR_DIR[model].

    Raises
    ------
    ValueError
        If 'model' is not 'log_reg' or 'ensemble'.
    """
    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model: {model!r}. "
            "Valid models: 'log_reg' or 'ensemble'"
        )

    # Original numeric features: cleaned, imputed by the median and
    # used to create the derived features
    median_features = [
        "Util",
        "Age",
        "Late30_59",
        "Debt",
        "Income",
        "OpenCred",
        "Late90",
        "RealEstate",
        "Late60_89",
    ]

    # 'passthrough' indicates to 'Pipeline' that a step must be ignored.
    # The same scaler object is used in both branches; ColumnTransformer
    # fits a separate copy for each one.
    scaler = "passthrough" if model == "ensemble" else StandardScaler()

    # No.Dep is a count: missing values get the most frequent value
    mode_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", scaler)
    ])

    # Order matters: cleaning (flags, codes 96/98, invalid Util) must run
    # before the imputer and before the derived features
    median_pipeline = Pipeline([
            ("cleaning", DataCleaning()),
            ("imputer", SimpleImputer(strategy="median")),
            ("feature_creater", FeatureCreation()),
            ("scaler", scaler)
    ])

    # verbose_feature_names_out=False keeps the plain column names
    # (e.g. 'Util' instead of 'median_pipeline__Util')
    preprocessor_pipeline = ColumnTransformer(
        [
            ("median_pipeline", median_pipeline, median_features),
            ("mode_pipeline", mode_pipeline, ["No.Dep"])
        ],
        verbose_feature_names_out=False,
    )

    # Output as DataFrame, with column names and the original index
    preprocessor_pipeline.set_output(transform="pandas")

    # All statistics (medians, mode, scaler) come from
    # the train set only
    preprocessor_pipeline.fit(X_train)

    joblib.dump(preprocessor_pipeline, PREPROCESSOR_DIR[model])


def process_data(model, X):
    """
    Applies the saved preprocessor of one model type to new data.

    Use it everywhere the models receive data (training, evaluation,
    calibration, notebooks), so all of them see exactly the same features.

    Parameters
    ----------
    model : str
        Model type, 'log_reg' or 'ensemble'.
    X : pandas.DataFrame
        Data with the 10 original columns (same as in 'preprocessing'),
        indexed by ID. Train, validation or test.

    Returns
    -------
    pandas.DataFrame
        17 named columns, with the same index as X: the 9 original numeric
        features (cleaned and imputed), IncomeMissing, SpecialCode,
        UtilInvalid, AnyLate, TotalLate, LateScore, UtilAbove100 and No.Dep.
        Standardized if model is 'log_reg'.

    Raises
    ------
    ValueError
        If 'model' is not 'log_reg' or 'ensemble'.
    """
    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model name {model!r}. Valid names are 'log_reg', 'ensemble'"
        )

    processor = joblib.load(PREPROCESSOR_DIR[model])
    X = processor.transform(X)

    return X


if __name__ == "__main__":

    # Preprocessors are fitted on the train set only
    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")

    preprocessing(model="log_reg", X_train=X_train)
    preprocessing(model="ensemble", X_train=X_train)