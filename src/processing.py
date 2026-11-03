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


# Path of the saved preprocessor of each model type
PREPROCESSOR_DIR = {
    "log_reg": PREPROCESSOR_LOG_REG,
    "ensemble": PREPROCESSOR_ENSEMBLE,
}


def preprocessing(model, X_train):

    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model: {model!r}."
            "Valid models: 'log_reg' or 'ensemble'"
        )

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

    # 'passthrough' indicates to 'Pipeline' that a step must be ignored
    scaler = "passthrough" if model == "ensemble" else StandardScaler()

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

    # All statistics (medians, mode, max valid Util, scaler) come from
    # the train set only
    preprocessor_pipeline.fit(X_train)

    joblib.dump(preprocessor_pipeline, PREPROCESSOR_DIR[model])


def process_data(model, X):

    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model name {model!r}. Valid names are 'log_reg', 'ensemble'"
        )

    processor = joblib.load(PREPROCESSOR_DIR[model])
    X = processor.transform(X)

    return X


if __name__ == "__main__":

    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")

    preprocessing(model="log_reg", X_train=X_train)
    preprocessing(model="ensemble", X_train=X_train)