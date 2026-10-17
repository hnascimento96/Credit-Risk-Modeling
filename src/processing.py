
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL, DATA_TRAIN, DATA_TEST, DATA_VAL
from src.CreditRiskPreprocessor import CreditRiskPreprocessor

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # Load the train and test sets from split
    train_data = pd.read_csv(SPLIT_TRAIN, index_col=0)
    test_data = pd.read_csv(SPLIT_TEST, index_col=0)
    val_data = pd.read_csv(SPLIT_VAL, index_col=0)

    # Impute missing values using statistics from the training set
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(train_data)

    train_data = preprocessor.transform(train_data)
    test_data = preprocessor.transform(test_data)
    val_data = preprocessor.transform(val_data)

    assert train_data[train_data.columns].isna().sum().sum() == 0
    assert test_data[test_data.columns].isna().sum().sum() == 0
    assert test_data[val_data.columns].isna().sum().sum() == 0

    # Save processed datasets
    train_data.to_csv(DATA_TRAIN, index=False)
    test_data.to_csv(DATA_TEST, index=False)
    val_data.to_csv(DATA_VAL, index=False)
