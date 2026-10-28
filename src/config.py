
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
DATA_CLEAN = PROJECT_ROOT / "data" / "processed" / "data_clean.csv"
DATA_FEATURES = PROJECT_ROOT / "data" / "processed" / "data_features.csv"

# Dataset splits
SPLIT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train_split.csv"
SPLIT_TEST = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
SPLIT_VAL = PROJECT_ROOT / "data" / "processed" / "validation_split.csv"

# Feature matrices
X_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "Xtrain.csv"
X_TEST_DIR = PROJECT_ROOT / "data" / "processed" / "Xtest.csv"
X_VAL_DIR = PROJECT_ROOT / "data" / "processed" / "Xeval.csv"

# Target vectors
Y_TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "Ytrain.csv"
Y_TEST_DIR = PROJECT_ROOT / "data" / "processed" / "Ytest.csv"
Y_VAL_DIR = PROJECT_ROOT / "data" / "processed" / "Yeval.csv"

# Preprocessors
PREPROCESSOR_LOG_REG = PROJECT_ROOT / "models" / "preprocessor_log_reg.joblib"
PREPROCESSOR_ENSEMBLE = PROJECT_ROOT / "models" / "preprocessor_ensemble.joblib"

# Trained models
MODEL_LOG_REG_DIR = PROJECT_ROOT / "models" / "log_reg.joblib"
MODEL_RANDOM_FOR_DIR = PROJECT_ROOT / "models" / "random_forest.joblib"
MODEL_CATBOOST_DIR = PROJECT_ROOT / "models" / "catboost.joblib"
MODEL_XGBOOST_DIR = PROJECT_ROOT / "models" / "xgboost.joblib"
MODEL_LIGHTGBM_DIR = PROJECT_ROOT / "models" / "lightgbm.joblib"

# Credit policy reports
CREDIT_POLICY_DIR = PROJECT_ROOT / "reports" / "credit_policy"

MODELS = {
    "log_reg": {
        "name": "Logistic Regression",
        "type": "log_reg",
        "path": MODEL_LOG_REG_DIR,
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "ensemble",
        "path": MODEL_RANDOM_FOR_DIR,
    },
    "catboost": {
        "name": "CatBoost",
        "type": "ensemble",
        "path": MODEL_CATBOOST_DIR,
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "ensemble",
        "path": MODEL_XGBOOST_DIR,
    },
    "lightgbm": {
        "name": "LightGBM",
        "type": "ensemble",
        "path": MODEL_LIGHTGBM_DIR,
    },
}

MODELS_LIST = [
    {
        "name": "Logistic Regression",
        "type": "log_reg",
        "path": MODEL_LOG_REG_DIR,
    },
    {
        "name": "Random Forest",
        "type": "ensemble",
        "path": MODEL_RANDOM_FOR_DIR,
    },
    {
        "name": "CatBoost",
        "type": "ensemble",
        "path": MODEL_CATBOOST_DIR,
    },
    {
        "name": "XGBoost",
        "type": "ensemble",
        "path": MODEL_XGBOOST_DIR,
    },
    {
        "name": "LightGBM",
        "type": "ensemble",
        "path": MODEL_LIGHTGBM_DIR,
    },
]

# Calibrated models
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

MODELS_CALIBRATED = {
    "log_reg": {
        "name": "Logistic Regression",
        "type": "log_reg",
        "path": CALIBRATION_DIR / "calibrated_model_Logistic Regression.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "ensemble",
        "path": CALIBRATION_DIR / "calibrated_model_Random Forest.joblib",
    },
    "catboost": {
        "name": "CatBoost",
        "type": "ensemble",
        "path": CALIBRATION_DIR / "calibrated_model_CatBoost.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "ensemble",
        "path": CALIBRATION_DIR / "calibrated_model_XGBoost.joblib",
    },
    "lightgbm": {
        "name": "LightGBM",
        "type": "ensemble",
        "path": CALIBRATION_DIR / "calibrated_model_LightGBM.joblib",
    },
}
