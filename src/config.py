from pathlib import Path

DATA_RAW = "data/raw/cs-training.csv"
DATA_CLEAN = "data/processed/data_clean.csv"
DATA_FEATURES = "data/processed/data_features.csv"

SPLIT_TRAIN = "data/processed/train_split.csv"
SPLIT_TEST = "data/processed/test_split.csv"
SPLIT_VAL = "data/processed/validation_split.csv"

X_TRAIN_DIR = "data/processed/Xtrain.csv"
X_TEST_DIR = "data/processed/Xtest.csv"
X_VAL_DIR = "data/processed/Xeval.csv" 

Y_TRAIN_DIR = "data/processed/Ytrain.csv"
Y_TEST_DIR = "data/processed/Ytest.csv"
Y_VAL_DIR = "data/processed/Yeval.csv"

PREPROCESSOR_LOG_REG = "models/preprocessor_log_reg.joblib"
PREPROCESSOR_ENSEMBLE = "models/preprocessor_ensemble.joblib"

MODEL_LOG_REG_DIR = "models/log_reg.joblib"
MODEL_RANDOM_FOR_DIR = "models/random_forest.joblib"
MODEL_CATBOOST_DIR = "models/catboost.joblib"
MODEL_XGBOOST_DIR = "models/xgboost.joblib"
MODEL_LIGHTGBM_DIR = "models/lightgbm.joblib"

MODELS = {
    "log_reg": {
        "name" : "Logistic Regression",
        "type" : "log_reg",
        "path" : MODEL_LOG_REG_DIR
    },
    "random_forest" : {
        "name" : "Random Forest",
        "type" : "ensemble",
        "path" : MODEL_RANDOM_FOR_DIR
    },
    "catboost" : {
        "name" : "CatBoost",
        "type" : "ensemble",
        "path" : MODEL_CATBOOST_DIR
    },
    "xgboost" : {
        "name" : "XGBoost",
        "type" : "ensemble",
        "path" : MODEL_XGBOOST_DIR
    },
    "lightgbm" : {
        "name" : "LightGBM",
        "type" : "ensemble",
        "path" : MODEL_LIGHTGBM_DIR
    }
}

MODELS_LIST = [
    {
        "name" : "Logistic Regression",
        "type" : "log_reg",
        "path" : MODEL_LOG_REG_DIR
    },
    {
        "name" : "Random Forest",
        "type" : "ensemble",
        "path" : MODEL_RANDOM_FOR_DIR
    },
    {
        "name" : "CatBoost",
        "type" : "ensemble",
        "path" : MODEL_CATBOOST_DIR
    },
    {
        "name" : "XGBoost",
        "type" : "ensemble",
        "path" : MODEL_XGBOOST_DIR
    },
    {
        "name" : "LightGBM",
        "type" : "ensemble",
        "path" : MODEL_LIGHTGBM_DIR
    },
]

CALIBRATION_DIR = Path("calibration")

MODELS_CALIBRATED = {
    "log_reg": {
        "name" : "Logistic Regression",
        "type" : "log_reg",
        "path" : CALIBRATION_DIR / "calibrated_model_Logistic Regression.joblib"
    },
    "random_forest" : {
        "name" : "Random Forest",
        "type" : "ensemble",
        "path" : CALIBRATION_DIR / "calibrated_model_Random Forest.joblib"
    },
    "catboost" : {
        "name" : "CatBoost",
        "type" : "ensemble",
        "path" : CALIBRATION_DIR / "calibrated_model_CatBoost.joblib"
    },
    "xgboost" : {
        "name" : "XGBoost",
        "type" : "ensemble",
        "path" : CALIBRATION_DIR / "calibrated_model_XGBoost.joblib"
    },
    "lightgbm" : {
        "name" : "LightGBM",
        "type" : "ensemble",
        "path" : CALIBRATION_DIR / "calibrated_model_LightGBM.joblib"
    }
}


