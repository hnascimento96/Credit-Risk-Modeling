import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve
)

from src.config import(
    X_TRAIN_DIR, Y_TRAIN_DIR,
    X_VAL_DIR, Y_VAL_DIR, 
    PREPROCESSOR_ENSEMBLE, PREPROCESSOR_LOG_REG,
    MODEL_LOG_REG_DIR, MODEL_CATBOOST_DIR, MODEL_RANDOM_FOR_DIR,
    MODEL_LIGHTGBM_DIR, MODEL_XGBOOST_DIR
) 

def evaluate_model(model, X, y, model_name, threshold=0.5):
    """
    Evaluates a binary classification model using ROC-AUC, precision,
    recall, Kolmogorov-Smirnov (KS), and Gini coefficient.

    Parameters
    ----------
    model : estimator
        Trained binary classification model with predict_proba().
    X : pandas.DataFrame or numpy.ndarray
        Input features used for evaluation.
    y : pandas.Series or numpy.ndarray
        True binary labels, where 1 represents default.
    model_name : str
        Name identifying the model.
    threshold : float, default=0.5
        Probability threshold used to classify observations.

    Returns
    -------
    dict
        Dictionary containing the model name and evaluation metrics.
    """
    # Predicted probabilities and classes
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    roc_auc = roc_auc_score(y, y_prob)

    # Precision and Recall
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)

    # KS statistic
    fpr, tpr, _ = roc_curve(y, y_prob)
    ks = np.max(tpr - fpr)

    # Gini coefficient
    gini = 2 * roc_auc - 1

    return {
        "Model": model_name,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "KS": ks,
        "Gini": gini
    }

def process_data(model, X):

    if model not in ("log_reg", "ensemble"):
        raise ValueError(
            f"Invalid model name {model!r}. Valid names are 'log_reg', 'ensemble'"
        )

    preprocessor_dir = {
        "log_reg" : PREPROCESSOR_LOG_REG,
        "ensemble" : PREPROCESSOR_ENSEMBLE
    }

    processor = joblib.load(preprocessor_dir[model])
    X = processor.transform(X)

    return X

def get_metrics(X, y):
    models_list = [
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
    
    results = []

    # Train metrics
    for model in models_list:
        X_processed = process_data(model["type"], X)

        # Evaluates model
        model["params"] = joblib.load(model["path"])
        result = evaluate_model(model["params"], X_processed, y, model["name"])

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df

if __name__ == "__main__":

    # Evaluation for threshold = 0.5
    # This choice has influence on values of 'Precision' and 'Recall' metrics
    # and no influence on ROC-AUC, KS, Gini values

    # Train metrics
    X_train = pd.read_csv(X_TRAIN_DIR, index_col="ID")
    
    # Converts DataFrame to 1D Series
    y_train = pd.read_csv(Y_TRAIN_DIR, index_col="ID").squeeze("columns")

    train_metrics_df = get_metrics(X_train, y_train)

    print("-"*100)
    print(f"Train metrics:")
    print(train_metrics_df)

    ##################
    # Validation metrics
    X_val = pd.read_csv(X_VAL_DIR, index_col="ID")
        
    # Converts DataFrame to 1D Series
    y_val = pd.read_csv(Y_VAL_DIR, index_col="ID").squeeze("columns")

    val_metrics_df = get_metrics(X_val, y_val)

    print("-"*100)
    print(f"Validation metrics:")
    print(val_metrics_df)


    ####################
    # Difference values

    # Merged dataframe
    merged_df = pd.merge(
        val_metrics_df.loc[:,["Model", "ROC-AUC", "KS"]],
        train_metrics_df.loc[:,["Model", "ROC-AUC", "KS"]],
        on="Model",
        how="inner"
    )

    merged_df["ROC-AUC (diff.)"] = merged_df["ROC-AUC_x"] - merged_df["ROC-AUC_y"]
    merged_df["KS (diff.)"] = merged_df["KS_x"] - merged_df["KS_y"]

    difference_df = merged_df.loc[:, ["Model", "ROC-AUC (diff.)", "KS (diff.)"]]

    print("-"*100)
    print(f"Difference values (train - validation):")
    print(difference_df)






