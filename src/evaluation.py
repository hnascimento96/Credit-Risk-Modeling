import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

from src.config import(
    X_TRAIN_DIR, Y_TRAIN_DIR,
    X_VAL_DIR, Y_VAL_DIR,
    MODELS_LIST, METRICS_DIR
)

# Single source of preprocessing: the same function used by training,
# calibration and the notebooks
from src.processing import process_data

def brier_score(y_pred, y_obs):

    y_pred = np.asarray(y_pred)
    y_obs = np.asarray(y_obs)

    bs = np.mean((y_pred - y_obs)**2)

    return bs

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

    # ROC-AUC
    roc_auc = roc_auc_score(y, y_prob)

    # KS statistic
    fpr, tpr, _ = roc_curve(y, y_prob)
    ks = np.max(tpr - fpr)

    # Gini coefficient
    gini = 2 * roc_auc - 1

    # Brier score
    brier_sc = brier_score(y_prob, y)

    return {
        "Model": model_name,
        "ROC-AUC": roc_auc,
        "KS": ks,
        "Gini": gini,
        "Brier score" : brier_sc
    }

def get_metrics(X, y):
    results = []

    for model_config in MODELS_LIST:
        X_processed = process_data(model_config["type"], X)

        # Loads the trained model into a local variable
        # (does not change the shared config dictionary)
        model = joblib.load(model_config["path"])
        result = evaluate_model(model, X_processed, y, model_config["name"])

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

    # Save metrics, so versions can be compared later
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    train_metrics_df.to_csv(METRICS_DIR / "train_metrics.csv", index=False)
    val_metrics_df.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)


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
    print(f"Difference values (validation - train):")
    print(difference_df)