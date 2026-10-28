import pandas as pd
import numpy as np
import joblib

from src.config import (
    X_VAL_DIR, Y_VAL_DIR, MODELS_CALIBRATED, CREDIT_POLICY_DIR
)
from src.processing import process_data

def scenario_analysis(y_prob, y, lim):

    y_df = pd.DataFrame({"y_prob" : y_prob, "default" : y})

    # Acceptance rate
    y_df["approved"] = np.where(y_df["y_prob"] <= lim, 1, 0)
    acpt_rate = float(y_df["approved"].mean())

    # Rejection rate
    rejct_rate = 1 - acpt_rate

    # Approved mean PD
    approved_mean_pd = float(y_df[y_df["approved"] == 1]["y_prob"].mean())

    # Default between approved rate
    count_approved = (y_df["approved"]==1).sum()
    count_default_approved = ((y_df["default"]==1) & (y_df["approved"]==1)).sum()
    default_approved_rate = (
        count_default_approved/count_approved
        if count_approved > 0
        else float("nan")
    )

    # Default between rejected rate
    count_rejected = (y_df["approved"]==0).sum()
    count_default_rejected = ((y_df["approved"] == 0) & 
                                      (y_df["default"] == 1)).sum()
    default_rejected_rate = (
        count_default_rejected/count_rejected
        if count_rejected > 0
        else float("nan")
    )

    # Defaults captured among rejected customers
    count_defaults = (y_df["default"] == 1).sum()
    count_defaults_rejected = (
        (y_df["default"] == 1) &
        (y_df["approved"] == 0)
    ).sum()

    defaults_captured_rate = (
        float (count_defaults_rejected / count_defaults)
        if count_defaults > 0
        else float("nan")
    )

    # Rejected non default customers rate
    count_non_default = (y_df["default"] == 0).sum()
    count_rejected_non_default = ((y_df["default"] == 0) & (y_df["approved"] == 0)).sum()

    rejected_non_default_rate = float(count_rejected_non_default/count_non_default
                                if count_non_default > 0 else float("nan")
    )

    # Metrics
    metrics = {"threshold" : lim,
               "acceptance_rate" : round(acpt_rate, 4),
               "rejection_rate" : round(rejct_rate, 4),
               "approved_mean_pd" : round(approved_mean_pd,4),
               "default_approved_rate" : round(default_approved_rate,4),
               "default_rejected_rate" : round(default_rejected_rate,4),
               "default_captured_rate" : round(defaults_captured_rate,4),
               "non_default_rejected_rate" : round(rejected_non_default_rate,4)
               }

    return metrics


if __name__ == "__main__":

    thresholds = [0.05, 0.1, 0.15, 0.20, 0.30, 0.4, 0.5]

    X_val = pd.read_csv(X_VAL_DIR, index_col="ID")
    y_val = pd.read_csv(Y_VAL_DIR, index_col="ID").squeeze("columns")

    model_config = MODELS_CALIBRATED["lightgbm"]

    X_val_processed = process_data(model=model_config["type"],
                                   X=X_val)

    model = joblib.load(model_config["path"])
    y_pred = model.predict_proba(X_val_processed)[:,1]

    print("-"*100)
    print("Scenario analysis")
    print("-"*100)

    metrics = []
    for threshold in thresholds:

        metrics.append(scenario_analysis(y_prob=y_pred, y=y_val, lim=threshold))

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(CREDIT_POLICY_DIR / "policy_comparison.csv", index=False)

    print(metrics_df)

    #print("y_pred:")
    #print(y_pred[:10])
