import pandas as pd
import numpy as np

def scenario_analysis(y_prob, y, lim):
    """
    Simulates a credit policy that approves a customer when the predicted
    PD is less than or equal to a threshold, and computes its metrics.

    Parameters
    ----------
    y_prob : numpy.ndarray
        Predicted probabilities of default (calibrated PDs).
    y : pandas.Series
        Observed target (1 = default), in the same row order as y_prob.
    lim : float
        PD threshold: customers with y_prob <= lim are approved.

    Returns
    -------
    dict
        Policy metrics, rounded to 4 decimals:
        - threshold: the PD threshold (lim);
        - acceptance_rate / rejection_rate: share of customers approved / rejected;
        - approved_mean_pd: mean predicted PD among approved (expected risk);
        - default_approved_rate: actual default rate among approved (realized risk);
        - default_rejected_rate: actual default rate among rejected;
        - default_captured_rate: share of all defaults that were rejected;
        - non_default_rejected_rate: share of good payers that were rejected.
    """

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