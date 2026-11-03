import shap


def explain_client(client_id, shap_values, X, pd_calibrated, y):
    """
    Shows the local explanation of one client: calibrated PD, actual outcome and waterfall plot.

    Parameters
    ----------
    client_id : ID of the client to explain.
    shap_values : SHAP Explanation object (2D: clients x features).
    X : DataFrame with the features used to compute shap_values (same row order).
    pd_calibrated : Series with the calibrated PD of each client, indexed by client ID.
    y : Series with the actual outcome (1 = default, 0 = no default), indexed by client ID.
    """
    # shap_values is indexed by position (0, 1, 2, ...), not by client ID,
    # so we find the client's position in X.
    position = X.index.get_loc(client_id)

    print(f"Client ID: {client_id}")
    print(f"Calibrated PD: {pd_calibrated[client_id]:.2%}")
    print(f"Actual outcome: {'default' if y[client_id] == 1 else 'no default'}")

    # max_display: number of features shown; the rest are grouped in one bar.
    shap.plots.waterfall(shap_values[position], max_display=10)