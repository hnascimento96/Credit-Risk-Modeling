"""
Figures for the README.

Builds the README figures from the saved artifacts (metrics CSVs, policy
table and trained models) and saves them in reports/figures/:

  - auc_v1_vs_v2.png: test ROC-AUC of each model, v1 vs v2;
  - calibration_deciles.png: predicted vs observed default rate by decile
    (calibrated Logistic Regression, test set);
  - credit_policy_tradeoff.png: approval and risk for each PD threshold;
  - feature_importance.png: share of the mean |SHAP| of each feature,
    Logistic Regression vs LightGBM.

Nothing is trained or calibrated here. Run after the other scripts:
    python -m src.report_figures
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    PROJECT_ROOT, METRICS_DIR, CREDIT_POLICY_DIR,
    X_VAL_DIR, Y_VAL_DIR, X_TEST_DIR, Y_TEST_DIR,
    MODELS, MODELS_CALIBRATED,
)
from src.processing import process_data

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Same colors in every figure: the chosen model and the challenger
COLOR_LOG_REG = "#2a78d6"     # blue
COLOR_LIGHTGBM = "#eb6834"    # orange
COLOR_V1 = "#a3a19b"          # gray: previous version, de-emphasized
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#e4e2dc"

# Delay features summed as one group in the importance figure: in the
# Logistic Regression, TotalLate and LateScore are exact linear combinations
# of the three delay counts, so only the group total is well defined
DELAY_GROUP = ["AnyLate", "LateScore", "TotalLate", "Late30_59", "Late60_89", "Late90"]

# Same SHAP sample as notebooks/03_shap_analysis.ipynb
SHAP_SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def set_style():
    """Applies a clean, light style shared by all figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLOR_GRID,
        "axes.labelcolor": COLOR_MUTED,
        "axes.titlecolor": COLOR_TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "legend.frameon": False,
        "font.size": 10,
    })


def save(fig, name):
    """
    Saves a figure in FIGURES_DIR and closes it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name (e.g. 'auc_v1_vs_v2.png').
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: reports/figures/{name}")


def plot_auc_v1_vs_v2():
    """
    Dot plot of the test ROC-AUC of each model, v1 vs v2.

    A dot plot is used instead of bars because the differences are small:
    bars would need an axis starting far from zero, which exaggerates them.
    """
    v1 = pd.read_csv(METRICS_DIR / "v1" / "uncalibrated_test_metrics.csv").set_index("Model")
    v2 = pd.read_csv(METRICS_DIR / "uncalibrated_test_metrics.csv").set_index("Model")
    models = list(v2.index)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(models))[::-1]

    for yi, model in zip(y, models):
        auc_v1, auc_v2 = v1.loc[model, "ROC-AUC"], v2.loc[model, "ROC-AUC"]
        color = COLOR_LOG_REG if model == "Logistic Regression" else COLOR_LIGHTGBM
        ax.plot([auc_v1, auc_v2], [yi, yi], color=COLOR_GRID, linewidth=2, zorder=1)
        ax.scatter(auc_v1, yi, s=60, color=COLOR_V1, zorder=2,
                   label="v1" if yi == y[0] else None)
        ax.scatter(auc_v2, yi, s=60, color=color, zorder=3)
        # Label to the right of both dots, so it never covers the v1 dot
        ax.annotate(f"{auc_v2:.3f}", (max(auc_v1, auc_v2), yi), xytext=(10, 0),
                    textcoords="offset points", va="center", color=COLOR_TEXT)

    # Legend entries for v2 (chosen model and ensembles)
    ax.scatter([], [], s=60, color=COLOR_LOG_REG, label="v2 – Logistic Regression")
    ax.scatter([], [], s=60, color=COLOR_LIGHTGBM, label="v2 – ensembles")

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC (test set)")
    ax.set_title("EDA treatments lift the Logistic Regression, not the ensembles")
    # Empty area of the plot: ensembles are on the right, the gap is on the left
    ax.legend(loc="lower left")
    # Room on the right for the value labels
    ax.set_xlim(right=ax.get_xlim()[1] + 0.004)
    save(fig, "auc_v1_vs_v2.png")


def plot_calibration_deciles():
    """
    Predicted vs observed default rate by PD decile for the calibrated
    Logistic Regression on the test set.
    """
    X_test = pd.read_csv(X_TEST_DIR, index_col="ID")
    y_test = pd.read_csv(Y_TEST_DIR, index_col="ID").squeeze("columns")

    config = MODELS_CALIBRATED["log_reg"]
    model = joblib.load(config["path"])
    pd_pred = model.predict_proba(process_data(config["type"], X_test))[:, 1]

    deciles = pd.DataFrame({"pd": pd_pred, "default": y_test.values})
    deciles["decile"] = pd.qcut(deciles["pd"], q=10, labels=False, duplicates="drop") + 1
    table = deciles.groupby("decile").agg(predicted=("pd", "mean"), observed=("default", "mean")) * 100

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(table.index, table["predicted"], width=0.6, color=COLOR_LOG_REG, label="Predicted PD (mean)")
    ax.scatter(table.index, table["observed"], s=50, color=COLOR_TEXT, zorder=3, label="Observed default rate")

    ax.set_xticks(table.index)
    ax.set_xlabel("PD decile (1 = lowest risk)")
    ax.set_ylabel("Default rate (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Calibrated Logistic Regression: predicted PD matches observed defaults")
    ax.legend(loc="upper left")
    save(fig, "calibration_deciles.png")


def plot_credit_policy():
    """
    Two panels for each PD threshold: approval and defaults blocked (left),
    default rate among approved customers (right). Two panels instead of
    one chart with two y-axes, since the rates have very different scales.
    """
    policy = pd.read_csv(CREDIT_POLICY_DIR / "policy_comparison.csv")
    y_test = pd.read_csv(Y_TEST_DIR, index_col="ID").squeeze("columns")
    threshold = policy["threshold"] * 100

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    left.plot(threshold, policy["acceptance_rate"] * 100, marker="o", linewidth=2,
              color=COLOR_LOG_REG, label="Customers approved")
    left.plot(threshold, policy["default_captured_rate"] * 100, marker="o", linewidth=2,
              color=COLOR_LIGHTGBM, label="Defaults blocked")
    left.set_ylim(0, 100)
    left.set_ylabel("%")
    left.set_title("Approval vs defaults blocked")
    left.legend(loc="center right")

    right.plot(threshold, policy["default_approved_rate"] * 100, marker="o", linewidth=2,
               color=COLOR_LOG_REG, label="Default rate among approved")
    right.axhline(y_test.mean() * 100, color=COLOR_MUTED, linestyle="--", linewidth=1)
    right.annotate(f"Overall default rate ({y_test.mean():.1%})", (threshold.max(), y_test.mean() * 100),
                   xytext=(0, 6), textcoords="offset points", ha="right", color=COLOR_MUTED)
    right.set_ylim(0, y_test.mean() * 100 * 1.25)
    right.set_ylabel("%")
    right.set_title("Risk of the approved portfolio")
    right.legend(loc="lower right")

    for ax in (left, right):
        # Recommended range of thresholds (see notebooks/04_credit_policy.ipynb)
        ax.axvspan(10, 15, color=COLOR_GRID, alpha=0.6, zorder=0)
        ax.set_xlabel("PD threshold (%)")
        ax.set_xticks(threshold)

    left.annotate("recommended", (12.5, 4), ha="center", color=COLOR_MUTED)
    save(fig, "credit_policy_tradeoff.png")


def importance_share(contributions):
    """
    Share (%) of the total mean |SHAP| of each feature, with the delay
    features summed as one group.

    Parameters
    ----------
    contributions : pandas.DataFrame
        SHAP values (log-odds), one row per customer and one column per feature.

    Returns
    -------
    pandas.Series
        Share of each feature (or group), in %.
    """
    values = contributions.copy()
    values["Delay history (group)"] = values[DELAY_GROUP].sum(axis=1)
    values = values.drop(columns=DELAY_GROUP)
    mean_abs = values.abs().mean()
    return mean_abs / mean_abs.sum() * 100


def plot_feature_importance():
    """
    Share of the mean |SHAP| of each feature in the Logistic Regression and
    in LightGBM, on the same validation sample as the SHAP notebook.

    - Logistic Regression: exact SHAP values of a linear model on standardized
      features (coefficient x standardized value).
    - LightGBM: exact tree SHAP values from LightGBM itself
      (predict with pred_contrib=True), the same algorithm used by the
      shap library for tree models.
    """
    X_val = pd.read_csv(X_VAL_DIR, index_col="ID")
    y_val = pd.read_csv(Y_VAL_DIR, index_col="ID").squeeze("columns")

    # Same stratified sample as the SHAP notebook
    X_val_ensemble = process_data(model="ensemble", X=X_val)
    X_sample, _, _, _ = train_test_split(
        X_val_ensemble, y_val, train_size=SHAP_SAMPLE_SIZE,
        stratify=y_val, random_state=RANDOM_STATE,
    )

    # LightGBM: last column of pred_contrib is the base value, so it is dropped
    lightgbm = joblib.load(MODELS["lightgbm"]["path"])
    lgbm_contrib = lightgbm.predict(X_sample, pred_contrib=True)[:, :-1]
    lgbm_shap = pd.DataFrame(lgbm_contrib, columns=X_sample.columns)

    # Logistic Regression: coefficient x standardized value
    log_reg = joblib.load(MODELS["log_reg"]["path"])
    X_sample_lr = process_data(model="log_reg", X=X_val).loc[X_sample.index]
    lr_shap = pd.DataFrame(X_sample_lr.values * log_reg.coef_[0], columns=X_sample_lr.columns)

    shares = pd.DataFrame({
        "Logistic Regression": importance_share(lr_shap),
        "LightGBM": importance_share(lgbm_shap),
    }).sort_values("LightGBM")
    print(shares.sort_values("LightGBM", ascending=False).round(1))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    y = np.arange(len(shares))
    height = 0.38
    ax.barh(y + height / 2, shares["Logistic Regression"], height=height,
            color=COLOR_LOG_REG, label="Logistic Regression (chosen)")
    ax.barh(y - height / 2, shares["LightGBM"], height=height,
            color=COLOR_LIGHTGBM, label="LightGBM (challenger)")

    ax.set_yticks(y)
    ax.set_yticklabels(shares.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.set_title("Both models rely on the same features, except Debt")
    ax.legend(loc="lower right")
    save(fig, "feature_importance.png")


if __name__ == "__main__":
    set_style()
    plot_auc_v1_vs_v2()
    plot_calibration_deciles()
    plot_credit_policy()
    plot_feature_importance()
