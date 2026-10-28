import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.image as mpimg
import math

from pathlib import Path
from src.config import(
    X_VAL_DIR, Y_VAL_DIR,
    X_TEST_DIR, Y_TEST_DIR,
    PREPROCESSOR_ENSEMBLE, PREPROCESSOR_LOG_REG,
    MODEL_LOG_REG_DIR, MODEL_CATBOOST_DIR, MODEL_RANDOM_FOR_DIR,
    MODEL_LIGHTGBM_DIR, MODEL_XGBOOST_DIR,
    MODELS, CALIBRATION_DIR
)

from src.processing import process_data
from src.evaluation import evaluate_model

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

def decile_analysis(model_name, y_pred, y_obs, calibrated):

    if calibrated not in (False, True):
        raise ValueError(
            f"Invalid argument {calibrated!r} for 'calibrated'\n Valid values: 'True', 'False'"
        )

    calibration_status = {
        False : "raw",
        True : "calibrated"
    }

    save_path = CALIBRATION_DIR / f"fig/{calibration_status[calibrated]}" / f"{model_name}.png"

    y = {"y_pred" : y_pred, "y_obs" : y_obs}

    y_df = pd.DataFrame(y)

    # Get deciles (drops empty intervals)

    y_df["decile"] = pd.qcut(y_df["y_pred"], 
                            q=10, labels=False, duplicates="drop")

    mean = {}
    mean["y_pred"] = y_df.groupby("decile")["y_pred"].mean()
    mean["y_obs"] = y_df.groupby("decile")["y_obs"].mean()

    deciles = np.arange(len(mean["y_pred"]))

    # Observed-to-predicted ratio
    mean["op_ratio"] = mean["y_obs"]/mean["y_pred"]

    mean_df = pd.DataFrame(mean)

    # plt.figure()
    # plt.plot(mean["y_obs"], mean["y_obs"], color="gray", label="Perfect calibration")
    # plt.plot(mean["y_obs"], mean["y_pred"], color="red", label="Model curve")
    # plt.xlabel("Observed quantile")
    # plt.ylabel("Predicted quantile")
    # plt.title(f"{model_name} calibration curve ({calibration_status[calibrated]})")
    # plt.legend()

    # Plots bar mean predicted values bar graph 
    plt.figure()
    plt.bar(deciles, mean["y_pred"]*100, label="Predicted default rate")
    plt.xticks(deciles)
    plt.xlabel("PD deciles")
    plt.ylabel("Default rate (%)")

    # Plots observed rate graph
    plt.plot(deciles, mean["y_obs"]*100, color="firebrick", marker="o",
             markersize=9,
             label="Observed rate"
             )

    plt.title(f"{model_name} default rate by decile ({calibration_status[calibrated]})")
    
    plt.legend()
    plt.savefig(save_path)
    

    #print(mean_df)

def evaluate_calibration(model, model_name, X, y, calibrated=False):
    """
    Evaluates a trained model using predicted probabilities and performs
    decile analysis.

    Parameters
    ----------
    model_config : dict
        Dictionary containing the model's name, type, and file path.
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Observed target values.
    calibrated : bool, default=False
        Indicates whether the model is calibrated.

    Returns
    -------
    None
    """
    y_pred = model.predict_proba(X)[:, 1]

    # print("-" * 100)
    # print("Statistics")
    # print(f"model_name: {model_name}")

    decile_analysis(
        model_name=model_name,
        y_obs=y,
        y_pred=y_pred,
        calibrated=calibrated
    )


def combine_figures(input_dir: str | Path, output_path: str | Path, n_cols: int = 3) -> Path:
    """Combine all PNG images in a folder into a single grid figure."""
    image_paths = sorted(Path(input_dir).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    n_cols = min(n_cols, len(image_paths))
    n_rows = math.ceil(len(image_paths) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)

    for ax, image_path in zip(axes.flat, image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.stem, fontsize=10)
        ax.axis("off")

    # Hide empty cells when the grid is larger than the number of images
    for ax in axes.flat[len(image_paths):]:
        ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return output_path

def calibrate_model(model, X, y):

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid"
    )

    calibrated_model.fit(X, y)

    joblib.dump(calibrated_model, CALIBRATION_DIR / f"calibrated_model_{model_config["name"]}.joblib")

    return calibrated_model

if __name__ == "__main__":

    X_val = pd.read_csv(X_VAL_DIR, index_col="ID")
    y_val = pd.read_csv(Y_VAL_DIR, index_col="ID").squeeze("columns")

    X_test = pd.read_csv(X_TEST_DIR, index_col="ID")
    y_test = pd.read_csv(Y_TEST_DIR, index_col="ID").squeeze("columns") 


    # Evaluate 'raw' models on validation set
    results = []

    print("-" * 100)
    print("Uncalibrated model analysis (validation set)")
    for model_config in MODELS.values():
        X_val_processed = process_data(model=model_config["type"], X=X_val)

        model = joblib.load(model_config["path"])

        evaluate_calibration(model=model,
                             model_name=model_config["name"],
                             X=X_val_processed,
                             y=y_val,
                             calibrated=False
                             )
        result = evaluate_model(model=model,
                               X=X_val_processed,
                               y=y_val,
                               model_name=model_config["name"]
                               )
        
        results.append(result)

    print("-" * 100)
    print(pd.DataFrame(results))

    # Evaluate uncalibrated models on test set
    results = []

    print("-" * 100)
    for model_config in MODELS.values():
        X_test_processed = process_data(model=model_config["type"], X=X_test)

        model = joblib.load(model_config["path"])

        evaluate_calibration(model=model,
                                model_name=model_config["name"],
                                X=X_test_processed,
                                y=y_test,
                                calibrated=False
                                )
        result = evaluate_model(model=model,
                                X=X_test_processed,
                                y=y_test,
                                model_name=model_config["name"]
                                )
        
        results.append(result)

    print(f"Uncalibrated model metrics (test set)")
    print("-" * 100)
    print(pd.DataFrame(results))
        

    # Evaluates 'calibrated' models on test set
    results = []

    for model_config in MODELS.values():
        X_val_processed = process_data(model=model_config["type"], X=X_val)

        model = joblib.load(model_config["path"])
        calibrated_model = calibrate_model(model=model, X=X_val_processed, y=y_val)

        X_test_processed = process_data(model=model_config["type"], X=X_test)
        
        result = evaluate_model(model=calibrated_model,
                       X=X_test_processed,
                       y=y_test,
                       model_name=model_config["name"]
                       )

        evaluate_calibration(model=calibrated_model,
                            model_name=model_config["name"],
                            X=X_test_processed,
                            y=y_test,
                            calibrated=True
                            )

        results.append(result)

        y_pred = calibrated_model.predict_proba(X_test_processed)[:, 1]


    print("-" * 100)
    print(f"Calibrated model metrics (test set)")
    print("-" * 100)
    print(pd.DataFrame(results))

    combine_figures(input_dir=CALIBRATION_DIR / "fig/raw",
                    output_path=CALIBRATION_DIR / "fig/raw_merged")

    combine_figures(input_dir=CALIBRATION_DIR / "fig/calibrated",
                    output_path=CALIBRATION_DIR / "fig/calibrated_merged")
    #plt.show()