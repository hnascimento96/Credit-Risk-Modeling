import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.image as mpimg
import math

from pathlib import Path
from src.config import(
    X_VAL_DIR, Y_VAL_DIR, 
    PREPROCESSOR_ENSEMBLE, PREPROCESSOR_LOG_REG,
    MODEL_LOG_REG_DIR, MODEL_CATBOOST_DIR, MODEL_RANDOM_FOR_DIR,
    MODEL_LIGHTGBM_DIR, MODEL_XGBOOST_DIR,
    MODELS_LIST, CALIBRATION_DIR
)

from src.processing import process_data

def decile_analysis(model_name, y_pred, y_obs, calibrated):

    if calibrated not in (False, True):
        raise ValueError(
            f"Invalid argument {calibrated!r} for 'calibrated'\n Valid values: 'True', 'False'"
        )

    calibration_status = {
        False : "raw",
        True : "calibrated"
    }

    save_path = CALIBRATION_DIR / f"{calibration_status[calibrated]}" / f"{model_name}.png"

    y = {"y_pred" : y_pred, "y_obs" : y_obs}

    y_df = pd.DataFrame(y)

    # Get deciles (drops empty intervals)
    y_df["decile"] = pd.qcut(y_df["y_pred"], 
                            q=10, labels=False, duplicates="drop")

    mean = {}
    mean["y_pred"] = y_df.groupby("decile")["y_pred"].mean()
    mean["y_obs"] = y_df.groupby("decile")["y_obs"].mean()

    # Observed-to-expected ratio
    mean["oe_ratio"] = mean["y_obs"]/mean["y_pred"]

    mean_df = pd.DataFrame(mean)

    plt.figure()
    plt.plot(mean["y_obs"], mean["y_obs"], color="gray", label="Perfect calibration")
    plt.plot(mean["y_obs"], mean["y_pred"], color="red", label="Model curve")
    plt.xlabel("Observed quantile")
    plt.ylabel("Predicted quantile")
    plt.title(f"{model_name} calibration curve ({calibration_status[calibrated]})")
    plt.legend()
    

    plt.savefig(save_path)
    plt.close()

    print(mean_df)



def calibration_analysis(calibrated, model_params):

    X_val_raw = pd.read_csv(X_VAL_DIR, index_col="ID")
    y_val = pd.read_csv(Y_VAL_DIR, index_col="ID").squeeze("columns")

    X_val = process_data(model_params["type"], X_val_raw)
    model = joblib.load(model_params["path"])
    y_val_pred = model.predict_proba(X_val)[:,1]


    print("-"*100)
    print("Statistics")
    print(f"model_name: {model_params["name"]}")
    decile_analysis(model_name=model_params["name"], y_obs=y_val, y_pred=y_val_pred,
                    calibrated=calibrated)

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

if __name__ == "__main__":

    for model in MODELS_LIST:
        calibration_analysis(calibrated=False, model_params=model)


    combine_figures(input_dir=CALIBRATION_DIR / "raw",
                    output_path=CALIBRATION_DIR / "raw_merged")
    plt.show()