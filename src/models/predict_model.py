import logging
import pickle
from pathlib import Path
from omegaconf import DictConfig
from darts.models import LightGBMModel

logger = logging.getLogger(__name__)


def predict_model(cfg: DictConfig, forecast_horizon: int, model_version: str):
    """
    Creates predictions for spesified forecast horizon
    """
    # Construct paths to the input and output directories
    input_dir = Path(cfg.paths.interim_data_path) / model_version
    output_dir = Path(cfg.paths.interim_data_path) / model_version
    model_dir = Path(cfg.paths.model_save_path)

    # Make paths
    model_path = model_dir / f"model_{model_version}.pkl"
    y_train_path = input_dir / "y_train.pkl"
    future_cov_holdout_path = input_dir / "future_cov_holdout.pkl"
    predictions_path = output_dir / "y_preds.pkl"

    # Load the model
    model = LightGBMModel.load(model_path)

    # Load y_train and future_cov_valid TimeSeries objects
    with open(y_train_path, "rb") as file:
        y_train = pickle.load(file)

    with open(future_cov_holdout_path, "rb") as file:
        future_cov_holdout = pickle.load(file)

    # Make predictions using the loaded model and TimeSeries objects
    y_pred = model.predict(
        n=forecast_horizon,
        series=y_train,
        future_covariates=future_cov_holdout,
    )

    logger.info("Prediction completed.")

    with open(predictions_path, "wb") as f:
        pickle.dump(y_pred, f)

    logger.info(f"Saved to {predictions_path}")
