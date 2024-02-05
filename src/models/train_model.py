import logging
import pickle
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from darts import TimeSeries
from darts.models import LightGBMModel

logger = logging.getLogger(__name__)


def train_model(cfg: DictConfig, forecast_horizon: int, model_version: str):
    """
    Trains model
    """
    logger.info(
        f"Starting the training process for model version {model_version}."
    )

    # Construct paths to the input and output directories
    input_dir = Path(cfg.paths.interim_data_path) / model_version
    output_dir = Path(cfg.paths.interim_data_path) / model_version
    model_dir = Path(cfg.paths.model_save_path)

    # Make paths
    input_data_path = input_dir / "train.pkl"
    model_save_path = model_dir / f"model_{model_version}.pkl"
    y_train_path = output_dir / "y_train.pkl"
    future_cov_train_path = output_dir / "future_cov_train.pkl"
    y_holdout_path = output_dir / "y_holdout.pkl"
    future_cov_holdout_path = output_dir / "future_cov_holdout.pkl"

    # Read the dataset
    with open(input_data_path, "rb") as file:
        data = pickle.load(file)

    # Data split
    cutoff_date = data["date"].max() - pd.Timedelta(forecast_horizon, unit="D")

    train = data[data["date"] <= cutoff_date].copy()
    holdout = data[data["date"] > cutoff_date].copy()

    # Model building
    static_cov_cols = list(cfg.train.static_cov_cols)
    future_cov_cols = list(
        set(train.columns)
        - set(static_cov_cols)
        - set(["id", "date", "sales"])
    )

    y_train = TimeSeries.from_group_dataframe(
        train,
        group_cols="id",
        time_col="date",
        static_cols=static_cov_cols,
        value_cols=["sales"],
        fill_missing_dates=True,
        freq="D",
        fillna_value=0,
    )

    future_cov_train = TimeSeries.from_group_dataframe(
        train,
        group_cols="id",
        time_col="date",
        static_cols=static_cov_cols,
        value_cols=future_cov_cols,
        fill_missing_dates=True,
        freq="D",
        fillna_value=0,
    )

    y_holdout = TimeSeries.from_group_dataframe(
        holdout,
        group_cols="id",
        time_col="date",
        static_cols=static_cov_cols,
        value_cols=["sales"],
        fill_missing_dates=True,
        freq="D",
        fillna_value=0,
    )

    future_cov_holdout = TimeSeries.from_group_dataframe(
        holdout,
        group_cols="id",
        time_col="date",
        static_cols=static_cov_cols,
        value_cols=future_cov_cols,
        fill_missing_dates=True,
        freq="D",
        fillna_value=0,
    )

    model = LightGBMModel(
        lags=list(cfg.train.target_lags),
        lags_future_covariates=[0],
        use_static_covariates=True,
        verbose=-1,
    )
    model.fit(series=y_train, future_covariates=future_cov_train)

    # Saving processes
    model.save(str(model_save_path))
    logger.info(f"Model saved to {model_save_path}")

    with open(y_train_path, "wb") as f:
        pickle.dump(y_train, f)
    logger.info(f"Y values for training saved to {y_train_path}")

    with open(future_cov_train_path, "wb") as f:
        pickle.dump(future_cov_train, f)
    logger.info(f"Future training covariates saved to {future_cov_train_path}")

    with open(y_holdout_path, "wb") as f:
        pickle.dump(y_holdout, f)
    logger.info(f"Actual Y values for holdout saved to {y_holdout_path}")

    with open(future_cov_holdout_path, "wb") as f:
        pickle.dump(future_cov_holdout, f)
    logger.info(
        f"Future holdout covariates saved to {future_cov_holdout_path}"
    )

    logger.info(
        f"Model and associated data saved for model version {model_version}."
    )
