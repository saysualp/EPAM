import logging
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def make_dataset(cfg: DictConfig):
    """
    Processes raw and external datasets according to specified configurations.
    """
    logger.info("Making final data set from raw data")

    # Construct paths to the input and output directories
    input_dir = Path(cfg.paths.input_data_path)
    output_dir = Path(cfg.paths.processed_data_path)
    external_dir = Path(cfg.paths.external_data_path)

    # Make paths
    raw_data_path = input_dir / "train.csv"
    stores_path = input_dir / "stores.csv"
    oil_path = external_dir / "oil.csv"
    output_path = output_dir / "train.csv"

    # Read the datasets
    raw_df = pd.read_csv(
        raw_data_path,
        usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
        dtype={
            "store_nbr": "category",
            "family": "category",
            "sales": "float32",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    raw_df["date"] = raw_df.date.dt.to_period("D")

    stores_df = pd.read_csv(
        stores_path,
        dtype={
            "store_nbr": "category",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "category",
        },
    )

    oil_df = pd.read_csv(
        oil_path,
        dtype={
            "dcoilwtico": "float32",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    oil_df["date"] = oil_df.date.dt.to_period("D")

    # Process the datasets
    oil_df.fillna(method=cfg.make_dataset.fillna_method, inplace=True)
    filtered_df = raw_df[raw_df["family"].isin(cfg.make_dataset.filter_family)]

    # Construct the final dataset
    final_df = filtered_df.merge(stores_df, on="store_nbr", how="left")
    final_df = final_df.merge(oil_df, on="date", how="left")
    final_df["dcoilwtico"] = final_df["dcoilwtico"].fillna(
        method=cfg.make_dataset.fillna_method
    )

    # Apply data cleaning based on configuration
    for rule in cfg.make_dataset.remove_unusual_data:
        families = rule.family
        before_date = rule.before_date
        condition = (final_df["family"].isin(families)) & (
            final_df["date"] <= before_date
        )
        final_df = final_df[~(condition)]

    # Create the unique identifier for time series
    group_by = cfg.make_dataset.group_by
    final_df["id"] = final_df[group_by].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    # Encode categorical columns
    label_encoder = LabelEncoder()
    for col in cfg.make_dataset.categorical_cols:
        final_df[col] = label_encoder.fit_transform(final_df[col])

    # Save the processed dataframe
    final_df.to_csv(output_path, index=False)

    logger.info(f"Dataset saved to {output_path}")
