import logging
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from tsfresh import extract_features
from datetime import datetime
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.timeseries.forecasting import WindowFeatures

logger = logging.getLogger(__name__)

def date_features(data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Adds date-related features to the given DataFrame based on the specified configuration.
    """
    if cfg.year:
        data['year'] = data['date'].dt.year
    if cfg.quarter:
        data['quarter'] = data['date'].dt.quarter
    if cfg.month:
        data['month'] = data['date'].dt.month
    if cfg.week:
        data['week'] = data['date'].dt.isocalendar().week
    if cfg.day_of_week:
        data['day_of_week'] = data['date'].dt.dayofweek
    if cfg.day_of_month:
        data['day_of_month'] = data['date'].dt.day
    if cfg.day_of_year:
        data['day_of_year'] = data['date'].dt.dayofyear
    if cfg.is_weekend:
        data['is_weekend'] = data['date'].dt.dayofweek >= 5
    if cfg.is_month_end:
        data['is_month_end'] = data['date'].dt.is_month_end
    if cfg.is_payroll:
        data['is_payroll'] = data['day_of_month'] == cfg.payroll_day

    earthquake_date = pd.to_datetime(cfg.earthquake_date) if cfg.earthquake_date else None
    if earthquake_date:
        start_of_week = earthquake_date - pd.Timedelta(days=earthquake_date.weekday())
        end_of_week = start_of_week + pd.Timedelta(days=6)
        data['is_earthquake'] = data['date'].between(start_of_week, end_of_week)

    # Convert boolean columns to integers for certain features
    for col in ['is_weekend', 'is_month_end', 'is_payroll', 'is_earthquake']:
        if col in data.columns:
            data[col] = data[col].astype(int)

    return data

def statistical_features(data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Creates statistical features beneficial for time series forecasting
    """
    feature_extraction_settings = {}
    
    for feature, settings in cfg.sales.items():
        if isinstance(settings, dict):
            feature_extraction_settings[feature] = {k: None for k in settings if settings[k]}
        elif isinstance(settings, list):
            feature_extraction_settings[feature] = settings
    
    statistical_features = extract_features(data[['id', 'date', 'sales']], 
                                            column_id='id', 
                                            column_sort='date', 
                                            kind_to_fc_parameters=feature_extraction_settings).reset_index()
    statistical_features = statistical_features.rename(columns={'index': 'id'})
    
    return statistical_features

def lag_features(data: pd.DataFrame, cfg: DictConfig, forecast_horizon: int) -> pd.DataFrame:
    """
    Creates lag features to the DataFrame based on the specified configuration.
    """
    pre_lags = cfg.lags

    # Calculate lags based on the forecast horizon and specified nonrecursive lags
    lags = [forecast_horizon + lag for lag in pre_lags]

    # Initialize the LagFeatures transformer
    lag_it = LagFeatures(periods=lags, drop_original=True)
    transformed_groups = []

    # Group the data by 'id' and apply lag transformation to each group
    for _, group in data.groupby('id'):
        transformed_group = lag_it.fit_transform(group[['sales']])
        transformed_groups.append(transformed_group)

    # Concatenate the transformed groups to form the final DataFrame with lagged features
    lagged_features = pd.concat(transformed_groups)
    
    return lagged_features

def window_features(data: pd.DataFrame, cfg: DictConfig, forecast_horizon: int) -> pd.DataFrame:
    """
    Creates window-based features (mean, max, std) to the DataFrame based on the specified configuration.
    """
    pre_windows = cfg.windows
    functions = list(cfg.functions)

    # Calculate windows based on the forecast horizon and specified rolling windows
    windows = [forecast_horizon + window for window in pre_windows]

    # Initialize the WindowFeatures transformer
    roll_it = WindowFeatures(window=windows, functions=functions, drop_original=True)
    transformed_groups = []

    # Group the data by 'id' and apply window transformation to each group
    for _, group in data.groupby('id'):
        transformed_group = roll_it.fit_transform(group[['sales']])
        transformed_groups.append(transformed_group)

    # Concatenate the transformed groups to form the final DataFrame with window features
    window_features = pd.concat(transformed_groups)
    
    return window_features

def build_features(cfg: DictConfig, forecast_horizon: int):
    """
    Orchestrates feature generation based on configuration.
    """
    logger.info("Starting feature generation process.")

    # Calculate the model version
    model_version = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Define input and output directories
    input_dir = Path(cfg.paths.processed_data_path)
    output_dir = Path(cfg.paths.interim_data_path) / model_version

    # Ensure the output directory exists
    if not output_dir.exists():
        logger.info(f"Creating directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Make paths
    input_data_path = input_dir / 'train.csv'
    processed_data_path = output_dir / "train.pkl"
    
    # Read the dataset
    data = pd.read_csv(input_data_path, parse_dates=['date'], infer_datetime_format=True)

    # Apply feature generation steps
    logger.info("Creating date features.")
    data = date_features(data, cfg.build_features.date_features)

    logger.info("Creating stastical features.")
    statistical_fea = statistical_features(data, cfg.build_features.statistical_features)

    logger.info("Creating lagged features.")
    lagged_fea = lag_features(data, cfg.build_features.lag_features, forecast_horizon)
        
    logger.info("Creating window features.")
    window_fea = window_features(data, cfg.build_features.window_features, forecast_horizon)

    # Merge the features back into the main DataFrame
    data = data.merge(statistical_fea, on='id', how='left')
    data = data.merge(lagged_fea, left_index=True, right_index=True,  how='left')
    data = data.merge(window_fea, left_index=True, right_index=True, how='left')

    # Drop rows with missing values that may have been introduced during feature generation
    data.dropna(inplace=True)

    # Save the processed data
    data.to_pickle(processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")

    logger.info("Feature generation complete.")

    return model_version
