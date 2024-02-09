import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.features.build_features import date_features


# Setup fixture for DataFrame
@pytest.fixture
def test_df():
    df = pd.DataFrame({
        "date": pd.date_range(start="1/1/2022", periods=10, freq="D"),
        "sales": range(10),
    })
    return df


# Setup fixture for mock configuration
@pytest.fixture
def mock_cfg_date():
    mock_cfg = MagicMock()
    mock_cfg.year = True
    mock_cfg.quarter = True
    mock_cfg.month = True
    mock_cfg.week = True
    mock_cfg.day_of_week = True
    mock_cfg.day_of_month = True
    mock_cfg.day_of_year = True
    mock_cfg.is_weekend = True
    mock_cfg.is_month_end = True
    mock_cfg.is_payroll = False
    mock_cfg.payroll_day = 15
    mock_cfg.earthquake_date = "2022-01-05"
    return mock_cfg


def test_date_features(test_df, mock_cfg_date):
    # Call the function with the test DataFrame and mock configuration
    result_df = date_features(test_df, mock_cfg_date)

    # Assertions to verify the added date features
    assert "year" in result_df.columns
