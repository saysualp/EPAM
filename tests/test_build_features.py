import unittest
from unittest.mock import MagicMock
import pandas as pd
from src.features.build_features import date_features


class TestFeatureGeneration(unittest.TestCase):

    def setUp(self):
        # Setup a simple DataFrame to test date_features
        self.test_df = pd.DataFrame({
            "date": pd.date_range(start="1/1/2022", periods=10, freq="D"),
            "sales": range(10),
        })
        # Ensure the date is in the correct format for comparison
        self.test_df["date"] = self.test_df["date"]

        # Setup a mock configuration for date_features
        self.mock_cfg_date = MagicMock()
        self.mock_cfg_date.year = True
        self.mock_cfg_date.quarter = True
        self.mock_cfg_date.month = True
        self.mock_cfg_date.week = True
        self.mock_cfg_date.day_of_week = True
        self.mock_cfg_date.day_of_month = True
        self.mock_cfg_date.day_of_year = True
        self.mock_cfg_date.is_weekend = True
        self.mock_cfg_date.is_month_end = True
        self.mock_cfg_date.is_payroll = False
        self.mock_cfg_date.payroll_day = 15
        self.mock_cfg_date.earthquake_date = "2022-01-05"

    def test_date_features(self):
        # Call the function with the test DataFrame and mock configuration
        result_df = date_features(self.test_df, self.mock_cfg_date)

        # Assertions to verify the added date features
        self.assertIn("year", result_df.columns)


if __name__ == "__main__":
    unittest.main()
