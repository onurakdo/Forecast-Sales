import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import zipfile
import pandas as pd

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from feature_data.create_feature_data import CreateFeatureData

class TestCreateFeatureData(unittest.TestCase):

    @patch('feature_data.create_feature_data.pd.read_csv')
    @patch('feature_data.create_feature_data.json.load')
    def test_getting_data(self, mock_json_load, mock_read_csv):
        # Mock configuration data
        mock_json_load.return_value = {"some_config": "value"}

        # Mock data for sales_train.csv
        mock_sales_train_df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'other_column': ['A', 'B', 'C']
        })

        # Mock data for items.csv
        mock_items_df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'item_name': ['Item1', 'Item2', 'Item3'],
            'item_category_id': [37, 40, 37]
        })

        # Set the return values for read_csv
        mock_read_csv.side_effect = [mock_sales_train_df, mock_items_df]

        # Instantiate the class
        feature_data = CreateFeatureData()

        # Call the method
        result_df = feature_data.getting_data()

        # Expected result
        expected_df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'other_column': ['A', 'B', 'C'],
            'item_category_id_37': [1, 0, 1]
        })

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()