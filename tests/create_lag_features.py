import unittest
from unittest.mock import patch
import pandas as pd
import os
import sys
# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from feature_data.create_feature_data import CreateFeatureData

class TestCreateFeatureData(unittest.TestCase):

    def test_create_lag_features(self):
        # Sample input data
        input_df = pd.DataFrame({
            'date_block_num': [0, 0, 1, 1, 2, 2],
            'shop_id': [1, 2, 1, 2, 1, 2],
            'sales_sum': [10, 20, 30, 40, 50, 60],
            'sales_item_price_mean': [100, 200, 300, 400, 500, 600],
            'item_category_id_37_ratio': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })

        # Create expected output DataFrame
        expected_df = pd.DataFrame({
            'date_block_num': [2, 2],
            'shop_id': [1, 2],
            'sales_sum': [50, 60],
            'sales_item_price_mean': [500, 600],
            'item_category_id_37_ratio': [0.9, 1.0],
            'sales_sum_lag_1': [30, 40],
            'sales_sum_lag_2': [10, 20],
            'sales_item_price_mean_lag_1': [300, 400],
            'sales_item_price_mean_lag_2': [100, 200],
            'item_category_id_37_ratio_lag_1': [0.7, 0.8],
            'item_category_id_37_ratio_lag_2': [0.5, 0.6]
        })

        # Instantiate the class
        feature_data = CreateFeatureData()

        # Manually set the config attribute
        feature_data.config = {'lag_features_list': [1, 2]}

        # Call the method
        result_df = feature_data.create_lag_features(input_df)

        # Correct dtypes for expected DataFrame
        expected_df = expected_df.astype({
            'date_block_num': 'int64',
            'shop_id': 'int64',
            'sales_sum': 'int64',
            'sales_item_price_mean': 'int64',
            'item_category_id_37_ratio': 'float64',
            'sales_sum_lag_1': 'float64',
            'sales_sum_lag_2': 'float64',
            'sales_item_price_mean_lag_1': 'float64',
            'sales_item_price_mean_lag_2': 'float64',
            'item_category_id_37_ratio_lag_1': 'float64',
            'item_category_id_37_ratio_lag_2': 'float64'
        })

        # Correct dtypes for result DataFrame
        result_df = result_df.astype({
            'date_block_num': 'int64',
            'shop_id': 'int64',
            'sales_sum': 'int64',
            'sales_item_price_mean': 'int64',
            'item_category_id_37_ratio': 'float64',
            'sales_sum_lag_1': 'float64',
            'sales_sum_lag_2': 'float64',
            'sales_item_price_mean_lag_1': 'float64',
            'sales_item_price_mean_lag_2': 'float64',
            'item_category_id_37_ratio_lag_1': 'float64',
            'item_category_id_37_ratio_lag_2': 'float64'
        })

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()