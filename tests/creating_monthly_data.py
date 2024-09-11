import unittest
from unittest.mock import patch
import pandas as pd
import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from feature_data.create_feature_data import CreateFeatureData

class TestCreateFeatureData(unittest.TestCase):

    def test_creating_monthly_data(self):
        # Sample input data
        input_df = pd.DataFrame({
            'date': ['01.01.2023', '15.01.2023', '22.02.2023', '05.03.2023'],
            'date_block_num': [0, 0, 1, 1],
            'shop_id': [1, 1, 1, 2],
            'item_price': [100, 150, 200, 250],
            'item_cnt_day': [1, 2, 3, 4],
            'item_category_id_37': [1, 0, 1, 1]
        })

        # Expected output data
        expected_df = pd.DataFrame({
            'date_block_num': [0, 1, 1],
            'month': [1, 2, 3],
            'shop_id': [1, 1, 2],
            'sales_item_price_mean': [125, 200, 250],
            'sales_sum': [3, 3, 4],
            'item_category_id_37_ratio': [0.5, 1.0, 1.0]
        })

        expected_df = expected_df.astype({
            'date_block_num': 'int64',
            'month': 'int32',
            'shop_id': 'int64',
            'sales_item_price_mean': 'float64',
            'sales_sum': 'int64',
            'item_category_id_37_ratio': 'float64'
        })

        # Instantiate the class
        feature_data = CreateFeatureData()

        # Call the method
        result_df = feature_data.creating_monthly_data(input_df)

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()
