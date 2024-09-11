import unittest
import pandas as pd
import os
import sys
# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from feature_data.create_feature_data import CreateFeatureData

class TestCreateFeatureData(unittest.TestCase):

    def test_fill_empty_months_where_sale_not_exist(self):
        # Sample input data
        input_df = pd.DataFrame({
            'date_block_num': [0, 1],
            'month': [1, 2],
            'shop_id': [1, 2],
            'item_price': [100, 200],
            'item_cnt_day': [1, 3],
            'item_category_id_37': [1, 0]
        })

        # Expected output data
        expected_df = pd.DataFrame({
            'date_block_num': [0, 0, 1, 1],
            'month': [1, 1, 2, 2],
            'shop_id': [1, 2, 1, 2],
            'item_price': [100, 0, 0, 200],
            'item_cnt_day': [1, 0, 0, 3],
            'item_category_id_37': [1, 0, 0, 0]
        })

        # Instantiate the class
        feature_data = CreateFeatureData()

        # Call the method
        result_df = feature_data.fill_empty_months_where_sale_not_exist(input_df)

        # Correct dtypes for expected DataFrame
        expected_df = expected_df.astype({
            'date_block_num': 'int64',
            'month': 'int64',
            'shop_id': 'int64',
            'item_price': 'float64',
            'item_cnt_day': 'int64',
            'item_category_id_37': 'int64'
        })

        # Correct dtypes for result DataFrame
        result_df = result_df.astype({
            'date_block_num': 'int64',
            'month': 'int64',
            'shop_id': 'int64',
            'item_price': 'float64',
            'item_cnt_day': 'int64',
            'item_category_id_37': 'int64'
        })

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()
