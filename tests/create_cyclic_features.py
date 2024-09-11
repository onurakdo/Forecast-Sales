
import unittest
import pandas as pd
import numpy as np
import os
import sys
# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from feature_data.create_feature_data import CreateFeatureData

class TestCreateFeatureData(unittest.TestCase):

    def test_create_cyclic_features(self):
        # Sample input data
        input_df = pd.DataFrame({
            'month': [1, 2, 3, 6, 12]  # Example months
        })

        # Expected output data
        expected_df = pd.DataFrame({
            'month': [1, 2, 3, 6, 12],
            'month_rad': [
                2 * np.pi * (1 - 1) / 12,  # 0
                2 * np.pi * (2 - 1) / 12,  # np.pi / 6
                2 * np.pi * (3 - 1) / 12,  # np.pi / 3
                2 * np.pi * (6 - 1) / 12,  # np.pi / 2
                2 * np.pi * (12 - 1) / 12  # 2 * np.pi
            ],
            'month_sin': [
                np.sin(2 * np.pi * (1 - 1) / 12),  # 0
                np.sin(2 * np.pi * (2 - 1) / 12),  # 0.5
                np.sin(2 * np.pi * (3 - 1) / 12),  # 0.86602540378
                np.sin(2 * np.pi * (6 - 1) / 12),  # 1
                np.sin(2 * np.pi * (12 - 1) / 12)  # 0
            ],
            'month_cos': [
                np.cos(2 * np.pi * (1 - 1) / 12),  # 1
                np.cos(2 * np.pi * (2 - 1) / 12),  # np.sqrt(3)/2
                np.cos(2 * np.pi * (3 - 1) / 12),  # 0.5
                np.cos(2 * np.pi * (6 - 1) / 12),  # 0
                np.cos(2 * np.pi * (12 - 1) / 12)  # 1
            ]
        })
        # Instantiate the class
        feature_data = CreateFeatureData()

        # Call the method
        result_df = feature_data.create_cyclic_features(input_df)

        # Correct dtypes for expected DataFrame
        expected_df = expected_df.astype({
            'month': 'int64',
            'month_rad': 'float64',
            'month_sin': 'float64',
            'month_cos': 'float64'
        })

        # Correct dtypes for result DataFrame
        result_df = result_df.astype({
            'month': 'int64',
            'month_rad': 'float64',
            'month_sin': 'float64',
            'month_cos': 'float64'
        })

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
