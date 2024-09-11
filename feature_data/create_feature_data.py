from unicodedata import category

import pandas as pd
import numpy as np
import json
import os

class CreateFeatureData:
    """
    A class to create and process features for machine learning models.

    Attributes:
        raw_data_path (str): Path to the raw data directory.
        feature_data_path (str): Path to the feature data directory.
        config (dict): Configuration settings loaded from a JSON file.
    """

    def __init__(self, raw_data_path='./raw_data', feature_data_path='./feature_data', config='config.json'):

        self.raw_data_path = raw_data_path
        self.feature_data_path = feature_data_path

        # Go to the main folder (parent directory of the current file's directory)
        main_folder = os.path.dirname(os.path.dirname(__file__))
        # Join the main folder path with the config file name
        config_path = os.path.join(main_folder, config)

        # Load configuration settings from a JSON file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def getting_data(self):

        """
        Loads and merges the sales and item data, and preprocesses it by dropping unnecessary columns
        and creating a binary feature based on item_category_id.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with merged sales and item data.

        item_categories.csv and shops.csv contain the names of each item and shop. 
        Although these names could be useful for extracting text-based information, 
        they are not utilized in the scope of the project to keep it manageable.
        
        items_df contains the category ID, which can be useful as a feature. 
        Therefore, items_df has been merged with sales_train_df.
        
        test_csv and sample_submission.csv are ignored in the analysis
        because their outputs will not be used for the Kaggle competition.
        """
        sales_train_df = pd.read_csv(self.raw_data_path + '/sales_train.csv')
        items_df = pd.read_csv(self.raw_data_path + '/items.csv')

        # Merge sales and item data on item_id
        sales_df = sales_train_df.merge(items_df, how='left', on='item_id')
        sales_df.drop(labels="item_name", axis=1, inplace=True)

        # item_category_id only contains 37 & 40 values, so creating a binary value is suitable
        sales_df["item_category_id_37"] = np.where(sales_df["item_category_id"] == 37, 1, 0)
        sales_df.drop(labels=['item_category_id'], axis=1, inplace=True)

        return sales_df

    def creating_monthly_data(self, df):
        """
        Groups the data by month and aggregates relevant features.

        Args:
            df (pd.DataFrame): The input DataFrame, which is daily sales data

        Returns:
            pd.DataFrame: Aggregated DataFrame with monthly data.
        """

        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        df['month']= df['date'].dt.month

        # Perform the group by and aggregation
        monthly_sales = df.groupby(['date_block_num', 'month',
                                    'shop_id']).agg({'item_price': 'mean',
                                                     'item_cnt_day':'sum',
                                                     'item_category_id_37':'mean'}).reset_index()

        # Rename columns for clarity
        monthly_sales.rename(columns={'item_price': 'sales_item_price_mean',
                                      'item_cnt_day': 'sales_sum',
                                      'item_category_id_37':'item_category_id_37_ratio'}, inplace=True)

        return monthly_sales

    def fill_empty_months_where_sale_not_exist(self, df):
        """
        Fills missing sales data by generating rows for months when no sales occurred.

        Args:
            df (pd.DataFrame): The input monthly sales dataFrame of shops with missing sales data

        Returns:
            pd.DataFrame: DataFrame with missing sales data filled with zeros.
        """

        # Extract unique date and shop values
        date_values = df[['date_block_num', 'month']].drop_duplicates()
        shop_values = df[['shop_id']].drop_duplicates()

        # Create a cross join (Cartesian product) between date_values and shop_values
        cross_df = date_values.merge(shop_values, how='cross')

        # Merge the cross product with the original DataFrame
        whole_df = cross_df.merge(df, on=['date_block_num', 'month', 'shop_id'], how='left')
        whole_df.fillna(0, inplace=True)

        return whole_df

    def create_lag_features(self, df):
        """
        Creates lag features for selected columns based on the configuration settings.

        Args:
            df (pd.DataFrame): The input DataFrame which is monthly sales dataframe

        Returns:
            pd.DataFrame: DataFrame with lag features added.
        """

        df = df.sort_values(by=['shop_id', 'date_block_num']).reset_index(drop=True)
        to_be_created_lag_features = ['sales_sum', 'sales_item_price_mean',
                                      'item_category_id_37_ratio']

        for col_name in to_be_created_lag_features:
            for lag_value in self.config["lag_features_list"]:

                # Create lag features for price
                df[f'{col_name}_lag_{lag_value}'] = df.groupby('shop_id')[col_name].shift(lag_value)

        # Since we have created lag features, we need to exclude the initial months.
        # Identify the values to drop and drop initial months to account for lag feature creation
        drop_threshold = max(self.config["lag_features_list"])
        df = df[df['date_block_num'] >= drop_threshold].reset_index(drop=True)

        return df

    def create_cyclic_features(self, df):
        """
         Converts the month into a cyclic feature using sine and cosine transformations.

         Args:
             df (pd.DataFrame): The input DataFrame containing monthly sales data

         Returns:
             pd.DataFrame: DataFrame with cyclic month features added.
         """

        # Convert month values (1-12) to radians
        df['month_rad'] = 2 * np.pi * (df['month'] - 1) / 12

        # Calculate sine and cosine
        df['month_sin'] = np.sin(df['month_rad'])
        df['month_cos'] = np.cos(df['month_rad'])

        return df

    def drop_irrelevant_features(self, df):
        """
        Drops columns that are no longer relevant for modeling.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with irrelevant features dropped.
        """

        df.drop(labels=['month', 'month_rad', 'date_block_num',
                        'shop_id', 'sales_item_price_mean'], axis=1, inplace=True)

        # Drop all columns related to item_category_id_37
        # They were created as variables in the modeling process, but
        # after generating the initial models, they were removed by returning to the process step because they caused noise."
        category_id_cols = df.filter(like="category_id_37").columns
        df.drop(labels=category_id_cols, axis=1, inplace=True)

        return df

    def process_features(self, df):
        """
        Processes the final feature set, including renaming and handling NaN values.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame ready for modeling.
        """

        # Fill NaN values, which are created after the lag operation, with 0
        df.fillna(0, inplace=True)

        # Changing item_cnt_month column name as target
        df.rename(columns={'sales_sum': 'target'}, inplace=True)

        return df

    def create_train_and_test_data(self, df):
        """
        Splits the data into training and testing sets and saves them as CSV files.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            tuple: Two DataFrames, one for training and one for testing.
        """

        # Split the DataFrame into training and testing sets
        test_df = df.sample(frac=self.config["test_train_split_ratio"])
        train_df = df.drop(test_df.index)

        # Reset indices
        test_df.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)

        # Save the train and test DataFrames
        test_df.to_csv(self.feature_data_path + '/test_df.csv', index=False)
        train_df.to_csv(self.feature_data_path + '/train_df.csv', index=False)

        return train_df, test_df
