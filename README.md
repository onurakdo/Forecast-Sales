# Forecast-Sales

## Overview

The **Forecast-Sales** project is a data science initiative aimed at predicting total sales for every store for the next month. This project utilizes a time-series dataset from the [Kaggle competition "Predict Future Sales"](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview), which serves as the final project for the "How to Win a Data Science Competition" Coursera course. The dataset, provided by the Russian software firm 1C Company, includes daily sales data, presenting a challenging forecasting task.

## Directory Structure

- **exploratory_data_analysis**: Contains scripts for analyzing the dataset, visualizing trends, and understanding the data distribution.
  - `analyze_data.py`: Script for data analysis and visualization.

- **feature_data**: Code for generating and processing features used in the modeling phase.
  - `create_feature_data.py`: Script for creating features from raw data.

- **model_evaluation**: Scripts for evaluating model performance, including metrics and comparison of different models.
  - `evaluate_models.py`: Script for evaluating and comparing model performance.

- **modelling**: Code for training and tuning machine learning models, as well as hyperparameter optimization.
  - `train_models.py`: Script for training machine learning models.

- **raw_data**: Code for extracting data from Kaggle and storing the raw data files used in the project.
  - `get_files_from_kaggle.py`: Script for downloading and extracting data from Kaggle.

- **tests**: Unit tests and validation scripts for ensuring code quality and correctness.
  - `create_cyclic_features.py`: Unit tests for `create_cyclic_features` in `CreateFeatureData` class.
  - `create_lag_features.py`: Unit tests for `create_lag_features` in `CreateFeatureData` class.
  - `creating_monthly_data.py`: Unit tests for `creating_monthly_data` in `CreateFeatureData` class.
  - `fill_empty_months_where_sale_not_exist.py`: Unit tests for `fill_empty_months_where_sale_not_exist` in `CreateFeatureData` class.
  - `getting_data.py`: Unit tests for `getting_data` file in `CreateFeatureData` class.

- **utility_functions**: Utility functions and helpers used across the project.
  - `mean_absolute_percentage_error.py`: Calculate MAPE for modeling and evaluation processes.

- **configs**: Configuration files for various aspects of the project.

- **main.ipynb**: Main notebook integrating outputs and analysis from different phases of the project.

- **LICENSE**: License file for the project.

- **README.md**: The top-level README for developers using this project.

- **requirements.txt**: The requirements file for reproducing the analysis environment.

## Main Notebook

The project is primarily executed and managed through the `main.ipynb` notebook. This notebook integrates the outputs and analysis from different phases of the project, including data preprocessing, feature engineering, model training, and evaluation. It serves as the central point for running the project and reviewing the results.

## Python Version
This project was developed using Python 3.9.10. It is recommended to use this specific version, as some dependencies might not be compatible with other Python versions.

Caution: If you are using a different version of Python, you may encounter issues when installing or running the required packages. Please ensure your Python environment is set to version 3.9.10 before proceeding.

## How to Run

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Forecast-Sales
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

   

