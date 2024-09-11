
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utility_functions.mean_absolute_percentage_error import mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence


class EvaluateModels:
    """
    A class to evaluate and compare machine learning models on a test dataset.

    Attributes:
        test_x (pd.DataFrame): Features of the test dataset.
        test_y (pd.Series): Target values of the test dataset.
        model_list (list): List of tuples containing models and their corresponding names.
    """

    def __init__(self, test_df, model_list):
        """
        Initializes the EvaluateModels class with test data and a list of models.

        Args:
            test_df (pd.DataFrame): DataFrame containing features and target column.
            model_list (list): List of tuples (model, model_name).
        """

        self.test_y = test_df[["target"]]
        self.test_x = test_df.drop(columns=["target"])
        self.model_list = model_list

    def compare_models_with_test_set(self):
        """
        Compares models using the test set and calculates performance metrics.

        Returns:
            pd.DataFrame: A DataFrame containing MAPE, MAE, and RMSE for each model.
        """
        # Initialize an empty list to store the results
        results = []

        for model, model_name in self.model_list:
            # Predict on the test set
            predictions = model.predict(self.test_x).ravel()

            # Calculate various metrics
            mape = mean_absolute_percentage_error(self.test_y.values.ravel(), predictions)
            mae = mean_absolute_error(self.test_y.values.ravel(), predictions)
            rmse = np.sqrt(mean_squared_error(self.test_y.values.ravel(), predictions))

            # Append the results to the list
            results.append({
                "Model": model_name,
                "MAPE (%)": mape,
                "MAE": mae,
                "RMSE": rmse
            })

        # Convert the list of results to a DataFrame and sort the dataFrame by MAPE
        results_df = pd.DataFrame(results).sort_values(by="MAPE (%)")

        return results_df

    def plot_feature_importance(self, model, model_name):
        """
        Plots the feature importance for a given model.

        Args:
            model: The machine learning model
            model_name (str): The name of the model, which is used for title of the plot.
        """
        # Checking model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': self.test_x.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Feature Importance for {model_name.upper()}')
            plt.show()
        else:
            print(f"{model_name} does not have the `feature_importances_` attribute.")

    def finding_best_model(self, model_name_string):
        """
        Finds and returns the best model by name from the model list.

        Args:
            model_name (str): The name of the model to find.

        Returns:
            model: The model instance if found, None otherwise.
        """
        for model, model_name in self.model_list:
            if model_name == model_name_string:
                # Returning the desired model instance
                return model

        print(f"Model '{model_name_string}' not found.")
        return None

    def plot_partial_dependence_plots_for_each_feature(self, model, model_name):
        """
        Plots partial dependence plots (PDP) for a given model.

        Args:
            model: The machine learning model to evaluate.
            model_name (str): The name of the model which is used for title of the plot.
        """
        feature_names = self.test_x.columns
        n_features = len(feature_names)

        # Determine grid size (e.g., 2 rows x 3 columns for 6 features)
        n_cols = 3  # Number of columns
        n_rows = (n_features + n_cols - 1) // n_cols  # Calculate number of rows required

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()

        # Plot partial dependence
        if hasattr(model, 'predict'):
            for i, feature in enumerate(feature_names):
                pdp = partial_dependence(model, self.test_x, features=[feature])
                pdp_values = pdp['average'][0]
                values = np.linspace(self.test_x[feature].min(), self.test_x[feature].max(), len(pdp_values))

                # Plot PDP on the corresponding axis
                axes[i].plot(values, pdp_values, marker='o')
                axes[i].set_title(f'{feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Partial Dependence')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Overall title for the figure
        plt.suptitle(f'Partial Dependence Plots for {model_name.upper()}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
