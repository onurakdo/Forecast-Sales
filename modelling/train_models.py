
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from utility_functions.mean_absolute_percentage_error import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Define MAPE scorer for use in model evaluation
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

class TrainModel:
    """
    A class for training machine learning models with hyperparameter tuning.

    Attributes:
        train_x (pd.DataFrame): Features for training.
        train_y (pd.Series): Target variable for training.
        config (dict): Configuration dictionary containing parameters for models and random search.
    """

    def __init__(self, train_df):
        """
         Initializes the TrainModel class with training data and loads configuration.

         Args:
             train_df (pd.DataFrame): DataFrame containing features and target column.
         """

        self.train_y = train_df[["target"]]
        self.train_x = train_df.drop(columns=["target"])

        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def random_search_hyper_parameter_tuning(self):
        """
        Performs randomized search for hyperparameter tuning on multiple models.

        Returns:
            list: A list containing the best parameters, model, and model name.
            pd.DataFrame: Features used for training.
            pd.Series: Target variable used for training.
        """
        # Two columns were generated from the month information, and it would be more logical to use them together.
        # Therefore, interaction_constraints were defined for the XGBoost model.
        # However, since this feature is not available in other models, it could not be used
        interaction_constraints = [['month_sin', 'month_cos']]
        model_list = [[xgb.XGBRegressor(verbose=0,  interaction_constraints=interaction_constraints),
                       self.config["xgb_param_dist"], "XGB"],
                      [lgb.LGBMRegressor(verbose=-1), self.config["lgb_param_dist"],  "LGB"],
                      [RandomForestRegressor(verbose=0), self.config["rf_param_dist"], "RF"]
                      ]

        model_best_param_list = []
        # Perform random search for each model
        for model, model_param_space, model_name in model_list:
            random_search = RandomizedSearchCV(
                model,
                param_distributions=model_param_space,
                n_iter=self.config["random_search_iter_size"],  # Number of random combinations to try
                scoring=mape_scorer,
                cv=self.config["cross_validation_fold_size"],  # Number of cross-validation folds
                random_state=42  # For reproducibility
            )

            random_search.fit(self.train_x, self.train_y)

            model_best_param_list.append([random_search.best_params_, model, model_name])
            print(f"Best parameters are found for {model_name}")
            print(f"Best parameters are {random_search.best_params_}")

        return model_best_param_list, self.train_x, self.train_y

    def train_model_with_best_params(self, model_best_param_list):
        """
        Trains each model using its best hyperparameters.

        Args:
            model_best_param_list (list): List containing the best parameters, model, and model name.

        Returns:
            list: A list of tuples, each containing the trained model and its name.
        """

        trained_model_list = []
        for model_best_params, model, model_name in model_best_param_list:
            model.set_params(**model_best_params)
            model.fit(self.train_x, self.train_y)

            print(f"{model_name} model trained successfully with the best parameters.")
            trained_model_list.append([model, model_name])

        return trained_model_list
