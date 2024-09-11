
import matplotlib.pyplot as plt
import seaborn as sns

class AnalyzeData:
    """
       A class to perform exploratory data analysis (EDA) on a DataFrame.

       This class provides various methods to analyze a given DataFrame, including:
       - Viewing basic data types.
       - Calculating and displaying descriptive statistics (mean, std, min, max).
       - Identifying and reporting missing values.
       - Computing and visualizing the correlation matrix using a heatmap.

       Attributes:
           df (pd.DataFrame): The DataFrame to be analyzed.
       """
    def __init__(self, df):
        self.df = df

    def get_descriptive_statistics(self):
        """Prints the shape and selected descriptive statistics (mean, min, max, std) of the DataFrame."""

        print("\n" + "-"*40)
        print(f"Shape of the DataFrame: {self.df.shape}")
        print("-"*40)

        # Display a heading
        print("\n" + "="*40)
        print(" Descriptive Statistics ".center(40, "="))
        print("="*40 + "\n")

        # Select only mean, min, max, and std from the descriptive statistics
        stats = self.df.describe().T[['mean', 'min', 'max']]
        print(stats)

    def get_missing_values(self):
        """Checks and prints any missing values in the DataFrame."""
        # Display a heading
        print("\n" + "="*40)
        print(" Missing Values ".center(40, "="))
        print("="*40 + "\n")

        # Check for missing values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            print(missing.sort_values(ascending=False))
        else:
            print("No missing values found!")

        # Print a separator
        print("\n" + "-"*40)

    def get_data_types(self):
        """Prints the data types of each column in the DataFrame."""
        # Display a heading
        print("\n" + "="*40)
        print(" Data Types ".center(40, "="))
        print("="*40 + "\n")

        # Print data types of each column
        print(self.df.dtypes)

        # Print a separator
        print("\n" + "-"*40)

    def get_correlation_matrix(self):
        """Prints and visualizes the correlation matrix using a heatmap."""
        # Display a heading
        print("\n" + "=" * 40)
        print(" Correlation Matrix ".center(40, "="))
        print("=" * 40 + "\n")

        # Calculate and print correlation matrix
        corr = self.df.corr(method='spearman')

        # Print a separator
        print("\n" + "-" * 40)

        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(10, 8))  # Adjust the size as needed
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.5)

        # Add a title
        plt.title('Correlation Matrix')

        # Display the plot
        plt.show()

    def create_analysis(self):
        """Runs a full analysis on the DataFrame."""
        self.get_data_types()
        self.get_descriptive_statistics()
        self.get_missing_values()
        self.get_correlation_matrix()