
import os
import zipfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GettingDataFromKaggle:
    """
    A class to download and extract data files from Kaggle.

    Attributes:
        kaggle_username (str): The Kaggle username loaded from environment variables.
        kaggle_key (str): The Kaggle API key loaded from environment variables.
        target_directory (str): The directory where data will be downloaded and extracted.
    """

    def __init__(self, target_directory='./raw_data'):
        """
        Initializes the GettingDataFromKaggle class with the target directory.

        Args:
            target_directory (str): Path to the directory where data will be stored. Default is './raw_data'.
        """

        # Get Kaggle credentials from environment variables
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = os.getenv('KAGGLE_KEY')
        self.target_directory = target_directory

        # Ensure target directory exists
        os.makedirs(self.target_directory, exist_ok=True)

    def _set_kaggle_credentials(self):
        """
        Set the Kaggle API credentials in the environment variables.
        Raises:
            ValueError: If Kaggle credentials are missing.
        """
        if self.kaggle_username and self.kaggle_key:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
            os.environ['KAGGLE_KEY'] = self.kaggle_key
        else:
            raise ValueError("Kaggle credentials are missing. Please check your .env file.")

    def download_data_from_kaggle(self):
        """
        Downloads the dataset from Kaggle using the Kaggle API.

        Raises:
            RuntimeError: If the download command fails.
        """
        self._set_kaggle_credentials()

        # Kaggle command to download the dataset
        command = (
            f'kaggle competitions download -c competitive-data-science-predict-future-sales '
            f'-p {self.target_directory} > /dev/null 2>&1'
        )
        result = os.system(command)

        if result == 0:
            print("Data file successfully downloaded from Kaggle.")
        else:
            raise RuntimeError("Failed to download data file from Kaggle.")

    def open_zip_file(self):
        """
        Extracts all zip files in the target directory.

        Raises:
            FileNotFoundError: If no zip files are found in the directory.
        """

        zip_files = [f for f in os.listdir(self.target_directory) if f.endswith(".zip")]

        if not zip_files:
            raise FileNotFoundError("No zip files found to extract.")

        for filename in zip_files:
            zip_file_path = os.path.join(self.target_directory, filename)

            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.target_directory)

            print(f"Extracted {filename} to {self.target_directory}")

    def run(self):
        """
        Orchestrates the process of downloading and extracting Kaggle data.
        """
        self.download_data_from_kaggle()
        self.open_zip_file()
