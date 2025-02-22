from abc import ABC, abstractmethod
import AbstractDataLoader
import pandas as pd

class AbstractFeatureEngineer(ABC):
    def __init__(self, dataloader: AbstractDataLoader):
        """
        Initialize with a DataLoader object.
        
        :param dataloader: An instance of a subclass of AbstractDataLoader.
        """
        self.dataloader = dataloader
        self.feature_path = None

    @abstractmethod
    def apply_feature_engineering(self):
        """
        Load data using the dataloader, apply feature engineering techniques,
        and output the processed features.
        
        :return: Processed features (e.g., a DataFrame, NumPy array, etc.)
        """
        raise NotImplementedError()

    def save_features(self, output_path: str, features):
        """
        Save the processed features to a file.
        
        :param output_path: The file path where features will be saved.
        :param features: The processed features (e.g., as a DataFrame).
        :return: The output file path.
        """
        features.to_csv(output_path, index=False)
        self.feature_path = output_path
        print(f"feature saved in {self.feature_path}")

    def load_features(self, feature_path: str):
        """
        Load processed features from a file.
        
        :param feature_path: The file path containing the processed features.
        :return: Loaded features (e.g., as a DataFrame).
        """
        if self.feature_path is not None:
            return pd.read_csv(feature_path)
        else:
            print("You have to save features first")
            return
