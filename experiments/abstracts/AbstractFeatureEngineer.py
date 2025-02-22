from abc import ABC, abstractmethod
from abstracts.AbstractDataLoader import AbstractDataLoader

class AbstractFeatureEngineer(ABC):
    def __init__(self, data_loader: AbstractDataLoader, feature_loader: AbstractDataLoader):
        """
        Initialize with a DataLoader object.
        
        :param dataloader: An instance of a subclass of AbstractDataLoader.
        """
        self.data_loader = data_loader
        self.feature_loader = feature_loader

    @abstractmethod
    def apply_feature_engineering(self, flag):
        """
        Load data using the dataloader, apply feature engineering techniques,
        and save the processed features to csv.
        Note: some feature may have different processing stages for train and (validation/test), you may want to seperate them.
        """
        if flag == "train":
            train_df = self.data_loader.load_train()
            pass # apply some feature engineering
            train_df.to_csv(self.feature_loader.train_path)
        elif flag == "valid":
            valid_df = self.data_loader.load_valid()
            pass # apply some feature engineering
            valid_df.to_csv(self.feature_loader.valid_path)
        else:
            test_df = self.data_loader.load_test()
            pass # apply some feature engineering
            test_df.to_csv(self.feature_loader.test_path)

    def get_feature_loader(self):
        """
        Load processed features from a file.
        
        :param feature_path: The file path containing the processed features.
        :return: Loaded features (e.g., as a DataFrame).
        """
        return self.feature_loader
