from abc import ABC, abstractmethod
from abstracts.AbstractDataLoader import AbstractDataLoader

class AbstractFeatureEngineer(ABC):
    def __init__(self, _dir: str, data_loader: AbstractDataLoader, prefix="fe"):
        """
        Initialize with a DataLoader object.
        
        :param dataloader: An instance of a subclass of AbstractDataLoader.
        """
        self.data_loader = data_loader
        self.feature_loader = AbstractDataLoader(_dir, prefix)

    @abstractmethod
    def apply_feature_engineering(self, flag):
        """
        Load data using the dataloader, apply feature engineering techniques,
        and save the processed features to csv.
        Note: some feature may have different processing stages for train and (validation/test), you may want to seperate them.
        """
        df = self.data_loader.load_data(flag)
        # apply some feature engineering
        if flag == "train":
            df.to_csv(self.feature_loader.train_path)
        elif flag == "valid":
            df.to_csv(self.feature_loader.valid_path)
        else:
            df.to_csv(self.feature_loader.test_path)

    def get_feature_loader(self):
        """
        Load processed features from a file.
        
        :param feature_path: The file path containing the processed features.
        :return: Loaded features (e.g., as a DataFrame).
        """
        return self.feature_loader
