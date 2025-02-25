from abc import ABC, abstractmethod
import os
import pandas as pd

class AbstractDataLoader(ABC):
    def __init__(self, _dir, prefix=""):
        """
        Initialize with the basic file path for the data.
        
        :param file_path: The path to the data file.
        """
        self.dir = _dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if prefix != "":
            prefix += "_"
        self.train_path = os.path.join(self.dir, f"{prefix}train.csv")
        self.valid_path = os.path.join(self.dir, f"{prefix}valid.csv")
        self.test_path = os.path.join(self.dir, f"{prefix}test.csv")
    
    def set_source_file(self, file_name):
        self.file_path = os.path.join(self.dir, file_name)

    def train_test_split(self):
        '''
        Load entire data file, filter invalid data and split train, valid and test.
        '''
        pass

    def load_data(self, flag=None):

        if flag == "train":
            return pd.read_csv(self.train_path)
        elif flag == "valid":
            return pd.read_csv(self.valid_path)
        elif flag == "test":
            return pd.read_csv(self.test_path)
        elif flag == "full":
            if self.file_path.endswith('.pkl'):
                return pd.read_pickle(self.file_path)
            else:
                return pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError("No valid file found")

    

