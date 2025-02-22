from abc import ABC, abstractmethod
import pandas as pd

class AbstractDataLoader(ABC):
    def __init__(self, file_path, train_path, valid_path, test_path):
        """
        Initialize with the basic file path for the data.
        
        :param file_path: The path to the data file.
        """
        self.file_path = file_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.feature_path = None  # This will store the final feature file path

    def load_and_split(self):
        '''
        Load entire data file, filter invalid data and split train, valid and test.
        '''
        pass

    def load_test(self):
        """
        Load the test data.
        
        :return: Test data in a format defined by the subclass.
        """

        return pd.read_csv(self.test_path)
        
            

    def load_train(self):
        """
        Load the training data.
        
        :return: Training data in a format defined by the subclass.
        """
        return pd.read_csv(self.train_path)
    

    def load_valid(self):
        """
        Load the validation data.
        
        :return: Validation data in a format defined by the subclass.
        """
        return pd.read_csv(self.valid_path)
    

