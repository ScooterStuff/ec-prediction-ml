from abstracts.AbstractDataLoader import AbstractDataLoader
import pandas as pd

class EC40_loader(AbstractDataLoader):

    def __init__(self, file_path, train_path, valid_path, test_path):
        super().__init__(file_path, train_path, valid_path, test_path)


    def load_and_split(self):
        ec40 = pd.read_pickle(self.file_path)
        # Filter rows where the 'ec' column is not empty
        ec40_filtered = ec40[ec40['ec'].apply(self.ec_not_empty)]

        # Get the distribution of the 'traintest' column in the filtered data
        distribution = ec40_filtered['traintest'].value_counts()
        print("Traintest distribution (raw counts):")
        print(distribution)

        # Optionally, also view the proportions
        proportions = ec40_filtered['traintest'].value_counts(normalize=True)
        print("\nTraintest distribution (proportions):")
        print(proportions)

        train_data = ec40_filtered[ec40_filtered['traintest'] == 0]
        train_data.to_csv(self.train_path)
        print(f"Training Set saved in {self.train_path}")

        test_data = ec40_filtered[ec40_filtered['traintest'] == 1]
        # Shuffle the test set randomly (set a random_state for reproducibility)
        shuffled_test = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        # Split the shuffled DataFrame into two halves
        split_index = len(shuffled_test) // 2
        validation_set = shuffled_test.iloc[:split_index]
        validation_set.to_csv(self.valid_path)
        print(f"Validation Set saved in {self.valid_path}")
        test_set = shuffled_test.iloc[split_index:]
        test_set.to_csv(self.test_path)
        print(f"Test Set saved in {self.test_path}")

    def ec_not_empty(self, x):
        try:
            # Convert string representation to list if necessary
            if isinstance(x, str):
                ec_list = eval(x)
                return len(ec_list) > 0
            elif isinstance(x, list):
                return len(x) > 0
            else:
                return False
        except Exception:
            return False