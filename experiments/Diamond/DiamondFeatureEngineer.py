from abstracts.AbstractFeatureEngineer import AbstractFeatureEngineer 
from abstracts.AbstractDataLoader import AbstractDataLoader
import itertools

class DiamondFeatureEngineer(AbstractFeatureEngineer):

    def apply_feature_engineering(self, flag):

        if flag == "train":
            self.filter_diamond_output(self.data_loader.train_path, self.feature_loader.train_path)
        elif flag == "valid":
            self.filter_diamond_output(self.data_loader.valid_path, self.feature_loader.valid_path)
        else:
            self.filter_diamond_output(self.data_loader.test_path, self.feature_loader.test_path)

    def filter_diamond_output(self, diamond_output_file, filtered_output_file):
        """
        Process the Diamond output file to filter records with identity <= 40.
        Additionally, for queries that are in the original input file but do not appear
        in the Diamond output, write a dummy record (all fields set to "-1").

        :param diamond_output_file: Path to the Diamond output file.
        :param filtered_output_file: Path where the filtered output should be written.
        :param original_query_file: Path to the original file containing all query IDs.
        """
        count = 0
        # Process the Diamond output file.
        with open(diamond_output_file, "r") as infile, open(filtered_output_file, "w") as outfile:
            # Group lines by query ID (assumed to be the first column, tab-delimited)
            for query, group in itertools.groupby(infile, key=lambda line: line.split("\t")[0]):
                lines = list(group)
                # Filter for lines where the identity (column 3) is <= 40.
                low_lines = [line for line in lines if float(line.strip().split("\t")[2]) <= 40]
                if low_lines:
                    for line in low_lines:
                        outfile.write(line)
                    count += 1

        print(f"finished writting {count} query to {filtered_output_file}")


    