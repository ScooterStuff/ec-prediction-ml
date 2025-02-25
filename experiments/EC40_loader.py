from abstracts.AbstractDataLoader import AbstractDataLoader
import constants
import pandas as pd
import numpy as np
import os

class EC40_loader(AbstractDataLoader):
    """
    A data loader class for the EC40 dataset, inheriting from AbstractDataLoader.
    It handles preprocessing, train-test splitting, and optional FASTA generation.
    """

    def get_fasta(self, flag):
        """
        Write sequences from the given DataFrame into a FASTA file.

        Args:
            df (pd.DataFrame): A DataFrame containing at least 'accession' and 'sequence' columns.
            fasta_path (str): Path to the output FASTA file.

        Returns:
            None
        """
        df = self.load_data(flag)
        fasta_path = os.path.join(self.dir, f"{flag}.fasta")
        print(f"Found {len(df)} sequences.")
        with open(fasta_path, "w") as fout:
            for index, row in df.iterrows():
                accession = row["accession"]
                sequence = row["sequence"]
                fout.write(f">{accession}\n{sequence}\n")
        print(f"Finished writting to {fasta_path}")
        return fasta_path

    def preprocess(self):
        """
        Preprocess the EC40 dataset by:
          1. Removing duplicates based on 'accession' and 'sequence'.
          2. Filtering out rows where 'ec' is an empty list.
          3. Loading a CD-HIT 40 classifier DataFrame and merging it with the EC40 data
             to keep only those accessions present in the classifier.
          4. Returning the merged DataFrame for further splitting.

        Returns:
            pd.DataFrame: The filtered and merged DataFrame containing EC40 data with classifier info.
        """
        # 1. Load original EC40 data
        ec40 = self.load_data(flag="full")

        # 2. Remove duplicates based on ['accession', 'sequence']
        ec40_unique = ec40.drop_duplicates(subset=["accession", "sequence"], keep="first")

        # 3. Filter out rows where 'ec' is an empty list
        ec40_unique = ec40_unique[ec40_unique['ec'].apply(lambda x: x.strip() != '[]')]

        # 4. Load the CD-HIT 40 classifier data
        cdhit40 = pd.read_pickle(constants.CDHIT40_PATH)

        # 5. Filter the classifier data to keep only those accessions present in ec40_unique
        filtered_classifier = cdhit40[cdhit40.index.isin(ec40_unique['accession'])]

        # 6. Further filter ec40_unique to retain rows with accessions in the classifier
        ec40_filtered = ec40_unique[ec40_unique['accession'].isin(filtered_classifier.index)]

        # 7. Merge the two DataFrames so we get classifier columns in ec40_filtered
        ec40_filtered = ec40_filtered.merge(
            filtered_classifier, 
            how='left',
            left_on='accession',
            right_index=True
        )

        return ec40_filtered

    def train_test_split(self, df, val_ratio=0.1, random_state=constants.RANDOM_SEED):
        """
        Split the preprocessed DataFrame into train, validation, and test sets based on:
          - A 'traintest' column (1 => test, 0 => train+validation).
          - A 'cluster_ID' column, ensuring that validation and train do not share clusters,
            and also removing any train clusters that appear in test.

        Args:
            df (pd.DataFrame): The DataFrame to be split. Must contain 'traintest' and 'cluster_ID' columns.
            val_ratio (float): Fraction of the train+validation data to allocate to validation. Defaults to 0.1.
            random_state (int): Random seed for reproducible shuffling. Defaults to constants.RANDOM_SEED.

        Returns:
            tuple: A tuple of (train_data, val_data, test_data) as three pd.DataFrames.
        """
        if df is None:
            df = self.load_data(flag="full")

        np.random.seed(random_state)

        # 1) Separate test set based on 'traintest' == 1
        test_data = df[df['traintest'] == 1].copy()

        # 2) Get the remaining data (train + validation)
        train_val_data = df[df['traintest'] == 0].copy()

        # 3) Shuffle clusters and compute cluster sizes
        cluster_counts = train_val_data.groupby('cluster_ID').size().reset_index(name='count')
        cluster_counts = cluster_counts.sample(frac=1, random_state=random_state).reset_index(drop=True)

        total_train_val = len(train_val_data)
        target_val_size = val_ratio * total_train_val

        selected_val_clusters = []
        running_sum = 0

        # 4) Select enough clusters to reach ~val_ratio of the total train+val size
        for i, row in cluster_counts.iterrows():
            c_id = row['cluster_ID']
            c_count = row['count']
            selected_val_clusters.append(c_id)
            running_sum += c_count
            if running_sum >= target_val_size:
                break

        # 5) Split out validation and train sets
        val_data = train_val_data[train_val_data['cluster_ID'].isin(selected_val_clusters)].copy()
        train_data = train_val_data[~train_val_data['cluster_ID'].isin(selected_val_clusters)].copy()

        # 6) Ensure no overlap between train and test clusters
        train_clusters_final = set(train_data['cluster_ID'].unique())
        test_clusters_final = set(test_data['cluster_ID'].unique())
        overlap = train_clusters_final.intersection(test_clusters_final)
        if len(overlap) > 0:
            print(f"Warning: {len(overlap)} cluster(s) appear in both train & test.")
            # Remove these overlapping clusters from train
            train_data = train_data[~train_data['cluster_ID'].isin(overlap)]
            print(f"Removed {len(overlap)} cluster(s) from train_val_data to avoid overlap with test.")

        # Print final sizes
        print(f"train: {len(train_data)}\nvalid: {len(val_data)}\ntest: {len(test_data)}")

        # 7) Save to CSV
        train_data.to_csv(self.train_path, index=False)
        val_data.to_csv(self.valid_path, index=False)
        test_data.to_csv(self.test_path, index=False)

        return train_data, val_data, test_data
