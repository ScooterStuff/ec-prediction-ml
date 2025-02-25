import pickle
import pandas as pd

# Load the pickle file
with open('ECPred_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure the loaded object is a Pandas DataFrame
if isinstance(data, pd.DataFrame):
    # Save DataFrame to CSV
    csv_filename = 'ECPred_data.csv'
    data.to_csv(csv_filename, index=False)
    print(f"Data successfully saved to {csv_filename}")
else:
    raise ValueError("Unexpected data format: Expected a Pandas DataFrame")
