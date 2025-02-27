import numpy as np
import pandas as pd

def parse_pssm(pssm_file):
    with open(pssm_file, 'r') as file:
        lines = file.readlines()

    pssm_data = []
    for line in lines:
        if line.strip() and line[0].isdigit():  # Extract only numerical data
            parts = line.split()
            values = list(map(int, parts[2:22]))  # PSSM scores (20 columns)
            pssm_data.append(values)

    pssm_array = np.array(pssm_data)

    # Extract Features
    avg_pssm = np.mean(pssm_array, axis=0)  # Mean PSSM for each AA
    max_pssm = np.max(pssm_array, axis=0)
    min_pssm = np.min(pssm_array, axis=0)

    features = np.concatenate([avg_pssm, max_pssm, min_pssm])

    return features

# Example Usage
features = parse_pssm("output.pssm")
print("Extracted Features:", features)
