import pandas as pd

# Load the CSV file
df = pd.read_csv("ec40.csv")

# Inspect the first few rows to confirm the format
print(df.head())

# Filter for test sequences
# (Adjust the filtering criteria if your CSV uses a different convention;
#  here we assume '0' indicates test sequences.)
test_df = df[df["traintest"] == 0]

print(f"Found {len(test_df)} test sequences.")

# Write the test sequences to a FASTA file.
# We will use the 'accession' column as the FASTA header and 'sequence' as the sequence.
with open("test_sequences.fasta", "w") as fout:
    for index, row in test_df.iterrows():
        accession = row["accession"]
        sequence = row["sequence"]
        fout.write(f">{accession}\n{sequence}\n")
