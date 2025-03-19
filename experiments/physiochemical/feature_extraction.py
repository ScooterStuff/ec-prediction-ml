import collections
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
import pandas as pd
import itertools
from tqdm import tqdm

# 1. Sequence Based Features
def compute_aac(seq):
    """Compute Amino Acid Composition (AAC) as a normalized frequency for 20 amino acids."""
    seq = seq.upper()
    aac = {aa: seq.count(aa) / len(seq) for aa in 'ACDEFGHIKLMNPQRSTVWY'''}
    return list(aac.values())

def compute_dipeptide_composition(seq):
    """Compute dipeptide composition (frequency of consecutive 2-residue pairs)."""
    seq = seq.upper()
    dipeptides = [seq[i:i+2] for i in range(len(seq)-1)]
    total = len(dipeptides)
    dipeptides_counts = collections.Counter(dipeptides)
    # Create all possible dipeptides (20x20 combinations)
    possible_dipeptides = [aa1+aa2 for aa1 in 'ACDEFGHIKLMNPQRSTVWY' for aa2 in 'ACDEFGHIKLMNPQRSTVWY']
    composition = {dp: dipeptides_counts.get(dp, 0)/total for dp in possible_dipeptides}
    return list(composition.values())


# 2. Physicochemical and Global Properties

def replace_ambiguous(sequence, replacement='A'):
    return sequence.upper().replace('X', replacement).replace('B', replacement).replace('O', replacement).replace('U', replacement).replace('Z', replacement)
def compute_physicochemical_properties(sequence):
    """Compute properties such as molecular weight, pI, GRAVY, aromaticity, instability, aliphatic and Boman index."""
    sequence = replace_ambiguous(sequence, replacement='A')
    sequence = sequence.upper()
    analysis = ProteinAnalysis(sequence)
    props = {}
    props['molecular_weight'] = analysis.molecular_weight()
    props['isoelectric_point'] = analysis.isoelectric_point()
    props['gravy'] = analysis.gravy()
    props['aromaticity'] = analysis.aromaticity()
    props['instability_index'] = analysis.instability_index()
    props['aliphatic_index'] = compute_aliphatic_index(sequence)
    props['boman_index'] = compute_boman_index(sequence)
    return props

def compute_aliphatic_index(sequence):
    """
    Calculate the aliphatic index.
    Formula: AI = (Ala% + 2.9 * Val% + 3.9 * (Ile% + Leu%)) 
    """
    sequence = sequence.upper()
    length = len(sequence)
    ala = sequence.count('A') / length * 100
    val = sequence.count('V') / length * 100
    ile = sequence.count('I') / length * 100
    leu = sequence.count('L') / length * 100
    ai = ala + 2.9 * val + 3.9 * (ile + leu)
    return ai

def compute_boman_index(sequence):
    """
    Calculate the Boman index (approximate protein binding potential).
    Values below are sample binding potentials for each amino acid.
    """
    boman_values = {
        'A':  0.17, 'C':  0.24, 'D': -0.46, 'E': -0.74,
        'F':  2.04, 'G':  0.0,  'H':  0.77, 'I':  2.22,
        'K': -1.23, 'L':  1.99, 'M':  1.41, 'N': -0.72,
        'P': -0.18, 'Q': -0.79, 'R': -2.53, 'S': -0.53,
        'T': -0.38, 'V':  2.08, 'W':  2.65, 'Y':  1.88
    }
    sequence = sequence.upper()
    total_score = sum(boman_values.get(aa, 0) for aa in sequence)
    return total_score / len(sequence)



if  __name__ == "__main__":
    # CSV is faster for smaller dataset
    data = "ec-prediction-ml/experiments/features_extraction/ec40.csv"
    # f = open(data, "r")
    # f.close()
    # with open(data, "r") as csvfile:
    #     csv_reader = csv.reader(csvfile)
    #     for x in csv_reader:
    #         print(x)

    ec40 = pd.read_csv(data)
    ec40_seq = ec40.iloc[:, 1].tolist()

    aac_result = [compute_aac(seq) for seq in tqdm(ec40_seq, desc="Computing AAC")]
    dc_result = [compute_dipeptide_composition(seq) for seq in tqdm(ec40_seq, desc="Computing DC")]
    pc_result = [compute_physicochemical_properties(seq) for seq in tqdm(ec40_seq, desc="Computing PC")]


    output_df = pd.DataFrame({
    "Original": ec40_seq,
    "AAC": aac_result,
    "DC": dc_result,
    })

    pc_df = pd.DataFrame(pc_result)

    output_df = pd.concat([output_df, pc_df], axis=1)

    csv_filename = "output_results.csv"
    output_df.to_csv(csv_filename, index=False)

    print('success')
