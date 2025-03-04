import collections
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
import pandas as pd

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

def compute_tripeptide_composition(sequence):
    """Compute tripeptide composition (frequency of consecutive 3-residue segments)."""
    sequence = sequence.upper()
    tripeptides = [sequence[i:i+3] for i in range(len(sequence)-2)]
    total = len(tripeptides)
    tripeptide_counts = collections.Counter(tripeptides)
    # Here we only return those tripeptides present in the sequence to avoid an 8000-length vector.
    composition = {tp: tripeptide_counts.get(tp, 0) / total for tp in tripeptides}
    return list(composition.values())

def compute_kmer_composition(sequence, k=3):
    """Compute k-mer composition for a given k (default is 3)."""
    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    total = len(kmers)
    kmer_counts = collections.Counter(kmers)
    composition = {kmer: kmer_counts[kmer] / total for kmer in kmer_counts}
    return list(composition.values())


# 2. Physicochemical and Global Properties

def compute_physicochemical_properties(sequence):
    """Compute properties such as molecular weight, pI, GRAVY, aromaticity, instability, aliphatic and Boman index."""
    print(sequence)
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
    aac_result = [compute_aac(seq) for seq in ec40_seq]
    dc_result = [compute_dipeptide_composition(seq) for seq in ec40_seq]
    tc_result = [compute_tripeptide_composition(seq) for seq in ec40_seq]
    kmerc_result = [compute_kmer_composition(seq, k=3) for seq in ec40_seq]

    output_df = pd.DataFrame({
    "Original": ec40_seq,
    "AAC": aac_result,
    "DC": dc_result,
    "TC":tc_result,
    "KMERC":kmerc_result
    })

    output_df.to_csv('output.csv', index=False)
    print('success')
