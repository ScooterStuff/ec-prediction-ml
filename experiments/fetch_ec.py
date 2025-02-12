import requests
import time
import csv
from tqdm import tqdm

def parse_best_hits(output_file):
    """Parses the DIAMOND output file and returns the best hit for each query."""
    best_hits = {}
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            query, subject = parts[0], parts[1]
            if query not in best_hits:
                best_hits[query] = []
            best_hits[query].append(subject)
    return best_hits

def get_ec_number(uniprot_id):
    """Fetches the EC number of a UniProt protein using the UniProt API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for ref in data.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", []):
            return ref.get("value")
    return None

def find_best_ec_numbers(output_file):
    """Iterates through hits and returns the first EC number found."""
    best_hits = parse_best_hits(output_file)
    results = []
    for query, subjects in tqdm(best_hits.items(), desc="Processing queries", unit="query"):
        for subject in subjects:
            uniprot_id = subject.split('_')[1]  # Extract UniProt ID
            ec_number = get_ec_number(uniprot_id)
            if ec_number:
                results.append((query, subject, ec_number))
                break  # Stop searching once the first EC number is found
            time.sleep(0.05)  # To avoid hitting API limits
        if not any(query in result for result in results):
            results.append((query, None, "No EC number found"))

    return results

def save_to_csv(results, output_csv="ec_results.csv"):
    """Saves the results to a CSV file."""
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Subject", "EC Number"])
        writer.writerows(results)

def fetch_ec(output_file, ec_results_file):
    
    print("Parsing DIAMOND output...")
    ec_results = find_best_ec_numbers(output_file)
    
    print("Saving results to CSV...")
    save_to_csv(ec_results, output_csv=ec_results_file)
    
    print(f"Results saved to '{ec_results_file}'")

# if __name__ == "__main__":
    # output_file = "../dataset/test_sequences/test_sequences_results.m8"
    # ec_result_path = "../dataset/test_sequences/test_sequences_ec_results.csv"

    # fetch_ec(output_file, ec_result_path)
