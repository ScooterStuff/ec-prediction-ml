import asyncio
import aiohttp
import async_timeout
import csv
import os
from tqdm import tqdm  # For progress bar

def parse_best_hits(output_file):
    """Parses the DIAMOND output file and returns the best hits for each query."""
    best_hits = {}
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            query, subject = parts[0], parts[1]
            best_hits.setdefault(query, []).append(subject)
    return best_hits

async def fetch_ec_number(session, uniprot_id, sem):
    """Asynchronously fetch the EC number for a given UniProt ID using a semaphore for rate limiting."""
    async with sem:  # Limit the number of concurrent requests
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            async with async_timeout.timeout(10):  # Timeout after 10 seconds
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Traverse the JSON structure to get EC numbers
                        for ref in data.get("proteinDescription", {}) \
                                       .get("recommendedName", {}) \
                                       .get("ecNumbers", []):
                            return ref.get("value")
        except Exception as e:
            print(f"Error fetching {uniprot_id}: {e}")
    return None

async def process_query(query, subjects, session, sem):
    """
    Process a single query: iterate over its subjects and return the first subject that has an EC number.
    If no subject returns an EC number, return a placeholder.
    """
    for subject in subjects:
        try:
            # Extract UniProt ID; adjust this if the format changes
            uniprot_id = subject.split('_')[1]
        except IndexError:
            continue  # Skip if format is unexpected
        ec_number = await fetch_ec_number(session, uniprot_id, sem)
        if ec_number:
            return (query, subject, ec_number)
    return (query, None, "No EC number found")

async def main_async(dimond_result_file):
    
    best_hits = parse_best_hits(dimond_result_file)
    
    tasks = []
    sem = asyncio.Semaphore(20)  # Limit to 20 concurrent requests
    
    async with aiohttp.ClientSession() as session:
        for query, subjects in best_hits.items():
            tasks.append(process_query(query, subjects, session, sem))
        
        results = []
        total_tasks = len(tasks)
        # Process tasks as they complete, updating the progress bar
        for coro in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="Processing queries"):
            result = await coro
            results.append(result)
    
    return results

def save_to_csv(results, output_csv="ec_results.csv"):
    """Saves the results to a CSV file."""
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Subject", "EC Number"])
        writer.writerows(results)

def fetch_ec_async(dimond_result_path, ec_result_path):
    
    print("Parsing DIAMOND output and fetching EC numbers concurrently...")
    results = asyncio.run(main_async(dimond_result_path))
    
    print("Saving results to CSV...")
    save_to_csv(results, ec_result_path)
    print(f"Results saved to '{ec_result_path}'")

if __name__ == "__main__":
    output_file = "../dataset/test_sequences/test_sequences_results.m8"
    ec_result_path = "../dataset/test_sequences/test_sequences_ec_results.csv"

    fetch_ec_async(output_file, ec_result_path)