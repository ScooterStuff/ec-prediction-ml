## Diamond Benchmark

This project use DIAMOND tool as benchmark. The pipeline utilizes UniRef50 as the reference database and processes protein sequences to generate alignment results.

1. Install DIAMOND
   Ensure you have DIAMOND installed on your system. You can download it from the official website.

2. Prepare the Database
   Download the UniRef50 database and create a DIAMOND database:
   `diamond.exe makedb --in uniref50.fasta -d uniref50_db`

- `uniref50.fasta`: The UniRef50 protein sequence database file.
- `uniref50_db`: The output DIAMOND database.

3. Run Protein Search
   Download ec40 dataset and transform it to fasta. Then use DIAMOND to perform a BLASTP search on the test sequences:
   `diamond.exe blastp --db uniref50_db.dmnd --query test_sequences.fasta --out diamond_results.m8 --threads 16`

- `test_sequences.fasta`: Input file containing query protein sequences.
- `diamond_results.m8`: Output file storing the search results.
- `--threads 16`: Uses 16 CPU threads for faster processing (This depend on your computer, mine max is 16).

### File Information

Some files are too large to be included in this repository, but you can generate them yourself following the steps above:

- `diamond.exe`: The DIAMOND executable.
- `uniref50.fasta`: The UniRef50 protein database.
- `test_sequences.fasta`: Test query sequences.
- `diamond_results.m8`: The result file generated after running DIAMOND.

### `fetch_ec.py`

This script is included in the repository. It is responsible for fetching Enzyme Commission (EC) numbers from the UniProt API.
