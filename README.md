# EC Prediction

## Diamond Benchmark

This project use DIAMOND tool as benchmark. The pipeline utilizes UniRef90 as the reference database and processes protein sequences to generate alignment results.

### Instructions

Goto `dimond.ipynb`, follow all the instructions
+ You may have to download several python packages to set everything up.
+ All code in there is written in `linux`. For different OS, you may need to rewrite some command
+ The output is already generated in `/stat` and `/dataset/test_sequences`, you may not have to run it again


### File Information

Some files are too large to be included in this repository, but you can generate them yourself following the steps above:

`dataset` : storing all data, including training, test and database sequences
- `diamond_db`: The DIAMOND database folder. (create script in `experiments/dimond.ipynb`)
   - `uniref90.fasta.gz` : original UniRef90 data zip
   - `uniref90.dmnd` : The UniRef90 database for DIAMOND to align
- `ec40` : All the ec40 original data
   - including .pkl, .csv formats
- `test_sequences`: Storing the original test sequence and intermidiate processed files
   - `test_sequences.fasta`: Test query sequences.
   - `test_sequences_results.m8` : DIAMOND alignment result file
   - `test_sequences_ec_results.csv` : EC number fetched from UniProt API according to DIMOND output
   - `evaluation_results.csv` : Prediction result for every sequences in `test_sequences.fasta`

`metrics` : Containing methods' statisitical metrics

   - `metrics.csv` : DIAMOND benchmark report

`experiments`: All methods and utils
   - `dimond` : DIMOND execution file, can be download in `dimond.ipynb` (for linux)
   - `dimond.ipynb` : Pipeline for DIAMOND benchmark (All stage included, no need to download any data your self!)
   - `evaluate_ec.py` : Evaluation util to compare predicted EC numbers and true EC numbers, then generate stats report
   - `fetch_ec_improved` and `fetch_ec` : Fetch utils to get EC numbers from DIAMOND results. Improved version implemented async stages to run fetch simultaneously.


