# Raw Data

This directory is intended for the raw Lichess game database files used for dataset generation and evaluation.

The original database files are not included in this repository because of their large file size.

## Data Source

The datasets are taken from the publicly available Lichess database:

https://database.lichess.org/

Only standard rated games are used.

## Expected File Format

Download the required monthly `.pgn.zst` files and place them directly in this directory.

Example:

data/raw/lichess_db_standard_rated_2019-01.pgn.zst

The Python scripts read the compressed `.zst` files directly. Manual decompression is not required.

## Files Used

The exact monthly database files used for the experiments should be documented here.

### Candidate Generation

- lichess_db_standard_rated_2016-01.pgn.zst

### Test Dataset

- lichess_db_standard_rated_2019-01.pgn.zst
- Further files will be listed here once the final test dataset has been generated.

## Note

Raw `.pgn` and `.pgn.zst` files are excluded from Git via `.gitignore`.