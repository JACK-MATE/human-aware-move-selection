# Human-Aware Move Selection in Chess

Code for the bachelor thesis:

**Human-Aware Move Selection in Chess: Evaluating Metrics for Error-Prone Decision Situations**

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
pip install numpy
```

Place a Stockfish 18 executable in:

```text
engines/stockfish/
```

Download the required Lichess database files from:

https://database.lichess.org/

and place them in:

```text
data/raw/
```

Required files:

```text
lichess_db_standard_rated_2016-01.pgn.zst
lichess_db_standard_rated_2019-01.pgn.zst
lichess_db_standard_rated_2019-02.pgn.zst
lichess_db_standard_rated_2019-03.pgn.zst
lichess_db_standard_rated_2019-04.pgn.zst
lichess_db_standard_rated_2019-05.pgn.zst
lichess_db_standard_rated_2019-06.pgn.zst
```

## Run the complete pipeline

Run the following scripts from the project root in this order:

```bash
python src/test_dataset_builder.py
python src/candidate_processor.py
python src/test_dataset_collector.py
python src/create_top_dataset.py
python src/run_all_analysis.py
python src/complexity_analyzer.py
python src/maia_analyzer.py
```

`run_all_analysis.py` runs the structural metric calculation and the Maia-based simulation.

The main results are stored in:

```text
data/results/
```

Structural analysis results are stored in:

```text
data/results/complexity_analysis/
```

Maia analysis results are stored in:

```text
data/results/maia_analysis/
```

## Holdout validation

Run:

```bash
python src/validation/run_pipeline.py
```

The validation results are stored in:

```text
data/results/complexity_validation/
```

## Re-running only the statistical analysis

The repository already contains the main intermediate result files. Therefore, the statistical analysis can be reproduced without processing the raw Lichess databases and without rerunning the computationally expensive Stockfish and Maia calculations:

```bash
python src/complexity_analyzer.py
python src/maia_analyzer.py
```

The Maia-based simulation uses `maia3-5m`. Stockfish 18 is used for the engine evaluations.