# Human-Aware Move Selection in Chess

Research code accompanying the bachelor thesis **“Human-Aware Move Selection in Chess: Evaluating Metrics for Error-Prone Decision Situations.”**

The project investigates whether structural properties of chess positions and human-like move probabilities can help identify difficult decisions and practically promising moves against human players.

The repository contains the complete analysis pipeline used for the thesis. It is research code rather than a reusable Python package, so most experiment settings are intentionally stored as constants near the top of the corresponding scripts.

## What the project does

The pipeline has two main analysis branches:

1. **Structural position metrics**
   - Good Move Ratio (GMR)
   - Number of good moves
   - Depth to Best Move Stability (DTBMS)
   - Equal-weight combinations of breadth and DTBMS

2. **Maia-3 move-tree simulation**
   - Uses rating-dependent Maia-3 move probabilities
   - Expands likely human continuations after an observed candidate move
   - Evaluates leaf positions with Stockfish
   - Compares the resulting expected score with empirical game results and the direct Stockfish WDL estimate

The repository also contains the scripts used to build the position dataset, evaluate the metrics, create figures and tables, and perform the structural holdout validation.

## Repository structure

```text
human-aware-move-selection/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Lichess .pgn.zst files (not tracked by Git)
│   ├── candidates/          # Intermediate candidate-position files
│   └── results/             # Aggregated datasets, metric results and analysis output
├── engines/
│   └── stockfish/           # Place the Stockfish executable here
└── src/
    ├── test_dataset_builder.py
    ├── candidate_processor.py
    ├── test_dataset_collector.py
    ├── create_top_dataset.py
    ├── metric_runner.py
    ├── complexity_analyzer.py
    ├── maia_runner.py
    ├── maia_analyzer.py
    ├── run_all_analysis.py
    ├── metrics/             # GMR, number of good moves, DTBMS, Stockfish helpers
    ├── maia/                # Maia-3 adapter and tree simulation
    └── validation/          # Structural holdout validation
```

## Requirements

The project is written in Python and was developed with the dependencies listed in `requirements.txt`.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`maia_analyzer.py` additionally imports NumPy. If it is not already installed as a dependency of another package, install it explicitly:

```bash
pip install numpy
```

### Maia-3 model

The simulation uses the model alias:

```text
maia3-5m
```

On first use, Maia-3 may download the corresponding model checkpoint. The adapter automatically uses CUDA if a compatible GPU is available and otherwise falls back to CPU.

## Stockfish setup

The experiments in the thesis use **Stockfish 18**. The development machine used the Windows x86-64 AVX2 build.

The binary is not included in the repository. Download Stockfish separately and place **exactly one** Stockfish executable in:

```text
engines/stockfish/
```

Example on Windows:

```text
engines/stockfish/stockfish-windows-x86-64-avx2.exe
```

The scripts search this directory automatically. The filename must begin with `stockfish`. On Unix-like systems the file must be executable.

See `engines/stockfish/README.md` for the engine-specific notes.

## Raw Lichess data

Raw Lichess database files are not committed because of their size. They can be downloaded from:

https://database.lichess.org/

Only standard rated games are used. The scripts read compressed `.pgn.zst` files directly; manual decompression is not required.

Place the required files in:

```text
data/raw/
```

The thesis pipeline uses:

### Candidate generation

```text
lichess_db_standard_rated_2016-01.pgn.zst
```

### Empirical test dataset

```text
lichess_db_standard_rated_2019-01.pgn.zst
lichess_db_standard_rated_2019-02.pgn.zst
lichess_db_standard_rated_2019-03.pgn.zst
lichess_db_standard_rated_2019-04.pgn.zst
lichess_db_standard_rated_2019-05.pgn.zst
lichess_db_standard_rated_2019-06.pgn.zst
```

## Two ways to use the repository

### 1. Re-run the statistical analysis from the committed intermediate results

This is the fastest way to inspect the thesis analysis. The repository already contains the central aggregated dataset and computed result files in `data/results/`.

To regenerate the structural analysis tables and figures:

```bash
python src/complexity_analyzer.py
```

Output is written to:

```text
data/results/complexity_analysis/
```

To regenerate the Maia analysis:

```bash
python src/maia_analyzer.py
```

Output is written to:

```text
data/results/maia_analysis/
```

This route does **not** require reprocessing the large raw Lichess databases or rerunning the expensive Stockfish/Maia calculations.

### 2. Rebuild the complete pipeline from the raw Lichess databases

Run all commands from the repository root and in the following order.

#### Step 1 — Generate raw candidate positions

```bash
python src/test_dataset_builder.py
```

Input:

```text
data/raw/lichess_db_standard_rated_2016-01.pgn.zst
```

Output:

```text
data/candidates/raw_candidate_positions.json
```

The script considers positions from full moves 11–20, keeps Blitz/Rapid/Classical games with a maximum rating difference of 200, requires at least two different observed moves, and retains up to 50,000 frequently occurring candidate positions.

#### Step 2 — Remove connected candidates and apply the Stockfish balance filter

```bash
python src/candidate_processor.py
```

Outputs:

```text
data/candidates/independent_candidate_positions.json
data/candidates/evaluated_candidate_positions.json
data/candidates/balanced_candidate_positions.json
```

The processor retains 5,000 independent candidate positions and evaluates them with Stockfish at depth 15. Positions within ±100 centipawns are kept as approximately balanced target positions.

#### Step 3 — Collect empirical observations from 2019

```bash
python src/test_dataset_collector.py
```

The collector reads the six monthly Lichess files from January through June 2019, searches for the balanced target positions, and stores the first matching target position per game.

Outputs:

```text
data/results/test_dataset_detailed.jsonl
data/results/test_dataset_aggregated.json
```

The detailed `.jsonl` file is intentionally excluded from Git because of its size.

#### Step 4 — Create the 500-position development dataset

```bash
python src/create_top_dataset.py
```

Output:

```text
data/results/test_dataset_aggregated_top500.json
```

The 500 positions with the highest number of observations are selected deterministically; FEN is used as the tie-breaker.

#### Step 5 — Calculate structural metrics and Maia simulations

Both expensive analysis runners can be started together with:

```bash
python src/run_all_analysis.py
```

This sequentially executes:

```text
src/metric_runner.py
src/maia_runner.py
```

and creates/updates:

```text
data/results/position_metrics.json
data/results/maia_results.json
```

They can also be run separately:

```bash
python src/metric_runner.py
python src/maia_runner.py
```

Both runners use checkpoint/resume logic by default. If an existing output file was created with different experiment parameters, the scripts intentionally stop instead of silently combining incompatible results.

#### Step 6 — Generate the evaluation tables and figures

```bash
python src/complexity_analyzer.py
python src/maia_analyzer.py
```

The corresponding output directories are:

```text
data/results/complexity_analysis/
data/results/maia_analysis/
```

## Structural holdout validation

The structural metrics are additionally evaluated on positions ranked 501–600 of the complete aggregated dataset. These positions do not overlap with the 500-position development set.

The complete validation pipeline can be run with:

```bash
python src/validation/run_pipeline.py
```

It performs three steps:

1. selects the holdout positions,
2. calculates the structural metrics for them,
3. compares the holdout results with the development-set results.

Individual validation steps are located in `src/validation/`.

## Main experimental settings

The most important constants are defined directly in the scripts so the experiment configuration can be inspected without a separate configuration system.

| File | Main settings |
| --- | --- |
| `test_dataset_builder.py` | moves 11–20, maximum rating difference 200, Blitz/Rapid/Classical, top 50,000 candidates |
| `candidate_processor.py` | top 5,000 independent candidates, Stockfish depth 15, balance range ±100 cp |
| `metric_runner.py` | good-move threshold 50 cp, good-move depth 15, DTBMS depths 6–24 |
| `maia_runner.py` | Maia-3-5M, 90% observed-move coverage, 6 simulated plies, Stockfish leaf depth 12 |
| `complexity_analyzer.py` | five rating buckets, minimum 10 observations, 10 plotting bins |
| `maia_analyzer.py` | minimum 10 observations per move, Dirichlet prior 0.5, 20,000 Monte Carlo samples, seed 20260819 |

For exact definitions and all secondary parameters, use the constants and comments in the respective source files. The thesis provides the methodological motivation and interpretation of these choices.

## Important output files

The central files used by the analysis are:

```text
data/results/test_dataset_aggregated.json
    Complete aggregated empirical dataset for the selected target positions.

data/results/test_dataset_aggregated_top500.json
    500-position development dataset used for the main analysis.

data/results/position_metrics.json
    Stockfish-based structural metrics for the development positions.

data/results/maia_results.json
    Maia-3 tree-simulation results.

data/results/complexity_analysis/
    Structural metric summaries, CSV files and plots.

data/results/maia_analysis/
    Maia evaluation summaries and CSV files.
```

## Reproducibility and checkpoints

Several calculations are computationally expensive. The metric and Maia runners therefore save checkpoints and resume existing calculations by default.

A few practical rules:

- Run scripts from the repository root.
- Keep exactly one Stockfish executable in `engines/stockfish/`.
- Do not overwrite input datasets with result files.
- If experiment constants are changed, use a new output file or deliberately move/remove the previous checkpoint.
- The Maia simulation can be much slower on CPU than on a CUDA-capable GPU.
- The committed result files make it possible to inspect the analysis without repeating all expensive computations.

## Data and software attribution

The raw chess games come from the public Lichess database:

https://database.lichess.org/

Stockfish is obtained separately from the official Stockfish project.

Human move probabilities are generated with Maia-3 / Chessformer through the `maia3` Python package.

The external raw datasets, Stockfish executable and Maia model weights are not distributed as part of this repository.
