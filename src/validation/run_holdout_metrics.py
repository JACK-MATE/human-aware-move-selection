from __future__ import annotations

import sys
from pathlib import Path


# =============================================================
# PATHS
# =============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


SRC_DIR = (
    PROJECT_ROOT
    / "src"
)


# Allows this script to import metric_runner.py from src/
# even though this file itself is inside src/validation/.
if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


from metric_runner import MetricRunner


# =============================================================
# VALIDATION FILES
# =============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "complexity_validation"
)


HOLDOUT_DATASET_PATH = (
    OUTPUT_DIR
    / "holdout_dataset_ranks_501_600.json"
)


HOLDOUT_METRICS_PATH = (
    OUTPUT_DIR
    / "holdout_position_metrics.json"
)


# =============================================================
# STOCKFISH
# =============================================================

STOCKFISH_DIRECTORY = (
    PROJECT_ROOT
    / "engines"
    / "stockfish"
)


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    print()

    print(
        "=" * 72
    )

    print(
        "CALCULATE HOLDOUT POSITION METRICS"
    )

    print(
        "=" * 72
    )


    if not HOLDOUT_DATASET_PATH.exists():

        raise FileNotFoundError(
            "Holdout dataset not found. "
            "Run create_holdout_dataset.py first:\n"
            f"{HOLDOUT_DATASET_PATH}"
        )


    runner = MetricRunner()


    # =========================================================
    # RUN EXISTING METRIC IMPLEMENTATION
    # =========================================================
    #
    # This uses exactly the same MetricRunner as the
    # development dataset.
    #
    # The output path is completely separate.
    #
    # MetricRunner.RESUME also remains active, meaning an
    # interrupted run can continue from this validation file.
    # =========================================================

    runner.run(

        dataset_path=
            HOLDOUT_DATASET_PATH,

        output_path=
            HOLDOUT_METRICS_PATH,

        stockfish_directory=
            STOCKFISH_DIRECTORY
    )


    print()

    print(
        "Created / updated:"
    )

    print(
        HOLDOUT_METRICS_PATH
    )


if __name__ == "__main__":

    main()