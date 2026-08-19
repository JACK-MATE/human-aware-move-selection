from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================
# PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Complete aggregated dataset.
#
# READ ONLY.
SOURCE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "test_dataset_aggregated.json"
)


# Existing development Top 500.
#
# This file is only used for a safety check.
# It is NEVER modified.
DEVELOPMENT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "test_dataset_aggregated_top500.json"
)


# All validation outputs go into their own folder.
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


SELECTION_INFO_PATH = (
    OUTPUT_DIR
    / "holdout_selection_info.json"
)


# =============================================================
# SETTINGS
# =============================================================

DEVELOPMENT_SIZE = 500

HOLDOUT_START_RANK = 501
HOLDOUT_END_RANK = 600


EXPECTED_HOLDOUT_SIZE = (
    HOLDOUT_END_RANK
    - HOLDOUT_START_RANK
    + 1
)


# =============================================================
# JSON HELPERS
# =============================================================

def load_json(
    path: Path
) -> Dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{path}"
        )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )


    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            f"Expected JSON object in:\n"
            f"{path}"
        )


    return data


def write_json(
    path: Path,
    data: Any
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Write through a temporary file first.
    #
    # This avoids leaving a broken JSON file behind if the
    # program is interrupted while writing.
    temp_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )


    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


    temp_path.replace(
        path
    )


# =============================================================
# RANKING
# =============================================================

def rank_positions(
    dataset: Dict[str, Any]
) -> List[
    Tuple[
        str,
        Dict[str, Any]
    ]
]:

    """
    Uses exactly the same ranking rule as create_top_dataset.py.

    Primary criterion:
        total_occurrences descending

    Tie-break:
        FEN alphabetically
    """

    return sorted(

        dataset.items(),

        key=lambda item: (

            -int(
                item[1].get(
                    "total_occurrences",
                    0
                )
            ),

            item[0]
        )
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
        "CREATE COMPLEXITY HOLDOUT DATASET"
    )

    print(
        "=" * 72
    )


    # =========================================================
    # LOAD DATA
    # =========================================================

    source_dataset = load_json(
        SOURCE_DATASET_PATH
    )


    development_dataset = load_json(
        DEVELOPMENT_DATASET_PATH
    )


    ranked_positions = rank_positions(
        source_dataset
    )


    if (
        len(ranked_positions)
        < HOLDOUT_END_RANK
    ):

        raise ValueError(
            f"Source dataset contains only "
            f"{len(ranked_positions):,} positions, "
            f"but rank {HOLDOUT_END_RANK} "
            f"is required."
        )


    # =========================================================
    # VERIFY DEVELOPMENT TOP 500
    # =========================================================
    #
    # We want to be absolutely sure that the existing
    # development dataset really consists of ranks 1-500
    # according to exactly the same ranking rule.
    # =========================================================

    expected_development_fens = {

        fen

        for (
            fen,
            position_data
        ) in ranked_positions[
            :DEVELOPMENT_SIZE
        ]
    }


    actual_development_fens = set(
        development_dataset.keys()
    )


    if (
        actual_development_fens
        != expected_development_fens
    ):

        missing = (
            expected_development_fens
            - actual_development_fens
        )


        unexpected = (
            actual_development_fens
            - expected_development_fens
        )


        raise ValueError(
            "Existing development Top 500 does not match "
            "ranks 1-500 of the source dataset.\n"
            f"Missing expected FENs: "
            f"{len(missing)}\n"
            f"Unexpected FENs: "
            f"{len(unexpected)}\n"
            "Validation selection was stopped."
        )


    # =========================================================
    # SELECT RANKS 501-600
    # =========================================================
    #
    # Python starts indexing at 0.
    #
    # Therefore:
    #
    #     [500:600]
    #
    # corresponds exactly to ranks:
    #
    #     501-600
    #
    # =========================================================

    holdout_items = ranked_positions[

        HOLDOUT_START_RANK - 1:
        HOLDOUT_END_RANK
    ]


    holdout_dataset = dict(
        holdout_items
    )


    if (
        len(holdout_dataset)
        != EXPECTED_HOLDOUT_SIZE
    ):

        raise RuntimeError(
            "Unexpected holdout size: "
            f"{len(holdout_dataset)} instead of "
            f"{EXPECTED_HOLDOUT_SIZE}."
        )


    # =========================================================
    # HARD OVERLAP CHECK
    # =========================================================

    overlap = (

        set(
            holdout_dataset.keys()
        )

        &

        actual_development_fens
    )


    if overlap:

        raise RuntimeError(
            "Development/holdout FEN overlap detected: "
            f"{len(overlap)} positions."
        )


    # =========================================================
    # STATISTICS
    # =========================================================

    occurrences = [

        int(
            position_data.get(
                "total_occurrences",
                0
            )
        )

        for position_data
        in holdout_dataset.values()
    ]


    selection_info = {

        "source_dataset":
            str(
                SOURCE_DATASET_PATH
            ),

        "development_dataset":
            str(
                DEVELOPMENT_DATASET_PATH
            ),

        "source_positions":
            len(
                source_dataset
            ),

        "development_ranks": [
            1,
            DEVELOPMENT_SIZE
        ],

        "holdout_ranks": [
            HOLDOUT_START_RANK,
            HOLDOUT_END_RANK
        ],

        "holdout_positions":
            len(
                holdout_dataset
            ),

        "development_holdout_fen_overlap":
            len(
                overlap
            ),

        "highest_holdout_occurrences":
            max(
                occurrences
            ),

        "lowest_holdout_occurrences":
            min(
                occurrences
            ),

        "total_holdout_observations":
            sum(
                occurrences
            ),

        "selection_rule":
            (
                "total_occurrences descending; "
                "FEN alphabetical tie-break"
            )
    }


    # =========================================================
    # SAVE
    # =========================================================

    write_json(
        HOLDOUT_DATASET_PATH,
        holdout_dataset
    )


    write_json(
        SELECTION_INFO_PATH,
        selection_info
    )


    # =========================================================
    # OUTPUT
    # =========================================================

    print()

    print(
        f"Source positions: "
        f"{len(source_dataset):,}"
    )


    print(
        "Development ranks: 1-500"
    )


    print(
        "Holdout ranks: 501-600"
    )


    print(
        f"Holdout positions: "
        f"{len(holdout_dataset):,}"
    )


    print(
        f"FEN overlap with development: "
        f"{len(overlap)}"
    )


    print(
        f"Highest holdout occurrences: "
        f"{max(occurrences):,}"
    )


    print(
        f"Lowest holdout occurrences: "
        f"{min(occurrences):,}"
    )


    print(
        f"Total holdout observations: "
        f"{sum(occurrences):,}"
    )


    print()

    print(
        "Created:"
    )

    print(
        HOLDOUT_DATASET_PATH
    )

    print(
        SELECTION_INFO_PATH
    )


if __name__ == "__main__":

    main()