import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================
# PROJECT ROOT
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# =============================================================
# CONFIGURATION
# =============================================================

TOP_POSITIONS = 1000


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "test_dataset_aggregated.json"
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "test_dataset_aggregated_tops.json"
)


# =============================================================
# LOAD DATASET
# =============================================================

def load_dataset(
    input_path: Path
) -> Dict[str, Any]:
    """
    Loads the original aggregated test dataset.

    IMPORTANT:
    The input file is opened in read-only mode.
    Nothing is written to it.
    """

    if not input_path.exists():

        raise FileNotFoundError(
            "Input dataset not found:\n"
            + str(input_path)
        )


    with open(
        input_path,
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
            "Expected test_dataset_aggregated.json "
            "to contain a JSON object."
        )


    return data


# =============================================================
# SELECT TOP POSITIONS
# =============================================================

def select_top_positions(
    dataset: Dict[str, Any],
    number_of_positions: int
) -> Dict[str, Any]:
    """
    Selects the positions with the highest total_occurrences.

    Sorting:

        1. total_occurrences descending
        2. FEN alphabetically as deterministic tie-breaker

    The complete original data belonging to each selected FEN
    is retained unchanged.
    """

    sorted_positions = sorted(

        dataset.items(),

        key=lambda item: (

            -item[1].get(
                "total_occurrences",
                0
            ),

            item[0]
        )
    )


    selected_positions = (
        sorted_positions[
            :number_of_positions
        ]
    )


    return dict(
        selected_positions
    )


# =============================================================
# SAVE NEW DATASET
# =============================================================

def save_dataset(
    dataset: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Saves the reduced dataset to a NEW file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False
        )


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    # =========================================================
    # SAFETY CHECK
    # =========================================================
    #
    # This prevents accidental overwriting of the original
    # dataset even if the paths are changed later.
    #

    if (
        INPUT_FILE.resolve()
        == OUTPUT_FILE.resolve()
    ):

        raise ValueError(
            "Input and output path are identical. "
            "The original dataset must never be overwritten."
        )


    # =========================================================
    # LOAD ORIGINAL DATASET
    # =========================================================

    print(
        "Loading original dataset..."
    )


    dataset = (
        load_dataset(
            INPUT_FILE
        )
    )


    print(
        f"Positions in original dataset: "
        f"{len(dataset):,}"
    )


    # =========================================================
    # SELECT TOP POSITIONS
    # =========================================================

    top_dataset = (
        select_top_positions(
            dataset=
                dataset,
            number_of_positions=
                TOP_POSITIONS
        )
    )


    print(
        f"Positions selected: "
        f"{len(top_dataset):,}"
    )


    # =========================================================
    # INFORMATION ABOUT FREQUENCIES
    # =========================================================

    if len(top_dataset) > 0:

        occurrence_values = [

            position_data.get(
                "total_occurrences",
                0
            )

            for position_data
            in top_dataset.values()
        ]


        print(
            f"Highest occurrence count: "
            f"{max(occurrence_values):,}"
        )


        print(
            f"Lowest occurrence count "
            f"among selected positions: "
            f"{min(occurrence_values):,}"
        )


    # =========================================================
    # SAVE NEW FILE
    # =========================================================

    save_dataset(
        dataset=
            top_dataset,
        output_path=
            OUTPUT_FILE
    )


    print()

    print(
        "=" * 60
    )

    print(
        "Top dataset created successfully."
    )

    print(
        "=" * 60
    )


    print(
        "Original file remains unchanged:"
    )

    print(
        INPUT_FILE
    )


    print()

    print(
        "New file:"
    )

    print(
        OUTPUT_FILE
    )


# =============================================================
# PROGRAM START
# =============================================================

if __name__ == "__main__":

    main()