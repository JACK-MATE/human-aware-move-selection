import json
from pathlib import Path
from typing import Any, Dict


# =============================================================
# PROJECT PATHS
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


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
    / "test_dataset_aggregated_top500.json"
)


# =============================================================
# SETTINGS
# =============================================================

TOP_POSITIONS = 500


# =============================================================
# LOAD ORIGINAL DATASET
# =============================================================

def load_dataset(
    input_path: Path
) -> Dict[str, Any]:
    """
    Loads the original aggregated dataset.

    IMPORTANT:
    The source file is opened READ ONLY and is never modified.
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
            "Expected the dataset to contain a JSON object."
        )


    return data


# =============================================================
# SELECT MOST FREQUENT POSITIONS
# =============================================================

def select_top_positions(
    dataset: Dict[str, Any],
    number_of_positions: int
) -> Dict[str, Any]:
    """
    Selects the positions with the highest total_occurrences.

    Ties are resolved alphabetically by FEN so that the result
    is deterministic.

    The complete original entry of every selected FEN is copied.
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


    return dict(
        sorted_positions[
            :number_of_positions
        ]
    )


# =============================================================
# SAVE NEW DATASET
# =============================================================

def save_dataset(
    dataset: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Writes the selected positions to a NEW JSON file.
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


    # Hard safety check:
    # input and output may never be the same file.
    if (
        INPUT_FILE.resolve()
        == OUTPUT_FILE.resolve()
    ):

        raise ValueError(
            "Input and output file must be different."
        )


    print(
        "Loading original dataset..."
    )


    dataset = (
        load_dataset(
            INPUT_FILE
        )
    )


    print(
        f"Original positions: "
        f"{len(dataset):,}"
    )


    top_dataset = (
        select_top_positions(
            dataset=
                dataset,
            number_of_positions=
                TOP_POSITIONS
        )
    )


    occurrences = [

        position_data.get(
            "total_occurrences",
            0
        )

        for position_data
        in top_dataset.values()
    ]


    print(
        f"Selected positions: "
        f"{len(top_dataset):,}"
    )


    if occurrences:

        print(
            f"Highest occurrence count: "
            f"{max(occurrences):,}"
        )

        print(
            f"Lowest occurrence count: "
            f"{min(occurrences):,}"
        )


    save_dataset(
        dataset=
            top_dataset,
        output_path=
            OUTPUT_FILE
    )


    print()

    print(
        "Created:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Original dataset was not modified."
    )


if __name__ == "__main__":

    main()