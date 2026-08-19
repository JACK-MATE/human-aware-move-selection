from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


# =============================================================
# PROJECT PATHS
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


# This file is inside:
#
#     src/validation/
#
# complexity_analyzer.py is inside:
#
#     src/
#
# Therefore src/ is added to the Python import path.
if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# =============================================================
# REUSE EXISTING DEVELOPMENT ANALYSIS FUNCTIONS
# =============================================================
#
# IMPORTANT:
#
# We do NOT run complexity_analyzer.py itself.
#
# We only reuse its definitions and helper functions so that:
#
# - complexity scores
# - empirical targets
# - rating buckets
# - Spearman calculation
#
# are implemented exactly the same way as during the
# development analysis.
# =============================================================

import src.complexity_analyzer as complexity


# =============================================================
# INPUT / OUTPUT
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


# -------------------------------------------------------------
# Detailed outputs
# -------------------------------------------------------------

POSITION_RESULTS_PATH = (
    OUTPUT_DIR
    / "validation_position_results.csv"
)


RATING_RESULTS_PATH = (
    OUTPUT_DIR
    / "validation_rating_results.csv"
)


# -------------------------------------------------------------
# Main validation outputs
# -------------------------------------------------------------

OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "validation_spearman_summary.csv"
)


RATING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "validation_rating_spearman_summary.csv"
)


ANALYSIS_JSON_PATH = (
    OUTPUT_DIR
    / "validation_analysis.json"
)


# =============================================================
# EXPECTED METRIC PARAMETERS
# =============================================================
#
# These are the parameters used during the development run.
#
# The validation run MUST use the same values.
#
# If metric_runner.py is accidentally changed before validation,
# this script will stop instead of comparing incompatible
# metrics.
# =============================================================

EXPECTED_GOOD_MOVE_THRESHOLD_CP = 50

EXPECTED_GOOD_MOVE_DEPTH = 15

EXPECTED_DTBMS_MIN_DEPTH = 6

EXPECTED_DTBMS_CANDIDATE_MAX_DEPTH = 20

EXPECTED_DTBMS_SEARCH_MAX_DEPTH = 24

EXPECTED_DTBMS_STEP = 2

EXPECTED_DTBMS_STABLE_STEPS = 3


# =============================================================
# JSON
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
# CSV
# =============================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, Any]]
) -> None:

    if not rows:

        return


    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=
                list(
                    rows[0].keys()
                )
        )


        writer.writeheader()


        writer.writerows(
            rows
        )


# =============================================================
# DISPLAY NAMES
# =============================================================

def display_model_name(
    model_name: str
) -> str:
    """
    The existing analyzer still uses "DTS" in some variable
    and model labels.

    The implemented metric is DTBMS.

    Only the OUTPUT LABEL is changed here.
    The underlying calculation is untouched.
    """

    return (
        model_name
        .replace(
            "DTS only",
            "DTBMS only"
        )
        .replace(
            "DTS +",
            "DTBMS +"
        )
    )


# =============================================================
# PARAMETER SAFETY CHECK
# =============================================================

def verify_metric_parameters(
    metric_output: Dict[str, Any]
) -> None:

    parameters = (
        metric_output.get(
            "metric_parameters",
            {}
        )
    )


    if not parameters:

        raise ValueError(
            "holdout_position_metrics.json contains no "
            "metric_parameters."
        )


    expected = {

        "good_move_max_loss_cp":
            EXPECTED_GOOD_MOVE_THRESHOLD_CP,

        "good_move_depth":
            EXPECTED_GOOD_MOVE_DEPTH,

        "dtbms_min_depth":
            EXPECTED_DTBMS_MIN_DEPTH,

        "dtbms_candidate_max_depth":
            EXPECTED_DTBMS_CANDIDATE_MAX_DEPTH,

        "dtbms_search_max_depth":
            EXPECTED_DTBMS_SEARCH_MAX_DEPTH,

        "dtbms_step":
            EXPECTED_DTBMS_STEP,

        "dtbms_stable_steps":
            EXPECTED_DTBMS_STABLE_STEPS,
    }


    problems = []


    for (
        parameter_name,
        expected_value
    ) in expected.items():

        actual_value = (
            parameters.get(
                parameter_name
            )
        )


        if (
            actual_value
            != expected_value
        ):

            problems.append(

                f"{parameter_name}: "
                f"expected {expected_value}, "
                f"found {actual_value}"
            )


    if problems:

        raise ValueError(

            "Holdout metrics were calculated with "
            "different parameters than expected.\n\n"

            + "\n".join(
                problems
            )
        )


# =============================================================
# SPEARMAN FOR ONE MODEL / TARGET
# =============================================================

def calculate_spearman_result(
    rows: List[Dict[str, Any]],
    model_name: str,
    target_name: str
) -> Dict[str, Any]:

    score_column = (
        complexity.MODEL_COLUMNS[
            model_name
        ]
    )


    target_column = (
        complexity.TARGET_COLUMNS[
            target_name
        ]
    )


    # ---------------------------------------------------------
    # Complexity values
    # ---------------------------------------------------------

    x = [

        float(
            row[
                score_column
            ]
        )

        for row in rows
    ]


    # ---------------------------------------------------------
    # Empirical error / not-best rates
    # ---------------------------------------------------------

    y = [

        float(
            row[
                target_column
            ]
        )

        for row in rows
    ]


    # ---------------------------------------------------------
    # Total number of underlying human observations
    # ---------------------------------------------------------

    observations = sum(

        int(
            row[
                "observations"
            ]
        )

        for row in rows
    )


    # ---------------------------------------------------------
    # Spearman
    # ---------------------------------------------------------
    #
    # No model is fitted here.
    #
    # The correlation is calculated completely independently
    # on the holdout positions.
    # ---------------------------------------------------------

    rho = (
        complexity.spearman_correlation(
            x,
            y
        )
    )


    return {

        "target":
            target_name,

        "model":
            display_model_name(
                model_name
            ),

        "positions":
            len(
                rows
            ),

        "observations":
            observations,

        "spearman_rho":
            rho
    }


# =============================================================
# OVERALL VALIDATION SUMMARY
# =============================================================

def calculate_overall_summary(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []


    for target_name in (
        complexity.TARGET_COLUMNS.keys()
    ):


        for model_name in (
            complexity.ALL_MODELS
        ):


            result = (
                calculate_spearman_result(

                    rows=
                        rows,

                    model_name=
                        model_name,

                    target_name=
                        target_name
                )
            )


            output.append(
                result
            )


    return output


# =============================================================
# RATING-SPECIFIC VALIDATION
# =============================================================

def calculate_rating_summary(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []


    for rating_bucket in (
        complexity.RATING_BUCKETS
    ):


        bucket_rows = [

            row

            for row in rows

            if (
                row[
                    "rating_bucket"
                ]
                == rating_bucket
            )
        ]


        # Spearman needs at least two positions.
        if len(
            bucket_rows
        ) < 2:

            continue


        for target_name in (
            complexity.TARGET_COLUMNS.keys()
        ):


            for model_name in (
                complexity.ALL_MODELS
            ):


                result = (
                    calculate_spearman_result(

                        rows=
                            bucket_rows,

                        model_name=
                            model_name,

                        target_name=
                            target_name
                    )
                )


                output.append(
                    {

                        "rating_bucket":
                            rating_bucket,

                        **result
                    }
                )


    return output


# =============================================================
# CONSOLE OUTPUT
# =============================================================

def format_rho(
    value: float
) -> str:

    if (
        isinstance(
            value,
            float
        )

        and

        math.isnan(
            value
        )
    ):

        return "n/a"


    return (
        f"{value:.3f}"
    )


def print_overall_summary(
    summary: List[Dict[str, Any]]
) -> None:

    print()

    print(
        "=" * 96
    )

    print(
        "HOLDOUT VALIDATION - OVERALL SPEARMAN RESULTS"
    )

    print(
        "=" * 96
    )


    for target_name in (
        complexity.TARGET_COLUMNS.keys()
    ):

        print()

        print(
            "#" * 96
        )

        print(
            f"TARGET: "
            f"{target_name}"
        )

        print(
            "#" * 96
        )


        print(

            f"{'Model':<32}"

            f"{'Spearman':>12}"

            f"{'Positions':>12}"

            f"{'Observations':>16}"
        )


        print(
            "-" * 72
        )


        target_rows = [

            row

            for row in summary

            if (
                row[
                    "target"
                ]
                == target_name
            )
        ]


        for row in target_rows:

            print(

                f"{row['model']:<32}"

                f"{format_rho(row['spearman_rho']):>12}"

                f"{row['positions']:>12}"

                f"{row['observations']:>16,}"
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
        "STRUCTURAL COMPLEXITY HOLDOUT VALIDATION"
    )

    print(
        "=" * 72
    )


    # =========================================================
    # LOAD HOLDOUT DATA
    # =========================================================

    print()

    print(
        "Loading holdout dataset..."
    )


    dataset = load_json(
        HOLDOUT_DATASET_PATH
    )


    print(
        f"Holdout positions: "
        f"{len(dataset):,}"
    )


    print()

    print(
        "Loading holdout position metrics..."
    )


    metric_output = load_json(
        HOLDOUT_METRICS_PATH
    )


    metric_positions = (
        metric_output.get(
            "positions",
            {}
        )
    )


    print(
        f"Metric positions: "
        f"{len(metric_positions):,}"
    )


    # =========================================================
    # VERIFY PARAMETERS
    # =========================================================

    verify_metric_parameters(
        metric_output
    )


    print()

    print(
        "Metric parameters match the development setup."
    )


    # =========================================================
    # BUILD ANALYSIS ROWS
    # =========================================================
    #
    # This is the same function used during development.
    #
    # It calculates:
    #
    # - GMR complexity
    # - DTBMS
    # - N transformations
    # - combined complexity variants
    # - >50 cp empirical error rate
    # - not-best-move rate
    #
    # =========================================================

    (
        overall_rows,
        rating_rows,
        diagnostics
    ) = (

        complexity.build_analysis_rows(
            dataset,
            metric_output
        )
    )


    if not overall_rows:

        raise RuntimeError(
            "No overall holdout analysis rows were created."
        )


    if not rating_rows:

        raise RuntimeError(
            "No rating-specific holdout rows were created."
        )


    # =========================================================
    # HARD COMPLETENESS CHECK
    # =========================================================

    missing_metric_positions = (
        diagnostics.get(
            "missing_metric_positions",
            []
        )
    )


    missing_best_move_positions = (
        diagnostics.get(
            "missing_best_move_positions",
            []
        )
    )


    missing_move_observations = int(
        diagnostics.get(
            "missing_move_observations",
            0
        )
    )


    if missing_metric_positions:

        raise RuntimeError(
            f"{len(missing_metric_positions)} holdout "
            "positions have no metric result."
        )


    if missing_best_move_positions:

        raise RuntimeError(
            f"{len(missing_best_move_positions)} holdout "
            "positions have no Stockfish best move."
        )


    if missing_move_observations > 0:

        print()

        print(
            "WARNING:"
        )

        print(
            f"{missing_move_observations:,} human "
            "observations could not be matched to "
            "stored Stockfish move data."
        )


    # =========================================================
    # SPEARMAN VALIDATION
    # =========================================================

    overall_summary = (
        calculate_overall_summary(
            overall_rows
        )
    )


    rating_summary = (
        calculate_rating_summary(
            rating_rows
        )
    )


    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # =========================================================
    # SAVE DETAILED DATA
    # =========================================================

    write_csv(
        POSITION_RESULTS_PATH,
        overall_rows
    )


    write_csv(
        RATING_RESULTS_PATH,
        rating_rows
    )


    # =========================================================
    # SAVE SPEARMAN RESULTS
    # =========================================================

    write_csv(
        OVERALL_SUMMARY_PATH,
        overall_summary
    )


    write_csv(
        RATING_SUMMARY_PATH,
        rating_summary
    )


    # =========================================================
    # SAVE JSON
    # =========================================================

    write_json(

        ANALYSIS_JSON_PATH,

        {

            "validation_type":
                (
                    "Position-level holdout validation "
                    "using ranks 501-600."
                ),

            "holdout_positions":
                len(
                    overall_rows
                ),

            "rating_specific_rows":
                len(
                    rating_rows
                ),

            "method":
                (
                    "The same structural metric definitions "
                    "and empirical targets as in the development "
                    "analysis are recalculated on previously "
                    "unused holdout positions. Validation is "
                    "based on Spearman rank correlation only. "
                    "No model parameters are fitted on the "
                    "holdout dataset."
                ),

            "metric_parameters":
                metric_output.get(
                    "metric_parameters",
                    {}
                ),

            "diagnostics":
                diagnostics,

            "overall_spearman_results":
                overall_summary,

            "rating_spearman_results":
                rating_summary
        }
    )


    # =========================================================
    # CONSOLE
    # =========================================================

    print_overall_summary(
        overall_summary
    )


    print()

    print(
        "=" * 72
    )

    print(
        "VALIDATION FINISHED"
    )

    print(
        "=" * 72
    )


    print()

    print(
        "Most important result:"
    )

    print(
        OVERALL_SUMMARY_PATH
    )


    print()

    print(
        "Complete validation:"
    )

    print(
        ANALYSIS_JSON_PATH
    )


if __name__ == "__main__":

    main()