from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# =============================================================
# PATHS
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "test_dataset_aggregated_top500.json"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "position_metrics.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "complexity_analysis"
)


# =============================================================
# SETTINGS
# =============================================================

RATING_BUCKETS = [
    "1400-1599",
    "1600-1799",
    "1800-1999",
    "2000-2199",
    "2200-2399",
]

RATING_MIDPOINTS = {
    "1400-1599": 1500,
    "1600-1799": 1700,
    "1800-1999": 1900,
    "2000-2199": 2100,
    "2200-2399": 2300,
}

ERROR_THRESHOLD_CP = 50
MIN_RATING_BUCKET_OBSERVATIONS = 10
N_BINS = 10
SHOW_PLOTS = False


# =============================================================
# TARGETS
# =============================================================

TARGET_ERROR = "Next-move error > 50 cp"

TARGET_NOT_BEST = (
    "Did not play Stockfish best move"
)

TARGET_COLUMNS = {
    TARGET_ERROR:
        "actual_error_rate_50cp",

    TARGET_NOT_BEST:
        "actual_not_best_move_rate",
}

TARGET_SHORT_NAMES = {
    TARGET_ERROR:
        "error_50cp",

    TARGET_NOT_BEST:
        "not_best_move",
}


# =============================================================
# MODELS
# =============================================================

MODEL_GMR = (
    "GMR: 1 - GMR"
)

MODEL_DTS_INV_N = (
    "DTS + 1/N"
)

MODEL_DTS_INV_SQRT_N = (
    "DTS + 1/sqrt(N)"
)

MODEL_DTS_INV_LOG_N = (
    "DTS + 1/log2(N+1)"
)


PRIMARY_MODELS = [
    MODEL_GMR,
    MODEL_DTS_INV_N,
    MODEL_DTS_INV_SQRT_N,
    MODEL_DTS_INV_LOG_N,
]


# =============================================================
# DIAGNOSTIC COMPONENTS
# =============================================================
#
# These are NOT extra proposed final complexity metrics.
#
# They only help us understand which part of the
# DTS + N combination carries information.
# =============================================================

MODEL_DTS_ONLY = (
    "DTS only"
)

MODEL_INV_N_ONLY = (
    "1/N only"
)

MODEL_INV_SQRT_N_ONLY = (
    "1/sqrt(N) only"
)

MODEL_INV_LOG_N_ONLY = (
    "1/log2(N+1) only"
)


DIAGNOSTIC_MODELS = [
    MODEL_DTS_ONLY,
    MODEL_INV_N_ONLY,
    MODEL_INV_SQRT_N_ONLY,
    MODEL_INV_LOG_N_ONLY,
]


ALL_MODELS = (
    PRIMARY_MODELS
    + DIAGNOSTIC_MODELS
)


MODEL_COLUMNS = {

    MODEL_GMR:
        "complexity_gmr",

    MODEL_DTS_INV_N:
        "complexity_dts_inv_n",

    MODEL_DTS_INV_SQRT_N:
        "complexity_dts_inv_sqrt_n",

    MODEL_DTS_INV_LOG_N:
        "complexity_dts_inv_log_n",

    MODEL_DTS_ONLY:
        "dts_normalized",

    MODEL_INV_N_ONLY:
        "breadth_inv_n",

    MODEL_INV_SQRT_N_ONLY:
        "breadth_inv_sqrt_n",

    MODEL_INV_LOG_N_ONLY:
        "breadth_inv_log_n",
}


# =============================================================
# LOAD JSON
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
            f"Expected a JSON object in:\n"
            f"{path}"
        )


    return data


# =============================================================
# RATING BUCKET HELPERS
# =============================================================

def normalize_bucket_name(
    name: str
) -> str:

    return (
        name
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )


def get_bucket_data(
    root_data: Dict[str, Any],
    wanted_bucket: str
) -> Optional[Dict[str, Any]]:

    wanted = normalize_bucket_name(
        wanted_bucket
    )


    for (
        bucket_name,
        bucket_data
    ) in root_data.get(
        "rating_buckets",
        {}
    ).items():


        if (
            normalize_bucket_name(
                bucket_name
            )
            == wanted
        ):

            return bucket_data


    return None


# =============================================================
# DTS NORMALIZATION
# =============================================================

def get_dts_normalization_bounds(
    metric_output: Dict[str, Any]
) -> Tuple[
    float,
    float
]:
    """
    Current DTBMS setup:

    Candidate depths:
        6, 8, 10, ..., 20

    If no stable candidate <= 20:
        stored value = 22

    Normalization:

        6  -> 0.0
        22 -> 1.0
    """

    parameters = metric_output.get(
        "metric_parameters",
        {}
    )


    min_depth = float(
        parameters.get(
            "dtbms_min_depth",
            6
        )
    )


    candidate_max = float(
        parameters.get(
            "dtbms_candidate_max_depth",
            20
        )
    )


    step = float(
        parameters.get(
            "dtbms_step",
            2
        )
    )


    max_depth = (
        candidate_max
        + step
    )


    if max_depth <= min_depth:

        raise ValueError(
            "Invalid DTS normalization bounds."
        )


    return (
        min_depth,
        max_depth
    )


def normalize_dts(
    raw_dts: float,
    min_depth: float,
    max_depth: float
) -> float:

    value = (
        (
            float(
                raw_dts
            )
            - min_depth
        )
        /
        (
            max_depth
            - min_depth
        )
    )


    return min(
        1.0,
        max(
            0.0,
            value
        )
    )


# =============================================================
# COMPLEXITY SCORES
# =============================================================

def calculate_complexities(
    metric_data: Dict[str, Any],
    dts_min: float,
    dts_max: float
) -> Dict[str, float]:

    gmr = float(
        metric_data[
            "good_move_ratio"
        ]
    )


    number_of_good_moves = int(
        metric_data[
            "number_of_good_moves"
        ]
    )


    raw_dts = float(
        metric_data[
            "best_move_dts"
        ]
    )


    if number_of_good_moves <= 0:

        raise ValueError(
            "number_of_good_moves must be >= 1."
        )


    # =========================================================
    # NORMALIZED DTS
    # =========================================================

    dts = normalize_dts(
        raw_dts,
        dts_min,
        dts_max
    )


    # =========================================================
    # BREADTH TRANSFORMATIONS
    # =========================================================

    inv_n = (
        1.0
        / number_of_good_moves
    )


    inv_sqrt_n = (
        1.0
        / math.sqrt(
            number_of_good_moves
        )
    )


    inv_log_n = (
        1.0
        /
        math.log2(
            number_of_good_moves
            + 1.0
        )
    )


    # =========================================================
    # GMR COMPLEXITY
    # =========================================================

    complexity_gmr = (
        1.0
        - gmr
    )


    # =========================================================
    # DTS + N COMPLEXITY
    #
    # Initial assumption:
    #
    # 50 % depth
    # 50 % breadth
    # =========================================================

    complexity_dts_inv_n = (
        dts
        + inv_n
    ) / 2.0


    complexity_dts_inv_sqrt_n = (
        dts
        + inv_sqrt_n
    ) / 2.0


    complexity_dts_inv_log_n = (
        dts
        + inv_log_n
    ) / 2.0


    return {

        "gmr":
            gmr,

        "number_of_good_moves":
            number_of_good_moves,

        "dts_raw":
            raw_dts,

        "dts_normalized":
            dts,

        "breadth_inv_n":
            inv_n,

        "breadth_inv_sqrt_n":
            inv_sqrt_n,

        "breadth_inv_log_n":
            inv_log_n,

        "complexity_gmr":
            complexity_gmr,

        "complexity_dts_inv_n":
            complexity_dts_inv_n,

        "complexity_dts_inv_sqrt_n":
            complexity_dts_inv_sqrt_n,

        "complexity_dts_inv_log_n":
            complexity_dts_inv_log_n,
    }


# =============================================================
# EMPIRICAL OUTCOMES
# =============================================================

def calculate_bucket_outcomes(
    bucket_data: Dict[str, Any],
    stockfish_moves: Dict[str, Any],
    stockfish_best_move: str
) -> Dict[str, Any]:
    """
    Calculates two empirical targets.

    TARGET 1:
        Immediate next-move error.

        A move counts as an error if:
            Stockfish loss > 50 cp


    TARGET 2:
        Failure to find the final best move.

        A move counts as failure if:
            played move != Stockfish best move


    The second target is particularly relevant to DTBMS,
    because DTBMS measures how early the identity of the
    best move becomes stable.
    """

    observations = 0

    errors_50cp = 0

    best_move_hits = 0


    missing_move_observations = 0

    missing_moves = []


    for (
        move_uci,
        observed_data
    ) in bucket_data.get(
        "moves",
        {}
    ).items():


        count = int(
            observed_data.get(
                "count",
                0
            )
        )


        if count <= 0:

            continue


        stockfish_move_data = (
            stockfish_moves.get(
                move_uci
            )
        )


        # -----------------------------------------------------
        # Normally this should never happen.
        #
        # If it does, we exclude the observation from both
        # targets so both targets use the same denominator.
        # -----------------------------------------------------

        if stockfish_move_data is None:

            missing_move_observations += (
                count
            )

            missing_moves.append(
                move_uci
            )

            continue


        observations += (
            count
        )


        # =====================================================
        # BEST MOVE TARGET
        # =====================================================

        if (
            move_uci
            == stockfish_best_move
        ):

            best_move_hits += (
                count
            )


        # =====================================================
        # 50 CP ERROR TARGET
        # =====================================================

        loss_cp = float(
            stockfish_move_data[
                "loss_cp"
            ]
        )


        if (
            loss_cp
            > ERROR_THRESHOLD_CP
        ):

            errors_50cp += (
                count
            )


    if observations <= 0:

        return {

            "observations":
                0,

            "errors_50cp":
                0,

            "actual_error_rate_50cp":
                None,

            "best_move_hits":
                0,

            "actual_best_move_rate":
                None,

            "not_best_move_count":
                0,

            "actual_not_best_move_rate":
                None,

            "missing_move_observations":
                missing_move_observations,

            "missing_moves":
                missing_moves,
        }


    not_best_move_count = (
        observations
        - best_move_hits
    )


    return {

        "observations":
            observations,

        "errors_50cp":
            errors_50cp,

        "actual_error_rate_50cp":
            (
                errors_50cp
                / observations
            ),

        "best_move_hits":
            best_move_hits,

        "actual_best_move_rate":
            (
                best_move_hits
                / observations
            ),

        "not_best_move_count":
            not_best_move_count,

        "actual_not_best_move_rate":
            (
                not_best_move_count
                / observations
            ),

        "missing_move_observations":
            missing_move_observations,

        "missing_moves":
            missing_moves,
    }


# =============================================================
# BUILD ANALYSIS ROWS
# =============================================================

def build_analysis_rows(
    dataset: Dict[str, Any],
    metric_output: Dict[str, Any]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any]
]:

    metric_positions = (
        metric_output.get(
            "positions",
            {}
        )
    )


    (
        dts_min,
        dts_max
    ) = get_dts_normalization_bounds(
        metric_output
    )


    overall_rows = []

    rating_rows = []


    diagnostics = {

        "dataset_positions":
            len(
                dataset
            ),

        "metric_positions":
            len(
                metric_positions
            ),

        "matched_positions":
            0,

        "missing_metric_positions":
            [],

        "missing_best_move_positions":
            [],

        "missing_move_observations":
            0,

        "missing_moves":
            [],

        "dts_normalization_min":
            dts_min,

        "dts_normalization_max":
            dts_max,
    }


    # =========================================================
    # POSITIONS
    # =========================================================

    for (
        fen,
        root_data
    ) in dataset.items():


        metric_data = (
            metric_positions.get(
                fen
            )
        )


        if metric_data is None:

            diagnostics[
                "missing_metric_positions"
            ].append(
                fen
            )

            continue


        stockfish_data = (
            metric_data.get(
                "stockfish",
                {}
            )
        )


        stockfish_best_move = (
            stockfish_data.get(
                "best_move"
            )
        )


        if not stockfish_best_move:

            diagnostics[
                "missing_best_move_positions"
            ].append(
                fen
            )

            continue


        diagnostics[
            "matched_positions"
        ] += 1


        # =====================================================
        # COMPLEXITY SCORES
        # =====================================================

        complexities = (
            calculate_complexities(
                metric_data,
                dts_min,
                dts_max
            )
        )


        stockfish_moves = (
            stockfish_data.get(
                "moves",
                {}
            )
        )


        # =====================================================
        # TOTAL COUNTERS FOR THIS FEN
        # =====================================================

        total_observations = 0

        total_errors_50cp = 0

        total_best_move_hits = 0


        # =====================================================
        # RATING BUCKETS
        # =====================================================

        for bucket in RATING_BUCKETS:


            bucket_data = (
                get_bucket_data(
                    root_data,
                    bucket
                )
            )


            if bucket_data is None:

                continue


            outcomes = (
                calculate_bucket_outcomes(
                    bucket_data,
                    stockfish_moves,
                    stockfish_best_move
                )
            )


            observations = (
                outcomes[
                    "observations"
                ]
            )


            if observations <= 0:

                continue


            diagnostics[
                "missing_move_observations"
            ] += (
                outcomes[
                    "missing_move_observations"
                ]
            )


            for move_uci in (
                outcomes[
                    "missing_moves"
                ]
            ):

                diagnostics[
                    "missing_moves"
                ].append(
                    {

                        "fen":
                            fen,

                        "rating_bucket":
                            bucket,

                        "move":
                            move_uci,
                    }
                )


            # =================================================
            # ADD TO OVERALL FEN COUNTERS
            # =================================================

            total_observations += (
                observations
            )


            total_errors_50cp += (
                outcomes[
                    "errors_50cp"
                ]
            )


            total_best_move_hits += (
                outcomes[
                    "best_move_hits"
                ]
            )


            # =================================================
            # RATING-SPECIFIC ROW
            # =================================================

            if (
                observations
                >= MIN_RATING_BUCKET_OBSERVATIONS
            ):

                rating_rows.append(
                    {

                        "fen":
                            fen,

                        "rating_bucket":
                            bucket,

                        "rating_midpoint":
                            RATING_MIDPOINTS[
                                bucket
                            ],

                        "observations":
                            observations,

                        "errors_50cp":
                            outcomes[
                                "errors_50cp"
                            ],

                        "actual_error_rate_50cp":
                            outcomes[
                                "actual_error_rate_50cp"
                            ],

                        "best_move_hits":
                            outcomes[
                                "best_move_hits"
                            ],

                        "actual_best_move_rate":
                            outcomes[
                                "actual_best_move_rate"
                            ],

                        "not_best_move_count":
                            outcomes[
                                "not_best_move_count"
                            ],

                        "actual_not_best_move_rate":
                            outcomes[
                                "actual_not_best_move_rate"
                            ],

                        "stockfish_best_move":
                            stockfish_best_move,

                        **complexities,
                    }
                )


        # =====================================================
        # OVERALL ROW FOR FEN
        # =====================================================

        if total_observations > 0:


            total_not_best = (
                total_observations
                - total_best_move_hits
            )


            overall_rows.append(
                {

                    "fen":
                        fen,

                    "observations":
                        total_observations,

                    "errors_50cp":
                        total_errors_50cp,

                    "actual_error_rate_50cp":
                        (
                            total_errors_50cp
                            / total_observations
                        ),

                    "best_move_hits":
                        total_best_move_hits,

                    "actual_best_move_rate":
                        (
                            total_best_move_hits
                            / total_observations
                        ),

                    "not_best_move_count":
                        total_not_best,

                    "actual_not_best_move_rate":
                        (
                            total_not_best
                            / total_observations
                        ),

                    "stockfish_best_move":
                        stockfish_best_move,

                    **complexities,
                }
            )


    return (
        overall_rows,
        rating_rows,
        diagnostics
    )


# =============================================================
# RANKS
# =============================================================

def average_ranks(
    values: List[float]
) -> List[float]:

    indexed = sorted(
        enumerate(
            values
        ),
        key=lambda item:
            item[1]
    )


    ranks = [
        0.0
        for _ in values
    ]


    i = 0


    while i < len(
        indexed
    ):


        j = (
            i
            + 1
        )


        while (
            j
            < len(indexed)
            and
            indexed[j][1]
            == indexed[i][1]
        ):

            j += 1


        average_rank = (
            (
                i
                + 1
            )
            + j
        ) / 2.0


        for k in range(
            i,
            j
        ):

            ranks[
                indexed[k][0]
            ] = (
                average_rank
            )


        i = (
            j
        )


    return ranks


# =============================================================
# PEARSON
# =============================================================

def pearson_correlation(
    x: List[float],
    y: List[float]
) -> float:

    if len(x) != len(y):

        raise ValueError(
            "x and y must have equal length."
        )


    if len(x) < 2:

        return float(
            "nan"
        )


    mean_x = (
        sum(x)
        / len(x)
    )


    mean_y = (
        sum(y)
        / len(y)
    )


    numerator = sum(

        (
            value_x
            - mean_x
        )
        *
        (
            value_y
            - mean_y
        )

        for (
            value_x,
            value_y
        ) in zip(
            x,
            y
        )
    )


    denominator_x = math.sqrt(
        sum(

            (
                value
                - mean_x
            ) ** 2

            for value in x
        )
    )


    denominator_y = math.sqrt(
        sum(

            (
                value
                - mean_y
            ) ** 2

            for value in y
        )
    )


    denominator = (
        denominator_x
        * denominator_y
    )


    if denominator == 0:

        return float(
            "nan"
        )


    return (
        numerator
        / denominator
    )


# =============================================================
# SPEARMAN
# =============================================================

def spearman_correlation(
    x: List[float],
    y: List[float]
) -> float:

    if len(x) != len(y):

        raise ValueError(
            "x and y must have equal length."
        )


    if len(x) < 2:

        return float(
            "nan"
        )


    return pearson_correlation(

        average_ranks(
            x
        ),

        average_ranks(
            y
        )
    )


# =============================================================
# SIMPLE LINEAR PREDICTION
# =============================================================

def fit_linear_model(
    x: List[float],
    y: List[float]
) -> Tuple[
    float,
    float
]:

    if len(x) != len(y):

        raise ValueError(
            "x and y must have equal length."
        )


    if len(x) < 2:

        return (
            0.0,
            0.0
        )


    mean_x = (
        sum(x)
        / len(x)
    )


    mean_y = (
        sum(y)
        / len(y)
    )


    denominator = sum(

        (
            value
            - mean_x
        ) ** 2

        for value in x
    )


    if denominator == 0:

        return (
            mean_y,
            0.0
        )


    numerator = sum(

        (
            value_x
            - mean_x
        )
        *
        (
            value_y
            - mean_y
        )

        for (
            value_x,
            value_y
        ) in zip(
            x,
            y
        )
    )


    slope = (
        numerator
        / denominator
    )


    intercept = (
        mean_y
        - slope
        * mean_x
    )


    return (
        intercept,
        slope
    )


# =============================================================
# CLIP PROBABILITY
# =============================================================

def clip_probability(
    value: float
) -> float:

    return min(
        1.0,
        max(
            0.0,
            value
        )
    )


# =============================================================
# MODEL QUALITY
# =============================================================

def prediction_metrics(
    rows: List[Dict[str, Any]],
    score_column: str,
    target_column: str
) -> Dict[str, float]:

    x = [

        float(
            row[
                score_column
            ]
        )

        for row in rows
    ]


    y = [

        float(
            row[
                target_column
            ]
        )

        for row in rows
    ]


    weights = [

        int(
            row[
                "observations"
            ]
        )

        for row in rows
    ]


    # =========================================================
    # SPEARMAN
    # =========================================================

    rho = spearman_correlation(
        x,
        y
    )


    # =========================================================
    # LINEAR CALIBRATION
    # =========================================================

    (
        intercept,
        slope
    ) = fit_linear_model(
        x,
        y
    )


    predicted = [

        clip_probability(

            intercept
            + slope
            * complexity
        )

        for complexity in x
    ]


    absolute_errors = [

        abs(
            prediction
            - actual
        )

        for (
            prediction,
            actual
        ) in zip(
            predicted,
            y
        )
    ]


    # =========================================================
    # MAE
    # =========================================================

    if absolute_errors:

        mae = (
            sum(
                absolute_errors
            )
            /
            len(
                absolute_errors
            )
        )

    else:

        mae = float(
            "nan"
        )


    # =========================================================
    # WEIGHTED MAE
    # =========================================================

    total_weight = (
        sum(
            weights
        )
    )


    if total_weight > 0:

        weighted_mae = (
            sum(

                error
                * weight

                for (
                    error,
                    weight
                ) in zip(
                    absolute_errors,
                    weights
                )
            )
            /
            total_weight
        )

    else:

        weighted_mae = float(
            "nan"
        )


    return {

        "n_rows":
            len(
                rows
            ),

        "observations":
            total_weight,

        "spearman_rho":
            rho,

        "linear_intercept":
            intercept,

        "linear_slope":
            slope,

        "mae":
            mae,

        "weighted_mae":
            weighted_mae,
    }


# =============================================================
# OVERALL SUMMARY
# =============================================================

def calculate_overall_summary(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []


    for (
        target_name,
        target_column
    ) in TARGET_COLUMNS.items():


        for model_name in ALL_MODELS:


            result = prediction_metrics(

                rows,

                MODEL_COLUMNS[
                    model_name
                ],

                target_column
            )


            output.append(
                {

                    "target":
                        target_name,

                    "model":
                        model_name,

                    **result,
                }
            )


    return output


# =============================================================
# RATING SUMMARY
# =============================================================

def calculate_rating_summary(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []


    for bucket in RATING_BUCKETS:


        bucket_rows = [

            row

            for row in rows

            if (
                row[
                    "rating_bucket"
                ]
                == bucket
            )
        ]


        if len(
            bucket_rows
        ) < 2:

            continue


        for (
            target_name,
            target_column
        ) in TARGET_COLUMNS.items():


            for model_name in ALL_MODELS:


                result = prediction_metrics(

                    bucket_rows,

                    MODEL_COLUMNS[
                        model_name
                    ],

                    target_column
                )


                output.append(
                    {

                        "rating_bucket":
                            bucket,

                        "rating_midpoint":
                            RATING_MIDPOINTS[
                                bucket
                            ],

                        "target":
                            target_name,

                        "model":
                            model_name,

                        **result,
                    }
                )


    return output


# =============================================================
# EMPIRICAL TARGET RATES BY RATING
# =============================================================

def calculate_rating_target_rates(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []


    for bucket in RATING_BUCKETS:


        bucket_rows = [

            row

            for row in rows

            if (
                row[
                    "rating_bucket"
                ]
                == bucket
            )
        ]


        if not bucket_rows:

            continue


        observations = sum(

            int(
                row[
                    "observations"
                ]
            )

            for row in bucket_rows
        )


        errors = sum(

            int(
                row[
                    "errors_50cp"
                ]
            )

            for row in bucket_rows
        )


        best_move_hits = sum(

            int(
                row[
                    "best_move_hits"
                ]
            )

            for row in bucket_rows
        )


        output.append(
            {

                "rating_bucket":
                    bucket,

                "rating_midpoint":
                    RATING_MIDPOINTS[
                        bucket
                    ],

                "positions":
                    len(
                        bucket_rows
                    ),

                "observations":
                    observations,

                "error_rate_50cp":
                    (
                        errors
                        / observations
                    ),

                "best_move_rate":
                    (
                        best_move_hits
                        / observations
                    ),

                "not_best_move_rate":
                    (
                        observations
                        - best_move_hits
                    )
                    / observations,
            }
        )


    return output


# =============================================================
# BINNING
# =============================================================

def build_quantile_bins(
    rows: List[Dict[str, Any]],
    score_column: str,
    target_column: str
) -> List[Dict[str, float]]:

    ordered = sorted(

        rows,

        key=lambda row:
            float(
                row[
                    score_column
                ]
            )
    )


    if not ordered:

        return []


    bin_count = min(
        N_BINS,
        len(
            ordered
        )
    )


    bins = []


    for bin_index in range(
        bin_count
    ):


        start = (
            bin_index
            * len(
                ordered
            )
            // bin_count
        )


        end = (
            (
                bin_index
                + 1
            )
            * len(
                ordered
            )
            // bin_count
        )


        group = ordered[
            start:end
        ]


        observations = sum(

            int(
                row[
                    "observations"
                ]
            )

            for row in group
        )


        if observations <= 0:

            continue


        mean_complexity = (
            sum(

                float(
                    row[
                        score_column
                    ]
                )
                *
                int(
                    row[
                        "observations"
                    ]
                )

                for row in group
            )
            /
            observations
        )


        target_rate = (
            sum(

                float(
                    row[
                        target_column
                    ]
                )
                *
                int(
                    row[
                        "observations"
                    ]
                )

                for row in group
            )
            /
            observations
        )


        bins.append(
            {

                "bin":
                    bin_index
                    + 1,

                "mean_complexity":
                    mean_complexity,

                "target_rate":
                    target_rate,

                "observations":
                    observations,

                "positions":
                    len(
                        group
                    ),
            }
        )


    return bins


# =============================================================
# FILE NAME HELPER
# =============================================================

def safe_name(
    text: str
) -> str:

    return (
        text
        .lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(">", "gt")
    )


# =============================================================
# FIND SUMMARY ROW
# =============================================================

def find_summary_row(
    summary: List[Dict[str, Any]],
    target_name: str,
    model_name: str
) -> Optional[Dict[str, Any]]:


    for row in summary:


        if (
            row[
                "target"
            ]
            == target_name
            and
            row[
                "model"
            ]
            == model_name
        ):

            return row


    return None


# =============================================================
# PLOT COMPARISON
# =============================================================

def plot_comparison(
    summary: List[Dict[str, Any]],
    target_name: str,
    models: List[str],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path
) -> None:


    selected = []


    for model_name in models:


        row = find_summary_row(
            summary,
            target_name,
            model_name
        )


        if row is not None:

            selected.append(
                row
            )


    labels = [

        row[
            "model"
        ]

        for row in selected
    ]


    values = [

        row[
            metric_key
        ]

        for row in selected
    ]


    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5
        )
    )


    ax.bar(

        range(
            len(
                labels
            )
        ),

        values
    )


    ax.set_xticks(

        range(
            len(
                labels
            )
        )
    )


    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )


    if metric_key == "spearman_rho":

        ax.axhline(
            0,
            linewidth=1
        )


    ax.grid(
        axis="y",
        alpha=0.25
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=180
    )


    if SHOW_PLOTS:

        plt.show()


    plt.close(
        fig
    )


# =============================================================
# PLOT ONE MODEL OVERALL
# =============================================================

def plot_overall_model(
    rows: List[Dict[str, Any]],
    model_name: str,
    target_name: str,
    output_path: Path
) -> None:


    score_column = (
        MODEL_COLUMNS[
            model_name
        ]
    )


    target_column = (
        TARGET_COLUMNS[
            target_name
        ]
    )


    metrics = prediction_metrics(
        rows,
        score_column,
        target_column
    )


    bins = build_quantile_bins(
        rows,
        score_column,
        target_column
    )


    x_bins = [

        row[
            "mean_complexity"
        ]

        for row in bins
    ]


    y_bins = [

        row[
            "target_rate"
        ]

        for row in bins
    ]


    x_all = [

        float(
            row[
                score_column
            ]
        )

        for row in rows
    ]


    min_x = min(
        x_all
    )


    max_x = max(
        x_all
    )


    if max_x == min_x:

        line_x = [
            min_x,
            max_x
        ]

    else:

        line_x = [

            min_x
            +
            (
                max_x
                - min_x
            )
            * i
            / 100

            for i in range(
                101
            )
        ]


    line_y = [

        clip_probability(

            metrics[
                "linear_intercept"
            ]
            +
            metrics[
                "linear_slope"
            ]
            * value
        )

        for value in line_x
    ]


    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5
        )
    )


    ax.plot(
        x_bins,
        y_bins,
        marker="o",
        label=
            "Observed rate "
            "(10 groups)"
    )


    ax.plot(
        line_x,
        line_y,
        linestyle="--",
        label=
            "Linear prediction"
    )


    ax.set_xlabel(
        "Complexity score"
    )


    ax.set_ylabel(
        target_name
    )


    ax.set_ylim(
        0,
        1
    )


    ax.set_title(
        f"{model_name}\n"
        f"{target_name}\n"
        f"Spearman rho = "
        f"{metrics['spearman_rho']:.3f} | "
        f"MAE = "
        f"{metrics['mae']:.3f}"
    )


    ax.grid(
        alpha=0.25
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=180
    )


    if SHOW_PLOTS:

        plt.show()


    plt.close(
        fig
    )


# =============================================================
# PLOT ONE MODEL BY RATING
# =============================================================

def plot_rating_model(
    rows: List[Dict[str, Any]],
    model_name: str,
    target_name: str,
    output_path: Path
) -> None:


    score_column = (
        MODEL_COLUMNS[
            model_name
        ]
    )


    target_column = (
        TARGET_COLUMNS[
            target_name
        ]
    )


    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.5
        )
    )


    for bucket in RATING_BUCKETS:


        bucket_rows = [

            row

            for row in rows

            if (
                row[
                    "rating_bucket"
                ]
                == bucket
            )
        ]


        if not bucket_rows:

            continue


        bins = build_quantile_bins(
            bucket_rows,
            score_column,
            target_column
        )


        ax.plot(

            [
                row[
                    "mean_complexity"
                ]
                for row in bins
            ],

            [
                row[
                    "target_rate"
                ]
                for row in bins
            ],

            marker="o",

            label=
                bucket
        )


    ax.set_xlabel(
        "Complexity score"
    )


    ax.set_ylabel(
        target_name
    )


    ax.set_ylim(
        0,
        1
    )


    ax.set_title(
        f"{model_name} by rating\n"
        f"{target_name}"
    )


    ax.grid(
        alpha=0.25
    )


    ax.legend(
        title=
            "Rating"
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=180
    )


    if SHOW_PLOTS:

        plt.show()


    plt.close(
        fig
    )


# =============================================================
# SAVE CSV
# =============================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, Any]]
) -> None:


    if not rows:

        return


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
# SAVE JSON
# =============================================================

def write_json(
    path: Path,
    data: Any
) -> None:


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# =============================================================
# CONSOLE NUMBER FORMAT
# =============================================================

def format_number(
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


# =============================================================
# PRINT MODEL TABLE
# =============================================================

def print_model_table(
    summary: List[Dict[str, Any]],
    target_name: str,
    models: List[str],
    heading: str
) -> None:


    print()

    print(
        heading
    )


    print(

        f"{'Model':<28}"

        f"{'Spearman':>12}"

        f"{'MAE':>12}"

        f"{'Weighted MAE':>16}"

        f"{'Slope':>12}"
    )


    print(
        "-" * 90
    )


    rows = []


    for model_name in models:


        row = find_summary_row(
            summary,
            target_name,
            model_name
        )


        if row is None:

            continue


        rows.append(
            row
        )


        print(

            f"{row['model']:<28}"

            f"{format_number(row['spearman_rho']):>12}"

            f"{format_number(row['mae']):>12}"

            f"{format_number(row['weighted_mae']):>16}"

            f"{format_number(row['linear_slope']):>12}"
        )


    valid_rho = [

        row

        for row in rows

        if not math.isnan(
            row[
                "spearman_rho"
            ]
        )
    ]


    if valid_rho:


        best = max(

            valid_rho,

            key=lambda row:
                row[
                    "spearman_rho"
                ]
        )


        print()

        print(
            f"Best by Spearman: "
            f"{best['model']} "
            f"(rho = "
            f"{best['spearman_rho']:.3f})"
        )


    valid_mae = [

        row

        for row in rows

        if not math.isnan(
            row[
                "mae"
            ]
        )
    ]


    if valid_mae:


        best = min(

            valid_mae,

            key=lambda row:
                row[
                    "mae"
                ]
        )


        print(
            f"Best by MAE: "
            f"{best['model']} "
            f"(MAE = "
            f"{best['mae']:.3f})"
        )


# =============================================================
# PRINT OVERALL SUMMARY
# =============================================================

def print_overall_summary(
    summary: List[Dict[str, Any]]
) -> None:


    print()

    print(
        "=" * 94
    )

    print(
        "COMPLEXITY ANALYSIS - OVERALL RESULTS"
    )

    print(
        "=" * 94
    )


    for target_name in TARGET_COLUMNS:


        print()

        print(
            "#" * 94
        )

        print(
            f"TARGET: "
            f"{target_name}"
        )

        print(
            "#" * 94
        )


        print_model_table(

            summary,

            target_name,

            PRIMARY_MODELS,

            "Primary complexity models"
        )


        print_model_table(

            summary,

            target_name,

            DIAGNOSTIC_MODELS,

            (
                "Diagnostic components "
                "(not separate final metrics)"
            )
        )


# =============================================================
# PRINT EMPIRICAL RATES
# =============================================================

def print_rating_target_rates(
    rows: List[Dict[str, Any]]
) -> None:


    print()

    print(
        "=" * 94
    )

    print(
        "EMPIRICAL TARGET RATES BY RATING"
    )

    print(
        "=" * 94
    )


    print(

        f"{'Rating':<14}"

        f"{'Positions':>12}"

        f"{'Observations':>15}"

        f"{'Error >50cp':>15}"

        f"{'Best move':>14}"

        f"{'Not best':>14}"
    )


    print(
        "-" * 84
    )


    for row in rows:


        print(

            f"{row['rating_bucket']:<14}"

            f"{row['positions']:>12}"

            f"{row['observations']:>15,}"

            f"{row['error_rate_50cp']:>15.3f}"

            f"{row['best_move_rate']:>14.3f}"

            f"{row['not_best_move_rate']:>14.3f}"
        )


# =============================================================
# PRINT RATING MODEL RESULTS
# =============================================================

def print_rating_summary(
    summary: List[Dict[str, Any]]
) -> None:
    """
    Terminal output shows:

    - all primary complexity metrics
    - DTS alone

    The CSV still contains all diagnostic components.
    """

    models_to_print = (
        PRIMARY_MODELS
        + [
            MODEL_DTS_ONLY
        ]
    )


    print()

    print(
        "=" * 94
    )

    print(
        "RATING-SPECIFIC MODEL RESULTS"
    )

    print(
        "=" * 94
    )


    for target_name in TARGET_COLUMNS:


        print()

        print(
            "#" * 94
        )

        print(
            f"TARGET: "
            f"{target_name}"
        )

        print(
            "#" * 94
        )


        for bucket in RATING_BUCKETS:


            rows = [

                row

                for row in summary

                if (
                    row[
                        "rating_bucket"
                    ]
                    == bucket
                    and
                    row[
                        "target"
                    ]
                    == target_name
                    and
                    row[
                        "model"
                    ]
                    in models_to_print
                )
            ]


            rows.sort(

                key=lambda row:

                    models_to_print.index(
                        row[
                            "model"
                        ]
                    )
            )


            if not rows:

                continue


            print()

            print(
                f"Rating bucket "
                f"{bucket}"
            )


            print(

                f"{'Model':<28}"

                f"{'Spearman':>12}"

                f"{'MAE':>12}"

                f"{'Positions':>12}"

                f"{'Observations':>15}"
            )


            print(
                "-" * 82
            )


            for row in rows:


                print(

                    f"{row['model']:<28}"

                    f"{format_number(row['spearman_rho']):>12}"

                    f"{format_number(row['mae']):>12}"

                    f"{row['n_rows']:>12}"

                    f"{row['observations']:>15,}"
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
        "COMPLEXITY ANALYZER"
    )

    print(
        "=" * 72
    )


    # =========================================================
    # LOAD DATASET
    # =========================================================

    print()

    print(
        "Loading dataset..."
    )


    dataset = load_json(
        DATASET_PATH
    )


    print(
        f"Dataset positions: "
        f"{len(dataset):,}"
    )


    # =========================================================
    # LOAD METRICS
    # =========================================================

    print()

    print(
        "Loading position metrics..."
    )


    metric_output = load_json(
        METRICS_PATH
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
    # SAFETY CHECK
    # =========================================================

    stored_threshold = (
        metric_output
        .get(
            "metric_parameters",
            {}
        )
        .get(
            "good_move_max_loss_cp"
        )
    )


    if (
        stored_threshold
        is not None
        and
        int(
            stored_threshold
        )
        != ERROR_THRESHOLD_CP
    ):

        raise ValueError(

            "ERROR_THRESHOLD_CP does not match "
            "position_metrics.json.\n"

            f"Analyzer: "
            f"{ERROR_THRESHOLD_CP}\n"

            f"Metrics: "
            f"{stored_threshold}"
        )


    # =========================================================
    # BUILD ANALYSIS DATA
    # =========================================================

    (
        overall_rows,
        rating_rows,
        diagnostics
    ) = build_analysis_rows(
        dataset,
        metric_output
    )


    if not overall_rows:

        raise RuntimeError(
            "No overall analysis rows could be created."
        )


    if not rating_rows:

        raise RuntimeError(
            "No rating-specific analysis rows could be created."
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print()

    print(
        f"Overall position rows: "
        f"{len(overall_rows):,}"
    )


    print(
        f"Rating-specific rows "
        f"(minimum "
        f"{MIN_RATING_BUCKET_OBSERVATIONS} "
        f"observations): "
        f"{len(rating_rows):,}"
    )


    print(
        "DTS normalization: "
        f"{diagnostics['dts_normalization_min']:.0f} "
        "-> 0.0, "
        f"{diagnostics['dts_normalization_max']:.0f} "
        "-> 1.0"
    )


    # =========================================================
    # WARNINGS
    # =========================================================

    if (
        diagnostics[
            "missing_metric_positions"
        ]
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            f"{len(diagnostics['missing_metric_positions'])} "
            "positions had no metric result."
        )


    if (
        diagnostics[
            "missing_best_move_positions"
        ]
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            f"{len(diagnostics['missing_best_move_positions'])} "
            "positions had no stored Stockfish best move."
        )


    if (
        diagnostics[
            "missing_move_observations"
        ]
        > 0
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            f"{diagnostics['missing_move_observations']:,} "
            "observations could not be matched "
            "to Stockfish move data."
        )


    # =========================================================
    # CALCULATE RESULTS
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


    rating_target_rates = (
        calculate_rating_target_rates(
            rating_rows
        )
    )


    # =========================================================
    # SAVE CSV FILES
    # =========================================================

    write_csv(

        OUTPUT_DIR
        / "complexity_position_results.csv",

        overall_rows
    )


    write_csv(

        OUTPUT_DIR
        / "complexity_rating_results.csv",

        rating_rows
    )


    write_csv(

        OUTPUT_DIR
        / "complexity_summary.csv",

        overall_summary
    )


    write_csv(

        OUTPUT_DIR
        / "complexity_rating_summary.csv",

        rating_summary
    )


    write_csv(

        OUTPUT_DIR
        / "empirical_target_rates_by_rating.csv",

        rating_target_rates
    )


    # =========================================================
    # SAVE JSON
    # =========================================================

    write_json(

        OUTPUT_DIR
        / "complexity_analysis.json",

        {

            "settings": {

                "rating_buckets":
                    RATING_BUCKETS,

                "error_threshold_cp":
                    ERROR_THRESHOLD_CP,

                "minimum_rating_bucket_observations":
                    MIN_RATING_BUCKET_OBSERVATIONS,

                "number_of_plot_bins":
                    N_BINS,

                "targets": {

                    TARGET_ERROR:
                        (
                            "Observed move has "
                            "Stockfish loss > 50 cp."
                        ),

                    TARGET_NOT_BEST:
                        (
                            "Observed move differs from "
                            "the final Stockfish best move."
                        ),
                },

                "dts_normalization": {

                    "minimum_depth":
                        diagnostics[
                            "dts_normalization_min"
                        ],

                    "maximum_depth":
                        diagnostics[
                            "dts_normalization_max"
                        ],

                    "formula":
                        (
                            "(raw_dts - minimum_depth) "
                            "/ "
                            "(maximum_depth - minimum_depth)"
                        ),
                },

                "complexity_formulas": {

                    MODEL_GMR:
                        "1 - GMR",

                    MODEL_DTS_INV_N:
                        (
                            "(DTS_normalized + 1/N) / 2"
                        ),

                    MODEL_DTS_INV_SQRT_N:
                        (
                            "(DTS_normalized + "
                            "1/sqrt(N)) / 2"
                        ),

                    MODEL_DTS_INV_LOG_N:
                        (
                            "(DTS_normalized + "
                            "1/log2(N+1)) / 2"
                        ),
                },

                "diagnostic_components": {

                    MODEL_DTS_ONLY:
                        "DTS_normalized",

                    MODEL_INV_N_ONLY:
                        "1/N",

                    MODEL_INV_SQRT_N_ONLY:
                        "1/sqrt(N)",

                    MODEL_INV_LOG_N_ONLY:
                        "1/log2(N+1)",
                },
            },

            "diagnostics":
                diagnostics,

            "overall_summary":
                overall_summary,

            "rating_summary":
                rating_summary,

            "rating_target_rates":
                rating_target_rates,
        }
    )


    # =========================================================
    # PLOTS
    # =========================================================

    plot_number = 1


    for target_name in TARGET_COLUMNS:


        target_short = (
            TARGET_SHORT_NAMES[
                target_name
            ]
        )


        # -----------------------------------------------------
        # Primary metrics:
        # Spearman comparison
        # -----------------------------------------------------

        plot_comparison(

            overall_summary,

            target_name,

            PRIMARY_MODELS,

            "spearman_rho",

            "Spearman correlation",

            (
                "Primary complexity models\n"
                + target_name
            ),

            OUTPUT_DIR
            / (
                f"{plot_number:02d}_"
                f"{target_short}_"
                "primary_spearman.png"
            )
        )


        plot_number += 1


        # -----------------------------------------------------
        # Primary metrics:
        # MAE comparison
        # -----------------------------------------------------

        plot_comparison(

            overall_summary,

            target_name,

            PRIMARY_MODELS,

            "mae",

            "Mean absolute error",

            (
                "Primary complexity models\n"
                + target_name
            ),

            OUTPUT_DIR
            / (
                f"{plot_number:02d}_"
                f"{target_short}_"
                "primary_mae.png"
            )
        )


        plot_number += 1


        # -----------------------------------------------------
        # Diagnostic single components
        # -----------------------------------------------------

        plot_comparison(

            overall_summary,

            target_name,

            DIAGNOSTIC_MODELS,

            "spearman_rho",

            "Spearman correlation",

            (
                "Diagnostic components\n"
                + target_name
            ),

            OUTPUT_DIR
            / (
                f"{plot_number:02d}_"
                f"{target_short}_"
                "components_spearman.png"
            )
        )


        plot_number += 1


        # -----------------------------------------------------
        # Detailed plots
        #
        # Four primary models + DTS alone
        # -----------------------------------------------------

        detailed_models = (
            PRIMARY_MODELS
            + [
                MODEL_DTS_ONLY
            ]
        )


        for model_name in detailed_models:


            model_short = safe_name(
                model_name
            )


            plot_overall_model(

                overall_rows,

                model_name,

                target_name,

                OUTPUT_DIR
                / (
                    f"{plot_number:02d}_"
                    f"{target_short}_"
                    f"{model_short}_overall.png"
                )
            )


            plot_number += 1


            plot_rating_model(

                rating_rows,

                model_name,

                target_name,

                OUTPUT_DIR
                / (
                    f"{plot_number:02d}_"
                    f"{target_short}_"
                    f"{model_short}_by_rating.png"
                )
            )


            plot_number += 1


    # =========================================================
    # TERMINAL OUTPUT
    # =========================================================

    print_overall_summary(
        overall_summary
    )


    print_rating_target_rates(
        rating_target_rates
    )


    print_rating_summary(
        rating_summary
    )


    # =========================================================
    # FINISHED
    # =========================================================

    print()

    print(
        "=" * 94
    )

    print(
        "ANALYSIS FINISHED"
    )

    print(
        "=" * 94
    )


    print()

    print(
        "Results were written to:"
    )

    print(
        OUTPUT_DIR
    )


    print()

    print(
        "Most important files:"
    )


    print()

    print(
        "complexity_summary.csv"
    )

    print(
        "  -> both targets and all models"
    )


    print()

    print(
        "complexity_rating_summary.csv"
    )

    print(
        "  -> both targets separately by rating"
    )


    print()

    print(
        "empirical_target_rates_by_rating.csv"
    )

    print(
        "  -> actual >50cp error rate and "
        "actual best-move rate by rating"
    )


    print()

    print(
        "complexity_position_results.csv"
    )

    print(
        "  -> one row per FEN with both empirical targets"
    )


    print()

    print(
        "complexity_rating_results.csv"
    )

    print(
        "  -> one row per FEN x rating bucket"
    )


    print()

    print(
        "complexity_analysis.json"
    )

    print(
        "  -> formulas, settings and fitted models"
    )


    print()

    print(
        "Interpretation:"
    )


    print(
        "  Positive Spearman means:"
    )

    print(
        "    higher complexity -> higher failure rate."
    )


    print()

    print(
        "  Target 1:"
    )

    print(
        "    immediate move loses >50 cp."
    )


    print()

    print(
        "  Target 2:"
    )

    print(
        "    player did not choose the final "
        "Stockfish best move."
    )


    print()

    print(
        "  DTS only is included as a diagnostic "
        "because DTBMS specifically measures "
        "best-move stability."
    )


# =============================================================
# START
# =============================================================

if __name__ == "__main__":

    main()