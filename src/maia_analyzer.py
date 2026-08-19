from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "numpy fehlt. Installiere es in .venv310 mit:\n"
        "pip install numpy"
    ) from exc


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

POSITION_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "position_metrics.json"
)

MAIA_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "maia_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "maia_analysis"
)

Q1_MOVE_RESULTS_PATH = (
    OUTPUT_DIR
    / "q1_move_level_results.csv"
)

Q1_SUMMARY_PATH = (
    OUTPUT_DIR
    / "q1_summary.csv"
)

Q1_RATING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "q1_rating_summary.csv"
)

Q2_GROUP_RESULTS_PATH = (
    OUTPUT_DIR
    / "q2_group_results.csv"
)

Q2_MOVE_CONFIDENCE_PATH = (
    OUTPUT_DIR
    / "q2_move_confidence.csv"
)

Q2_SUMMARY_PATH = (
    OUTPUT_DIR
    / "q2_summary.csv"
)

Q2_RATING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "q2_rating_summary.csv"
)

ANALYSIS_JSON_PATH = (
    OUTPUT_DIR
    / "maia_analysis.json"
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


# Harte Untergrenze:
# Züge mit weniger als 10 empirischen Beobachtungen
# werden nicht ausgewertet.
MIN_MOVE_OBSERVATIONS = 10


# Jeffreys-Prior für Win / Draw / Loss.
DIRICHLET_PRIOR = 0.5


# Anzahl Ziehungen pro FEN x Ratinggruppe.
BAYES_MONTE_CARLO_SAMPLES = 20_000


# Ab dieser Wahrscheinlichkeit betrachten wir einen
# empirisch besten Zug als ausreichend sicher bestimmt.
HIGH_CONFIDENCE_THRESHOLD = 0.80


# Reproduzierbarkeit.
RANDOM_SEED = 20260819


TIE_TOLERANCE = 1e-12


# =============================================================
# BASIC HELPERS
# =============================================================

def load_json(
    path: Path
) -> Dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            f"Datei nicht gefunden:\n{path}"
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
            f"JSON-Objekt erwartet:\n{path}"
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
            ensure_ascii=False,
            allow_nan=False
        )

    temp_path.replace(
        path
    )


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8"
        )

        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def normalize_bucket_name(
    name: str
) -> str:

    return (
        str(name)
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )


def get_dataset_bucket(
    root_data: Dict[str, Any],
    wanted_bucket: str
) -> Optional[Dict[str, Any]]:

    wanted = (
        normalize_bucket_name(
            wanted_bucket
        )
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


def get_maia_bucket(
    position_data: Dict[str, Any],
    wanted_bucket: str
) -> Optional[Dict[str, Any]]:

    wanted = (
        normalize_bucket_name(
            wanted_bucket
        )
    )

    for (
        bucket_name,
        bucket_data
    ) in position_data.items():

        if (
            normalize_bucket_name(
                bucket_name
            )
            == wanted
        ):

            return bucket_data

    return None


def safe_int(
    value: Any
) -> int:

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return 0


# =============================================================
# SCORE DEFINITIONS
# =============================================================

def expected_score_from_wdl(
    wdl: Dict[str, Any]
) -> float:
    """
    WDL -> expected score:

        score =
            (win + 0.5 * draw)
            / total
    """

    win = float(
        wdl.get(
            "win",
            0.0
        )
    )

    draw = float(
        wdl.get(
            "draw",
            0.0
        )
    )

    loss = float(
        wdl.get(
            "loss",
            0.0
        )
    )

    total = (
        win
        + draw
        + loss
    )

    if total <= 0:

        raise ValueError(
            f"Ungültiges WDL: {wdl}"
        )

    return (
        win
        + 0.5 * draw
    ) / total


def empirical_counts_from_mover_pov(
    fen: str,
    move_data: Dict[str, Any]
) -> Tuple[
    int,
    int,
    int
]:
    """
    Das Dataset speichert:

        white_wins
        draws
        black_wins

    Für die Analyse brauchen wir dagegen:

        wins_for_player_who_moved
        draws
        losses_for_player_who_moved
    """

    parts = (
        fen.split()
    )

    if len(parts) < 2:

        raise ValueError(
            f"Ungültige FEN:\n{fen}"
        )

    white_wins = safe_int(
        move_data.get(
            "white_wins",
            0
        )
    )

    draws = safe_int(
        move_data.get(
            "draws",
            0
        )
    )

    black_wins = safe_int(
        move_data.get(
            "black_wins",
            0
        )
    )

    if parts[1] == "w":

        return (
            white_wins,
            draws,
            black_wins
        )

    if parts[1] == "b":

        return (
            black_wins,
            draws,
            white_wins
        )

    raise ValueError(
        f"Ungültige side-to-move-Angabe:\n{fen}"
    )


def empirical_score(
    wins: int,
    draws: int,
    losses: int
) -> float:

    n = (
        wins
        + draws
        + losses
    )

    if n <= 0:

        raise ValueError(
            "Empirische Beobachtungszahl ist 0."
        )

    return (
        wins
        + 0.5 * draws
    ) / n


# =============================================================
# QUESTION 1
#
# Wie nah liegen Maia und Stockfish an der tatsächlich
# beobachteten Punktausbeute?
# =============================================================

def build_q1_rows(
    dataset: Dict[str, Any],
    position_metrics: Dict[str, Any],
    maia_output: Dict[str, Any]
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any]
]:

    metric_positions = (
        position_metrics.get(
            "positions",
            {}
        )
    )

    maia_positions = (
        maia_output.get(
            "positions",
            {}
        )
    )

    rows: List[
        Dict[str, Any]
    ] = []

    diagnostics = {

        "maia_simulated_moves_total":
            0,

        "excluded_below_min_observations":
            0,

        "missing_dataset_position":
            0,

        "missing_dataset_bucket":
            0,

        "missing_dataset_move":
            0,

        "missing_position_metrics":
            0,

        "missing_stockfish_move":
            0,

        "maia_observation_count_mismatches":
            0,

        "included_move_rows":
            0,
    }


    # =========================================================
    # IMPORTANT
    # =========================================================
    #
    # Wir iterieren bewusst über maia_results.json.
    #
    # Dadurch betrachten wir exakt die Zugmenge,
    # die ursprünglich für die Maia-Simulation ausgewählt wurde.
    #
    # Es werden NICHT nachträglich alle anderen beobachteten
    # Züge aus dem Dataset aufgenommen.
    # =========================================================

    for (
        fen,
        maia_position
    ) in maia_positions.items():

        root_data = (
            dataset.get(
                fen
            )
        )

        if root_data is None:

            diagnostics[
                "missing_dataset_position"
            ] += 1

            continue


        metric_position = (
            metric_positions.get(
                fen
            )
        )

        if metric_position is None:

            diagnostics[
                "missing_position_metrics"
            ] += 1

            continue


        stockfish_moves = (

            metric_position
            .get(
                "stockfish",
                {}
            )
            .get(
                "moves",
                {}
            )
        )


        for rating_bucket in RATING_BUCKETS:

            maia_bucket = (
                get_maia_bucket(
                    maia_position,
                    rating_bucket
                )
            )

            if maia_bucket is None:

                continue


            dataset_bucket = (
                get_dataset_bucket(
                    root_data,
                    rating_bucket
                )
            )

            if dataset_bucket is None:

                diagnostics[
                    "missing_dataset_bucket"
                ] += 1

                continue


            dataset_moves = (
                dataset_bucket.get(
                    "moves",
                    {}
                )
            )


            maia_moves = (
                maia_bucket.get(
                    "moves",
                    {}
                )
            )


            for (
                move_uci,
                maia_move_data
            ) in maia_moves.items():

                diagnostics[
                    "maia_simulated_moves_total"
                ] += 1


                empirical_move = (
                    dataset_moves.get(
                        move_uci
                    )
                )

                if empirical_move is None:

                    diagnostics[
                        "missing_dataset_move"
                    ] += 1

                    continue


                observations = safe_int(
                    empirical_move.get(
                        "count",
                        0
                    )
                )


                # =================================================
                # HARD N >= 10 FILTER
                # =================================================

                if (
                    observations
                    < MIN_MOVE_OBSERVATIONS
                ):

                    diagnostics[
                        "excluded_below_min_observations"
                    ] += 1

                    continue


                stockfish_move = (
                    stockfish_moves.get(
                        move_uci
                    )
                )

                if stockfish_move is None:

                    diagnostics[
                        "missing_stockfish_move"
                    ] += 1

                    continue


                (
                    wins,
                    draws,
                    losses
                ) = (
                    empirical_counts_from_mover_pov(
                        fen,
                        empirical_move
                    )
                )


                if (
                    wins
                    + draws
                    + losses
                    != observations
                ):

                    raise ValueError(

                        "W/D/L stimmen nicht mit move count überein.\n\n"

                        f"FEN: {fen}\n"
                        f"Rating: {rating_bucket}\n"
                        f"Move: {move_uci}\n"
                        f"count: {observations}\n"
                        f"W+D+L: "
                        f"{wins + draws + losses}"
                    )


                maia_stored_n = safe_int(

                    maia_move_data.get(
                        "observations",
                        observations
                    )
                )


                if (
                    maia_stored_n
                    != observations
                ):

                    diagnostics[
                        "maia_observation_count_mismatches"
                    ] += 1


                simulation = (
                    maia_move_data.get(
                        "simulation",
                        {}
                    )
                )


                maia_wdl = (
                    simulation.get(
                        "average_wdl"
                    )
                )


                stockfish_wdl = (
                    stockfish_move.get(
                        "wdl"
                    )
                )


                if not isinstance(
                    maia_wdl,
                    dict
                ):

                    raise ValueError(

                        "Maia average_wdl fehlt für:\n"

                        f"{fen}\n"
                        f"{rating_bucket}\n"
                        f"{move_uci}"
                    )


                if not isinstance(
                    stockfish_wdl,
                    dict
                ):

                    raise ValueError(

                        "Stockfish-WDL fehlt für:\n"

                        f"{fen}\n"
                        f"{move_uci}"
                    )


                real_score = (
                    empirical_score(
                        wins,
                        draws,
                        losses
                    )
                )


                maia_score = (
                    expected_score_from_wdl(
                        maia_wdl
                    )
                )


                stockfish_score = (
                    expected_score_from_wdl(
                        stockfish_wdl
                    )
                )


                rows.append(
                    {

                        "fen":
                            fen,

                        "side_to_move":
                            fen.split()[1],

                        "rating_bucket":
                            rating_bucket,

                        "maia_elo":
                            safe_int(
                                maia_bucket.get(
                                    "maia_elo",
                                    0
                                )
                            ),

                        "move_uci":
                            move_uci,

                        "observations":
                            observations,

                        "wins_for_mover":
                            wins,

                        "draws":
                            draws,

                        "losses_for_mover":
                            losses,

                        "empirical_score":
                            real_score,

                        "maia_expected_score":
                            maia_score,

                        "stockfish_expected_score":
                            stockfish_score,

                        "maia_absolute_error":
                            abs(
                                maia_score
                                - real_score
                            ),

                        "stockfish_absolute_error":
                            abs(
                                stockfish_score
                                - real_score
                            ),

                        "maia_signed_error":
                            (
                                maia_score
                                - real_score
                            ),

                        "stockfish_signed_error":
                            (
                                stockfish_score
                                - real_score
                            ),

                        "retained_probability_mass":
                            float(
                                simulation.get(
                                    "retained_probability_mass",
                                    0.0
                                )
                            ),

                        "leaf_count":
                            safe_int(
                                simulation.get(
                                    "leaf_count",
                                    0
                                )
                            ),
                    }
                )


    diagnostics[
        "included_move_rows"
    ] = len(
        rows
    )


    rows.sort(

        key=lambda row: (

            row[
                "rating_bucket"
            ],

            row[
                "fen"
            ],

            row[
                "move_uci"
            ],
        )
    )


    return (
        rows,
        diagnostics
    )


def summarize_q1_model(
    rows: List[Dict[str, Any]],
    model_prefix: str
) -> Dict[str, Any]:

    if not rows:

        return {

            "data_points":
                0,

            "observations":
                0,

            "mae":
                None,

            "weighted_mae":
                None,

            "rmse":
                None,

            "weighted_rmse":
                None,

            "bias":
                None,

            "weighted_bias":
                None,
        }


    prediction_key = (
        f"{model_prefix}_expected_score"
    )


    actual = np.asarray(

        [
            float(
                row[
                    "empirical_score"
                ]
            )

            for row in rows
        ],

        dtype=float
    )


    predicted = np.asarray(

        [
            float(
                row[
                    prediction_key
                ]
            )

            for row in rows
        ],

        dtype=float
    )


    weights = np.asarray(

        [
            int(
                row[
                    "observations"
                ]
            )

            for row in rows
        ],

        dtype=float
    )


    error = (
        predicted
        - actual
    )


    abs_error = (
        np.abs(
            error
        )
    )


    squared_error = (
        error ** 2
    )


    return {

        "data_points":
            len(
                rows
            ),

        "observations":
            int(
                weights.sum()
            ),

        "mae":
            float(
                abs_error.mean()
            ),

        "weighted_mae":
            float(
                np.average(
                    abs_error,
                    weights=weights
                )
            ),

        "rmse":
            float(
                math.sqrt(
                    squared_error.mean()
                )
            ),

        "weighted_rmse":
            float(
                math.sqrt(
                    np.average(
                        squared_error,
                        weights=weights
                    )
                )
            ),

        "bias":
            float(
                error.mean()
            ),

        "weighted_bias":
            float(
                np.average(
                    error,
                    weights=weights
                )
            ),
    }


def build_q1_summaries(
    rows: List[Dict[str, Any]]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:

    models = [

        (
            "Maia simulation",
            "maia"
        ),

        (
            "Stockfish WDL",
            "stockfish"
        ),
    ]


    overall = [

        {

            "model":
                model_name,

            **summarize_q1_model(
                rows,
                prefix
            )
        }

        for (
            model_name,
            prefix
        ) in models
    ]


    by_rating: List[
        Dict[str, Any]
    ] = []


    for rating_bucket in RATING_BUCKETS:

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


        for (
            model_name,
            prefix
        ) in models:

            by_rating.append(
                {

                    "rating_bucket":
                        rating_bucket,

                    "model":
                        model_name,

                    **summarize_q1_model(
                        bucket_rows,
                        prefix
                    )
                }
            )


    return (
        overall,
        by_rating
    )


# =============================================================
# QUESTION 2
#
# Welches Modell erkennt den Zug mit den besten
# praktischen Erfolgsaussichten?
# =============================================================

def stable_group_seed(
    fen: str,
    rating_bucket: str
) -> int:

    text = (
        f"{RANDOM_SEED}|"
        f"{fen}|"
        f"{rating_bucket}"
    )


    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).digest()


    return int.from_bytes(

        digest[
            :8
        ],

        byteorder="big",

        signed=False
    )


def draw_score_posterior(
    wins: int,
    draws: int,
    losses: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Posterior:

        (pW, pD, pL)
        ~
        Dirichlet(
            W + 0.5,
            D + 0.5,
            L + 0.5
        )

    Punktausbeute:

        score =
            pW
            + 0.5 * pD
    """

    alpha = np.asarray(
        [
            wins
            + DIRICHLET_PRIOR,

            draws
            + DIRICHLET_PRIOR,

            losses
            + DIRICHLET_PRIOR,
        ],
        dtype=float
    )


    gamma_draws = rng.gamma(

        shape=alpha,

        scale=1.0,

        size=(
            BAYES_MONTE_CARLO_SAMPLES,
            3
        )
    )


    probabilities = (

        gamma_draws

        / gamma_draws.sum(
            axis=1,
            keepdims=True
        )
    )


    return (

        probabilities[
            :,
            0
        ]

        + 0.5
        * probabilities[
            :,
            1
        ]
    )


def select_highest_score_move(
    move_rows: List[Dict[str, Any]],
    score_key: str
) -> str:
    """
    Bei exakt gleichem Wert:
    alphabetischer UCI-Tie-Break.
    """

    ordered = sorted(

        move_rows,

        key=lambda row:
            row[
                "move_uci"
            ]
    )


    return max(

        ordered,

        key=lambda row:
            float(
                row[
                    score_key
                ]
            )
    )[
        "move_uci"
    ]


def build_q2_results(
    q1_rows: List[Dict[str, Any]]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any]
]:

    grouped: Dict[
        Tuple[str, str],
        List[Dict[str, Any]]
    ] = {}


    for row in q1_rows:

        grouped.setdefault(
            (
                row[
                    "fen"
                ],

                row[
                    "rating_bucket"
                ]
            ),
            []
        ).append(
            row
        )


    group_results: List[
        Dict[str, Any]
    ] = []


    move_confidence_rows: List[
        Dict[str, Any]
    ] = []


    diagnostics = {

        "groups_after_n10_filter":
            len(
                grouped
            ),

        "excluded_groups_with_fewer_than_two_moves":
            0,

        "included_q2_groups":
            0,
    }


    for (
        (
            fen,
            rating_bucket
        ),
        group_move_rows
    ) in sorted(

        grouped.items(),

        key=lambda item: (
            item[0][1],
            item[0][0]
        )
    ):


        move_rows = sorted(

            group_move_rows,

            key=lambda row:
                row[
                    "move_uci"
                ]
        )


        # =====================================================
        # Q2 requires at least two candidate moves
        # =====================================================

        if len(
            move_rows
        ) < 2:

            diagnostics[
                "excluded_groups_with_fewer_than_two_moves"
            ] += 1

            continue


        rng = (
            np.random.default_rng(
                stable_group_seed(
                    fen,
                    rating_bucket
                )
            )
        )


        # =====================================================
        # BAYES POSTERIOR FOR EACH MOVE
        # =====================================================

        posterior_columns = [

            draw_score_posterior(

                wins=int(
                    row[
                        "wins_for_mover"
                    ]
                ),

                draws=int(
                    row[
                        "draws"
                    ]
                ),

                losses=int(
                    row[
                        "losses_for_mover"
                    ]
                ),

                rng=rng
            )

            for row in move_rows
        ]


        # Matrix:
        #
        # rows = Monte Carlo samples
        # columns = candidate moves

        score_samples = (
            np.column_stack(
                posterior_columns
            )
        )


        sampled_best_indices = (
            np.argmax(
                score_samples,
                axis=1
            )
        )


        best_counts = (
            np.bincount(

                sampled_best_indices,

                minlength=len(
                    move_rows
                )
            )
        )


        probability_best = (

            best_counts

            / BAYES_MONTE_CARLO_SAMPLES
        )


        posterior_mean_scores = (
            score_samples.mean(
                axis=0
            )
        )


        sampled_best_scores = (
            score_samples.max(
                axis=1
            )
        )


        # =====================================================
        # MODEL CHOICES
        # =====================================================

        maia_choice = (
            select_highest_score_move(
                move_rows,
                "maia_expected_score"
            )
        )


        stockfish_choice = (
            select_highest_score_move(
                move_rows,
                "stockfish_expected_score"
            )
        )


        move_to_index = {

            row[
                "move_uci"
            ]:
                index

            for (
                index,
                row
            ) in enumerate(
                move_rows
            )
        }


        maia_index = (
            move_to_index[
                maia_choice
            ]
        )


        stockfish_index = (
            move_to_index[
                stockfish_choice
            ]
        )


        # =====================================================
        # RAW EMPIRICAL BEST MOVE
        # =====================================================

        best_raw_score = max(

            float(
                row[
                    "empirical_score"
                ]
            )

            for row in move_rows
        )


        raw_best_moves = sorted(
            [

                row[
                    "move_uci"
                ]

                for row in move_rows

                if abs(
                    float(
                        row[
                            "empirical_score"
                        ]
                    )
                    - best_raw_score
                )
                <= TIE_TOLERANCE
            ]
        )


        # =====================================================
        # POSTERIOR BEST MOVE
        # =====================================================

        posterior_best_index = int(

            np.argmax(
                probability_best
            )
        )


        posterior_best_move = (

            move_rows[
                posterior_best_index
            ][
                "move_uci"
            ]
        )


        posterior_best_confidence = float(

            probability_best[
                posterior_best_index
            ]
        )


        high_confidence_case = (

            posterior_best_confidence
            >= HIGH_CONFIDENCE_THRESHOLD
        )


        # =====================================================
        # POSTERIOR EXPECTED REGRET
        # =====================================================
        #
        # In jeder Monte-Carlo-Ziehung:
        #
        # best score in that simulated reality
        # minus
        # score of model-selected move
        #
        # Danach Mittelwert.
        # =====================================================

        maia_posterior_regret = float(

            np.mean(

                sampled_best_scores

                - score_samples[
                    :,
                    maia_index
                ]
            )
        )


        stockfish_posterior_regret = float(

            np.mean(

                sampled_best_scores

                - score_samples[
                    :,
                    stockfish_index
                ]
            )
        )


        row_by_move = {

            row[
                "move_uci"
            ]:
                row

            for row in move_rows
        }


        # =====================================================
        # RAW EMPIRICAL REGRET
        # =====================================================

        maia_raw_regret = (

            best_raw_score

            - float(
                row_by_move[
                    maia_choice
                ][
                    "empirical_score"
                ]
            )
        )


        stockfish_raw_regret = (

            best_raw_score

            - float(
                row_by_move[
                    stockfish_choice
                ][
                    "empirical_score"
                ]
            )
        )


        # =====================================================
        # GROUP OUTPUT
        # =====================================================

        group_results.append(
            {

                "fen":
                    fen,

                "rating_bucket":
                    rating_bucket,

                "eligible_moves":
                    len(
                        move_rows
                    ),

                "eligible_observations":
                    sum(
                        int(
                            row[
                                "observations"
                            ]
                        )
                        for row in move_rows
                    ),

                "empirical_raw_best_moves":
                    "|".join(
                        raw_best_moves
                    ),

                "empirical_raw_best_score":
                    best_raw_score,

                "posterior_best_move":
                    posterior_best_move,

                "posterior_best_confidence":
                    posterior_best_confidence,

                "high_confidence_case":
                    high_confidence_case,

                "maia_choice":
                    maia_choice,

                "stockfish_choice":
                    stockfish_choice,

                "maia_and_stockfish_same_choice":
                    (
                        maia_choice
                        == stockfish_choice
                    ),

                "maia_raw_top1_correct":
                    (
                        maia_choice
                        in raw_best_moves
                    ),

                "stockfish_raw_top1_correct":
                    (
                        stockfish_choice
                        in raw_best_moves
                    ),

                "maia_posterior_top1_correct":
                    (
                        maia_choice
                        == posterior_best_move
                    ),

                "stockfish_posterior_top1_correct":
                    (
                        stockfish_choice
                        == posterior_best_move
                    ),

                "maia_choice_probability_best":
                    float(
                        probability_best[
                            maia_index
                        ]
                    ),

                "stockfish_choice_probability_best":
                    float(
                        probability_best[
                            stockfish_index
                        ]
                    ),

                "maia_posterior_expected_regret":
                    maia_posterior_regret,

                "stockfish_posterior_expected_regret":
                    stockfish_posterior_regret,

                "maia_raw_empirical_regret":
                    maia_raw_regret,

                "stockfish_raw_empirical_regret":
                    stockfish_raw_regret,
            }
        )


        # =====================================================
        # MOVE-LEVEL CONFIDENCE OUTPUT
        # =====================================================

        for (
            index,
            row
        ) in enumerate(
            move_rows
        ):

            move_confidence_rows.append(
                {

                    "fen":
                        fen,

                    "rating_bucket":
                        rating_bucket,

                    "move_uci":
                        row[
                            "move_uci"
                        ],

                    "observations":
                        row[
                            "observations"
                        ],

                    "empirical_score":
                        row[
                            "empirical_score"
                        ],

                    "posterior_mean_score":
                        float(
                            posterior_mean_scores[
                                index
                            ]
                        ),

                    "probability_best":
                        float(
                            probability_best[
                                index
                            ]
                        ),

                    "maia_expected_score":
                        row[
                            "maia_expected_score"
                        ],

                    "stockfish_expected_score":
                        row[
                            "stockfish_expected_score"
                        ],

                    "is_maia_choice":
                        (
                            row[
                                "move_uci"
                            ]
                            == maia_choice
                        ),

                    "is_stockfish_choice":
                        (
                            row[
                                "move_uci"
                            ]
                            == stockfish_choice
                        ),
                }
            )


    diagnostics[
        "included_q2_groups"
    ] = len(
        group_results
    )


    return (
        group_results,
        move_confidence_rows,
        diagnostics
    )


def summarize_q2_model(
    group_rows: List[Dict[str, Any]],
    model_prefix: str
) -> Dict[str, Any]:

    if not group_rows:

        return {

            "groups":
                0,

            "raw_top1_accuracy":
                None,

            "posterior_top1_accuracy":
                None,

            "mean_probability_choice_is_best":
                None,

            "mean_posterior_expected_regret":
                None,

            "mean_raw_empirical_regret":
                None,

            "high_confidence_groups":
                0,

            "high_confidence_accuracy":
                None,
        }


    raw_accuracy = np.mean(
        [
            float(
                row[
                    f"{model_prefix}_raw_top1_correct"
                ]
            )

            for row in group_rows
        ]
    )


    posterior_accuracy = np.mean(
        [
            float(
                row[
                    f"{model_prefix}_posterior_top1_correct"
                ]
            )

            for row in group_rows
        ]
    )


    mean_probability_best = np.mean(
        [
            float(
                row[
                    f"{model_prefix}_choice_probability_best"
                ]
            )

            for row in group_rows
        ]
    )


    mean_posterior_regret = np.mean(
        [
            float(
                row[
                    f"{model_prefix}_posterior_expected_regret"
                ]
            )

            for row in group_rows
        ]
    )


    mean_raw_regret = np.mean(
        [
            float(
                row[
                    f"{model_prefix}_raw_empirical_regret"
                ]
            )

            for row in group_rows
        ]
    )


    high_confidence_rows = [

        row

        for row in group_rows

        if row[
            "high_confidence_case"
        ]
    ]


    if high_confidence_rows:

        high_confidence_accuracy = (

            sum(
                1

                for row in high_confidence_rows

                if row[
                    f"{model_prefix}_posterior_top1_correct"
                ]
            )

            / len(
                high_confidence_rows
            )
        )

    else:

        high_confidence_accuracy = (
            None
        )


    return {

        "groups":
            len(
                group_rows
            ),

        "raw_top1_accuracy":
            float(
                raw_accuracy
            ),

        "posterior_top1_accuracy":
            float(
                posterior_accuracy
            ),

        "mean_probability_choice_is_best":
            float(
                mean_probability_best
            ),

        "mean_posterior_expected_regret":
            float(
                mean_posterior_regret
            ),

        "mean_raw_empirical_regret":
            float(
                mean_raw_regret
            ),

        "high_confidence_groups":
            len(
                high_confidence_rows
            ),

        "high_confidence_accuracy":
            (
                float(
                    high_confidence_accuracy
                )

                if (
                    high_confidence_accuracy
                    is not None
                )

                else None
            ),
    }


def build_q2_summaries(
    group_rows: List[Dict[str, Any]]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:

    models = [

        (
            "Maia simulation",
            "maia"
        ),

        (
            "Stockfish WDL",
            "stockfish"
        ),
    ]


    overall = [

        {

            "model":
                model_name,

            **summarize_q2_model(
                group_rows,
                prefix
            )
        }

        for (
            model_name,
            prefix
        ) in models
    ]


    by_rating: List[
        Dict[str, Any]
    ] = []


    for rating_bucket in RATING_BUCKETS:

        bucket_rows = [

            row

            for row in group_rows

            if (
                row[
                    "rating_bucket"
                ]
                == rating_bucket
            )
        ]


        for (
            model_name,
            prefix
        ) in models:

            by_rating.append(
                {

                    "rating_bucket":
                        rating_bucket,

                    "model":
                        model_name,

                    **summarize_q2_model(
                        bucket_rows,
                        prefix
                    )
                }
            )


    return (
        overall,
        by_rating
    )


# =============================================================
# CONSOLE OUTPUT
# =============================================================

def format_value(
    value: Optional[float],
    digits: int = 4
) -> str:

    if value is None:

        return "n/a"

    return (
        f"{value:.{digits}f}"
    )


def print_q1_summary(
    summary: List[Dict[str, Any]]
) -> None:

    print()

    print(
        "=" * 82
    )

    print(
        "QUESTION 1 - ACTUAL POINT SCORE"
    )

    print(
        "=" * 82
    )


    print(

        f"{'Model':<22}"

        f"{'Rows':>8}"

        f"{'Obs.':>12}"

        f"{'MAE':>10}"

        f"{'wMAE':>10}"

        f"{'wBias':>10}"
    )


    print(
        "-" * 72
    )


    for row in summary:

        print(

            f"{row['model']:<22}"

            f"{row['data_points']:>8,}"

            f"{row['observations']:>12,}"

            f"{format_value(row['mae']):>10}"

            f"{format_value(row['weighted_mae']):>10}"

            f"{format_value(row['weighted_bias']):>10}"
        )


def print_q2_summary(
    summary: List[Dict[str, Any]]
) -> None:

    print()

    print(
        "=" * 108
    )

    print(
        "QUESTION 2 - PRACTICALLY BEST MOVE"
    )

    print(
        "=" * 108
    )


    print(

        f"{'Model':<22}"

        f"{'Groups':>8}"

        f"{'RawAcc':>10}"

        f"{'PostAcc':>10}"

        f"{'P(best)':>10}"

        f"{'PostRegret':>13}"

        f"{'HC Acc':>10}"
    )


    print(
        "-" * 83
    )


    for row in summary:

        print(

            f"{row['model']:<22}"

            f"{row['groups']:>8,}"

            f"{format_value(row['raw_top1_accuracy'], 3):>10}"

            f"{format_value(row['posterior_top1_accuracy'], 3):>10}"

            f"{format_value(row['mean_probability_choice_is_best'], 3):>10}"

            f"{format_value(row['mean_posterior_expected_regret'], 4):>13}"

            f"{format_value(row['high_confidence_accuracy'], 3):>10}"
        )


    print()

    print(
        "P(best) = mittlere posterior Wahrscheinlichkeit, "
        "dass die Modellwahl wirklich der praktisch beste Zug ist."
    )

    print(
        "HC Acc = Trefferquote nur bei Gruppen mit mindestens "
        f"{HIGH_CONFIDENCE_THRESHOLD:.0%} Confidence."
    )


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    print()

    print(
        "=" * 80
    )

    print(
        "MAIA ANALYZER"
    )

    print(
        "=" * 80
    )


    # =========================================================
    # LOAD FILES
    # =========================================================

    dataset = (
        load_json(
            DATASET_PATH
        )
    )


    position_metrics = (
        load_json(
            POSITION_METRICS_PATH
        )
    )


    maia_output = (
        load_json(
            MAIA_RESULTS_PATH
        )
    )


    print(
        f"Dataset positions: "
        f"{len(dataset):,}"
    )


    print(
        f"Metric positions: "
        f"{len(position_metrics.get('positions', {})):,}"
    )


    print(
        f"Maia result positions: "
        f"{len(maia_output.get('positions', {})):,}"
    )


    # =========================================================
    # QUESTION 1
    # =========================================================

    print()

    print(
        f"Building move rows with "
        f"N >= {MIN_MOVE_OBSERVATIONS}..."
    )


    (
        q1_rows,
        q1_diagnostics
    ) = (
        build_q1_rows(
            dataset,
            position_metrics,
            maia_output
        )
    )


    if not q1_rows:

        raise RuntimeError(
            "Keine auswertbaren Maia-Züge gefunden."
        )


    (
        q1_summary,
        q1_rating_summary
    ) = (
        build_q1_summaries(
            q1_rows
        )
    )


    # =========================================================
    # QUESTION 2
    # =========================================================

    print()

    print(
        "Running Bayesian best-move analysis with "
        f"{BAYES_MONTE_CARLO_SAMPLES:,} "
        "samples per group..."
    )


    (
        q2_group_rows,
        q2_move_confidence_rows,
        q2_diagnostics
    ) = (
        build_q2_results(
            q1_rows
        )
    )


    if not q2_group_rows:

        raise RuntimeError(

            "Keine FEN x Ratinggruppe besitzt "
            "mindestens zwei auswertbare Züge."
        )


    (
        q2_summary,
        q2_rating_summary
    ) = (
        build_q2_summaries(
            q2_group_rows
        )
    )


    # =========================================================
    # SAVE RESULTS
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    write_csv(
        Q1_MOVE_RESULTS_PATH,
        q1_rows
    )


    write_csv(
        Q1_SUMMARY_PATH,
        q1_summary
    )


    write_csv(
        Q1_RATING_SUMMARY_PATH,
        q1_rating_summary
    )


    write_csv(
        Q2_GROUP_RESULTS_PATH,
        q2_group_rows
    )


    write_csv(
        Q2_MOVE_CONFIDENCE_PATH,
        q2_move_confidence_rows
    )


    write_csv(
        Q2_SUMMARY_PATH,
        q2_summary
    )


    write_csv(
        Q2_RATING_SUMMARY_PATH,
        q2_rating_summary
    )


    # =========================================================
    # MASTER JSON
    # =========================================================

    analysis = {

        "analysis_type":
            "Maia practical outcome evaluation",

        "minimum_move_observations":
            MIN_MOVE_OBSERVATIONS,


        "question_1": {

            "question":
                (
                    "How closely do Maia and regular "
                    "Stockfish WDL predict the actually "
                    "observed point score?"
                ),

            "empirical_score_definition":
                (
                    "(wins + 0.5 * draws) / observations "
                    "from the perspective of the player "
                    "who made the observed move."
                ),

            "primary_metric":
                "weighted_mae",

            "secondary_metrics": [
                "mae",
                "rmse",
                "weighted_rmse",
                "bias",
                "weighted_bias",
            ],

            "diagnostics":
                q1_diagnostics,

            "overall_results":
                q1_summary,

            "rating_results":
                q1_rating_summary,
        },


        "question_2": {

            "question":
                (
                    "How accurately do Maia and regular "
                    "Stockfish WDL identify the move with "
                    "the best practical point-scoring "
                    "prospects among the already simulated "
                    "frequent moves?"
                ),

            "minimum_eligible_moves_per_group":
                2,

            "dirichlet_prior": [
                DIRICHLET_PRIOR,
                DIRICHLET_PRIOR,
                DIRICHLET_PRIOR,
            ],

            "monte_carlo_samples_per_group":
                BAYES_MONTE_CARLO_SAMPLES,

            "random_seed":
                RANDOM_SEED,

            "high_confidence_threshold":
                HIGH_CONFIDENCE_THRESHOLD,

            "primary_metrics": [
                "mean_probability_choice_is_best",
                "mean_posterior_expected_regret",
                "high_confidence_accuracy",
            ],

            "secondary_metrics": [
                "raw_top1_accuracy",
                "posterior_top1_accuracy",
                "mean_raw_empirical_regret",
            ],

            "diagnostics":
                q2_diagnostics,

            "overall_results":
                q2_summary,

            "rating_results":
                q2_rating_summary,
        },


        "maia_parameters":
            maia_output.get(
                "parameters",
                {}
            ),


        "position_metric_parameters":
            position_metrics.get(
                "metric_parameters",
                {}
            ),
    }


    write_json(
        ANALYSIS_JSON_PATH,
        analysis
    )


    # =========================================================
    # CONSOLE
    # =========================================================

    print_q1_summary(
        q1_summary
    )


    print_q2_summary(
        q2_summary
    )


    print()

    print(
        "=" * 80
    )

    print(
        "MAIA ANALYSIS FINISHED"
    )

    print(
        "=" * 80
    )


    print()

    print(
        "Main JSON:"
    )

    print(
        ANALYSIS_JSON_PATH
    )


    print()

    print(
        "Question 1 summary:"
    )

    print(
        Q1_SUMMARY_PATH
    )


    print()

    print(
        "Question 2 summary:"
    )

    print(
        Q2_SUMMARY_PATH
    )


if __name__ == "__main__":

    main()