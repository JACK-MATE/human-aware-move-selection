from __future__ import annotations

from typing import Any, Dict

import chess


# =============================================================
# SCORE BOUNDS
# =============================================================

def clamp_score(
    score: float
) -> float:

    if score < 0.0:
        return 0.0

    if score > 1.0:
        return 1.0

    return score


# =============================================================
# COMPLEXITY -> WHITE EXPECTED SCORE
# =============================================================

def calculate_white_expected_score_from_complexity(
    white_complexity: float,
    black_complexity: float,
    scale: float = 1.0
) -> float:
    """
    Calculates White's expected score from the difference
    between the complexity faced by White and Black.

    Formula:

        E_white =
            0.5
            +
            scale
            * (C_black - C_white)
            / 200

    Complexity range:

        0 ... 100

    Examples:

        C_white = 50
        C_black = 50
            -> 0.50

        C_white = 40
        C_black = 60
            -> 0.60

        C_white = 20
        C_black = 80
            -> 0.80
    """

    expected_score = (
        0.5
        +
        scale
        * (
            black_complexity
            - white_complexity
        )
        / 200.0
    )


    return clamp_score(
        expected_score
    )


# =============================================================
# GMR -> WHITE EXPECTED SCORE
# =============================================================

def calculate_white_expected_score_from_gmr(
    white_gmr: float,
    black_gmr: float,
    scale: float = 1.0
) -> float:
    """
    Calculates White's expected score directly from GMR.

    High GMR means an easier decision.

    Formula:

        E_white =
            0.5
            +
            scale
            * (GMR_white - GMR_black)
            / 2
    """

    expected_score = (
        0.5
        +
        scale
        * (
            white_gmr
            - black_gmr
        )
        / 2.0
    )


    return clamp_score(
        expected_score
    )


# =============================================================
# COMPLEXITY FAMILY
# =============================================================

def calculate_complexity_family_expected_scores(
    white_scores: Dict[str, float],
    black_scores: Dict[str, float],
    scale: float = 1.0
) -> Dict[str, float]:

    if (
        set(
            white_scores.keys()
        )
        !=
        set(
            black_scores.keys()
        )
    ):

        raise ValueError(
            "White and Black complexity variants do not match."
        )


    results = {}


    for weighting in white_scores.keys():

        results[
            weighting
        ] = (
            calculate_white_expected_score_from_complexity(
                white_complexity=
                    white_scores[
                        weighting
                    ],
                black_complexity=
                    black_scores[
                        weighting
                    ],
                scale=
                    scale
            )
        )


    return results


# =============================================================
# ROOT + CHILD -> EXPECTED SCORES
# =============================================================

def calculate_expected_scores_for_move(
    root_fen: str,
    root_metrics: Dict[str, Any],
    child_metrics: Dict[str, Any],
    scale: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates expected scores for one move.

    root_fen:
        position before the move

    root_metrics:
        metrics of the player who is currently deciding

    child_metrics:
        metrics of the opponent's resulting position after
        that move


    Example:

        root FEN:
            White to move

        root complexity:
            Complexity White

        after move:
            Black to move

        child complexity:
            Complexity Black


    If Black is initially to move, the assignment is reversed
    automatically.
    """

    board = (
        chess.Board(
            root_fen
        )
    )


    # =========================================================
    # ASSIGN ROOT / CHILD TO WHITE AND BLACK
    # =========================================================

    if board.turn == chess.WHITE:

        white_metrics = (
            root_metrics
        )

        black_metrics = (
            child_metrics
        )

        mover_is_white = True


    else:

        black_metrics = (
            root_metrics
        )

        white_metrics = (
            child_metrics
        )

        mover_is_white = False


    # =========================================================
    # GMR EXPECTED SCORE
    # =========================================================

    white_gmr_score = (
        calculate_white_expected_score_from_gmr(
            white_gmr=
                white_metrics[
                    "good_move_ratio"
                ],
            black_gmr=
                black_metrics[
                    "good_move_ratio"
                ],
            scale=
                scale
        )
    )


    # =========================================================
    # BEST-MOVE DTS COMPLEXITY
    # =========================================================

    best_move_scores = (
        calculate_complexity_family_expected_scores(
            white_scores=
                white_metrics[
                    "complexity_scores"
                ][
                    "best_move_dts"
                ],
            black_scores=
                black_metrics[
                    "complexity_scores"
                ][
                    "best_move_dts"
                ],
            scale=
                scale
        )
    )


    # =========================================================
    # EVALUATION DTS COMPLEXITY
    # =========================================================

    eval_scores = (
        calculate_complexity_family_expected_scores(
            white_scores=
                white_metrics[
                    "complexity_scores"
                ][
                    "eval_dts"
                ],
            black_scores=
                black_metrics[
                    "complexity_scores"
                ][
                    "eval_dts"
                ],
            scale=
                scale
        )
    )


    # =========================================================
    # SCORE FROM MOVER'S PERSPECTIVE
    # =========================================================

    if mover_is_white:

        mover_gmr_score = (
            white_gmr_score
        )

        mover_best_scores = dict(
            best_move_scores
        )

        mover_eval_scores = dict(
            eval_scores
        )


    else:

        mover_gmr_score = (
            1.0
            - white_gmr_score
        )


        mover_best_scores = {

            key:
                1.0 - value

            for key, value
            in best_move_scores.items()
        }


        mover_eval_scores = {

            key:
                1.0 - value

            for key, value
            in eval_scores.items()
        }


    return {

        "white_expected_score": {

            "good_move_ratio":
                white_gmr_score,

            "complexity_scores": {

                "best_move_dts":
                    best_move_scores,

                "eval_dts":
                    eval_scores
            }
        },


        "mover_expected_score": {

            "good_move_ratio":
                mover_gmr_score,

            "complexity_scores": {

                "best_move_dts":
                    mover_best_scores,

                "eval_dts":
                    mover_eval_scores
            }
        }
    }