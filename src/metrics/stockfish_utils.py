from __future__ import annotations

from typing import Any, Dict, List

import chess
import chess.engine


# =============================================================
# CONSTANTS
# =============================================================

# Large value used when converting mate evaluations into
# centipawn-like numerical values.
MATE_SCORE_CP = 100_000


# =============================================================
# SCORE CONVERSION
# =============================================================

def score_to_cp(
    score,
    pov_color: chess.Color
) -> int:
    """
    Converts a Stockfish score into a numerical centipawn value.

    The result is always returned from the perspective of
    pov_color.

    Positive:
        better for pov_color

    Negative:
        worse for pov_color

    Mate scores are converted using a very large centipawn
    value. The distance to mate is preserved by python-chess.
    """

    pov_score = score.pov(
        pov_color
    )

    cp = pov_score.score(
        mate_score=MATE_SCORE_CP
    )

    if cp is None:
        return 0

    return cp


# =============================================================
# SIMPLE POSITION EVALUATION
# =============================================================

def evaluate_cp(
    engine,
    board: chess.Board,
    pov_color: chess.Color,
    depth: int
) -> int:
    """
    Evaluates one position with Stockfish at a fixed depth.

    This helper remains available for metrics or debugging that
    require a simple single-position evaluation.
    """

    info = engine.analyse(
        board,
        chess.engine.Limit(
            depth=depth
        )
    )

    return score_to_cp(
        score=info["score"],
        pov_color=pov_color
    )


# =============================================================
# MULTIPV ANALYSIS
# =============================================================

def analyse_legal_moves_multipv(
    engine,
    board: chess.Board,
    depth: int
) -> Dict[chess.Move, int]:
    """
    Evaluates ALL legal moves from the SAME starting position
    using one Stockfish MultiPV search.

    Returns:

        {
            chess.Move: evaluation_cp,
            ...
        }

    All evaluations are measured from the perspective of the
    player who is to move in the original position.

    This makes the evaluations directly comparable and allows
    the best legal move to be used as the reference.
    """

    legal_moves = list(
        board.legal_moves
    )

    if len(legal_moves) == 0:
        return {}


    pov_color = board.turn


    # ---------------------------------------------------------
    # MultiPV search
    # ---------------------------------------------------------

    infos = engine.analyse(
        board,
        chess.engine.Limit(
            depth=depth
        ),
        multipv=len(
            legal_moves
        )
    )


    move_evaluations = {}


    # ---------------------------------------------------------
    # Extract the first move of every principal variation.
    # ---------------------------------------------------------

    for info in infos:

        pv = info.get(
            "pv",
            []
        )

        if len(pv) == 0:
            continue


        move = pv[0]


        evaluation = score_to_cp(
            score=info["score"],
            pov_color=pov_color
        )


        move_evaluations[
            move
        ] = evaluation


    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------
    #
    # We explicitly requested one PV for every legal move.
    # Missing moves would invalidate Number of Good Moves.
    #

    if len(move_evaluations) != len(legal_moves):

        raise RuntimeError(
            "Stockfish MultiPV did not return an evaluation "
            "for every legal move. "
            f"Expected {len(legal_moves)}, "
            f"received {len(move_evaluations)}."
        )


    return move_evaluations


# =============================================================
# DEPTH SERIES FOR DTS
# =============================================================

def analyse_depth_series(
    engine,
    board: chess.Board,
    min_depth: int,
    max_depth: int,
    step: int
) -> Dict[int, Dict[str, Any]]:
    """
    Performs ONE continuous Stockfish search up to max_depth
    and records the best move and evaluation at selected depths.

    Example with:

        min_depth = 6
        max_depth = 20
        step = 2

    recorded depths are:

        6, 8, 10, 12, 14, 16, 18, 20

    Returns:

        {
            6: {
                "best_move": "e2e4",
                "evaluation_cp": 23
            },
            8: {
                ...
            }
        }

    Best-Move DTS and Evaluation DTS can therefore use exactly
    the same Stockfish search rather than performing two
    independent searches.
    """

    if step <= 0:

        raise ValueError(
            "step must be greater than 0."
        )


    if min_depth > max_depth:

        raise ValueError(
            "min_depth must not be greater than max_depth."
        )


    target_depths = list(
        range(
            min_depth,
            max_depth + 1,
            step
        )
    )


    target_depth_set = set(
        target_depths
    )


    pov_color = board.turn


    depth_data = {}


    # ---------------------------------------------------------
    # ONE iterative Stockfish search
    # ---------------------------------------------------------

    with engine.analysis(
        board,
        chess.engine.Limit(
            depth=max_depth
        )
    ) as analysis:

        for info in analysis:

            depth = info.get(
                "depth"
            )


            if depth not in target_depth_set:
                continue


            score = info.get(
                "score"
            )

            pv = info.get(
                "pv",
                []
            )


            # Some intermediate UCI info messages do not yet
            # contain both score and PV.
            if score is None:
                continue

            if len(pv) == 0:
                continue


            # At one search depth Stockfish may emit several
            # updates. Overwriting the entry means that the
            # latest available information for this depth is
            # retained.
            depth_data[
                depth
            ] = {

                "best_move":
                    pv[0].uci(),

                "evaluation_cp":
                    score_to_cp(
                        score=score,
                        pov_color=pov_color
                    )
            }


    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    missing_depths = [

        depth
        for depth
        in target_depths

        if depth not in depth_data
    ]


    if len(missing_depths) > 0:

        raise RuntimeError(
            "Stockfish search did not provide complete DTS "
            "data for depths: "
            + ", ".join(
                str(depth)
                for depth
                in missing_depths
            )
        )


    return depth_data