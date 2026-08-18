from __future__ import annotations

from typing import Any, Dict

import chess
import chess.engine


# =============================================================
# CONSTANTS
# =============================================================

MATE_SCORE_CP = 100_000


# =============================================================
# FEN HANDLING
# =============================================================

def board_from_position_key(
    fen: str
) -> chess.Board:
    """
    Accepts both:

        4-field project FEN
        6-field normal FEN
    """

    parts = fen.split()

    if len(parts) == 4:

        fen = (
            fen
            + " 0 1"
        )

    elif len(parts) != 6:

        raise ValueError(
            "Unexpected FEN format:\n"
            + fen
        )

    return chess.Board(
        fen
    )


# =============================================================
# SCORE CONVERSION
# =============================================================

def score_to_cp(
    score,
    pov_color: chess.Color
) -> int:
    """
    Converts a Stockfish score to centipawns from one fixed POV.

    Positive:
        better for pov_color

    Negative:
        worse for pov_color
    """

    cp = (
        score
        .pov(pov_color)
        .score(
            mate_score=MATE_SCORE_CP
        )
    )

    if cp is None:

        return 0

    return cp


# =============================================================
# WDL CONVERSION
# =============================================================

def wdl_to_dict(
    wdl,
    pov_color: chess.Color
) -> Dict[str, int]:
    """
    Converts Stockfish WDL into:

        {
            "win": ...,
            "draw": ...,
            "loss": ...
        }

    The result is always stored from the perspective of
    pov_color.

    In this project, pov_color is normally the player to move
    in the original FEN.
    """

    pov_wdl = (
        wdl.pov(
            pov_color
        )
    )

    return {

        "win":
            pov_wdl.wins,

        "draw":
            pov_wdl.draws,

        "loss":
            pov_wdl.losses
    }


# =============================================================
# MULTIPV:
# ALL LEGAL MOVES
# =============================================================

def analyse_legal_moves_multipv(
    engine,
    board: chess.Board,
    depth: int
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates ALL legal moves in ONE MultiPV search.

    For every legal move, we retain:

        evaluation_cp
        WDL

    Example:

        {
            "e2e4": {
                "evaluation_cp": 35,
                "wdl": {
                    "win": 310,
                    "draw": 520,
                    "loss": 170
                }
            },

            ...
        }

    All values are measured from the perspective of the player
    to move in the ORIGINAL position.

    Therefore all legal moves remain directly comparable.

    IMPORTANT:

    This does NOT perform an additional WDL search.

    WDL is simply retained from the existing MultiPV analysis.
    """

    legal_moves = list(
        board.legal_moves
    )

    if len(legal_moves) == 0:

        return {}


    pov_color = (
        board.turn
    )


    # =========================================================
    # ONE MULTIPV SEARCH FOR ALL LEGAL MOVES
    # =========================================================

    infos = (
        engine.analyse(
            board,
            chess.engine.Limit(
                depth=depth
            ),
            multipv=len(
                legal_moves
            )
        )
    )


    move_data = {}


    # =========================================================
    # EXTRACT EACH LEGAL MOVE
    # =========================================================

    for info in infos:

        pv = info.get(
            "pv",
            []
        )

        score = info.get(
            "score"
        )

        wdl = info.get(
            "wdl"
        )


        if (
            len(pv) == 0
            or score is None
        ):

            continue


        # -----------------------------------------------------
        # WDL is required because it is one of the values we
        # deliberately want to retain for every legal move.
        # -----------------------------------------------------

        if wdl is None:

            raise RuntimeError(
                "Stockfish MultiPV did not return WDL. "
                "Make sure UCI_ShowWDL is enabled."
            )


        move_uci = (
            pv[0].uci()
        )


        move_data[
            move_uci
        ] = {

            "evaluation_cp":
                score_to_cp(
                    score=score,
                    pov_color=pov_color
                ),

            "wdl":
                wdl_to_dict(
                    wdl=wdl,
                    pov_color=pov_color
                )
        }


    # =========================================================
    # SAFETY CHECK
    # =========================================================
    #
    # We requested one PV for every legal move.
    # Missing moves would invalidate N, GMR and the later
    # empirical analysis of actually played moves.
    #

    if (
        len(move_data)
        != len(legal_moves)
    ):

        raise RuntimeError(
            "Stockfish MultiPV did not return every legal move. "
            f"Expected {len(legal_moves)}, "
            f"received {len(move_data)}."
        )


    return move_data


# =============================================================
# DEPTH SERIES FOR BEST-MOVE DTS
# =============================================================

def analyse_depth_series(
    engine,
    board: chess.Board,
    min_depth: int,
    max_depth: int,
    step: int
) -> Dict[int, Dict[str, Any]]:
    """
    Performs ONE continuous Stockfish search.

    Example:

        6, 8, 10, ..., 24

    For every selected depth we store:

        best move
        evaluation
        WDL

    Depths 22 and 24 can later be used to confirm DTBMS values
    of 18 or 20.
    """

    if step <= 0:

        raise ValueError(
            "step must be greater than 0."
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

    pov_color = (
        board.turn
    )

    depth_data = {}


    # =========================================================
    # ONE CONTINUOUS SEARCH
    # =========================================================

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


            pv = info.get(
                "pv",
                []
            )

            score = info.get(
                "score"
            )


            if (
                len(pv) == 0
                or score is None
            ):

                continue


            entry = {

                "best_move":
                    pv[0].uci(),

                "evaluation_cp":
                    score_to_cp(
                        score=score,
                        pov_color=pov_color
                    )
            }


            # =================================================
            # WDL
            # =================================================

            wdl = info.get(
                "wdl"
            )

            if wdl is not None:

                entry[
                    "wdl"
                ] = (
                    wdl_to_dict(
                        wdl=wdl,
                        pov_color=pov_color
                    )
                )


            # Stockfish can send several updates for the same
            # depth. The latest complete one is retained.
            depth_data[
                depth
            ] = entry


    # =========================================================
    # SAFETY CHECK
    # =========================================================

    missing_depths = [

        depth

        for depth
        in target_depths

        if depth
        not in depth_data
    ]


    if len(missing_depths) > 0:

        raise RuntimeError(
            "Stockfish did not provide complete depth data for: "
            + ", ".join(
                str(depth)
                for depth
                in missing_depths
            )
        )


    return depth_data