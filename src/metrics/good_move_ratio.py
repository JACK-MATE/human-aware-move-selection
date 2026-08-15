from __future__ import annotations

from typing import Optional

import chess

from .number_of_good_moves import (
    calculate_number_of_good_moves
)


def calculate_good_move_ratio(
    engine,
    fen: str,
    max_eval_loss_cp: int = 50,
    depth: int = 15,
    precomputed_good_moves: Optional[int] = None
) -> float:
    """
    Calculates the Good Move Ratio (GMR).

    Formula:

        GMR =
            number of good moves
            --------------------
            number of legal moves

    Good moves are defined relative to Stockfish's best legal
    move at the specified search depth.

    A low value means that only a small fraction of the legal
    moves is acceptable.

    A high value means that many legal alternatives are
    acceptable.

    precomputed_good_moves allows the MetricRunner to reuse the
    already calculated Number of Good Moves and prevents a
    second expensive MultiPV search.
    """

    board = chess.Board(
        fen
    )


    legal_moves = (
        board.legal_moves.count()
    )


    if legal_moves == 0:

        return 0.0


    # ---------------------------------------------------------
    # Reuse existing result when available.
    # ---------------------------------------------------------

    if precomputed_good_moves is not None:

        good_moves = (
            precomputed_good_moves
        )


    else:

        good_moves = (
            calculate_number_of_good_moves(
                engine=engine,
                fen=fen,
                max_eval_loss_cp=
                    max_eval_loss_cp,
                depth=depth
            )
        )


    return (
        good_moves
        / legal_moves
    )