from __future__ import annotations

import chess

from .stockfish_utils import (
    analyse_legal_moves_multipv
)


def calculate_number_of_good_moves(
    engine,
    fen: str,
    max_eval_loss_cp: int = 50,
    depth: int = 15
) -> int:
    """
    Calculates the number of good legal moves.

    All legal moves are evaluated with Stockfish MultiPV from
    the SAME original position.

    The best legal move is used as the reference.

    A move is considered good if:

        best_evaluation - move_evaluation
            <= max_eval_loss_cp

    Example:

        Best move:      +0.80
        Move A:         +0.62
        Loss:              18 cp -> good

        Move B:         +0.35
        Loss:              45 cp -> good

        Move C:         -0.10
        Loss:              90 cp -> not good

    If legal moves exist, the best move itself always has an
    evaluation loss of 0 and is therefore always a good move.
    """

    board = chess.Board(
        fen
    )


    legal_move_count = (
        board.legal_moves.count()
    )


    if legal_move_count == 0:

        return 0


    # ---------------------------------------------------------
    # Evaluate every legal move from the same root position.
    # ---------------------------------------------------------

    move_evaluations = (
        analyse_legal_moves_multipv(
            engine=engine,
            board=board,
            depth=depth
        )
    )


    # ---------------------------------------------------------
    # Best legal move becomes the reference.
    # ---------------------------------------------------------

    best_evaluation = max(
        move_evaluations.values()
    )


    # ---------------------------------------------------------
    # Count acceptable alternatives.
    # ---------------------------------------------------------

    good_moves = 0


    for evaluation in move_evaluations.values():

        evaluation_loss = (
            best_evaluation
            - evaluation
        )


        if (
            evaluation_loss
            <= max_eval_loss_cp
        ):

            good_moves += 1


    return good_moves