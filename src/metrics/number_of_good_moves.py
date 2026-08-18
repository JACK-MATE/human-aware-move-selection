from __future__ import annotations

from typing import Any, Dict, Tuple


# =============================================================
# NUMBER OF GOOD MOVES
# =============================================================

def calculate_good_move_data(
    move_evaluations: Dict[
        str,
        Dict[str, Any]
    ],
    max_eval_loss_cp: int = 50
) -> Tuple[
    int,
    Dict[str, Dict[str, Any]]
]:
    """
    Uses the already completed MultiPV analysis to calculate:

        Number of Good Moves

    and to retain for EVERY legal move:

        evaluation_cp
        loss_cp
        is_good
        WDL


    A move is defined as good if:

        best_evaluation - move_evaluation
            <= max_eval_loss_cp


    No Stockfish analysis happens in this function.
    It only processes already calculated values.
    """

    if len(move_evaluations) == 0:

        return (
            0,
            {}
        )


    # =========================================================
    # BEST LEGAL MOVE EVALUATION
    # =========================================================

    best_evaluation = max(

        move_info[
            "evaluation_cp"
        ]

        for move_info
        in move_evaluations.values()
    )


    # =========================================================
    # ANALYSE EVERY LEGAL MOVE
    # =========================================================

    number_of_good_moves = 0

    move_data = {}


    for (
        move_uci,
        move_info
    ) in move_evaluations.items():


        evaluation = (
            move_info[
                "evaluation_cp"
            ]
        )


        # -----------------------------------------------------
        # Evaluation loss relative to the best legal move.
        # -----------------------------------------------------

        loss_cp = (
            best_evaluation
            - evaluation
        )


        # -----------------------------------------------------
        # Good move definition.
        # -----------------------------------------------------

        is_good = (
            loss_cp
            <= max_eval_loss_cp
        )


        if is_good:

            number_of_good_moves += 1


        # -----------------------------------------------------
        # Retain all useful move-level raw information.
        # -----------------------------------------------------

        move_data[
            move_uci
        ] = {

            "evaluation_cp":
                evaluation,

            "loss_cp":
                loss_cp,

            "is_good":
                is_good,

            "wdl":
                move_info[
                    "wdl"
                ]
        }


    return (
        number_of_good_moves,
        move_data
    )