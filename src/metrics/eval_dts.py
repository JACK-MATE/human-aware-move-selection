from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import chess

from .stockfish_utils import (
    analyse_depth_series
)


def calculate_eval_dts(
    engine,
    fen: str,
    min_depth: int = 6,
    max_depth: int = 20,
    step: int = 2,
    stable_steps: int = 3,
    max_eval_change_cp: int = 30,
    precomputed_depth_data:
        Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[int, bool]:
    """
    Calculates Evaluation Depth to Stability.

    Definition:

    Evaluation DTS is the earliest tested depth from which all
    remaining Stockfish evaluations stay inside one allowed
    evaluation range.

    The evaluation range is:

        maximum evaluation
        -
        minimum evaluation

    and must be <= max_eval_change_cp.

    Example:

        Depth 6   -> +10
        Depth 8   -> +18
        Depth 10  -> +14
        Depth 12  -> +62
        Depth 14  -> +55
        Depth 16  -> +58
        Depth 18  -> +57
        Depth 20  -> +59

    Although depths 6/8/10 initially look stable, the
    evaluation changes substantially afterwards.

    From depth 12 onwards:

        max = 62
        min = 55
        range = 7 cp

    Therefore:

        Evaluation DTS = 12

    At least stable_steps measurements are required.

    If no stable suffix exists within the investigated range,
    the returned DTS is one step above the maximum tested depth
    and stabilized is False.

    Example:

        22, False

    means:

        DTS > 20
    """

    if stable_steps <= 0:

        raise ValueError(
            "stable_steps must be greater than 0."
        )


    if max_eval_change_cp < 0:

        raise ValueError(
            "max_eval_change_cp must not be negative."
        )


    # ---------------------------------------------------------
    # Reuse shared depth data when available.
    # ---------------------------------------------------------

    if precomputed_depth_data is not None:

        depth_data = (
            precomputed_depth_data
        )


    else:

        board = chess.Board(
            fen
        )

        depth_data = (
            analyse_depth_series(
                engine=engine,
                board=board,
                min_depth=min_depth,
                max_depth=max_depth,
                step=step
            )
        )


    tested_depths = sorted(
        depth_data.keys()
    )


    if len(tested_depths) == 0:

        raise RuntimeError(
            "No depth data available for Evaluation DTS."
        )


    # ---------------------------------------------------------
    # Find earliest stable suffix.
    # ---------------------------------------------------------

    for index, depth in enumerate(
        tested_depths
    ):

        remaining_depths = (
            tested_depths[
                index:
            ]
        )


        if (
            len(remaining_depths)
            < stable_steps
        ):

            break


        evaluations = [

            int(
                depth_data[
                    later_depth
                ][
                    "evaluation_cp"
                ]
            )

            for later_depth
            in remaining_depths
        ]


        evaluation_range = (
            max(
                evaluations
            )
            -
            min(
                evaluations
            )
        )


        if (
            evaluation_range
            <= max_eval_change_cp
        ):

            return (
                depth,
                True
            )


    # ---------------------------------------------------------
    # Not stabilized inside investigated range.
    # ---------------------------------------------------------

    censored_dts = (
        tested_depths[-1]
        + step
    )


    return (
        censored_dts,
        False
    )