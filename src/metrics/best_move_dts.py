from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import chess

from .stockfish_utils import (
    analyse_depth_series
)


def calculate_best_move_dts(
    engine,
    fen: str,
    min_depth: int = 6,
    max_depth: int = 20,
    step: int = 2,
    stable_steps: int = 3,
    precomputed_depth_data:
        Optional[Dict[int, Dict[str, Any]]] = None
) -> Tuple[int, bool]:
    """
    Calculates Best-Move Depth to Stability.

    Definition:

    Best-Move DTS is the earliest tested depth from which the
    best move remains unchanged for ALL remaining tested depths.

    At least stable_steps observations must remain.

    Example:

        Depth 6   -> Move A
        Depth 8   -> Move A
        Depth 10  -> Move A
        Depth 12  -> Move B
        Depth 14  -> Move B
        Depth 16  -> Move B
        Depth 18  -> Move B
        Depth 20  -> Move B

    Result:

        DTS = 12

    The temporary stability at 6/8/10 is ignored because the
    best move changes later.

    If no sufficiently long stable suffix exists within the
    investigated depth range, the function returns:

        max tested depth + step

    together with:

        stabilized = False

    Example:

        return 22, False

    This means:

        DTS > 20

    NOT that the exact DTS is known to be 22.
    """

    if stable_steps <= 0:

        raise ValueError(
            "stable_steps must be greater than 0."
        )


    # ---------------------------------------------------------
    # Use the shared Stockfish depth search when supplied.
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
            "No depth data available for Best-Move DTS."
        )


    # ---------------------------------------------------------
    # Search for the earliest stable suffix.
    # ---------------------------------------------------------

    for index, depth in enumerate(
        tested_depths
    ):

        remaining_depths = (
            tested_depths[
                index:
            ]
        )


        # We require enough observations to actually call the
        # result stable.
        if (
            len(remaining_depths)
            < stable_steps
        ):

            break


        reference_move = (
            depth_data[
                depth
            ][
                "best_move"
            ]
        )


        stable = all(

            depth_data[
                later_depth
            ][
                "best_move"
            ]
            == reference_move

            for later_depth
            in remaining_depths
        )


        if stable:

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