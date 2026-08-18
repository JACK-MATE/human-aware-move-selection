from __future__ import annotations

from typing import Dict, Tuple


# =============================================================
# BEST-MOVE DEPTH TO STABILITY
# =============================================================

def calculate_best_move_dts(
    best_move_by_depth: Dict[int, str],
    candidate_max_depth: int = 20,
    stable_steps: int = 3,
    step: int = 2
) -> Tuple[int, bool]:
    """
    Best-Move Depth to Stability (DTBMS).

    Definition:

    Earliest candidate depth from which the best move remains
    unchanged at ALL later observed depths.

    At least stable_steps observations must be available.

    Example:

        14 -> e2e4
        16 -> e2e4
        18 -> e2e4
        20 -> e2e4
        22 -> e2e4
        24 -> e2e4

    gives:

        DTBMS = 14


    IMPORTANT:

    Candidate depths only go up to 20.

    Depths 22 and 24 exist solely as confirmation depths.

    Therefore:

        20 / 22 / 24 identical

    allows DTBMS = 20.

    If no candidate depth <= 20 can be confirmed:

        return 22, False

    meaning:

        DTBMS > 20

    NOT:

        DTBMS = 22
    """

    if stable_steps <= 0:

        raise ValueError(
            "stable_steps must be greater than 0."
        )


    depths = sorted(
        best_move_by_depth.keys()
    )


    if len(depths) == 0:

        raise ValueError(
            "best_move_by_depth is empty."
        )


    # =========================================================
    # TEST EACH POSSIBLE START DEPTH
    # =========================================================

    for (
        index,
        depth
    ) in enumerate(
        depths
    ):

        # 22 and 24 are confirmation depths only.
        if depth > candidate_max_depth:

            break


        remaining_depths = (
            depths[
                index:
            ]
        )


        if (
            len(remaining_depths)
            < stable_steps
        ):

            continue


        reference_move = (
            best_move_by_depth[
                depth
            ]
        )


        stable = all(

            best_move_by_depth[
                later_depth
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


    # =========================================================
    # NOT STABILIZED BY DEPTH 20
    # =========================================================

    return (
        candidate_max_depth
        + step,
        False
    )