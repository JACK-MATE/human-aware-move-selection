from __future__ import annotations

from typing import Dict, List, Tuple


# =============================================================
# DTS NORMALIZATION
# =============================================================

def normalize_dts(
    depth_to_stability: int,
    stabilized: bool,
    min_depth: int = 6,
    max_depth: int = 20,
    step: int = 2
) -> float:
    """
    Normalizes a DTS value to the range [0, 1].

    With the current settings:

        DTS 6   -> 0.000
        DTS 8   -> 0.125
        DTS 10  -> 0.250
        DTS 12  -> 0.375
        DTS 14  -> 0.500
        DTS 16  -> 0.625
        DTS 18  -> 0.750
        DTS 20  -> 0.875
        DTS >20 -> 1.000

    If the metric has not stabilized within the tested range,
    the next theoretical tested depth is used as a coded value.

    With max_depth=20 and step=2:

        DTS >20 -> coded as 22

    IMPORTANT:
        This does NOT mean that the actual DTS was measured
        as exactly 22.

        It is only a numerical representation of the censored
        category "DTS > 20" for the Complexity Score.
    """

    censored_depth = (
        max_depth
        + step
    )


    # ---------------------------------------------------------
    # Determine DTS value used for normalization
    # ---------------------------------------------------------

    if stabilized:

        coded_dts = (
            depth_to_stability
        )

    else:

        coded_dts = (
            censored_depth
        )


    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    normalized = (
        (
            coded_dts
            - min_depth
        )
        /
        (
            censored_depth
            - min_depth
        )
    )


    # ---------------------------------------------------------
    # Safety bounds
    # ---------------------------------------------------------

    if normalized < 0.0:

        return 0.0


    if normalized > 1.0:

        return 1.0


    return normalized


# =============================================================
# SINGLE COMPLEXITY SCORE
# =============================================================

def calculate_complexity_score(
    depth_to_stability: int,
    stabilized: bool,
    good_move_ratio: float,
    dts_weight: float,
    gmr_weight: float,
    min_depth: int = 6,
    max_depth: int = 20,
    step: int = 2
) -> float:
    """
    Calculates one Complexity Score.

    Formula:

        Complexity =
            100 * (
                dts_weight * normalized_DTS
                +
                gmr_weight * (1 - GMR)
            )

    Interpretation:

        High DTS
            -> higher complexity

        Low GMR
            -> higher complexity

        High GMR
            -> lower complexity

    The function is generic.

    It can therefore be used with:

        Best-Move DTS

    or:

        Evaluation DTS

    This allows both Complexity definitions to be tested
    independently using exactly the same formula.
    """

    # ---------------------------------------------------------
    # Validate weights
    # ---------------------------------------------------------

    if abs(
        (
            dts_weight
            + gmr_weight
        )
        - 1.0
    ) > 0.000001:

        raise ValueError(
            "dts_weight and gmr_weight must sum to 1."
        )


    if (
        dts_weight < 0.0
        or
        gmr_weight < 0.0
    ):

        raise ValueError(
            "Complexity weights must not be negative."
        )


    # ---------------------------------------------------------
    # Validate GMR
    # ---------------------------------------------------------

    if (
        good_move_ratio < 0.0
        or
        good_move_ratio > 1.0
    ):

        raise ValueError(
            "good_move_ratio must be between 0 and 1."
        )


    # ---------------------------------------------------------
    # Normalize DTS
    # ---------------------------------------------------------

    normalized_dts = (
        normalize_dts(
            depth_to_stability=
                depth_to_stability,
            stabilized=
                stabilized,
            min_depth=
                min_depth,
            max_depth=
                max_depth,
            step=
                step
        )
    )


    # ---------------------------------------------------------
    # Convert GMR into move narrowness
    # ---------------------------------------------------------
    #
    # GMR = 1.0
    #     -> all legal moves are good
    #     -> narrowness = 0
    #
    # GMR = 0.1
    #     -> only 10 % of moves are good
    #     -> narrowness = 0.9
    #

    move_narrowness = (
        1.0
        - good_move_ratio
    )


    # ---------------------------------------------------------
    # Calculate final score
    # ---------------------------------------------------------

    complexity = (
        100.0
        * (
            dts_weight
            * normalized_dts

            +

            gmr_weight
            * move_narrowness
        )
    )


    return complexity


# =============================================================
# MULTIPLE WEIGHTINGS
# =============================================================

def calculate_complexity_scores(
    depth_to_stability: int,
    stabilized: bool,
    good_move_ratio: float,
    weightings: List[Tuple[float, float]],
    min_depth: int = 6,
    max_depth: int = 20,
    step: int = 2
) -> Dict[str, float]:
    """
    Calculates Complexity Scores for several DTS/GMR
    weight combinations.

    Example:

        (0.0, 1.0)
        (0.1, 0.9)
        ...
        (0.5, 0.5)
        ...
        (1.0, 0.0)

    The result can later be compared with human error data
    without performing any additional Stockfish analysis.

    Example output:

        {
            "dts_0.4_gmr_0.6": 61.23,
            "dts_0.5_gmr_0.5": 66.81,
            "dts_0.6_gmr_0.4": 72.39
        }
    """

    results = {}


    for (
        dts_weight,
        gmr_weight
    ) in weightings:

        score = (
            calculate_complexity_score(
                depth_to_stability=
                    depth_to_stability,
                stabilized=
                    stabilized,
                good_move_ratio=
                    good_move_ratio,
                dts_weight=
                    dts_weight,
                gmr_weight=
                    gmr_weight,
                min_depth=
                    min_depth,
                max_depth=
                    max_depth,
                step=
                    step
            )
        )


        key = (
            f"dts_{dts_weight:.1f}"
            f"_gmr_{gmr_weight:.1f}"
        )


        results[
            key
        ] = score


    return results