# =============================================================
# GOOD MOVE RATIO
# =============================================================

def calculate_good_move_ratio(
    number_of_good_moves: int,
    legal_moves: int
) -> float:
    """
    Formula:

        GMR =
            Number of Good Moves
            --------------------
            Number of Legal Moves
    """

    if legal_moves == 0:

        return 0.0


    return (
        number_of_good_moves
        / legal_moves
    )