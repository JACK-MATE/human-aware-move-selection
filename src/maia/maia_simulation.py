from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine

from src.metrics.stockfish_utils import (
    MATE_SCORE_CP,
    board_from_position_key,
    score_to_cp,
    wdl_to_dict
)


# =============================================================
# OPTIONAL STOCKFISH LEAF CACHE
# =============================================================

LeafCache = Dict[
    Tuple[str, bool, int],
    Dict[str, Any]
]


# =============================================================
# SELECT OWN MOVES
# =============================================================

def select_own_moves(
    move_probabilities: Dict[chess.Move, float],
    min_probability: float
) -> List[Tuple[chess.Move, float]]:
    """
    Selection rule for moves of the player whose observed
    root move is being evaluated.

    Keep every Maia move with probability > min_probability.

    If none reaches the threshold:
        keep the single most probable move.

    Original Maia probabilities are preserved.
    """

    if not move_probabilities:
        return []


    ordered_moves = sorted(
        move_probabilities.items(),
        key=lambda item: (
            -item[1],
            item[0].uci()
        )
    )


    selected = [
        (move, probability)
        for move, probability
        in ordered_moves
        if probability > min_probability
    ]


    # Never allow an empty non-terminal branch.
    if not selected:
        selected = [
            ordered_moves[0]
        ]


    return selected


# =============================================================
# SELECT OPPONENT MOVES
# =============================================================

def select_opponent_moves(
    move_probabilities: Dict[chess.Move, float],
    top_n: int,
    min_probability: float
) -> List[Tuple[chess.Move, float]]:
    """
    Selection rule for opponent moves.

    1. Sort moves by Maia probability.
    2. Keep at most top_n moves.
    3. A move must have probability >= min_probability.
    4. If none survives, keep the single most probable move.

    Example:

        A 50%
        B 25%
        C 10%
        D  4%

    with top_n=3 and min_probability=5%:

        A, B, C
    """

    if not move_probabilities:
        return []


    ordered_moves = sorted(
        move_probabilities.items(),
        key=lambda item: (
            -item[1],
            item[0].uci()
        )
    )


    selected = [
        (move, probability)
        for move, probability
        in ordered_moves
        if probability >= min_probability
    ][:top_n]


    if not selected:
        selected = [
            ordered_moves[0]
        ]


    return selected


# =============================================================
# STOCKFISH LEAF EVALUATION
# =============================================================

def evaluate_leaf(
    engine,
    board: chess.Board,
    root_pov: chess.Color,
    depth: int,
    leaf_cache: Optional[LeafCache],
    cache_stats: Dict[str, int]
) -> Dict[str, Any]:
    """
    Evaluates ONE final leaf position.

    IMPORTANT:
    Stockfish is called only in this function.

    Intermediate Maia tree nodes are NOT analysed by Stockfish.

    Evaluation and WDL are always returned from the perspective
    of the player whose observed root move is being evaluated.
    """

    # =========================================================
    # TERMINAL POSITION
    # =========================================================

    outcome = board.outcome(
        claim_draw=False
    )


    if outcome is not None:

        if outcome.winner is None:

            return {
                "evaluation_cp": 0,
                "wdl": {
                    "win": 0,
                    "draw": 1000,
                    "loss": 0
                }
            }


        if outcome.winner == root_pov:

            return {
                "evaluation_cp": MATE_SCORE_CP,
                "wdl": {
                    "win": 1000,
                    "draw": 0,
                    "loss": 0
                }
            }


        return {
            "evaluation_cp": -MATE_SCORE_CP,
            "wdl": {
                "win": 0,
                "draw": 0,
                "loss": 1000
            }
        }


    # =========================================================
    # CACHE
    # =========================================================

    cache_key = (
        board.fen(),
        bool(root_pov),
        depth
    )


    if (
        leaf_cache is not None
        and cache_key in leaf_cache
    ):

        cache_stats["hits"] += 1

        return leaf_cache[
            cache_key
        ]


    # =========================================================
    # STOCKFISH
    # =========================================================

    cache_stats[
        "stockfish_evaluations"
    ] += 1


    info = engine.analyse(
        board,
        chess.engine.Limit(
            depth=depth
        )
    )


    score = info.get(
        "score"
    )

    wdl = info.get(
        "wdl"
    )


    if score is None:

        raise RuntimeError(
            "Stockfish returned no score for Maia leaf."
        )


    if wdl is None:

        raise RuntimeError(
            "Stockfish returned no WDL for Maia leaf."
        )


    result = {
        "evaluation_cp":
            score_to_cp(
                score=score,
                pov_color=root_pov
            ),

        "wdl":
            wdl_to_dict(
                wdl=wdl,
                pov_color=root_pov
            )
    }


    if leaf_cache is not None:

        leaf_cache[
            cache_key
        ] = result


    return result


# =============================================================
# SIMULATE ONE OBSERVED MOVE
# =============================================================

def simulate_observed_move(
    maia,
    stockfish_engine,
    root_fen: str,
    observed_move_uci: str,
    elo: int,
    plies_after_move: int = 6,

    own_min_probability: float = 0.25,

    opponent_top_n: int = 3,
    opponent_min_probability: float = 0.05,

    leaf_stockfish_depth: int = 12,

    leaf_cache: Optional[LeafCache] = None
) -> Dict[str, Any]:
    """
    Practical evaluation of one actually observed move.

    ROOT PLAYER:
        keep moves with >25% Maia probability.
        If none -> Top 1.

    OPPONENT:
        keep up to Top 3 moves,
        but only if probability >=5%.
        If none -> Top 1.

    After the observed move, the Maia tree continues for
    plies_after_move half-moves.

    Stockfish evaluates ONLY final leaf positions.


    Probability handling:

        p_leaf =
            p(move_1)
            * p(move_2)
            * ...
            * p(move_n)

    Original Maia probabilities are retained.

    Final value:

        sum(p_leaf * leaf_value)
        ------------------------
             sum(p_leaf)

    The denominator is saved as retained_probability_mass.
    """

    if plies_after_move < 0:

        raise ValueError(
            "plies_after_move must not be negative."
        )


    # =========================================================
    # ROOT
    # =========================================================

    board = board_from_position_key(
        root_fen
    )


    # Player whose observed move we are evaluating.
    root_pov = board.turn


    observed_move = chess.Move.from_uci(
        observed_move_uci
    )


    if observed_move not in board.legal_moves:

        raise ValueError(
            "Observed move is illegal in root FEN:\n"
            + observed_move_uci
            + "\n"
            + root_fen
        )


    # =========================================================
    # HISTORY
    # =========================================================

    history = maia.initial_history(
        board
    )


    # =========================================================
    # FIXED OBSERVED MOVE
    # =========================================================

    board.push(
        observed_move
    )


    history = maia.history_after_move(
        history,
        board
    )


    # =========================================================
    # ACCUMULATORS
    # =========================================================

    weighted_evaluation = 0.0

    weighted_win = 0.0
    weighted_draw = 0.0
    weighted_loss = 0.0

    retained_probability_mass = 0.0

    leaf_count = 0
    maia_nodes = 0

    own_nodes = 0
    opponent_nodes = 0


    cache_stats = {
        "hits": 0,
        "stockfish_evaluations": 0
    }


    # =========================================================
    # RECURSIVE TREE
    # =========================================================

    def recurse(
        current_history,
        plies_left: int,
        path_probability: float
    ) -> None:

        nonlocal weighted_evaluation
        nonlocal weighted_win
        nonlocal weighted_draw
        nonlocal weighted_loss

        nonlocal retained_probability_mass

        nonlocal leaf_count
        nonlocal maia_nodes

        nonlocal own_nodes
        nonlocal opponent_nodes


        # =====================================================
        # LEAF
        # =====================================================

        if (
            plies_left == 0
            or board.is_game_over(
                claim_draw=False
            )
        ):

            leaf = evaluate_leaf(
                engine=stockfish_engine,
                board=board,
                root_pov=root_pov,
                depth=leaf_stockfish_depth,
                leaf_cache=leaf_cache,
                cache_stats=cache_stats
            )


            weighted_evaluation += (
                path_probability
                * leaf["evaluation_cp"]
            )


            weighted_win += (
                path_probability
                * leaf["wdl"]["win"]
            )


            weighted_draw += (
                path_probability
                * leaf["wdl"]["draw"]
            )


            weighted_loss += (
                path_probability
                * leaf["wdl"]["loss"]
            )


            retained_probability_mass += (
                path_probability
            )


            leaf_count += 1

            return


        # =====================================================
        # MAIA POLICY
        # =====================================================

        move_probabilities = (
            maia.get_move_probabilities(
                board=board,
                self_elo=elo,
                opponent_elo=elo,
                history=current_history
            )
        )


        maia_nodes += 1


        # =====================================================
        # OWN PLAYER OR OPPONENT?
        # =====================================================

        if board.turn == root_pov:

            own_nodes += 1


            selected_moves = (
                select_own_moves(
                    move_probabilities=
                        move_probabilities,

                    min_probability=
                        own_min_probability
                )
            )


        else:

            opponent_nodes += 1


            selected_moves = (
                select_opponent_moves(
                    move_probabilities=
                        move_probabilities,

                    top_n=
                        opponent_top_n,

                    min_probability=
                        opponent_min_probability
                )
            )


        if not selected_moves:

            raise RuntimeError(
                "No Maia move selected in non-terminal position."
            )


        # =====================================================
        # CHILDREN
        # =====================================================

        for (
            move,
            move_probability
        ) in selected_moves:


            board.push(
                move
            )


            child_history = (
                maia.history_after_move(
                    current_history,
                    board
                )
            )


            recurse(
                current_history=
                    child_history,

                plies_left=
                    plies_left - 1,

                path_probability=(
                    path_probability
                    * move_probability
                )
            )


            board.pop()


    # =========================================================
    # START
    # =========================================================

    recurse(
        current_history=history,
        plies_left=plies_after_move,
        path_probability=1.0
    )


    # =========================================================
    # SAFETY
    # =========================================================

    if retained_probability_mass <= 0:

        raise RuntimeError(
            "Maia simulation retained zero probability mass."
        )


    if retained_probability_mass > 1.0 + 1e-6:

        raise RuntimeError(
            "Retained probability mass is greater than 1."
        )


    # =========================================================
    # GLOBAL NORMALIZATION
    # =========================================================

    average_evaluation = (
        weighted_evaluation
        / retained_probability_mass
    )


    average_wdl = {
        "win":
            weighted_win
            / retained_probability_mass,

        "draw":
            weighted_draw
            / retained_probability_mass,

        "loss":
            weighted_loss
            / retained_probability_mass
    }


    # =========================================================
    # WDL CHECK
    # =========================================================

    wdl_sum = (
        average_wdl["win"]
        + average_wdl["draw"]
        + average_wdl["loss"]
    )


    if abs(
        wdl_sum - 1000.0
    ) > 1e-3:

        raise RuntimeError(
            "Average WDL does not sum to 1000. "
            f"Received: {wdl_sum}"
        )


    # =========================================================
    # CACHE STATS
    # =========================================================

    total_leaf_requests = (
        cache_stats["hits"]
        + cache_stats["stockfish_evaluations"]
    )


    if total_leaf_requests > 0:

        cache_hit_rate = (
            cache_stats["hits"]
            / total_leaf_requests
        )

    else:

        cache_hit_rate = 0.0


    # =========================================================
    # RESULT
    # =========================================================

    return {
        "average_evaluation_cp":
            average_evaluation,

        "average_wdl":
            average_wdl,

        "retained_probability_mass":
            retained_probability_mass,

        "leaf_count":
            leaf_count,

        "maia_nodes":
            maia_nodes,

        "own_maia_nodes":
            own_nodes,

        "opponent_maia_nodes":
            opponent_nodes,

        "own_min_probability":
            own_min_probability,

        "opponent_top_n":
            opponent_top_n,

        "opponent_min_probability":
            opponent_min_probability,

        "leaf_cache_hits":
            cache_stats["hits"],

        "leaf_cache_hit_rate":
            cache_hit_rate,

        "stockfish_leaf_evaluations":
            cache_stats[
                "stockfish_evaluations"
            ]
    }