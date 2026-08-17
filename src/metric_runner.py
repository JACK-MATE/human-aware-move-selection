from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import chess
import chess.engine


from metrics.number_of_good_moves import (
    calculate_number_of_good_moves
)

from metrics.good_move_ratio import (
    calculate_good_move_ratio
)

from metrics.best_move_dts import (
    calculate_best_move_dts
)

from metrics.eval_dts import (
    calculate_eval_dts
)

from metrics.complexity_score import (
    calculate_complexity_scores
)

from metrics.stockfish_utils import (
    analyse_depth_series
)


# =============================================================
# PROJECT ROOT
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# =============================================================
# METRIC RUNNER
# =============================================================

class MetricRunner:

    # =========================================================
    # GOOD MOVE PARAMETERS
    # =========================================================

    GOOD_MOVE_MAX_LOSS_CP = 50
    GOOD_MOVE_DEPTH = 15


    # =========================================================
    # DTS PARAMETERS
    # =========================================================

    DTS_MIN_DEPTH = 6
    DTS_MAX_DEPTH = 20
    DTS_STEP = 2
    DTS_STABLE_STEPS = 3


    # =========================================================
    # EVALUATION DTS
    # =========================================================

    EVAL_DTS_MAX_CHANGE_CP = 30


    # =========================================================
    # COMPLEXITY WEIGHTINGS
    # =========================================================

    COMPLEXITY_WEIGHTINGS = [

        (0.0, 1.0),
        (0.1, 0.9),
        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2),
        (0.9, 0.1),
        (1.0, 0.0)

    ]


    # =========================================================
    # CHILD POSITION MODE
    # =========================================================
    #
    # "observed":
    #
    #     Only moves that actually occurred in the test
    #     dataset are analysed.
    #
    #     This is what we currently want for testing.
    #
    #
    # "legal":
    #
    #     Every legal move from every root FEN is analysed.
    #
    #     This can later be used for a complete human-aware
    #     move selector.
    #

    CHILD_MOVE_MODE = "observed"


    # =========================================================
    # STOCKFISH
    # =========================================================

    STOCKFISH_THREADS = 1
    STOCKFISH_HASH_MB = 32


    # =========================================================
    # TEST LIMIT
    # =========================================================
    #
    # IMPORTANT:
    #
    # This limits ROOT positions.
    #
    # Each root may produce several child FENs, so the actual
    # number of positions analysed can be substantially larger.
    #
    # Set to None for the complete dataset.
    #

    MAX_ROOT_POSITIONS = 10


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self
    ):

        self.dataset = {}

        self.stockfish_name = None


    # =========================================================
    # MAIN
    # =========================================================

    def run(
        self,
        dataset_path: Path,
        output_path: Path,
        stockfish_directory: Path
    ) -> None:

        start_time = (
            time.perf_counter()
        )


        # =====================================================
        # LOAD DATASET
        # =====================================================

        print(
            "Loading test dataset..."
        )


        self.dataset = (
            self.load_dataset(
                dataset_path
            )
        )


        print(
            f"Unique root positions in dataset: "
            f"{len(self.dataset):,}"
        )


        # =====================================================
        # SELECT ROOT POSITIONS
        # =====================================================

        root_positions = list(
            self.dataset.keys()
        )


        if self.MAX_ROOT_POSITIONS is not None:

            root_positions = root_positions[
                :self.MAX_ROOT_POSITIONS
            ]


        print(
            f"Root positions selected: "
            f"{len(root_positions):,}"
        )


        # =====================================================
        # COLLECT ROOT + CHILD POSITIONS
        # =====================================================
        #
        # We need metrics for:
        #
        #     P
        #
        # and:
        #
        #     P after move m
        #
        # because expected score compares the complexity faced
        # by the player before the move with the complexity
        # faced by the opponent after the move.
        #

        positions = (
            self.collect_required_positions(
                root_positions=
                    root_positions
            )
        )


        print(
            f"Unique metric positions "
            f"(roots + children): "
            f"{len(positions):,}"
        )


        # =====================================================
        # STOCKFISH
        # =====================================================

        stockfish_path = (
            self.find_stockfish_executable(
                stockfish_directory
            )
        )


        print()

        print(
            "Stockfish:"
        )

        print(
            stockfish_path
        )


        engine = (
            chess.engine.SimpleEngine.popen_uci(
                str(
                    stockfish_path
                )
            )
        )


        self.stockfish_name = (
            engine.id.get(
                "name",
                "Unknown"
            )
        )


        print(
            f"Engine ID: "
            f"{self.stockfish_name}"
        )


        results = {}


        try:

            # =================================================
            # ENGINE CONFIGURATION
            # =================================================

            if "Threads" in engine.options:

                engine.configure({
                    "Threads":
                        self.STOCKFISH_THREADS
                })


            if "Hash" in engine.options:

                engine.configure({
                    "Hash":
                        self.STOCKFISH_HASH_MB
                })


            # =================================================
            # ANALYSE ALL REQUIRED FENS
            # =================================================

            total_positions = len(
                positions
            )


            for index, fen in enumerate(
                positions,
                start=1
            ):

                position_start = (
                    time.perf_counter()
                )


                print()

                print(
                    "=" * 60
                )


                print(
                    f"Position "
                    f"{index}/{total_positions}"
                )


                print(
                    fen
                )


                metrics = (
                    self.calculate_metrics_for_fen(
                        engine=engine,
                        fen=fen
                    )
                )


                results[
                    fen
                ] = metrics


                # =============================================
                # CONSOLE OUTPUT
                # =============================================

                runtime = (
                    time.perf_counter()
                    - position_start
                )


                print()

                print(
                    f"Legal moves: "
                    f"{metrics['legal_moves']}"
                )


                print(
                    f"Good moves: "
                    f"{metrics['number_of_good_moves']}"
                )


                print(
                    f"Good Move Ratio: "
                    f"{metrics['good_move_ratio']:.4f}"
                )


                if metrics[
                    "best_move_dts_stabilized"
                ]:

                    print(
                        f"Best-Move DTS: "
                        f"{metrics['best_move_dts']}"
                    )

                else:

                    print(
                        f"Best-Move DTS: "
                        f">{self.DTS_MAX_DEPTH}"
                    )


                if metrics[
                    "eval_dts_stabilized"
                ]:

                    print(
                        f"Evaluation DTS: "
                        f"{metrics['eval_dts']}"
                    )

                else:

                    print(
                        f"Evaluation DTS: "
                        f">{self.DTS_MAX_DEPTH}"
                    )


                print(
                    f"Position runtime: "
                    f"{runtime:.2f} s"
                )


                # =============================================
                # CHECKPOINT
                # =============================================

                self.save_results(
                    output_path=
                        output_path,
                    results=
                        results
                )


        finally:

            engine.quit()


        # =====================================================
        # FINISHED
        # =====================================================

        runtime = (
            time.perf_counter()
            - start_time
        )


        print()

        print(
            "=" * 60
        )

        print(
            "Metric calculation finished"
        )

        print(
            "=" * 60
        )


        print(
            f"Metric positions analysed: "
            f"{len(results):,}"
        )


        print(
            f"Runtime in minutes: "
            f"{runtime / 60:.2f}"
        )


    # =========================================================
    # CALCULATE METRICS FOR ONE FEN
    # =========================================================

    def calculate_metrics_for_fen(
        self,
        engine,
        fen: str
    ) -> Dict[str, Any]:

        board = (
            chess.Board(
                fen
            )
        )


        legal_moves = (
            board.legal_moves.count()
        )


        # =====================================================
        # GOOD MOVES / GMR
        # =====================================================

        self.clear_stockfish_hash(
            engine
        )


        good_moves = (
            calculate_number_of_good_moves(
                engine=engine,
                fen=fen,
                max_eval_loss_cp=
                    self.GOOD_MOVE_MAX_LOSS_CP,
                depth=
                    self.GOOD_MOVE_DEPTH
            )
        )


        good_move_ratio = (
            calculate_good_move_ratio(
                engine=engine,
                fen=fen,
                max_eval_loss_cp=
                    self.GOOD_MOVE_MAX_LOSS_CP,
                depth=
                    self.GOOD_MOVE_DEPTH,
                precomputed_good_moves=
                    good_moves
            )
        )


        # =====================================================
        # DTS
        # =====================================================

        self.clear_stockfish_hash(
            engine
        )


        depth_data = (
            analyse_depth_series(
                engine=engine,
                board=board,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_MAX_DEPTH,
                step=
                    self.DTS_STEP
            )
        )


        (
            best_move_dts,
            best_move_dts_stabilized
        ) = (
            calculate_best_move_dts(
                engine=engine,
                fen=fen,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_MAX_DEPTH,
                step=
                    self.DTS_STEP,
                stable_steps=
                    self.DTS_STABLE_STEPS,
                precomputed_depth_data=
                    depth_data
            )
        )


        (
            eval_dts,
            eval_dts_stabilized
        ) = (
            calculate_eval_dts(
                engine=engine,
                fen=fen,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_MAX_DEPTH,
                step=
                    self.DTS_STEP,
                stable_steps=
                    self.DTS_STABLE_STEPS,
                max_eval_change_cp=
                    self.EVAL_DTS_MAX_CHANGE_CP,
                precomputed_depth_data=
                    depth_data
            )
        )


        # =====================================================
        # COMPLEXITY – BEST-MOVE DTS
        # =====================================================

        best_complexity = (
            calculate_complexity_scores(
                depth_to_stability=
                    best_move_dts,
                stabilized=
                    best_move_dts_stabilized,
                good_move_ratio=
                    good_move_ratio,
                weightings=
                    self.COMPLEXITY_WEIGHTINGS,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_MAX_DEPTH,
                step=
                    self.DTS_STEP
            )
        )


        # =====================================================
        # COMPLEXITY – EVAL DTS
        # =====================================================

        eval_complexity = (
            calculate_complexity_scores(
                depth_to_stability=
                    eval_dts,
                stabilized=
                    eval_dts_stabilized,
                good_move_ratio=
                    good_move_ratio,
                weightings=
                    self.COMPLEXITY_WEIGHTINGS,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_MAX_DEPTH,
                step=
                    self.DTS_STEP
            )
        )


        return {

            "legal_moves":
                legal_moves,

            "number_of_good_moves":
                good_moves,

            "good_move_ratio":
                good_move_ratio,

            "best_move_dts":
                best_move_dts,

            "best_move_dts_stabilized":
                best_move_dts_stabilized,

            "eval_dts":
                eval_dts,

            "eval_dts_stabilized":
                eval_dts_stabilized,

            "complexity_scores": {

                "best_move_dts":
                    best_complexity,

                "eval_dts":
                    eval_complexity
            }
        }


    # =========================================================
    # COLLECT REQUIRED POSITIONS
    # =========================================================

    def collect_required_positions(
        self,
        root_positions: List[str]
    ) -> List[str]:
        """
        Collects all unique positions whose metrics are required.

        This always includes every selected root FEN.

        Depending on CHILD_MOVE_MODE it additionally includes
        child positions after:

            "observed"
                -> moves actually present in the dataset

            "legal"
                -> every legal move
        """

        positions = []

        seen = set()


        for root_fen in root_positions:

            if root_fen not in seen:

                positions.append(
                    root_fen
                )

                seen.add(
                    root_fen
                )


            moves = (
                self.get_moves_for_root(
                    root_fen=
                        root_fen
                )
            )


            for move_uci in moves:

                child_fen = (
                    self.create_child_fen(
                        root_fen=
                            root_fen,
                        move_uci=
                            move_uci
                    )
                )


                if child_fen in seen:

                    continue


                positions.append(
                    child_fen
                )

                seen.add(
                    child_fen
                )


        return positions


    # =========================================================
    # GET MOVES FOR ROOT
    # =========================================================

    def get_moves_for_root(
        self,
        root_fen: str
    ) -> List[str]:

        if self.CHILD_MOVE_MODE == "legal":

            board = chess.Board(
                root_fen
            )

            return [

                move.uci()

                for move
                in board.legal_moves
            ]


        if self.CHILD_MOVE_MODE != "observed":

            raise ValueError(
                "CHILD_MOVE_MODE must be "
                "'observed' or 'legal'."
            )


        root_data = (
            self.dataset[
                root_fen
            ]
        )


        observed_moves = set()


        rating_buckets = (
            root_data.get(
                "rating_buckets",
                {}
            )
        )


        for bucket_data in (
            rating_buckets.values()
        ):

            moves = (
                bucket_data.get(
                    "moves",
                    {}
                )
            )


            for move_uci in moves.keys():

                observed_moves.add(
                    move_uci
                )


        return sorted(
            observed_moves
        )


    # =========================================================
    # CREATE CHILD FEN
    # =========================================================

    def create_child_fen(
        self,
        root_fen: str,
        move_uci: str
    ) -> str:
        """
        Applies one move to a root FEN and returns the resulting
        position in the same four-field FEN format used by the
        dataset:

            board
            side to move
            castling rights
            en-passant square

        Halfmove and fullmove counters are deliberately omitted.
        """

        board = (
            chess.Board(
                root_fen
            )
        )


        move = (
            chess.Move.from_uci(
                move_uci
            )
        )


        if move not in board.legal_moves:

            raise ValueError(
                f"Illegal move {move_uci} "
                f"for FEN:\n{root_fen}"
            )


        board.push(
            move
        )


        turn = (
            "w"
            if board.turn
            else "b"
        )


        castling = (
            board.castling_xfen()
        )


        if board.ep_square is None:

            ep_square = "-"

        else:

            ep_square = (
                chess.square_name(
                    board.ep_square
                )
            )


        return (
            f"{board.board_fen()} "
            f"{turn} "
            f"{castling} "
            f"{ep_square}"
        )


    # =========================================================
    # CLEAR HASH
    # =========================================================

    def clear_stockfish_hash(
        self,
        engine
    ) -> None:

        if "Clear Hash" in engine.options:

            engine.configure({
                "Clear Hash": None
            })


    # =========================================================
    # LOAD DATASET
    # =========================================================

    def load_dataset(
        self,
        dataset_path: Path
    ) -> Dict[str, Any]:

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n"
                f"{dataset_path}"
            )


        with open(
            dataset_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )


    # =========================================================
    # FIND STOCKFISH
    # =========================================================

    def find_stockfish_executable(
        self,
        directory: Path
    ) -> Path:

        if not directory.exists():

            raise FileNotFoundError(
                f"Stockfish directory not found:\n"
                f"{directory}"
            )


        candidates = []


        for file_path in (
            directory.iterdir()
        ):

            if not file_path.is_file():

                continue


            if not file_path.name.lower().startswith(
                "stockfish"
            ):

                continue


            if (
                file_path.suffix.lower()
                == ".exe"
            ):

                candidates.append(
                    file_path
                )

                continue


            if os.access(
                str(
                    file_path
                ),
                os.X_OK
            ):

                candidates.append(
                    file_path
                )


        if len(candidates) == 0:

            raise FileNotFoundError(
                "No Stockfish executable found."
            )


        if len(candidates) > 1:

            raise RuntimeError(
                "More than one Stockfish executable found:\n"
                + "\n".join(
                    str(path)
                    for path
                    in candidates
                )
            )


        return candidates[
            0
        ]


    # =========================================================
    # SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        output_path: Path,
        results: Dict[str, Dict[str, Any]]
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        output = {

            "metric_parameters": {

                "stockfish_name":
                    self.stockfish_name,

                "stockfish_threads":
                    self.STOCKFISH_THREADS,

                "stockfish_hash_mb":
                    self.STOCKFISH_HASH_MB,

                "good_move_max_loss_cp":
                    self.GOOD_MOVE_MAX_LOSS_CP,

                "good_move_depth":
                    self.GOOD_MOVE_DEPTH,

                "good_move_analysis":
                    "MultiPV",

                "good_move_reference":
                    "best_move_evaluation",

                "dts_min_depth":
                    self.DTS_MIN_DEPTH,

                "dts_max_depth":
                    self.DTS_MAX_DEPTH,

                "dts_step":
                    self.DTS_STEP,

                "dts_stable_steps":
                    self.DTS_STABLE_STEPS,

                "dts_censored_value":
                    (
                        self.DTS_MAX_DEPTH
                        + self.DTS_STEP
                    ),

                "eval_dts_max_change_cp":
                    self.EVAL_DTS_MAX_CHANGE_CP,

                "complexity_formula":
                    (
                        "100 * (dts_weight * normalized_dts "
                        "+ gmr_weight * (1 - gmr))"
                    ),

                "complexity_dts_variants": [

                    "best_move_dts",
                    "eval_dts"

                ],

                "complexity_weightings":
                    self.COMPLEXITY_WEIGHTINGS,

                "child_move_mode":
                    self.CHILD_MOVE_MODE
            },


            "positions":
                results
        }


        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )


# =============================================================
# START
# =============================================================

if __name__ == "__main__":

    test_dataset_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated.json"
    )


    metric_output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "metric_results.json"
    )


    stockfish_directory = (

        PROJECT_ROOT
        / "engines"
        / "stockfish"
    )


    runner = (
        MetricRunner()
    )


    runner.run(
        dataset_path=
            test_dataset_file,
        output_path=
            metric_output_file,
        stockfish_directory=
            stockfish_directory
    )