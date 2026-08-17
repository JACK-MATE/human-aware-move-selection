from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import chess
import chess.engine


from metric_runner import MetricRunner

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

from metrics.expected_score import (
    calculate_expected_scores_for_move
)


# =============================================================
# PROJECT ROOT
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# =============================================================
# EXPECTED SCORE RUNNER
# =============================================================

class ExpectedScoreRunner:

    # =========================================================
    # SCORE SCALE
    # =========================================================
    #
    # Current baseline:
    #
    #     1.0
    #
    # Complexity formula:
    #
    #     E_white =
    #         0.5
    #         + scale * (C_black - C_white) / 200
    #
    # GMR formula:
    #
    #     E_white =
    #         0.5
    #         + scale * (GMR_white - GMR_black) / 2
    #

    SCORE_SCALE = 1.0


    # =========================================================
    # TEST LIMIT
    # =========================================================
    #
    # Number of ROOT positions.
    #
    # Important:
    #
    # One root position may contain several observed moves.
    # Therefore considerably more child positions may have to
    # be analysed.
    #
    # None = complete dataset
    #

    MAX_ROOT_POSITIONS = 10


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self
    ):

        # -----------------------------------------------------
        # We reuse exactly the same metric parameters and
        # Stockfish helpers as metric_runner.py.
        # -----------------------------------------------------

        self.metric_runner = (
            MetricRunner()
        )


        # -----------------------------------------------------
        # Runtime cache.
        #
        # If several root/move combinations lead to the same
        # child FEN, its metrics are calculated only once.
        # -----------------------------------------------------

        self.child_metric_cache = {}


        # -----------------------------------------------------
        # Counters for console output.
        # -----------------------------------------------------

        self.children_calculated = 0

        self.children_from_cache = 0

        self.children_from_root_metrics = 0


    # =========================================================
    # MAIN
    # =========================================================

    def run(
        self,
        dataset_path: Path,
        metric_results_path: Path,
        output_path: Path,
        stockfish_directory: Path
    ) -> None:

        start_time = (
            time.perf_counter()
        )


        # =====================================================
        # LOAD TEST DATASET
        # =====================================================

        print(
            "Loading test dataset..."
        )


        dataset = (
            self.load_json(
                dataset_path
            )
        )


        # =====================================================
        # LOAD ROOT METRICS
        # =====================================================

        print(
            "Loading metric results..."
        )


        metric_file = (
            self.load_json(
                metric_results_path
            )
        )


        if "positions" not in metric_file:

            raise KeyError(
                "metric_results.json does not contain "
                "a 'positions' object."
            )


        root_metrics = (
            metric_file[
                "positions"
            ]
        )


        print(
            f"Dataset root positions: "
            f"{len(dataset):,}"
        )


        print(
            f"Available root metrics: "
            f"{len(root_metrics):,}"
        )


        # =====================================================
        # SELECT ROOT POSITIONS
        # =====================================================

        root_positions = list(
            dataset.keys()
        )


        if self.MAX_ROOT_POSITIONS is not None:

            root_positions = (
                root_positions[
                    :self.MAX_ROOT_POSITIONS
                ]
            )


        print(
            f"Root positions to process: "
            f"{len(root_positions):,}"
        )


        # =====================================================
        # STOCKFISH
        # =====================================================

        stockfish_path = (
            self.metric_runner.find_stockfish_executable(
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


        print(
            f"Engine ID: "
            f"{engine.id.get('name', 'Unknown')}"
        )


        # =====================================================
        # ENGINE CONFIGURATION
        # =====================================================

        if "Threads" in engine.options:

            engine.configure({

                "Threads":
                    self.metric_runner.STOCKFISH_THREADS

            })


        if "Hash" in engine.options:

            engine.configure({

                "Hash":
                    self.metric_runner.STOCKFISH_HASH_MB

            })


        # =====================================================
        # RESULT STORAGE
        # =====================================================

        results = {}


        try:

            # =================================================
            # ROOT POSITIONS
            # =================================================

            for index, root_fen in enumerate(
                root_positions,
                start=1
            ):

                root_start = (
                    time.perf_counter()
                )


                print()

                print(
                    "=" * 60
                )


                print(
                    f"Root position "
                    f"{index}/{len(root_positions)}"
                )


                print(
                    root_fen
                )


                # =============================================
                # ROOT METRICS MUST ALREADY EXIST
                # =============================================

                if root_fen not in root_metrics:

                    raise KeyError(
                        "Root FEN missing from "
                        "metric_results.json:\n\n"
                        f"{root_fen}\n\n"
                        "metric_runner.py must first calculate "
                        "the metrics of this root position."
                    )


                root_position_metrics = (
                    root_metrics[
                        root_fen
                    ]
                )


                root_data = (
                    dataset[
                        root_fen
                    ]
                )


                # =============================================
                # OBSERVED MOVES + CHILD FENS
                # =============================================

                observed_moves = (
                    self.get_observed_moves(
                        root_fen=
                            root_fen,
                        root_data=
                            root_data
                    )
                )


                print(
                    f"Observed different moves: "
                    f"{len(observed_moves)}"
                )


                move_results = {}


                # =============================================
                # EACH OBSERVED MOVE
                # =============================================

                for (
                    move_index,
                    (
                        move_uci,
                        child_fen
                    )
                ) in enumerate(
                    observed_moves.items(),
                    start=1
                ):

                    print()

                    print(
                        f"  Move "
                        f"{move_index}/"
                        f"{len(observed_moves)}: "
                        f"{move_uci}"
                    )


                    print(
                        f"  Child FEN: "
                        f"{child_fen}"
                    )


                    # =========================================
                    # CHILD METRICS
                    # =========================================
                    #
                    # Priority:
                    #
                    # 1. Already contained in metric_results
                    # 2. Already calculated during this run
                    # 3. Calculate now with Stockfish
                    #

                    if child_fen in root_metrics:

                        child_position_metrics = (
                            root_metrics[
                                child_fen
                            ]
                        )


                        self.children_from_root_metrics += 1


                        print(
                            "  Child metrics: "
                            "already available in metric_results"
                        )


                    elif child_fen in self.child_metric_cache:

                        child_position_metrics = (
                            self.child_metric_cache[
                                child_fen
                            ]
                        )


                        self.children_from_cache += 1


                        print(
                            "  Child metrics: "
                            "runtime cache"
                        )


                    else:

                        child_start = (
                            time.perf_counter()
                        )


                        print(
                            "  Child metrics: "
                            "calculating with Stockfish..."
                        )


                        child_position_metrics = (
                            self.calculate_metrics_for_fen(
                                engine=engine,
                                fen=child_fen
                            )
                        )


                        self.child_metric_cache[
                            child_fen
                        ] = (
                            child_position_metrics
                        )


                        self.children_calculated += 1


                        child_runtime = (
                            time.perf_counter()
                            - child_start
                        )


                        print(
                            f"  Child metric runtime: "
                            f"{child_runtime:.2f} s"
                        )


                    # =========================================
                    # EXPECTED SCORE
                    # =========================================

                    predicted_scores = (
                        calculate_expected_scores_for_move(
                            root_fen=
                                root_fen,
                            root_metrics=
                                root_position_metrics,
                            child_metrics=
                                child_position_metrics,
                            scale=
                                self.SCORE_SCALE
                        )
                    )


                    # =========================================
                    # STORE MOVE RESULT
                    # =========================================

                    move_results[
                        move_uci
                    ] = {

                        "child_fen":
                            child_fen,

                        "predicted_scores":
                            predicted_scores
                    }


                # =============================================
                # ROOT RESULT
                # =============================================

                board = (
                    self.board_from_position_key(
                        root_fen
                    )
                )


                results[
                    root_fen
                ] = {

                    "side_to_move":
                        (
                            "white"
                            if board.turn == chess.WHITE
                            else "black"
                        ),

                    "moves":
                        move_results
                }


                # =============================================
                # CHECKPOINT
                # =============================================

                self.save_results(
                    output_path=
                        output_path,
                    results=
                        results
                )


                root_runtime = (
                    time.perf_counter()
                    - root_start
                )


                print()

                print(
                    f"Root runtime: "
                    f"{root_runtime:.2f} s"
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
            "Expected-score calculation finished"
        )

        print(
            "=" * 60
        )


        print(
            f"Root positions processed: "
            f"{len(results):,}"
        )


        print(
            f"Child positions newly calculated: "
            f"{self.children_calculated:,}"
        )


        print(
            f"Child positions reused from metric_results: "
            f"{self.children_from_root_metrics:,}"
        )


        print(
            f"Child positions reused from runtime cache: "
            f"{self.children_from_cache:,}"
        )


        print(
            f"Runtime in minutes: "
            f"{runtime / 60:.2f}"
        )


    # =========================================================
    # GET OBSERVED MOVES + CHILD FENS
    # =========================================================

    def get_observed_moves(
        self,
        root_fen: str,
        root_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Returns:

            {
                "move_uci": "child_fen",
                ...
            }

        If test_dataset_aggregated.json explicitly contains
        "child_fen", that value is used.

        If not, the child FEN is reconstructed from:

            root FEN + played move

        The fallback also makes the code compatible with older
        versions of the aggregated dataset.
        """

        observed_moves = {}


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


            for (
                move_uci,
                move_data
            ) in moves.items():

                child_fen = None


                # =============================================
                # CHILD FEN STORED DIRECTLY IN DATASET
                # =============================================

                if isinstance(
                    move_data,
                    dict
                ):

                    child_fen = (
                        move_data.get(
                            "child_fen"
                        )
                    )


                # =============================================
                # FALLBACK:
                # RECONSTRUCT CHILD POSITION
                # =============================================

                if not child_fen:

                    child_fen = (
                        self.create_child_fen(
                            root_fen=
                                root_fen,
                            move_uci=
                                move_uci
                        )
                    )


                child_fen = (
                    self.normalize_position_key(
                        child_fen
                    )
                )


                # =============================================
                # CONSISTENCY CHECK
                # =============================================
                #
                # The same UCI move may occur in several rating
                # buckets, but it must always lead to exactly
                # the same child position.
                #

                if (
                    move_uci in observed_moves
                    and observed_moves[
                        move_uci
                    ] != child_fen
                ):

                    raise ValueError(
                        "Same move produced different child "
                        "FENs across rating buckets.\n\n"
                        f"Root:\n{root_fen}\n\n"
                        f"Move:\n{move_uci}\n\n"
                        f"First child:\n"
                        f"{observed_moves[move_uci]}\n\n"
                        f"Second child:\n"
                        f"{child_fen}"
                    )


                observed_moves[
                    move_uci
                ] = (
                    child_fen
                )


        return dict(
            sorted(
                observed_moves.items()
            )
        )


    # =========================================================
    # CALCULATE ALL METRICS FOR CHILD FEN
    # =========================================================

    def calculate_metrics_for_fen(
        self,
        engine,
        fen: str
    ) -> Dict[str, Any]:
        """
        Calculates exactly the same metrics as metric_runner.py,
        but only for one child position.
        """

        board = (
            self.board_from_position_key(
                fen
            )
        )


        legal_moves = (
            board.legal_moves.count()
        )


        # =====================================================
        # CLEAR HASH BEFORE GOOD MOVE ANALYSIS
        # =====================================================

        self.metric_runner.clear_stockfish_hash(
            engine
        )


        # =====================================================
        # NUMBER OF GOOD MOVES
        # =====================================================

        good_moves = (
            calculate_number_of_good_moves(
                engine=engine,
                fen=fen,
                max_eval_loss_cp=
                    self.metric_runner.GOOD_MOVE_MAX_LOSS_CP,
                depth=
                    self.metric_runner.GOOD_MOVE_DEPTH
            )
        )


        # =====================================================
        # GOOD MOVE RATIO
        # =====================================================

        good_move_ratio = (
            calculate_good_move_ratio(
                engine=engine,
                fen=fen,
                max_eval_loss_cp=
                    self.metric_runner.GOOD_MOVE_MAX_LOSS_CP,
                depth=
                    self.metric_runner.GOOD_MOVE_DEPTH,
                precomputed_good_moves=
                    good_moves
            )
        )


        # =====================================================
        # CLEAR HASH BEFORE DTS
        # =====================================================

        self.metric_runner.clear_stockfish_hash(
            engine
        )


        # =====================================================
        # SHARED DTS SEARCH
        # =====================================================

        depth_data = (
            analyse_depth_series(
                engine=engine,
                board=board,
                min_depth=
                    self.metric_runner.DTS_MIN_DEPTH,
                max_depth=
                    self.metric_runner.DTS_MAX_DEPTH,
                step=
                    self.metric_runner.DTS_STEP
            )
        )


        # =====================================================
        # BEST-MOVE DTS
        # =====================================================

        (
            best_move_dts,
            best_move_dts_stabilized
        ) = (
            calculate_best_move_dts(
                engine=engine,
                fen=fen,
                min_depth=
                    self.metric_runner.DTS_MIN_DEPTH,
                max_depth=
                    self.metric_runner.DTS_MAX_DEPTH,
                step=
                    self.metric_runner.DTS_STEP,
                stable_steps=
                    self.metric_runner.DTS_STABLE_STEPS,
                precomputed_depth_data=
                    depth_data
            )
        )


        # =====================================================
        # EVALUATION DTS
        # =====================================================

        (
            eval_dts,
            eval_dts_stabilized
        ) = (
            calculate_eval_dts(
                engine=engine,
                fen=fen,
                min_depth=
                    self.metric_runner.DTS_MIN_DEPTH,
                max_depth=
                    self.metric_runner.DTS_MAX_DEPTH,
                step=
                    self.metric_runner.DTS_STEP,
                stable_steps=
                    self.metric_runner.DTS_STABLE_STEPS,
                max_eval_change_cp=
                    self.metric_runner.EVAL_DTS_MAX_CHANGE_CP,
                precomputed_depth_data=
                    depth_data
            )
        )


        # =====================================================
        # BEST-MOVE DTS COMPLEXITY
        # =====================================================

        best_move_complexity_scores = (
            calculate_complexity_scores(
                depth_to_stability=
                    best_move_dts,
                stabilized=
                    best_move_dts_stabilized,
                good_move_ratio=
                    good_move_ratio,
                weightings=
                    self.metric_runner.COMPLEXITY_WEIGHTINGS,
                min_depth=
                    self.metric_runner.DTS_MIN_DEPTH,
                max_depth=
                    self.metric_runner.DTS_MAX_DEPTH,
                step=
                    self.metric_runner.DTS_STEP
            )
        )


        # =====================================================
        # EVALUATION DTS COMPLEXITY
        # =====================================================

        eval_complexity_scores = (
            calculate_complexity_scores(
                depth_to_stability=
                    eval_dts,
                stabilized=
                    eval_dts_stabilized,
                good_move_ratio=
                    good_move_ratio,
                weightings=
                    self.metric_runner.COMPLEXITY_WEIGHTINGS,
                min_depth=
                    self.metric_runner.DTS_MIN_DEPTH,
                max_depth=
                    self.metric_runner.DTS_MAX_DEPTH,
                step=
                    self.metric_runner.DTS_STEP
            )
        )


        # =====================================================
        # RETURN
        # =====================================================

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
                    best_move_complexity_scores,

                "eval_dts":
                    eval_complexity_scores
            }
        }


    # =========================================================
    # CREATE CHILD FEN
    # =========================================================

    def create_child_fen(
        self,
        root_fen: str,
        move_uci: str
    ) -> str:

        board = (
            self.board_from_position_key(
                root_fen
            )
        )


        try:

            move = (
                chess.Move.from_uci(
                    move_uci
                )
            )

        except ValueError:

            raise ValueError(
                f"Invalid UCI move:\n"
                f"{move_uci}\n\n"
                f"Root FEN:\n"
                f"{root_fen}"
            )


        if move not in board.legal_moves:

            raise ValueError(
                f"Illegal move:\n"
                f"{move_uci}\n\n"
                f"Root FEN:\n"
                f"{root_fen}"
            )


        board.push(
            move
        )


        return (
            self.get_position_key(
                board
            )
        )


    # =========================================================
    # BOARD FROM POSITION KEY
    # =========================================================

    def board_from_position_key(
        self,
        fen_key: str
    ) -> chess.Board:

        parts = (
            fen_key.split()
        )


        if len(parts) == 4:

            full_fen = (
                fen_key
                + " 0 1"
            )

        elif len(parts) == 6:

            full_fen = (
                fen_key
            )

        else:

            raise ValueError(
                "Unexpected FEN format:\n"
                f"{fen_key}"
            )


        return chess.Board(
            full_fen
        )


    # =========================================================
    # POSITION KEY
    # =========================================================

    def get_position_key(
        self,
        board: chess.Board
    ) -> str:

        return " ".join([

            board.board_fen(),

            (
                "w"
                if board.turn == chess.WHITE
                else "b"
            ),

            board.castling_xfen(),

            (
                chess.square_name(
                    board.ep_square
                )
                if board.ep_square is not None
                else "-"
            )
        ])


    # =========================================================
    # NORMALIZE FEN
    # =========================================================

    def normalize_position_key(
        self,
        fen: str
    ) -> str:
        """
        Converts both four-field and six-field FENs into the
        four-field position key used throughout the project.
        """

        board = (
            self.board_from_position_key(
                fen
            )
        )


        return (
            self.get_position_key(
                board
            )
        )


    # =========================================================
    # LOAD JSON
    # =========================================================

    def load_json(
        self,
        path: Path
    ) -> Dict[str, Any]:

        if not path.exists():

            raise FileNotFoundError(
                f"File not found:\n"
                f"{path}"
            )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = (
                json.load(
                    file
                )
            )


        if not isinstance(
            data,
            dict
        ):

            raise ValueError(
                f"Expected JSON object:\n"
                f"{path}"
            )


        return data


    # =========================================================
    # SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        output_path: Path,
        results: Dict[str, Any]
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        output = {

            "score_parameters": {

                "scale":
                    self.SCORE_SCALE,

                "complexity_formula":
                    (
                        "E_white = 0.5 + scale * "
                        "(C_black - C_white) / 200"
                    ),

                "gmr_formula":
                    (
                        "E_white = 0.5 + scale * "
                        "(GMR_white - GMR_black) / 2"
                    ),

                "objective_evaluation_used":
                    False,

                "child_metrics":
                    (
                        "Calculated directly from child FENs "
                        "using the same metric parameters as "
                        "metric_runner.py"
                    )
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
# PROGRAM START
# =============================================================

if __name__ == "__main__":

    # =========================================================
    # TEST DATASET
    # =========================================================

    dataset_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated.json"
    )


    # =========================================================
    # ROOT METRICS
    # =========================================================

    metric_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "metric_results.json"
    )


    # =========================================================
    # EXPECTED SCORE OUTPUT
    # =========================================================

    output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "expected_score_results.json"
    )


    # =========================================================
    # STOCKFISH
    # =========================================================

    stockfish_directory = (

        PROJECT_ROOT
        / "engines"
        / "stockfish"
    )


    # =========================================================
    # RUN
    # =========================================================

    runner = (
        ExpectedScoreRunner()
    )


    runner.run(
        dataset_path=
            dataset_file,

        metric_results_path=
            metric_file,

        output_path=
            output_file,

        stockfish_directory=
            stockfish_directory
    )