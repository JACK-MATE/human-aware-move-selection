from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

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
    #
    # All legal moves are evaluated from the SAME starting
    # position using Stockfish MultiPV.
    #
    # The best legal move is used as the reference.
    #

    GOOD_MOVE_MAX_LOSS_CP = 50

    GOOD_MOVE_DEPTH = 15


    # =========================================================
    # DTS PARAMETERS
    # =========================================================
    #
    # Tested depths:
    #
    # 6, 8, 10, 12, 14, 16, 18, 20
    #

    DTS_MIN_DEPTH = 6

    DTS_MAX_DEPTH = 20

    DTS_STEP = 2

    DTS_STABLE_STEPS = 3


    # =========================================================
    # EVALUATION DTS PARAMETERS
    # =========================================================

    EVAL_DTS_MAX_CHANGE_CP = 30


    # =========================================================
    # COMPLEXITY SCORE WEIGHTINGS
    # =========================================================
    #
    # Tuple:
    #
    #     (DTS weight, GMR weight)
    #
    #
    # Both Complexity variants use these same weightings:
    #
    #     1. Best-Move DTS + GMR
    #     2. Evaluation DTS + GMR
    #
    #
    # This makes it possible to compare:
    #
    #     - which DTS works better
    #     - which weighting works better
    #     - whether this changes by rating bucket
    #

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
    # STOCKFISH PARAMETERS
    # =========================================================

    STOCKFISH_THREADS = 1

    STOCKFISH_HASH_MB = 32


    # =========================================================
    # TEST LIMIT
    # =========================================================
    #
    # Set to:
    #
    #     None
    #
    # for the complete dataset.
    #

    MAX_POSITIONS = 10


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
        # LOAD TEST DATASET
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
            f"Unique positions in dataset: "
            f"{len(self.dataset):,}"
        )


        # =====================================================
        # ONLY FENS ARE REQUIRED
        # =====================================================
        #
        # Occurrences and rating buckets are deliberately not
        # copied into metric_results.json.
        #

        positions = list(
            self.dataset.keys()
        )


        if self.MAX_POSITIONS is not None:

            positions = positions[
                :self.MAX_POSITIONS
            ]


        print(
            f"Positions to analyze: "
            f"{len(positions):,}"
        )


        # =====================================================
        # FIND STOCKFISH
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


        # =====================================================
        # START STOCKFISH
        # =====================================================

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


        # =====================================================
        # RESULT STORAGE
        # =====================================================

        results = {}


        try:

            # =================================================
            # ENGINE CONFIGURATION
            # =================================================

            options = (
                engine.options
            )


            if "Threads" in options:

                engine.configure({
                    "Threads":
                        self.STOCKFISH_THREADS
                })


            if "Hash" in options:

                engine.configure({
                    "Hash":
                        self.STOCKFISH_HASH_MB
                })


            # =================================================
            # ANALYZE POSITIONS
            # =================================================

            total_positions = (
                len(
                    positions
                )
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


                # =============================================
                # BOARD
                # =============================================

                board = (
                    chess.Board(
                        fen
                    )
                )


                legal_moves = (
                    board.legal_moves.count()
                )


                # =============================================
                # CLEAR HASH BEFORE GOOD MOVE ANALYSIS
                # =============================================

                self.clear_stockfish_hash(
                    engine
                )


                # =============================================
                # NUMBER OF GOOD MOVES
                # =============================================

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


                # =============================================
                # GOOD MOVE RATIO
                # =============================================

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


                # =============================================
                # CLEAR HASH BEFORE DTS
                # =============================================
                #
                # Good-Move MultiPV has already searched this
                # position deeply.
                #
                # DTS must start with a fresh transposition
                # table.
                #

                self.clear_stockfish_hash(
                    engine
                )


                # =============================================
                # SHARED DTS SEARCH
                # =============================================
                #
                # ONE continuous search provides BOTH:
                #
                #     best move
                #     evaluation
                #
                # at:
                #
                #     6, 8, 10, ..., 20
                #

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


                # =============================================
                # BEST-MOVE DTS
                # =============================================

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


                # =============================================
                # EVALUATION DTS
                # =============================================

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


                # =============================================
                # COMPLEXITY:
                # BEST-MOVE DTS + GMR
                # =============================================

                best_move_complexity_scores = (
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


                # =============================================
                # COMPLEXITY:
                # EVALUATION DTS + GMR
                # =============================================
                #
                # This uses the exact same Complexity formula,
                # but replaces Best-Move DTS with Evaluation
                # DTS.
                #
                # No additional Stockfish calculation is
                # required.
                #

                eval_complexity_scores = (
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


                # =============================================
                # STORE RESULT
                # =============================================

                results[
                    fen
                ] = {

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


                # =============================================
                # CONSOLE OUTPUT
                # =============================================

                position_runtime = (
                    time.perf_counter()
                    - position_start
                )


                print()


                print(
                    f"Legal moves: "
                    f"{legal_moves}"
                )


                print(
                    f"Good moves: "
                    f"{good_moves}"
                )


                print(
                    f"Good Move Ratio: "
                    f"{good_move_ratio:.4f}"
                )


                # ---------------------------------------------
                # Best-Move DTS
                # ---------------------------------------------

                if best_move_dts_stabilized:

                    print(
                        f"Best-Move DTS: "
                        f"{best_move_dts}"
                    )

                else:

                    print(
                        f"Best-Move DTS: "
                        f">{self.DTS_MAX_DEPTH}"
                    )


                # ---------------------------------------------
                # Evaluation DTS
                # ---------------------------------------------

                if eval_dts_stabilized:

                    print(
                        f"Evaluation DTS: "
                        f"{eval_dts}"
                    )

                else:

                    print(
                        f"Evaluation DTS: "
                        f">{self.DTS_MAX_DEPTH}"
                    )


                # ---------------------------------------------
                # Best-Move Complexity Scores
                # ---------------------------------------------

                print(
                    "Complexity Scores "
                    "(Best-Move DTS + GMR):"
                )


                for (
                    weighting,
                    score
                ) in (
                    best_move_complexity_scores.items()
                ):

                    print(
                        f"  {weighting}: "
                        f"{score:.2f}"
                    )


                # ---------------------------------------------
                # Evaluation Complexity Scores
                # ---------------------------------------------

                print(
                    "Complexity Scores "
                    "(Evaluation DTS + GMR):"
                )


                for (
                    weighting,
                    score
                ) in (
                    eval_complexity_scores.items()
                ):

                    print(
                        f"  {weighting}: "
                        f"{score:.2f}"
                    )


                print(
                    f"Position runtime: "
                    f"{position_runtime:.2f} s"
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
            f"Positions analyzed: "
            f"{len(results):,}"
        )


        print(
            f"Runtime in minutes: "
            f"{runtime / 60:.2f}"
        )


    # =========================================================
    # CLEAR STOCKFISH HASH
    # =========================================================

    def clear_stockfish_hash(
        self,
        engine
    ) -> None:
        """
        Clears Stockfish's transposition table.

        This is done:

            1. before Good Move / GMR
            2. before DTS

        so that the two engine-based metric groups do not
        influence each other through cached search results.
        """

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
        """
        Loads the aggregated test dataset.

        Only its FEN keys are required for metric calculation.
        """

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Test dataset not found:\n"
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
    # FIND STOCKFISH EXECUTABLE
    # =========================================================

    def find_stockfish_executable(
        self,
        directory: Path
    ) -> Path:
        """
        Finds exactly one Stockfish executable.
        """

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


            # Windows
            if (
                file_path.suffix.lower()
                == ".exe"
            ):

                candidates.append(
                    file_path
                )

                continue


            # Linux / macOS
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
                "No Stockfish executable found in:\n"
                f"{directory}"
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
        """
        Saves metric_results.json.

        The file contains:

            metric_parameters
                -> calculation settings

            positions
                -> FEN
                    -> metrics

        Dataset-specific information such as occurrences and
        rating buckets is deliberately excluded.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        output = {

            # =================================================
            # PARAMETER OVERVIEW
            # =================================================

            "metric_parameters": {

                # ---------------------------------------------
                # Stockfish
                # ---------------------------------------------

                "stockfish_name":
                    self.stockfish_name,

                "stockfish_threads":
                    self.STOCKFISH_THREADS,

                "stockfish_hash_mb":
                    self.STOCKFISH_HASH_MB,

                "clear_hash_between_metric_groups":
                    True,


                # ---------------------------------------------
                # Good Moves / GMR
                # ---------------------------------------------

                "good_move_max_loss_cp":
                    self.GOOD_MOVE_MAX_LOSS_CP,

                "good_move_depth":
                    self.GOOD_MOVE_DEPTH,

                "good_move_analysis":
                    "MultiPV",

                "good_move_reference":
                    "best_move_evaluation",


                # ---------------------------------------------
                # DTS
                # ---------------------------------------------

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


                # ---------------------------------------------
                # Evaluation DTS
                # ---------------------------------------------

                "eval_dts_max_change_cp":
                    self.EVAL_DTS_MAX_CHANGE_CP,


                # ---------------------------------------------
                # Complexity
                # ---------------------------------------------

                "complexity_formula":
                    (
                        "100 * (dts_weight * normalized_dts "
                        "+ gmr_weight * (1 - gmr))"
                    ),

                "complexity_dts_normalization":
                    (
                        "(coded_dts - 6) / (22 - 6)"
                    ),

                "complexity_non_stabilized_dts":
                    (
                        "DTS > 20 is coded as 22"
                    ),

                "complexity_dts_variants": [

                    "best_move_dts",
                    "eval_dts"

                ],

                "complexity_weightings":
                    self.COMPLEXITY_WEIGHTINGS
            },


            # =================================================
            # POSITION METRICS
            # =================================================

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
    # INPUT
    # =========================================================

    test_dataset_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated.json"
    )


    # =========================================================
    # OUTPUT
    # =========================================================

    metric_output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "metric_results.json"
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