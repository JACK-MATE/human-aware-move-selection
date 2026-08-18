from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import chess.engine


from metrics.stockfish_utils import (
    analyse_depth_series,
    analyse_legal_moves_multipv,
    board_from_position_key
)

from metrics.number_of_good_moves import (
    calculate_good_move_data
)

from metrics.good_move_ratio import (
    calculate_good_move_ratio
)

from metrics.best_move_dts import (
    calculate_best_move_dts
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
    # BEST-MOVE DTS PARAMETERS
    # =========================================================
    #
    # Candidate DTBMS:
    #
    #     6, 8, ..., 20
    #
    # Actual Stockfish search:
    #
    #     6, 8, ..., 24
    #
    # 22 and 24 only confirm late stabilization.
    #

    DTS_MIN_DEPTH = 6

    DTS_CANDIDATE_MAX_DEPTH = 20

    DTS_SEARCH_MAX_DEPTH = 24

    DTS_STEP = 2

    DTS_STABLE_STEPS = 3


    # =========================================================
    # STOCKFISH
    # =========================================================

    STOCKFISH_THREADS = 1

    STOCKFISH_HASH_MB = 32


    # =========================================================
    # TEST LIMIT
    # =========================================================
    #
    # First test:
    #
    #     10
    #
    # Complete dataset:
    #
    #     None
    #

    MAX_POSITIONS = None


    # =========================================================
    # CHECKPOINT
    # =========================================================

    CHECKPOINT_EVERY = 10

    RESUME = True


    # =========================================================
    # MAIN
    # =========================================================

    def run(
        self,
        dataset_path: Path,
        output_path: Path,
        stockfish_directory: Path
    ) -> None:


        # =====================================================
        # SAFETY:
        # INPUT MUST NEVER BE OUTPUT
        # =====================================================

        if (
            dataset_path.resolve()
            == output_path.resolve()
        ):

            raise ValueError(
                "Input and output path are identical. "
                "The test dataset must never be overwritten."
            )


        # =====================================================
        # LOAD DATASET
        # =====================================================

        print(
            "Loading dataset..."
        )


        dataset = (
            self.load_dataset(
                dataset_path
            )
        )


        positions = list(
            dataset.keys()
        )


        if self.MAX_POSITIONS is not None:

            positions = (
                positions[
                    :self.MAX_POSITIONS
                ]
            )


        print(
            f"Dataset positions: "
            f"{len(dataset):,}"
        )

        print(
            f"Positions selected: "
            f"{len(positions):,}"
        )


        # =====================================================
        # PARAMETERS
        # =====================================================

        parameters = (
            self.get_metric_parameters()
        )


        # =====================================================
        # RESUME OLD OUTPUT IF AVAILABLE
        # =====================================================

        results = (
            self.load_checkpoint(
                output_path=
                    output_path,
                parameters=
                    parameters
            )
        )


        print(
            f"Already completed: "
            f"{len(results):,}"
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


        try:

            self.configure_engine(
                engine
            )


            total_start = (
                time.perf_counter()
            )


            # =================================================
            # POSITIONS
            # =================================================

            for (
                index,
                fen
            ) in enumerate(
                positions,
                start=1
            ):


                # ---------------------------------------------
                # Already calculated
                # ---------------------------------------------

                if fen in results:

                    continue


                position_start = (
                    time.perf_counter()
                )


                # ---------------------------------------------
                # CALCULATE
                # ---------------------------------------------

                metrics = (
                    self.calculate_metrics_for_fen(
                        engine=engine,
                        fen=fen
                    )
                )


                results[
                    fen
                ] = metrics


                # ---------------------------------------------
                # OUTPUT
                # ---------------------------------------------

                runtime = (
                    time.perf_counter()
                    - position_start
                )


                print()

                print(
                    f"{index}/{len(positions)}"
                )

                print(
                    fen
                )

                print(
                    f"Runtime: "
                    f"{runtime:.2f} s"
                )

                print(
                    f"Legal moves: "
                    f"{metrics['legal_moves']}"
                )

                print(
                    f"Good moves: "
                    f"{metrics['number_of_good_moves']}"
                )

                print(
                    f"GMR: "
                    f"{metrics['good_move_ratio']:.4f}"
                )


                if metrics[
                    "best_move_dts_stabilized"
                ]:

                    print(
                        f"DTBMS: "
                        f"{metrics['best_move_dts']}"
                    )

                else:

                    print(
                        f"DTBMS: "
                        f">{self.DTS_CANDIDATE_MAX_DEPTH}"
                    )


                # ---------------------------------------------
                # CHECKPOINT
                # ---------------------------------------------

                if (
                    len(results)
                    % self.CHECKPOINT_EVERY
                    == 0
                ):

                    self.save_results(
                        output_path=
                            output_path,
                        parameters=
                            parameters,
                        results=
                            results
                    )


            # =================================================
            # FINAL SAVE
            # =================================================

            self.save_results(
                output_path=
                    output_path,
                parameters=
                    parameters,
                results=
                    results
            )


            total_runtime = (
                time.perf_counter()
                - total_start
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
                f"Positions calculated: "
                f"{len(results):,}"
            )

            print(
                f"Runtime: "
                f"{total_runtime / 60:.2f} min"
            )


        finally:

            engine.quit()


    # =========================================================
    # CALCULATE ONE POSITION
    # =========================================================

    def calculate_metrics_for_fen(
        self,
        engine,
        fen: str
    ) -> Dict[str, Any]:


        board = (
            board_from_position_key(
                fen
            )
        )


        legal_moves = (
            board.legal_moves.count()
        )


        if legal_moves == 0:

            raise ValueError(
                "Position contains no legal moves:\n"
                + fen
            )


        # =====================================================
        # 1. ALL LEGAL MOVES
        # =====================================================
        #
        # ONE MultiPV search provides:
        #
        #     Number of Good Moves
        #     GMR
        #     evaluation of every legal move
        #     loss of every legal move
        #     good / not good
        #

        self.clear_hash(
            engine
        )


        move_evaluations = (
            analyse_legal_moves_multipv(
                engine=engine,
                board=board,
                depth=
                    self.GOOD_MOVE_DEPTH
            )
        )


        (
            number_of_good_moves,
            move_data
        ) = (
            calculate_good_move_data(
                move_evaluations=
                    move_evaluations,
                max_eval_loss_cp=
                    self.GOOD_MOVE_MAX_LOSS_CP
            )
        )


        good_move_ratio = (
            calculate_good_move_ratio(
                number_of_good_moves=
                    number_of_good_moves,
                legal_moves=
                    legal_moves
            )
        )


        # =====================================================
        # 2. BEST-MOVE DTS + STOCKFISH CONTROL VALUE
        # =====================================================
        #
        # ONE continuous single-PV search.
        #

        self.clear_hash(
            engine
        )


        depth_data = (
            analyse_depth_series(
                engine=engine,
                board=board,
                min_depth=
                    self.DTS_MIN_DEPTH,
                max_depth=
                    self.DTS_SEARCH_MAX_DEPTH,
                step=
                    self.DTS_STEP
            )
        )


        # =====================================================
        # BEST MOVE BY DEPTH
        # =====================================================

        best_move_by_depth = {

            depth:
                depth_data[
                    depth
                ][
                    "best_move"
                ]

            for depth
            in sorted(
                depth_data.keys()
            )
        }


        # =====================================================
        # DTBMS
        # =====================================================

        (
            best_move_dts,
            best_move_dts_stabilized
        ) = (
            calculate_best_move_dts(
                best_move_by_depth=
                    best_move_by_depth,
                candidate_max_depth=
                    self.DTS_CANDIDATE_MAX_DEPTH,
                stable_steps=
                    self.DTS_STABLE_STEPS,
                step=
                    self.DTS_STEP
            )
        )


        # =====================================================
        # STOCKFISH CONTROL VALUE
        # =====================================================
        #
        # Taken from the deepest analysed depth.
        #
        # This is NOT part of the Complexity calculation.
        #

        final_stockfish = (
            depth_data[
                self.DTS_SEARCH_MAX_DEPTH
            ]
        )


        if "wdl" not in final_stockfish:

            raise RuntimeError(
                "Stockfish did not return WDL at final depth."
            )


        # =====================================================
        # RESULT
        # =====================================================

        return {

            "legal_moves":
                legal_moves,

            "number_of_good_moves":
                number_of_good_moves,

            "good_move_ratio":
                good_move_ratio,

            "best_move_dts":
                best_move_dts,

            "best_move_dts_stabilized":
                best_move_dts_stabilized,

            "best_move_by_depth": {

                str(depth):
                    move

                for (
                    depth,
                    move
                ) in best_move_by_depth.items()
            },


            # =================================================
            # STOCKFISH CONTROL VALUES
            # =================================================

            "stockfish": {

                "pov":
                    "side_to_move",

                "best_move":
                    final_stockfish[
                        "best_move"
                    ],

                "evaluation_cp":
                    final_stockfish[
                        "evaluation_cp"
                    ],

                "wdl":
                    final_stockfish[
                        "wdl"
                    ],

                "moves":
                    move_data
            }
        }


    # =========================================================
    # ENGINE CONFIGURATION
    # =========================================================

    def configure_engine(
        self,
        engine
    ) -> None:


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


        # Required for the WDL control value.
        if "UCI_ShowWDL" not in engine.options:

            raise RuntimeError(
                "Stockfish does not expose UCI_ShowWDL."
            )


        engine.configure({

            "UCI_ShowWDL":
                True
        })


    # =========================================================
    # CLEAR HASH
    # =========================================================

    def clear_hash(
        self,
        engine
    ) -> None:


        if "Clear Hash" in engine.options:

            engine.configure({

                "Clear Hash":
                    None
            })


    # =========================================================
    # METRIC PARAMETERS
    # =========================================================

    def get_metric_parameters(
        self
    ) -> Dict[str, Any]:


        return {

            "good_move_max_loss_cp":
                self.GOOD_MOVE_MAX_LOSS_CP,

            "good_move_depth":
                self.GOOD_MOVE_DEPTH,

            "dtbms_min_depth":
                self.DTS_MIN_DEPTH,

            "dtbms_candidate_max_depth":
                self.DTS_CANDIDATE_MAX_DEPTH,

            "dtbms_search_max_depth":
                self.DTS_SEARCH_MAX_DEPTH,

            "dtbms_step":
                self.DTS_STEP,

            "dtbms_stable_steps":
                self.DTS_STABLE_STEPS,

            "stockfish_threads":
                self.STOCKFISH_THREADS,

            "stockfish_hash_mb":
                self.STOCKFISH_HASH_MB,

            "stockfish_wdl_pov":
                "side_to_move"
        }


    # =========================================================
    # LOAD DATASET
    # =========================================================

    def load_dataset(
        self,
        path: Path
    ) -> Dict[str, Any]:
        """
        IMPORTANT:

        The source dataset is opened ONLY in read mode.

        This function cannot write to or modify the dataset.
        """


        if not path.exists():

            raise FileNotFoundError(
                "Dataset not found:\n"
                + str(path)
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
                "Expected dataset to contain a JSON object."
            )


        return data


    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================

    def load_checkpoint(
        self,
        output_path: Path,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:


        if (
            not self.RESUME
            or not output_path.exists()
        ):

            return {}


        with open(

            output_path,

            "r",

            encoding="utf-8"

        ) as file:

            old_output = (
                json.load(
                    file
                )
            )


        # Do not accidentally combine results produced with
        # different metric definitions.
        if (
            old_output.get(
                "metric_parameters"
            )
            != parameters
        ):

            raise ValueError(
                "Existing position_metrics.json was created "
                "with different metric parameters.\n"
                "Rename or delete the old output before "
                "starting this version."
            )


        return (
            old_output.get(
                "positions",
                {}
            )
        )


    # =========================================================
    # SAVE OUTPUT
    # =========================================================

    def save_results(
        self,
        output_path: Path,
        parameters: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """
        Writes ONLY to position_metrics.json.

        The source test dataset is never touched.
        """


        output_path.parent.mkdir(

            parents=True,

            exist_ok=True
        )


        output = {

            "metric_parameters":
                parameters,

            "positions":
                results
        }


        # Write temporary file first so an interruption during
        # json.dump cannot destroy the previous checkpoint.
        temporary_path = (
            output_path.with_suffix(
                output_path.suffix
                + ".tmp"
            )
        )


        with open(

            temporary_path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                output,

                file,

                indent=2,

                ensure_ascii=False
            )


        temporary_path.replace(
            output_path
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
                "Stockfish directory not found:\n"
                + str(directory)
            )


        candidates = []


        for path in directory.iterdir():

            if not path.is_file():

                continue


            if not path.name.lower().startswith(
                "stockfish"
            ):

                continue


            if (
                path.suffix.lower() == ".exe"
                or os.access(
                    str(path),
                    os.X_OK
                )
            ):

                candidates.append(
                    path
                )


        if len(candidates) == 0:

            raise FileNotFoundError(
                "No Stockfish executable found in:\n"
                + str(directory)
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


        return candidates[0]


# =============================================================
# PROGRAM START
# =============================================================

if __name__ == "__main__":


    # =========================================================
    # INPUT
    # =========================================================
    #
    # READ ONLY.
    #

    dataset_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated_top500.json"
    )


    # =========================================================
    # OUTPUT
    # =========================================================
    #
    # Completely separate NEW JSON.
    #

    output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "position_metrics.json"
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
            dataset_file,

        output_path=
            output_file,

        stockfish_directory=
            stockfish_directory
    )