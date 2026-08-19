from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess.engine

from maia.maia_adapter import Maia3Adapter
from maia.maia_simulation import simulate_observed_move


# =============================================================
# PROJECT ROOT
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# =============================================================
# RATING BUCKETS
# =============================================================

RATING_BUCKETS = {
    "1400-1599": 1500,
    "1600-1799": 1700,
    "1800-1999": 1900,
    "2000-2199": 2100,
    "2200-2399": 2300,
}


# =============================================================
# DATA SELECTION
# =============================================================

MIN_BUCKET_OBSERVATIONS = 10

# Only the most frequent actually observed moves are simulated
# until >= 90% of the observations in FEN x bucket are covered.
OBSERVED_MOVE_COVERAGE = 0.90


# =============================================================
# MAIA TREE
# =============================================================

# Own continuations:
# keep moves with Maia probability > 20%.
# If none exists, keep Maia's most likely move.
OWN_MIN_MOVE_PROBABILITY = 0.20


# Opponent continuations:
# maximum Top 3, each with at least 5% probability.
# If none exists, keep Maia's most likely move.
OPPONENT_TOP_N = 3
OPPONENT_MIN_MOVE_PROBABILITY = 0.05


# Three complete moves AFTER the observed move.
MAIA_PLIES_AFTER_OBSERVED_MOVE = 6


# =============================================================
# STOCKFISH LEAVES
# =============================================================

STOCKFISH_LEAF_DEPTH = 12
STOCKFISH_THREADS = 1
STOCKFISH_HASH_MB = 32


# =============================================================
# MAIA MODEL
# =============================================================

MAIA_MODEL = "maia3-5m"


# =============================================================
# LEAF CACHE
# =============================================================

USE_LEAF_CACHE = True


# =============================================================
# TEST LIMIT
# =============================================================

# None = complete dataset.
MAX_SIMULATIONS = None


# =============================================================
# CHECKPOINT / SAVE SETTINGS
# =============================================================

# Saving after every single simulation caused many thousands of
# file replacements and triggered a transient Windows file-lock.
#
# Saving every 10 simulations is much less aggressive.
# In the worst case only the last <=9 simulations have to be
# repeated after an unexpected crash.
CHECKPOINT_EVERY = 10

RESUME = True


# Windows can temporarily lock a JSON file, e.g. because of
# indexing, antivirus scanning or another short-lived file access.
#
# Instead of immediately aborting, retry the atomic replacement.
SAVE_RETRIES = 20
SAVE_RETRY_DELAY_SECONDS = 0.5


# =============================================================
# BUCKET HELPERS
# =============================================================

def normalize_bucket_name(
    name: str
) -> str:

    return (
        name
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )


def get_bucket_data(
    root_data: Dict[str, Any],
    wanted_bucket: str
) -> Optional[Dict[str, Any]]:

    rating_buckets = root_data.get(
        "rating_buckets",
        {}
    )

    wanted = normalize_bucket_name(
        wanted_bucket
    )

    for (
        bucket_name,
        bucket_data
    ) in rating_buckets.items():

        if (
            normalize_bucket_name(
                bucket_name
            )
            == wanted
        ):
            return bucket_data

    return None


# =============================================================
# OBSERVED MOVES
# =============================================================

def get_move_counts(
    bucket_data: Dict[str, Any]
) -> Dict[str, int]:

    counts = {}

    for (
        move_uci,
        move_data
    ) in bucket_data.get(
        "moves",
        {}
    ).items():

        count = int(
            move_data.get(
                "count",
                0
            )
        )

        if count > 0:
            counts[
                move_uci
            ] = count

    return counts


def select_observed_moves(
    move_counts: Dict[str, int],
    required_coverage: float
) -> Tuple[
    List[Tuple[str, int]],
    int
]:

    total_observations = sum(
        move_counts.values()
    )

    if total_observations == 0:
        return [], 0

    ordered_moves = sorted(
        move_counts.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    selected = []
    selected_observations = 0

    for (
        move_uci,
        count
    ) in ordered_moves:

        selected.append(
            (
                move_uci,
                count
            )
        )

        selected_observations += count

        if (
            selected_observations
            / total_observations
            >= required_coverage
        ):
            break

    return (
        selected,
        selected_observations
    )


# =============================================================
# RUNNER
# =============================================================

class MaiaRunner:

    def run(
        self,
        dataset_path: Path,
        output_path: Path,
        stockfish_directory: Path
    ) -> None:

        # =====================================================
        # SAFETY
        # =====================================================

        if (
            dataset_path.resolve()
            == output_path.resolve()
        ):
            raise ValueError(
                "Input and output path must be different."
            )

        # =====================================================
        # LOAD DATA
        # =====================================================

        dataset = self.load_dataset(
            dataset_path
        )

        parameters = self.get_parameters()

        results = self.load_checkpoint(
            output_path,
            parameters
        )

        # =====================================================
        # BUILD COMPLETE WORKLOAD
        # =====================================================

        tasks = self.build_tasks(
            dataset
        )

        pending_tasks = [
            task
            for task in tasks
            if not self.task_is_done(
                results,
                task
            )
        ]

        completed_before_run = (
            len(tasks)
            - len(pending_tasks)
        )

        print(
            f"Top dataset FENs: "
            f"{len(dataset):,}"
        )

        print(
            f"Planned Maia simulations: "
            f"{len(tasks):,}"
        )

        print(
            f"Already completed: "
            f"{completed_before_run:,}"
        )

        # =====================================================
        # OPTIONAL TEST LIMIT
        # =====================================================

        if MAX_SIMULATIONS is not None:

            pending_tasks = pending_tasks[
                :MAX_SIMULATIONS
            ]

        print(
            f"Simulations this run: "
            f"{len(pending_tasks):,}"
        )

        if not pending_tasks:
            print(
                "Nothing left to calculate."
            )
            return

        # =====================================================
        # STOCKFISH PATH
        # =====================================================

        stockfish_path = self.find_stockfish(
            stockfish_directory
        )

        # =====================================================
        # LOAD MAIA ONCE
        # =====================================================

        print()
        print(
            "Loading Maia-3..."
        )

        maia = Maia3Adapter(
            model_alias=MAIA_MODEL
        )

        print(
            f"Maia model: {MAIA_MODEL}"
        )

        print(
            f"Maia device: {maia.device}"
        )

        # =====================================================
        # START STOCKFISH ONCE
        # =====================================================

        stockfish = (
            chess.engine.SimpleEngine.popen_uci(
                str(
                    stockfish_path
                )
            )
        )

        self.configure_stockfish(
            stockfish
        )

        try:

            current_root_fen = None
            leaf_cache = None

            # =================================================
            # RUN SIMULATIONS
            # =================================================

            for (
                run_index,
                task
            ) in enumerate(
                pending_tasks,
                start=1
            ):

                (
                    fen,
                    bucket,
                    elo,
                    bucket_observations,
                    selected_observations,
                    move_uci,
                    move_observations
                ) = task

                # Absolute progress across the entire dataset.
                absolute_index = (
                    completed_before_run
                    + run_index
                )

                # =============================================
                # CACHE PER ROOT FEN
                # =============================================

                if fen != current_root_fen:

                    current_root_fen = fen

                    leaf_cache = (
                        {}
                        if USE_LEAF_CACHE
                        else None
                    )

                # =============================================
                # SIMULATION
                # =============================================

                start = time.perf_counter()

                simulation = simulate_observed_move(

                    maia=maia,

                    stockfish_engine=
                        stockfish,

                    root_fen=
                        fen,

                    observed_move_uci=
                        move_uci,

                    elo=
                        elo,

                    plies_after_move=
                        MAIA_PLIES_AFTER_OBSERVED_MOVE,

                    own_min_probability=
                        OWN_MIN_MOVE_PROBABILITY,

                    opponent_top_n=
                        OPPONENT_TOP_N,

                    opponent_min_probability=
                        OPPONENT_MIN_MOVE_PROBABILITY,

                    leaf_stockfish_depth=
                        STOCKFISH_LEAF_DEPTH,

                    leaf_cache=
                        leaf_cache
                )

                runtime = (
                    time.perf_counter()
                    - start
                )

                # =============================================
                # STORE RESULT IN MEMORY
                # =============================================

                self.store_result(

                    results=
                        results,

                    fen=
                        fen,

                    bucket=
                        bucket,

                    elo=
                        elo,

                    bucket_observations=
                        bucket_observations,

                    selected_observations=
                        selected_observations,

                    move_uci=
                        move_uci,

                    move_observations=
                        move_observations,

                    simulation=
                        simulation,

                    runtime_seconds=
                        runtime
                )

                # =============================================
                # OUTPUT
                # =============================================

                print()

                print(
                    f"{absolute_index}/"
                    f"{len(tasks)}"
                )

                print(
                    f"{bucket} | "
                    f"Maia Elo {elo}"
                )

                print(
                    fen
                )

                print(
                    f"Observed move: "
                    f"{move_uci} "
                    f"({move_observations}x)"
                )

                print(
                    f"Runtime: "
                    f"{runtime:.2f} s"
                )

                print(
                    f"Leaves: "
                    f"{simulation['leaf_count']}"
                )

                print(
                    f"Maia nodes: "
                    f"{simulation['maia_nodes']}"
                )

                print(
                    f"Own Maia nodes: "
                    f"{simulation['own_maia_nodes']}"
                )

                print(
                    f"Opponent Maia nodes: "
                    f"{simulation['opponent_maia_nodes']}"
                )

                print(
                    f"Retained probability mass: "
                    f"{simulation['retained_probability_mass']:.4f}"
                )

                print(
                    f"Leaf cache hits: "
                    f"{simulation['leaf_cache_hits']}"
                )

                print(
                    f"Leaf cache hit rate: "
                    f"{simulation['leaf_cache_hit_rate']:.2%}"
                )

                print(
                    f"Stockfish leaf evaluations: "
                    f"{simulation['stockfish_leaf_evaluations']}"
                )

                print(
                    f"Average evaluation: "
                    f"{simulation['average_evaluation_cp']:.2f} cp"
                )

                print(
                    "Average WDL: "
                    f"{simulation['average_wdl']['win']:.1f} / "
                    f"{simulation['average_wdl']['draw']:.1f} / "
                    f"{simulation['average_wdl']['loss']:.1f}"
                )

                # =============================================
                # CHECKPOINT
                # =============================================

                if (
                    run_index
                    % CHECKPOINT_EVERY
                    == 0
                ):

                    print(
                        f"Saving checkpoint at "
                        f"{absolute_index}/{len(tasks)}..."
                    )

                    self.save_results(
                        output_path,
                        parameters,
                        results
                    )

            # =================================================
            # FINAL SAVE
            # =================================================

            print()
            print(
                "Saving final Maia results..."
            )

            self.save_results(
                output_path,
                parameters,
                results
            )

            print(
                "Maia analysis finished successfully."
            )

        finally:

            stockfish.quit()


    # =========================================================
    # BUILD TASKS
    # =========================================================

    def build_tasks(
        self,
        dataset: Dict[str, Any]
    ) -> List[Tuple]:

        tasks = []

        for (
            fen,
            root_data
        ) in dataset.items():

            for (
                bucket,
                elo
            ) in RATING_BUCKETS.items():

                bucket_data = get_bucket_data(
                    root_data,
                    bucket
                )

                if bucket_data is None:
                    continue

                move_counts = get_move_counts(
                    bucket_data
                )

                bucket_observations = sum(
                    move_counts.values()
                )

                if (
                    bucket_observations
                    < MIN_BUCKET_OBSERVATIONS
                ):
                    continue

                (
                    selected_moves,
                    selected_observations
                ) = select_observed_moves(

                    move_counts=
                        move_counts,

                    required_coverage=
                        OBSERVED_MOVE_COVERAGE
                )

                for (
                    move_uci,
                    move_observations
                ) in selected_moves:

                    tasks.append(
                        (
                            fen,
                            bucket,
                            elo,
                            bucket_observations,
                            selected_observations,
                            move_uci,
                            move_observations
                        )
                    )

        return tasks


    # =========================================================
    # STORE RESULT
    # =========================================================

    def store_result(
        self,
        results,
        fen,
        bucket,
        elo,
        bucket_observations,
        selected_observations,
        move_uci,
        move_observations,
        simulation,
        runtime_seconds
    ):

        position_result = results.setdefault(
            fen,
            {}
        )

        bucket_result = (
            position_result.setdefault(
                bucket,
                {
                    "maia_elo":
                        elo,

                    "bucket_observations":
                        bucket_observations,

                    "selected_observations":
                        selected_observations,

                    "selected_observation_coverage":
                        (
                            selected_observations
                            / bucket_observations
                        ),

                    "moves":
                        {}
                }
            )
        )

        bucket_result[
            "moves"
        ][
            move_uci
        ] = {

            "observations":
                move_observations,

            "observed_share":
                (
                    move_observations
                    / bucket_observations
                ),

            "runtime_seconds":
                runtime_seconds,

            "simulation":
                simulation
        }


    # =========================================================
    # RESUME CHECK
    # =========================================================

    def task_is_done(
        self,
        results,
        task
    ) -> bool:

        (
            fen,
            bucket,
            _,
            _,
            _,
            move_uci,
            _
        ) = task

        return (
            move_uci
            in results
            .get(fen, {})
            .get(bucket, {})
            .get("moves", {})
        )


    # =========================================================
    # PARAMETERS
    # =========================================================

    def get_parameters(
        self
    ) -> Dict[str, Any]:

        # IMPORTANT:
        #
        # Do NOT include checkpoint frequency or retry settings
        # here. They do not alter the scientific calculation and
        # therefore should not invalidate an existing checkpoint.

        return {

            "schema_version":
                4,

            "rating_buckets":
                RATING_BUCKETS,

            "min_bucket_observations":
                MIN_BUCKET_OBSERVATIONS,

            "observed_move_coverage":
                OBSERVED_MOVE_COVERAGE,

            "own_min_move_probability":
                OWN_MIN_MOVE_PROBABILITY,

            "own_empty_threshold_fallback":
                "single_most_probable_move",

            "opponent_top_n":
                OPPONENT_TOP_N,

            "opponent_min_move_probability":
                OPPONENT_MIN_MOVE_PROBABILITY,

            "opponent_empty_threshold_fallback":
                "single_most_probable_move",

            "maia_plies_after_observed_move":
                MAIA_PLIES_AFTER_OBSERVED_MOVE,

            "stockfish_leaf_depth":
                STOCKFISH_LEAF_DEPTH,

            "maia_model":
                MAIA_MODEL,

            "stockfish_threads":
                STOCKFISH_THREADS,

            "stockfish_hash_mb":
                STOCKFISH_HASH_MB,

            "leaf_cache":
                USE_LEAF_CACHE,

            "leaf_weighting":
                "raw_path_probability_with_global_normalization",

            "wdl_pov":
                "player_who_made_observed_move"
        }


    # =========================================================
    # LOAD DATASET
    # =========================================================

    def load_dataset(
        self,
        path
    ):

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

            return json.load(
                file
            )


    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================

    def load_checkpoint(
        self,
        output_path,
        parameters
    ):

        if (
            not RESUME
            or not output_path.exists()
        ):
            return {}

        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as file:

            old_output = json.load(
                file
            )

        if (
            old_output.get(
                "parameters"
            )
            != parameters
        ):

            raise ValueError(
                "Existing maia_results.json uses different "
                "parameters.\n"
                "Do NOT delete it automatically. Check the "
                "parameters first."
            )

        return old_output.get(
            "positions",
            {}
        )


    # =========================================================
    # ROBUST ATOMIC SAVE
    # =========================================================

    def save_results(
        self,
        output_path,
        parameters,
        results
    ) -> None:
        """
        Writes the complete current checkpoint to a temporary
        file and atomically replaces maia_results.json.

        Windows may temporarily lock the destination file.
        In that case the replacement is retried automatically.

        The old valid maia_results.json remains untouched until
        the new JSON has been written completely.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output = {
            "parameters":
                parameters,

            "positions":
                results
        }

        # Unique temporary path for this Python process.
        temporary_path = output_path.with_name(
            output_path.name
            + f".tmp.{os.getpid()}"
        )

        # -----------------------------------------------------
        # WRITE COMPLETE TEMP FILE
        # -----------------------------------------------------

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

            # Force Python's buffer to disk before replacement.
            file.flush()
            os.fsync(
                file.fileno()
            )

        # -----------------------------------------------------
        # ATOMIC REPLACE WITH WINDOWS RETRIES
        # -----------------------------------------------------

        last_error = None

        for attempt in range(
            1,
            SAVE_RETRIES + 1
        ):

            try:

                os.replace(
                    temporary_path,
                    output_path
                )

                return

            except PermissionError as error:

                last_error = error

            except OSError as error:

                # Windows:
                # 5  = Access denied
                # 32 = Sharing violation
                if getattr(
                    error,
                    "winerror",
                    None
                ) not in (
                    5,
                    32
                ):
                    raise

                last_error = error

            if attempt < SAVE_RETRIES:

                print(
                    f"Checkpoint file temporarily locked. "
                    f"Retry {attempt}/{SAVE_RETRIES}..."
                )

                time.sleep(
                    SAVE_RETRY_DELAY_SECONDS
                )

        # -----------------------------------------------------
        # EMERGENCY CHECKPOINT
        # -----------------------------------------------------

        # If Windows blocked the normal file for the entire retry
        # period, preserve the newest valid JSON under a separate
        # filename before stopping.

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        emergency_path = output_path.with_name(
            output_path.stem
            + "_emergency_"
            + timestamp
            + output_path.suffix
        )

        try:

            os.replace(
                temporary_path,
                emergency_path
            )

            raise RuntimeError(
                "Could not replace maia_results.json after "
                f"{SAVE_RETRIES} attempts.\n"
                "The newest results were preserved here:\n"
                + str(
                    emergency_path
                )
            ) from last_error

        except RuntimeError:
            raise

        except Exception as emergency_error:

            raise RuntimeError(
                "Could not save the normal checkpoint and could "
                "not create the emergency checkpoint.\n"
                "Temporary file may still exist here:\n"
                + str(
                    temporary_path
                )
            ) from emergency_error


    # =========================================================
    # STOCKFISH
    # =========================================================

    def configure_stockfish(
        self,
        engine
    ):

        if "Threads" in engine.options:

            engine.configure(
                {
                    "Threads":
                        STOCKFISH_THREADS
                }
            )

        if "Hash" in engine.options:

            engine.configure(
                {
                    "Hash":
                        STOCKFISH_HASH_MB
                }
            )

        if "UCI_ShowWDL" not in engine.options:

            raise RuntimeError(
                "Stockfish does not expose UCI_ShowWDL."
            )

        engine.configure(
            {
                "UCI_ShowWDL":
                    True
            }
        )


    def find_stockfish(
        self,
        directory
    ):

        if not directory.exists():

            raise FileNotFoundError(
                "Stockfish directory not found:\n"
                + str(
                    directory
                )
            )

        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.lower().startswith(
                "stockfish"
            )
            and (
                path.suffix.lower()
                == ".exe"
                or os.access(
                    str(path),
                    os.X_OK
                )
            )
        ]

        if len(candidates) != 1:

            raise RuntimeError(
                "Expected exactly one Stockfish executable in:\n"
                + str(
                    directory
                )
            )

        return candidates[0]


# =============================================================
# START
# =============================================================

if __name__ == "__main__":

    dataset_file = (
        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated_top500.json"
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "results"
        / "maia_results.json"
    )

    stockfish_directory = (
        PROJECT_ROOT
        / "engines"
        / "stockfish"
    )

    MaiaRunner().run(
        dataset_path=
            dataset_file,

        output_path=
            output_file,

        stockfish_directory=
            stockfish_directory
    )