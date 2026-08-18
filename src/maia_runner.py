from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess.engine

from maia.maia_adapter import (
    Maia3Adapter
)

from maia.maia_simulation import (
    simulate_observed_move
)


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
    "2200-2399": 2300
}


# =============================================================
# DATA SELECTION
# =============================================================

MIN_BUCKET_OBSERVATIONS = 10


# For the actually observed ROOT moves:
# simulate only the most frequent moves until >=90% of real
# observations in FEN × bucket are covered.
OBSERVED_MOVE_COVERAGE = 0.90


# =============================================================
# MAIA TREE
# =============================================================

# -------------------------------------------------------------
# Player whose observed move is being evaluated
# -------------------------------------------------------------

OWN_MIN_MOVE_PROBABILITY = 0.20


# -------------------------------------------------------------
# Opponent
# -------------------------------------------------------------

OPPONENT_TOP_N = 3

OPPONENT_MIN_MOVE_PROBABILITY = 0.05


# -------------------------------------------------------------
# Tree depth
# -------------------------------------------------------------

# 3 complete moves AFTER the observed move.
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

# Keep at 1 for the next runtime test.
MAX_SIMULATIONS = None


# =============================================================
# CHECKPOINT
# =============================================================

CHECKPOINT_EVERY = 1

RESUME = True


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


        if (
            dataset_path.resolve()
            == output_path.resolve()
        ):

            raise ValueError(
                "Input and output path must be different."
            )


        dataset = self.load_dataset(
            dataset_path
        )


        parameters = self.get_parameters()


        results = self.load_checkpoint(
            output_path,
            parameters
        )


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
            f"{len(tasks) - len(pending_tasks):,}"
        )


        if MAX_SIMULATIONS is not None:

            pending_tasks = pending_tasks[
                :MAX_SIMULATIONS
            ]


        print(
            f"Simulations this run: "
            f"{len(pending_tasks):,}"
        )


        if not pending_tasks:

            return


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
        # STOCKFISH
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


            for (
                index,
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


                start = time.perf_counter()


                # =============================================
                # SIMULATION
                # =============================================

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
                    f"{index}/"
                    f"{len(pending_tasks)}"
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


                if (
                    index
                    % CHECKPOINT_EVERY
                    == 0
                ):

                    self.save_results(
                        output_path,
                        parameters,
                        results
                    )


            self.save_results(
                output_path,
                parameters,
                results
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
                ) = (
                    select_observed_moves(
                        move_counts=
                            move_counts,

                        required_coverage=
                            OBSERVED_MOVE_COVERAGE
                    )
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
    # STORE
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
    # RESUME
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
    # JSON
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
            old_output.get("parameters")
            != parameters
        ):

            raise ValueError(
                "Existing maia_results.json uses different "
                "parameters.\n"
                "Delete or rename it before starting."
            )


        return old_output.get(
            "positions",
            {}
        )


    def save_results(
        self,
        output_path,
        parameters,
        results
    ):


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        temporary_path = output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )


        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "parameters":
                        parameters,

                    "positions":
                        results
                },
                file,
                indent=2,
                ensure_ascii=False
            )


        temporary_path.replace(
            output_path
        )


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


        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.lower().startswith(
                "stockfish"
            )
            and (
                path.suffix.lower() == ".exe"
                or os.access(
                    str(path),
                    os.X_OK
                )
            )
        ]


        if len(candidates) != 1:

            raise RuntimeError(
                "Expected exactly one Stockfish executable in:\n"
                + str(directory)
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