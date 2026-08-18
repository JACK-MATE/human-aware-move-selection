from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


# =============================================================
# PROJECT PATHS
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


SRC_DIRECTORY = (
    PROJECT_ROOT
    / "src"
)


METRIC_RUNNER = (
    SRC_DIRECTORY
    / "metric_runner.py"
)


MAIA_RUNNER = (
    SRC_DIRECTORY
    / "maia_runner.py"
)


# =============================================================
# RUN ONE SCRIPT
# =============================================================

def run_script(
    script_path: Path,
    name: str
) -> None:
    """
    Runs one Python script with the SAME Python interpreter
    that started this file.

    If the script fails, the complete pipeline stops.
    """

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print()

    start_time = time.perf_counter()


    result = subprocess.run(
        [
            sys.executable,
            str(script_path)
        ]
    )


    runtime = (
        time.perf_counter()
        - start_time
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"{name} failed with exit code "
            f"{result.returncode}."
        )


    print()
    print(
        f"{name} finished successfully."
    )

    print(
        f"Runtime: "
        f"{runtime / 60:.2f} minutes"
    )


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    total_start = time.perf_counter()


    # ---------------------------------------------------------
    # 1. Structural + Stockfish metrics
    # ---------------------------------------------------------

    run_script(
        script_path=
            METRIC_RUNNER,

        name=
            "PHASE 1: Metric Runner"
    )


    # ---------------------------------------------------------
    # 2. Maia practical move evaluation
    # ---------------------------------------------------------

    run_script(
        script_path=
            MAIA_RUNNER,

        name=
            "PHASE 2: Maia Runner"
    )


    total_runtime = (
        time.perf_counter()
        - total_start
    )


    print()
    print("=" * 70)
    print("ALL ANALYSIS FINISHED")
    print("=" * 70)

    print(
        f"Total runtime: "
        f"{total_runtime / 3600:.2f} hours"
    )


if __name__ == "__main__":

    main()