from __future__ import annotations

from create_holdout_dataset import (
    main as create_holdout
)

from run_holdout_metrics import (
    main as run_metrics
)

from validate_complexity import (
    main as validate_complexity
)


# =============================================================
# PIPELINE SETTINGS
# =============================================================
#
# If a step has already finished successfully, you can set it
# to False and start the pipeline again.
#
# The metric calculation itself also uses the existing
# MetricRunner checkpoint/resume functionality.
# =============================================================

RUN_HOLDOUT_SELECTION = True

RUN_METRIC_CALCULATION = True

RUN_COMPLEXITY_VALIDATION = True


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    print()

    print(
        "#" * 72
    )

    print(
        "STRUCTURAL COMPLEXITY VALIDATION PIPELINE"
    )

    print(
        "#" * 72
    )


    # =========================================================
    # STEP 1:
    # SELECT RANKS 501-600
    # =========================================================

    if RUN_HOLDOUT_SELECTION:

        create_holdout()


    # =========================================================
    # STEP 2:
    # CALCULATE GMR / N / DTBMS
    # =========================================================

    if RUN_METRIC_CALCULATION:

        run_metrics()


    # =========================================================
    # STEP 3:
    # VALIDATE DEVELOPMENT RESULTS
    # =========================================================

    if RUN_COMPLEXITY_VALIDATION:

        validate_complexity()


    # =========================================================
    # DONE
    # =========================================================

    print()

    print(
        "#" * 72
    )

    print(
        "VALIDATION PIPELINE FINISHED"
    )

    print(
        "#" * 72
    )


if __name__ == "__main__":

    main()