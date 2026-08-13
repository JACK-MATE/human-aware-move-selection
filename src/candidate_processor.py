from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chess
import chess.engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CandidateProcessor:

    # =========================================================
    # CONFIGURATION
    # =========================================================

    # Number of independent positions retained before
    # applying the Stockfish balance filter.
    FINAL_INDEPENDENT_CANDIDATES = 5_000

    # A position is considered approximately balanced if
    # its Stockfish evaluation is between -1.00 and +1.00.
    MAX_ABS_EVAL_CP = 100

    # Search depth used only for the balance filter.
    STOCKFISH_DEPTH = 15

    # Fixed engine settings improve reproducibility.
    STOCKFISH_THREADS = 1
    STOCKFISH_HASH_MB = 32


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        self.candidates = []

        self.candidates_by_fen = {}

        # Cache of direct candidate children.
        # This avoids reconstructing the same transitions
        # repeatedly during connectivity analysis.
        self.children_cache = {}

        # Should normally stay at 0.
        # If it is larger than 0, we should investigate.
        self.illegal_candidate_moves = 0


    # =========================================================
    # MAIN METHOD
    # =========================================================

    def run(
        self,
        input_path: Path,
        independent_output_path: Path,
        evaluated_output_path: Path,
        balanced_output_path: Path,
        stockfish_directory: Path
    ) -> None:

        start_time = time.perf_counter()

        # =====================================================
        # LOAD RAW CANDIDATES
        # =====================================================

        print("Loading raw candidates...")

        self.candidates = self.load_candidates(
            input_path
        )

        self.candidates_by_fen = {
            candidate["fen"]: candidate
            for candidate in self.candidates
        }

        # Duplicate FENs should not exist because the first
        # builder already aggregated positions by FEN.
        if (
            len(self.candidates_by_fen)
            != len(self.candidates)
        ):
            raise ValueError(
                "Duplicate FENs found in raw candidate file."
            )

        print(
            f"Raw candidates loaded: "
            f"{len(self.candidates):,}"
        )


        # =====================================================
        # STEP 1 + 2:
        # REMOVE CONNECTED POSITIONS AND SELECT TOP 5,000
        # =====================================================

        print()
        print(
            "Removing connected positions and "
            "selecting independent candidates..."
        )

        (
            independent_candidates,
            marked_for_removal
        ) = self.select_independent_candidates()

        print()
        print(
            f"Independent candidates selected: "
            f"{len(independent_candidates):,}"
        )

        print(
            f"Connected positions marked for removal: "
            f"{len(marked_for_removal):,}"
        )

        if self.illegal_candidate_moves > 0:
            print()
            print(
                f"WARNING: {self.illegal_candidate_moves:,} "
                f"stored candidate moves could not be "
                f"reconstructed as legal moves."
            )

        # Save the 5,000 independent candidates BEFORE
        # starting Stockfish.
        #
        # Therefore this work is not lost even if Stockfish
        # later fails or is interrupted.
        self.save_candidates(
            independent_candidates,
            independent_output_path
        )

        print(
            f"Independent candidate file saved:\n"
            f"{independent_output_path}"
        )


        # =====================================================
        # STEP 3:
        # STOCKFISH BALANCE FILTER
        # =====================================================

        print()
        print("Searching for Stockfish executable...")

        stockfish_path = self.find_stockfish_executable(
            stockfish_directory
        )

        print(
            f"Stockfish found:\n"
            f"{stockfish_path}"
        )

        print()
        print(
            "Evaluating independent candidates "
            "with Stockfish..."
        )

        (
            evaluated_candidates,
            balanced_candidates
        ) = self.evaluate_and_filter_positions(
            independent_candidates,
            stockfish_path
        )

        # Store ALL 5,000 evaluations.
        #
        # This means that changing the +/-1.00 threshold later
        # does not require another Stockfish run.
        self.save_candidates(
            evaluated_candidates,
            evaluated_output_path
        )

        # Store only the candidates that pass the balance filter.
        self.save_candidates(
            balanced_candidates,
            balanced_output_path
        )


        # =====================================================
        # FINAL INFORMATION
        # =====================================================

        runtime = (
            time.perf_counter()
            - start_time
        )

        print()
        print("Candidate processing finished")
        print("----------------------------------")

        print(
            "Raw candidates:",
            f"{len(self.candidates):,}"
        )

        print(
            "Independent candidates:",
            f"{len(independent_candidates):,}"
        )

        print(
            "Balanced candidates:",
            f"{len(balanced_candidates):,}"
        )

        print(
            "Runtime in minutes:",
            round(runtime / 60, 2)
        )


    # =========================================================
    # INPUT
    # =========================================================

    def load_candidates(
        self,
        input_path: Path
    ) -> List[Dict[str, Any]]:

        if not input_path.exists():
            raise FileNotFoundError(
                f"Raw candidate file not found:\n"
                f"{input_path}"
            )

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Raw candidate file must contain a JSON list."
            )

        return data


    # =========================================================
    # STEP 1 + 2:
    # CONNECTIVITY FILTER
    # =========================================================

    def select_independent_candidates(
        self
    ) -> Tuple[
        List[Dict[str, Any]],
        Set[str]
    ]:

        selected = []

        selected_fens = set()

        # Positions are NOT deleted directly.
        # They are only marked here.
        marked_for_removal = set()

        # The raw JSON is already ranked by occurrences.
        for raw_rank, candidate in enumerate(
            self.candidates,
            start=1
        ):

            fen = candidate["fen"]

            # This position is connected to a better-ranked
            # candidate and therefore cannot become a root.
            if fen in marked_for_removal:
                continue

            # ---------------------------------------------
            # Accept candidate
            # ---------------------------------------------

            candidate_copy = dict(
                candidate
            )

            candidate_copy[
                "raw_rank"
            ] = raw_rank

            candidate_copy[
                "independent_rank"
            ] = (
                len(selected) + 1
            )

            selected.append(
                candidate_copy
            )

            selected_fens.add(
                fen
            )

            # ---------------------------------------------
            # Find all candidate descendants of this root
            # ---------------------------------------------

            descendants = (
                self.find_candidate_descendants(
                    root_fen=fen,
                    protected_fens=selected_fens
                )
            )

            marked_for_removal.update(
                descendants
            )

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if len(selected) % 500 == 0:

                print(
                    f"Selected: "
                    f"{len(selected):,} | "
                    f"Connected positions marked: "
                    f"{len(marked_for_removal):,}"
                )

            # ---------------------------------------------
            # We only need the first 5,000 independent
            # candidates.
            # ---------------------------------------------

            if (
                len(selected)
                >= self.FINAL_INDEPENDENT_CANDIDATES
            ):
                break

        if (
            len(selected)
            < self.FINAL_INDEPENDENT_CANDIDATES
        ):
            print()
            print(
                f"WARNING: Only {len(selected):,} "
                f"independent candidates could be selected."
            )

        return (
            selected,
            marked_for_removal
        )


    def find_candidate_descendants(
        self,
        root_fen: str,
        protected_fens: Set[str]
    ) -> Set[str]:

        descendants = set()

        # Prevent loops caused by reversible move sequences
        # or transpositions.
        visited = {
            root_fen
        }

        stack = [
            root_fen
        ]

        while stack:

            current_fen = stack.pop()

            children = self.get_candidate_children(
                current_fen
            )

            for child_fen in children:

                if child_fen in visited:
                    continue

                visited.add(
                    child_fen
                )

                # A previously accepted higher-ranked root
                # must never be removed.
                if child_fen in protected_fens:
                    continue

                # Mark child for removal.
                descendants.add(
                    child_fen
                )

                # IMPORTANT:
                #
                # Even though the child is already marked,
                # continue searching from it.
                #
                # Example:
                #
                # Root -> A -> A1 -> A2
                #
                # A, A1 and A2 are all removed if they are
                # themselves raw candidates.
                stack.append(
                    child_fen
                )

        return descendants


    # =========================================================
    # DIRECT CHILDREN
    # =========================================================

    def get_candidate_children(
        self,
        fen_key: str
    ) -> List[str]:

        # Return cached result if this candidate was already
        # analysed before.
        if fen_key in self.children_cache:
            return self.children_cache[
                fen_key
            ]

        candidate = self.candidates_by_fen[
            fen_key
        ]

        board = self.board_from_position_key(
            fen_key
        )

        children = set()

        for move_uci in candidate[
            "moves"
        ].keys():

            try:
                move = chess.Move.from_uci(
                    move_uci
                )

            except ValueError:

                self.illegal_candidate_moves += 1
                continue

            # The move should always be legal because it came
            # directly from an actual Lichess game.
            if move not in board.legal_moves:

                self.illegal_candidate_moves += 1
                continue

            # Play move.
            board.push(
                move
            )

            child_fen = self.get_position_key(
                board
            )

            # Restore original position.
            board.pop()

            # We only care about the child if the child itself
            # is one of the 50,000 raw candidates.
            if child_fen in self.candidates_by_fen:

                children.add(
                    child_fen
                )

        result = list(
            children
        )

        self.children_cache[
            fen_key
        ] = result

        return result


    # =========================================================
    # POSITION HANDLING
    # =========================================================

    def board_from_position_key(
        self,
        fen_key: str
    ) -> chess.Board:

        # The first dataset builder deliberately stored only:
        #
        # 1. Piece placement
        # 2. Side to move
        # 3. Castling rights
        # 4. En-passant square
        #
        # python-chess expects the full six-field FEN.
        #
        # Halfmove and fullmove counters are irrelevant for
        # identifying these candidate positions, therefore
        # dummy values are added.
        full_fen = (
            fen_key
            + " 0 1"
        )

        return chess.Board(
            full_fen
        )


    def get_position_key(
        self,
        board: chess.Board
    ) -> str:

        # IMPORTANT:
        #
        # This function must remain identical to the function
        # used in test_dataset_builder.py.
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
    # STOCKFISH EXECUTABLE
    # =========================================================

    def find_stockfish_executable(
        self,
        stockfish_directory: Path
    ) -> Path:

        if not stockfish_directory.exists():

            raise FileNotFoundError(
                f"Stockfish directory does not exist:\n"
                f"{stockfish_directory}"
            )

        candidates = []

        for path in stockfish_directory.iterdir():

            if not path.is_file():
                continue

            if not path.name.lower().startswith(
                "stockfish"
            ):
                continue

            # Windows executable or executable Unix file.
            if (
                path.suffix.lower() == ".exe"
                or os.access(str(path), os.X_OK)
            ):
                candidates.append(
                    path
                )

        if len(candidates) == 0:

            raise FileNotFoundError(
                "No Stockfish executable found in:\n"
                f"{stockfish_directory}"
            )

        if len(candidates) > 1:

            names = "\n".join(
                str(path.name)
                for path in candidates
            )

            raise RuntimeError(
                "Multiple Stockfish executables found. "
                "Please keep only the version that should "
                "be used for the experiment:\n"
                f"{names}"
            )

        return candidates[0]


    # =========================================================
    # STEP 3:
    # STOCKFISH EVALUATION
    # =========================================================

    def evaluate_and_filter_positions(
        self,
        candidates: List[Dict[str, Any]],
        stockfish_path: Path
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]]
    ]:

        evaluated_candidates = []

        balanced_candidates = []

        engine = chess.engine.SimpleEngine.popen_uci(
            str(stockfish_path)
        )

        try:

            # ---------------------------------------------
            # Fixed engine configuration
            # ---------------------------------------------

            engine_configuration = {}

            if "Threads" in engine.options:

                engine_configuration[
                    "Threads"
                ] = self.STOCKFISH_THREADS

            if "Hash" in engine.options:

                engine_configuration[
                    "Hash"
                ] = self.STOCKFISH_HASH_MB

            if engine_configuration:

                engine.configure(
                    engine_configuration
                )

            engine_name = "Unknown"

            if hasattr(engine, "id"):

                engine_name = engine.id.get(
                    "name",
                    "Unknown"
                )

            print(
                f"Engine: {engine_name}"
            )

            print(
                f"Depth: {self.STOCKFISH_DEPTH}"
            )

            print(
                f"Threads: {self.STOCKFISH_THREADS}"
            )

            print(
                f"Hash: {self.STOCKFISH_HASH_MB} MB"
            )

            print()

            total = len(
                candidates
            )

            for index, candidate in enumerate(
                candidates,
                start=1
            ):

                evaluation = self.evaluate_position(
                    engine,
                    candidate["fen"]
                )

                candidate_with_eval = dict(
                    candidate
                )

                candidate_with_eval.update(
                    evaluation
                )

                # -----------------------------------------
                # Determine whether position is balanced
                # -----------------------------------------

                evaluation_cp = (
                    candidate_with_eval.get(
                        "stockfish_eval_cp_white"
                    )
                )

                is_balanced = (
                    evaluation_cp is not None
                    and abs(evaluation_cp)
                    <= self.MAX_ABS_EVAL_CP
                )

                candidate_with_eval[
                    "is_balanced"
                ] = is_balanced

                candidate_with_eval[
                    "stockfish_depth"
                ] = self.STOCKFISH_DEPTH

                evaluated_candidates.append(
                    candidate_with_eval
                )

                if is_balanced:

                    balanced_candidates.append(
                        candidate_with_eval
                    )

                # -----------------------------------------
                # Progress
                # -----------------------------------------

                if index % 100 == 0:

                    print(
                        f"Stockfish: "
                        f"{index:,}/{total:,} | "
                        f"Balanced: "
                        f"{len(balanced_candidates):,}"
                    )

        finally:

            engine.quit()

        return (
            evaluated_candidates,
            balanced_candidates
        )


    def evaluate_position(
        self,
        engine,
        fen_key: str
    ) -> Dict[str, Any]:

        board = self.board_from_position_key(
            fen_key
        )

        # A new game object is deliberately supplied for every
        # position so the individual Stockfish analyses are
        # treated as separate analysis contexts.
        info = engine.analyse(
            board,
            chess.engine.Limit(
                depth=self.STOCKFISH_DEPTH
            ),
            game=object()
        )

        pov_score = info.get(
            "score"
        )

        if pov_score is None:

            return {
                "stockfish_eval_type":
                    "unknown",

                "stockfish_eval_cp_white":
                    None,

                "stockfish_eval_white":
                    None
            }

        # Always store the evaluation from White's perspective.
        score = pov_score.pov(
            chess.WHITE
        )

        # Mate evaluations are automatically excluded from
        # the +/-1.00 balance range.
        if score.is_mate():

            return {
                "stockfish_eval_type":
                    "mate",

                "stockfish_eval_cp_white":
                    None,

                "stockfish_eval_white":
                    None,

                "stockfish_mate_white":
                    score.mate()
            }

        centipawns = score.score()

        if centipawns is None:

            return {
                "stockfish_eval_type":
                    "unknown",

                "stockfish_eval_cp_white":
                    None,

                "stockfish_eval_white":
                    None
            }

        return {
            "stockfish_eval_type":
                "cp",

            "stockfish_eval_cp_white":
                centipawns,

            "stockfish_eval_white":
                round(
                    centipawns / 100.0,
                    2
                )
        }


    # =========================================================
    # OUTPUT
    # =========================================================

    def save_candidates(
        self,
        candidates: List[Dict[str, Any]],
        output_path: Path
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                candidates,
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

    raw_candidate_file = (
        PROJECT_ROOT
        / "data"
        / "candidates"
        / "raw_candidate_positions.json"
    )


    # =========================================================
    # OUTPUT: TOP 5,000 INDEPENDENT POSITIONS
    # =========================================================

    independent_output_file = (
        PROJECT_ROOT
        / "data"
        / "candidates"
        / "independent_candidate_positions.json"
    )


    # =========================================================
    # OUTPUT: ALL 5,000 WITH STOCKFISH EVALUATIONS
    # =========================================================

    evaluated_output_file = (
        PROJECT_ROOT
        / "data"
        / "candidates"
        / "evaluated_candidate_positions.json"
    )


    # =========================================================
    # OUTPUT: ONLY POSITIONS BETWEEN -1.00 AND +1.00
    # =========================================================

    balanced_output_file = (
        PROJECT_ROOT
        / "data"
        / "candidates"
        / "balanced_candidate_positions.json"
    )


    # =========================================================
    # STOCKFISH
    # =========================================================
    #
    # Put exactly one Stockfish executable in:
    #
    # engines/stockfish/
    #
    # The filename itself does not matter as long as it begins
    # with "stockfish".
    #
    stockfish_directory = (
        PROJECT_ROOT
        / "engines"
        / "stockfish"
    )


    # =========================================================
    # RUN
    # =========================================================

    processor = CandidateProcessor()

    processor.run(
        input_path=raw_candidate_file,
        independent_output_path=independent_output_file,
        evaluated_output_path=evaluated_output_file,
        balanced_output_path=balanced_output_file,
        stockfish_directory=stockfish_directory
    )