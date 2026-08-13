from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, cast

import chess
import chess.pgn
import zstandard as zstd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDatasetBuilder:

    # =========================================================
    # CONFIGURATION
    # =========================================================

    # Only positions between move 11 and move 20 are considered.
    START_FULLMOVE = 11
    END_FULLMOVE = 20

    # Games must contain at least 20 full moves (= 40 plies).
    MIN_FULLMOVES = 20

    # A candidate position must have at least two different
    # moves that were played in the complete small dataset.
    MIN_DIFFERENT_MOVES = 2

    # Number of raw candidates retained after ranking.
    TOP_CANDIDATES = 50_000

    # Only these Lichess game types are accepted.
    ALLOWED_SPEED_KEYWORDS = {
        "Blitz",
        "Rapid",
        "Classical"
    }

    # Maximum allowed Elo difference between both players.
    MAX_RATING_DIFF = 200


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        self.positions = defaultdict(
            lambda: {
                "occurrences": 0,
                "moves": defaultdict(int)
            }
        )


    # =========================================================
    # MAIN METHOD
    # =========================================================

    def run(
        self,
        input_path: Path,
        output_path: Path
    ) -> None:

        start_time = time.perf_counter()

        binary_file, reader, text_stream = self.open_pgn_text_stream(
            input_path
        )

        total_games = 0
        suitable_games = 0

        try:

            while True:

                game = cast(
                    Optional[chess.pgn.Game],
                    chess.pgn.read_game(text_stream)
                )

                if game is None:
                    break

                total_games += 1

                # Apply header-based filters first.
                if not self.is_suitable_game(game):
                    continue

                # Only extract all moves after the cheaper filters.
                moves = list(
                    game.mainline_moves()
                )

                # Game must contain at least 20 full moves.
                if not self.has_minimum_length(moves):
                    continue

                suitable_games += 1

                # Collect positions and played moves.
                self.handle_suitable_game(
                    game,
                    moves
                )

                # Progress output for long runs.
                if suitable_games % 100_000 == 0:
                    print(
                        f"Suitable games processed: "
                        f"{suitable_games:,}"
                    )

        finally:

            text_stream.close()

            if reader is not None:
                reader.close()

            if binary_file is not None:
                binary_file.close()


        # =====================================================
        # BUILD CANDIDATE LIST
        # =====================================================

        candidates = self.build_candidate_list()

        eligible_positions = self.count_eligible_positions()

        if eligible_positions < self.TOP_CANDIDATES:
            print()
            print(
                f"WARNING: Only {eligible_positions:,} eligible "
                f"positions were found instead of the requested "
                f"{self.TOP_CANDIDATES:,}."
            )

        self.save_candidates(
            candidates,
            output_path
        )

        runtime = time.perf_counter() - start_time


        # =====================================================
        # FINAL INFORMATION
        # =====================================================

        print()
        print("Candidate generation finished")
        print("----------------------------------")

        print(
            "Total games read:",
            f"{total_games:,}"
        )

        print(
            "Suitable games:",
            f"{suitable_games:,}"
        )

        print(
            "Different positions found:",
            f"{len(self.positions):,}"
        )

        print(
            "Eligible positions:",
            f"{eligible_positions:,}"
        )

        print(
            "Raw candidates saved:",
            f"{len(candidates):,}"
        )

        print(
            "Runtime in seconds:",
            round(runtime, 2)
        )


    # =========================================================
    # FILE HANDLING
    # =========================================================

    def open_pgn_text_stream(
        self,
        file_path: Path
    ):

        path = file_path

        # Lichess database compressed as .zst.
        if path.suffix == ".zst":

            binary_file = open(
                path,
                "rb"
            )

            decompressor = zstd.ZstdDecompressor()

            reader = decompressor.stream_reader(
                binary_file
            )

            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8",
                errors="replace"
            )

            return (
                binary_file,
                reader,
                text_stream
            )

        # Normal uncompressed PGN file.
        text_file = open(
            path,
            "r",
            encoding="utf-8",
            errors="replace"
        )

        return (
            None,
            None,
            text_file
        )


    # =========================================================
    # HELPER FUNCTIONS
    # =========================================================

    def parse_int(
        self,
        value: Optional[str]
    ) -> Optional[int]:

        if value is None:
            return None

        value = value.strip()

        if value == "":
            return None

        try:
            return int(value)

        except ValueError:
            return None


    # =========================================================
    # GAME FILTERS
    # =========================================================

    def is_allowed_variant(
        self,
        game
    ) -> bool:

        variant = game.headers.get(
            "Variant",
            "Standard"
        )

        return variant == "Standard"


    def is_allowed_event(
        self,
        game
    ) -> bool:

        event = game.headers.get(
            "Event",
            ""
        )

        return any(
            keyword in event
            for keyword in self.ALLOWED_SPEED_KEYWORDS
        )


    def is_allowed_termination(
        self,
        game
    ) -> bool:

        termination = game.headers.get(
            "Termination",
            ""
        )

        return termination == "Normal"


    def is_allowed_rating_diff(
        self,
        game
    ) -> bool:

        white_elo = self.parse_int(
            game.headers.get(
                "WhiteElo"
            )
        )

        black_elo = self.parse_int(
            game.headers.get(
                "BlackElo"
            )
        )

        # Games without valid ratings are ignored.
        if white_elo is None or black_elo is None:
            return False

        rating_difference = abs(
            white_elo - black_elo
        )

        return (
            rating_difference
            <= self.MAX_RATING_DIFF
        )


    def is_suitable_game(
        self,
        game
    ) -> bool:

        if not self.is_allowed_variant(game):
            return False

        if not self.is_allowed_event(game):
            return False

        if not self.is_allowed_termination(game):
            return False

        if not self.is_allowed_rating_diff(game):
            return False

        return True


    # =========================================================
    # GAME LENGTH FILTER
    # =========================================================

    def has_minimum_length(
        self,
        moves
    ) -> bool:

        minimum_plies = (
            self.MIN_FULLMOVES * 2
        )

        return (
            len(moves)
            >= minimum_plies
        )


    # =========================================================
    # POSITION IDENTIFICATION
    # =========================================================

    def get_position_key(
        self,
        board: chess.Board
    ) -> str:

        # The halfmove clock and fullmove number are excluded.
        #
        # The position key contains:
        # 1. Piece placement
        # 2. Side to move
        # 3. Castling rights
        # 4. En-passant square
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
    # POSITION COLLECTION
    # =========================================================

    def handle_suitable_game(
        self,
        game,
        moves
    ) -> None:

        board = game.board()

        for move in moves:

            # Record the position before the move is played.
            if (
                self.START_FULLMOVE
                <= board.fullmove_number
                <= self.END_FULLMOVE
            ):

                fen_key = self.get_position_key(
                    board
                )

                move_uci = move.uci()

                position_entry = self.positions[
                    fen_key
                ]

                position_entry[
                    "occurrences"
                ] += 1

                position_entry[
                    "moves"
                ][move_uci] += 1

            board.push(move)


    # =========================================================
    # CANDIDATE GENERATION
    # =========================================================

    def build_candidate_list(
        self
    ):

        candidates = []

        # This filter is applied only after the complete
        # small dataset has been processed.
        for fen, data in self.positions.items():

            different_moves = len(
                data["moves"]
            )

            if different_moves < self.MIN_DIFFERENT_MOVES:
                continue

            candidates.append({
                "fen": fen,
                "occurrences": data["occurrences"],
                "different_moves": different_moves,
                "moves": dict(
                    data["moves"]
                )
            })

        # Rank positions by total occurrence frequency.
        candidates.sort(
            key=lambda candidate:
                candidate["occurrences"],
            reverse=True
        )

        # Keep at most 50,000 raw candidates.
        return candidates[
            :self.TOP_CANDIDATES
        ]


    # =========================================================
    # STATISTICS
    # =========================================================

    def count_eligible_positions(
        self
    ) -> int:

        return sum(
            1
            for data in self.positions.values()
            if len(data["moves"])
            >= self.MIN_DIFFERENT_MOVES
        )


    # =========================================================
    # OUTPUT
    # =========================================================

    def save_candidates(
        self,
        candidates,
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

    input_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "lichess_db_standard_rated_2016-01.pgn.zst"
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "candidates"
        / "raw_candidate_positions.json"
    )

    builder = TestDatasetBuilder()

    builder.run(
        input_path=input_file,
        output_path=output_file
    )