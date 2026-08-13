from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn
import zstandard as zstd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================
# FAST PGN VISITOR
# =============================================================

class FastGameVisitor(chess.pgn.BaseVisitor):

    def __init__(
        self,
        collector,
        source_file: str
    ):

        self.collector = collector
        self.source_file = source_file

        self.headers = chess.pgn.Headers({})

        self.header_suitable = False
        self.parse_error = False

        self.mainline_plies = 0

        # Moves leading to the currently inspected position.
        # Move objects are stored instead of UCI strings because
        # this avoids unnecessary string creation for every game.
        self.previous_moves = []

        # First target position found in this game.
        self.observation = None

        # After the first 40 plies there is no more relevant work.
        self.stop_position_processing = False


    # =========================================================
    # HEADERS
    # =========================================================

    def begin_headers(self):

        self.headers = chess.pgn.Headers({})

        return self.headers


    def visit_header(
        self,
        tagname: str,
        tagvalue: str
    ) -> None:

        self.headers[
            tagname
        ] = tagvalue


    def end_headers(self):

        self.header_suitable = (
            self.collector.is_suitable_headers(
                self.headers
            )
        )

        # IMPORTANT:
        #
        # If the cheap header filters fail, python-chess uses
        # its fast game-skipping path instead of parsing all
        # chess moves.
        if not self.header_suitable:
            return chess.pgn.SKIP

        return None


    # =========================================================
    # VARIATIONS
    # =========================================================

    def begin_variation(self):

        # Lichess database analysis only uses the main line.
        return chess.pgn.SKIP


    # =========================================================
    # OPTIONAL FAST PATH FOR NEWER PYTHON-CHESS VERSIONS
    # =========================================================

    def begin_parse_san(
        self,
        board: chess.Board,
        san: str
    ):

        # Once 40 plies have been parsed, the game is known to
        # satisfy the minimum-length requirement and all relevant
        # positions (moves 11-20) have already been checked.
        #
        # Newer python-chess versions call this method before
        # parsing SAN. Returning SKIP avoids unnecessary SAN
        # parsing for all later moves.
        #
        # Older python-chess versions simply do not call this
        # method, so it remains harmless and compatible.
        if self.stop_position_processing:
            return chess.pgn.SKIP

        return None


    # =========================================================
    # MOVES
    # =========================================================

    def visit_move(
        self,
        board: chess.Board,
        move: chess.Move
    ) -> None:

        # On older python-chess versions the parser may still
        # visit later moves. No additional work is required.
        if self.stop_position_processing:
            return

        self.mainline_plies += 1


        # =====================================================
        # CHECK TARGET POSITION
        # =====================================================

        if (
            self.observation is None
            and self.collector.START_FULLMOVE
            <= board.fullmove_number
            <= self.collector.END_FULLMOVE
        ):

            # Much cheaper than constructing a FEN string
            # for every inspected position.
            position_key = (
                self.collector.get_fast_position_key(
                    board
                )
            )

            target_fen = (
                self.collector.target_positions.get(
                    position_key
                )
            )

            if target_fen is not None:

                self.observation = (
                    self.build_observation(
                        board=board,
                        move=move,
                        target_fen=target_fen
                    )
                )


        # =====================================================
        # STORE PREVIOUS MOVES
        # =====================================================

        # Once a target has been found, the move sequence is
        # already stored in the observation and no longer needs
        # to be extended.
        if self.observation is None:

            self.previous_moves.append(
                move
            )


        # =====================================================
        # STOP RELEVANT PROCESSING AFTER 40 PLIES
        # =====================================================

        minimum_plies = (
            self.collector.MIN_FULLMOVES
            * 2
        )

        if self.mainline_plies >= minimum_plies:

            self.stop_position_processing = True


    # =========================================================
    # BUILD MATCH OBSERVATION
    # =========================================================

    def build_observation(
        self,
        board: chess.Board,
        move: chess.Move,
        target_fen: str
    ) -> Dict[str, Any]:

        white_elo = (
            self.collector.parse_int(
                self.headers.get(
                    "WhiteElo"
                )
            )
        )

        black_elo = (
            self.collector.parse_int(
                self.headers.get(
                    "BlackElo"
                )
            )
        )


        if board.turn == chess.WHITE:

            side_to_move = "white"

            player_rating = white_elo
            opponent_rating = black_elo

        else:

            side_to_move = "black"

            player_rating = black_elo
            opponent_rating = white_elo


        rating_bucket = (
            self.collector.get_rating_bucket(
                player_rating
            )
        )


        # UCI conversion of the previous moves happens ONLY
        # when an actual target position is found.
        moves_to_position_uci = [

            previous_move.uci()

            for previous_move
            in self.previous_moves
        ]


        observation = {

            "fen":
                target_fen,

            "fullmove_number":
                board.fullmove_number,

            "side_to_move":
                side_to_move,

            "played_move_uci":
                move.uci(),

            "played_move_san":
                board.san(
                    move
                ),

            "moves_to_position_uci":
                moves_to_position_uci,

            "white_rating":
                white_elo,

            "black_rating":
                black_elo,

            "player_rating":
                player_rating,

            "opponent_rating":
                opponent_rating,

            "rating_bucket":
                rating_bucket,

            # Original result from the PGN.
            "result":
                self.headers.get(
                    "Result",
                    ""
                ),

            "game_url":
                self.headers.get(
                    "Site",
                    ""
                ),

            "time_control":
                self.headers.get(
                    "TimeControl",
                    ""
                ),

            "source_file":
                self.source_file
        }

        return observation


    # =========================================================
    # ERROR HANDLING
    # =========================================================

    def handle_error(
        self,
        error: Exception
    ) -> None:

        # An error inside the relevant first 40 plies makes the
        # game unusable.
        #
        # Errors after the relevant range are irrelevant for
        # this experiment.
        if not self.stop_position_processing:

            self.parse_error = True


    # =========================================================
    # RESULT
    # =========================================================

    def result(self):

        minimum_plies = (
            self.collector.MIN_FULLMOVES
            * 2
        )

        suitable = (
            self.header_suitable
            and not self.parse_error
            and self.mainline_plies
            >= minimum_plies
        )

        if not suitable:

            return (
                False,
                None
            )

        return (
            True,
            self.observation
        )


# =============================================================
# TEST DATASET COLLECTOR
# =============================================================

class TestDatasetCollector:

    # =========================================================
    # CONFIGURATION
    # =========================================================

    START_FULLMOVE = 11
    END_FULLMOVE = 20

    MIN_FULLMOVES = 20

    ALLOWED_SPEED_KEYWORDS = {
        "Blitz",
        "Rapid",
        "Classical"
    }

    MAX_RATING_DIFF = 200

    VALID_RESULTS = {
        "1-0",
        "0-1",
        "1/2-1/2"
    }

    RATING_BUCKET_SIZE = 200

    PROGRESS_INTERVAL = 100_000


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        # Maps a fast board representation to the original
        # four-field FEN stored in balanced_candidate_positions.
        #
        # fast position key -> original FEN
        #
        self.target_positions = {}


        self.aggregated_data = defaultdict(
            lambda: {

                "total_occurrences": 0,

                "rating_buckets": defaultdict(
                    lambda: {

                        "occurrences": 0,

                        "moves": defaultdict(
                            lambda: {

                                "count": 0,

                                "white_wins": 0,

                                "draws": 0,

                                "black_wins": 0
                            }
                        )
                    }
                )
            }
        )


    # =========================================================
    # MAIN METHOD
    # =========================================================

    def run(
        self,
        candidate_path: Path,
        input_paths: List[Path],
        detailed_output_path: Path,
        aggregated_output_path: Path
    ) -> None:

        start_time = time.perf_counter()


        # =====================================================
        # LOAD TARGET POSITIONS
        # =====================================================

        print(
            "Loading final target positions..."
        )

        self.target_positions = (
            self.load_target_positions(
                candidate_path
            )
        )

        print(
            f"Target positions loaded: "
            f"{len(self.target_positions):,}"
        )

        if len(self.target_positions) == 0:

            raise ValueError(
                "No target positions were loaded."
            )


        # =====================================================
        # OUTPUT DIRECTORIES
        # =====================================================

        detailed_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        aggregated_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # =====================================================
        # GLOBAL COUNTERS
        # =====================================================

        total_games = 0
        suitable_games = 0
        total_matches = 0


        # =====================================================
        # DETAILED JSONL OUTPUT
        # =====================================================

        with open(
            detailed_output_path,
            "w",
            encoding="utf-8"
        ) as detailed_file:


            # =================================================
            # MONTHLY FILES
            # =================================================

            for input_path in input_paths:

                if not input_path.exists():

                    raise FileNotFoundError(
                        f"Input database not found:\n"
                        f"{input_path}"
                    )


                print()
                print(
                    "=" * 60
                )

                print(
                    f"Processing: "
                    f"{input_path.name}"
                )

                print(
                    "=" * 60
                )


                file_games = 0
                file_suitable_games = 0
                file_matches = 0


                (
                    binary_file,
                    reader,
                    text_stream
                ) = self.open_pgn_text_stream(
                    input_path
                )


                # Visitor factory used for every game
                # in this monthly database.
                visitor_factory = lambda: FastGameVisitor(
                    self,
                    input_path.name
                )


                try:

                    while True:

                        # =====================================
                        # FAST PGN PARSING
                        # =====================================

                        scan_result = (
                            chess.pgn.read_game(
                                text_stream,
                                Visitor=visitor_factory
                            )
                        )


                        # End of compressed PGN file.
                        if scan_result is None:
                            break


                        total_games += 1
                        file_games += 1


                        suitable, observation = (
                            scan_result
                        )


                        # =====================================
                        # SUITABLE GAME
                        # =====================================

                        if suitable:

                            suitable_games += 1
                            file_suitable_games += 1


                        # =====================================
                        # MATCH
                        # =====================================

                        if observation is not None:

                            total_matches += 1
                            file_matches += 1


                            # ---------------------------------
                            # Detailed observation
                            # ---------------------------------

                            detailed_file.write(

                                json.dumps(
                                    observation,
                                    ensure_ascii=False
                                )

                                + "\n"
                            )


                            # ---------------------------------
                            # Aggregation
                            # ---------------------------------

                            self.add_to_aggregation(
                                observation
                            )


                            # ---------------------------------
                            # Periodic disk flush
                            # ---------------------------------

                            if (
                                total_matches
                                % 1_000
                                == 0
                            ):

                                detailed_file.flush()


                        # =====================================
                        # PROGRESS
                        # =====================================

                        if (
                            total_games
                            % self.PROGRESS_INTERVAL
                            == 0
                        ):

                            self.print_progress(
                                total_games=
                                    total_games,

                                suitable_games=
                                    suitable_games,

                                matches=
                                    total_matches,

                                start_time=
                                    start_time
                            )


                finally:

                    text_stream.close()

                    if reader is not None:
                        reader.close()

                    if binary_file is not None:
                        binary_file.close()


                # =================================================
                # MONTH FINISHED
                # =================================================

                detailed_file.flush()


                # Checkpoint of aggregate after every month.
                self.save_aggregated_data(
                    aggregated_output_path
                )


                print()

                print(
                    f"Finished "
                    f"{input_path.name}"
                )

                print(
                    f"Games read: "
                    f"{file_games:,}"
                )

                print(
                    f"Suitable games: "
                    f"{file_suitable_games:,}"
                )

                print(
                    f"Matching games: "
                    f"{file_matches:,}"
                )


        # =====================================================
        # FINAL AGGREGATED OUTPUT
        # =====================================================

        self.save_aggregated_data(
            aggregated_output_path
        )


        # =====================================================
        # FINAL INFORMATION
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
            "Test dataset collection finished"
        )

        print(
            "=" * 60
        )


        print(
            "Total games read:",
            f"{total_games:,}"
        )

        print(
            "Suitable games:",
            f"{suitable_games:,}"
        )

        print(
            "Matching games:",
            f"{total_matches:,}"
        )

        print(
            "Target positions:",
            f"{len(self.target_positions):,}"
        )

        print(
            "Runtime in minutes:",
            round(
                runtime / 60,
                2
            )
        )

        if runtime > 0:

            print(
                "Average games per second:",
                round(
                    total_games
                    / runtime,
                    1
                )
            )


    # =========================================================
    # TARGET POSITIONS
    # =========================================================

    def load_target_positions(
        self,
        candidate_path: Path
    ):

        if not candidate_path.exists():

            raise FileNotFoundError(
                f"Balanced candidate file not found:\n"
                f"{candidate_path}"
            )


        with open(
            candidate_path,
            "r",
            encoding="utf-8"
        ) as file:

            candidates = json.load(
                file
            )


        target_positions = {}


        for candidate in candidates:

            fen_key = candidate[
                "fen"
            ]

            # Candidate FEN contains four fields.
            # Add irrelevant counters to create a Board.
            board = chess.Board(
                fen_key
                + " 0 1"
            )

            fast_key = (
                self.get_fast_position_key(
                    board
                )
            )


            if (
                fast_key in target_positions
                and target_positions[
                    fast_key
                ] != fen_key
            ):

                raise ValueError(
                    "Two target FENs produced the same "
                    "fast position key."
                )


            target_positions[
                fast_key
            ] = fen_key


        return target_positions


    # =========================================================
    # FAST POSITION KEY
    # =========================================================

    def get_fast_position_key(
        self,
        board: chess.Board
    ) -> Tuple[Any, ...]:

        # Equivalent position information to the four FEN
        # fields used by the previous programs:
        #
        # - piece placement
        # - side to move
        # - castling rights
        # - en-passant square
        #
        # Unlike board_fen(), this avoids creating a large
        # string for every position.

        return (

            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,

            board.occupied_co[
                chess.WHITE
            ],

            board.occupied_co[
                chess.BLACK
            ],

            board.turn,

            board.castling_rights,

            board.ep_square
        )


    # =========================================================
    # FILE HANDLING
    # =========================================================

    def open_pgn_text_stream(
        self,
        file_path: Path
    ):

        if file_path.suffix == ".zst":

            binary_file = open(
                file_path,
                "rb"
            )

            decompressor = (
                zstd.ZstdDecompressor()
            )

            reader = (
                decompressor.stream_reader(
                    binary_file
                )
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


        text_file = open(
            file_path,
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
    # HELPER
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

            return int(
                value
            )

        except ValueError:

            return None


    # =========================================================
    # HEADER FILTERS
    # =========================================================

    def is_suitable_headers(
        self,
        headers
    ) -> bool:

        # ---------------------------------------------
        # Variant
        # ---------------------------------------------

        variant = headers.get(
            "Variant",
            "Standard"
        )

        if variant != "Standard":
            return False


        # ---------------------------------------------
        # Speed
        # ---------------------------------------------

        event = headers.get(
            "Event",
            ""
        )

        allowed_speed = any(

            keyword in event

            for keyword
            in self.ALLOWED_SPEED_KEYWORDS
        )

        if not allowed_speed:
            return False


        # ---------------------------------------------
        # Termination
        # ---------------------------------------------

        termination = headers.get(
            "Termination",
            ""
        )

        if termination != "Normal":
            return False


        # ---------------------------------------------
        # Ratings
        # ---------------------------------------------

        white_elo = self.parse_int(
            headers.get(
                "WhiteElo"
            )
        )

        black_elo = self.parse_int(
            headers.get(
                "BlackElo"
            )
        )

        if (
            white_elo is None
            or black_elo is None
        ):
            return False


        if (
            abs(
                white_elo
                - black_elo
            )
            > self.MAX_RATING_DIFF
        ):
            return False


        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        result = headers.get(
            "Result",
            ""
        )

        if (
            result
            not in self.VALID_RESULTS
        ):
            return False


        # ---------------------------------------------
        # Optional PlyCount shortcut
        # ---------------------------------------------
        #
        # If a PGN happens to contain a PlyCount header,
        # short games can already be skipped here.
        #
        # Lichess files do not need to contain this field,
        # so absence does not reject the game.

        ply_count = self.parse_int(
            headers.get(
                "PlyCount"
            )
        )

        if (
            ply_count is not None
            and ply_count
            < self.MIN_FULLMOVES * 2
        ):
            return False


        return True


    # =========================================================
    # RATING BUCKETS
    # =========================================================

    def get_rating_bucket(
        self,
        rating: Optional[int]
    ) -> str:

        if rating is None:

            return "unknown"


        lower_bound = (

            rating
            // self.RATING_BUCKET_SIZE

            * self.RATING_BUCKET_SIZE
        )


        upper_bound = (

            lower_bound

            + self.RATING_BUCKET_SIZE

            - 1
        )


        return (
            f"{lower_bound}-"
            f"{upper_bound}"
        )


    # =========================================================
    # AGGREGATION
    # =========================================================

    def add_to_aggregation(
        self,
        observation: Dict[str, Any]
    ) -> None:

        fen = observation[
            "fen"
        ]

        rating_bucket = observation[
            "rating_bucket"
        ]

        move_uci = observation[
            "played_move_uci"
        ]

        result = observation[
            "result"
        ]


        fen_data = (
            self.aggregated_data[
                fen
            ]
        )


        fen_data[
            "total_occurrences"
        ] += 1


        bucket_data = (
            fen_data[
                "rating_buckets"
            ][
                rating_bucket
            ]
        )


        bucket_data[
            "occurrences"
        ] += 1


        move_data = (
            bucket_data[
                "moves"
            ][
                move_uci
            ]
        )


        move_data[
            "count"
        ] += 1


        # Results remain in their original PGN meaning.

        if result == "1-0":

            move_data[
                "white_wins"
            ] += 1


        elif result == "1/2-1/2":

            move_data[
                "draws"
            ] += 1


        elif result == "0-1":

            move_data[
                "black_wins"
            ] += 1


    # =========================================================
    # BUILD AGGREGATED OUTPUT
    # =========================================================

    def build_aggregated_output(
        self
    ) -> Dict[str, Any]:

        output = {}


        for (
            fen,
            fen_data
        ) in self.aggregated_data.items():


            output[
                fen
            ] = {

                "total_occurrences":
                    fen_data[
                        "total_occurrences"
                    ],

                "rating_buckets":
                    {}
            }


            for (
                rating_bucket,
                bucket_data
            ) in fen_data[
                "rating_buckets"
            ].items():


                output[
                    fen
                ][
                    "rating_buckets"
                ][
                    rating_bucket
                ] = {

                    "occurrences":
                        bucket_data[
                            "occurrences"
                        ],

                    "moves":
                        {}
                }


                for (
                    move_uci,
                    move_data
                ) in bucket_data[
                    "moves"
                ].items():


                    output[
                        fen
                    ][
                        "rating_buckets"
                    ][
                        rating_bucket
                    ][
                        "moves"
                    ][
                        move_uci
                    ] = {

                        "count":
                            move_data[
                                "count"
                            ],

                        "white_wins":
                            move_data[
                                "white_wins"
                            ],

                        "draws":
                            move_data[
                                "draws"
                            ],

                        "black_wins":
                            move_data[
                                "black_wins"
                            ]
                    }


        return output


    # =========================================================
    # SAVE AGGREGATED OUTPUT
    # =========================================================

    def save_aggregated_data(
        self,
        output_path: Path
    ) -> None:

        output = (
            self.build_aggregated_output()
        )


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


    # =========================================================
    # PROGRESS
    # =========================================================

    def print_progress(
        self,
        total_games: int,
        suitable_games: int,
        matches: int,
        start_time: float
    ) -> None:

        runtime = (
            time.perf_counter()
            - start_time
        )


        if runtime > 0:

            games_per_second = (
                total_games
                / runtime
            )

        else:

            games_per_second = 0.0


        print(

            f"Games read: "
            f"{total_games:,} | "

            f"Suitable: "
            f"{suitable_games:,} | "

            f"Matches: "
            f"{matches:,} | "

            f"Speed: "
            f"{games_per_second:.1f} games/s | "

            f"Runtime: "
            f"{runtime / 60:.1f} min"
        )


# =============================================================
# PROGRAM START
# =============================================================

if __name__ == "__main__":

    # =========================================================
    # FINAL BALANCED CANDIDATES
    # =========================================================

    candidate_file = (

        PROJECT_ROOT
        / "data"
        / "candidates"
        / "balanced_candidate_positions.json"
    )


    # =========================================================
    # LARGE TEST DATASETS
    # =========================================================
    #
    # Put downloaded .pgn.zst files in:
    #
    # data/raw/
    #
    MONTHLY_FILES = [

        "lichess_db_standard_rated_2019-01.pgn.zst",

        # Later:
        #
        # "lichess_db_standard_rated_2019-02.pgn.zst",
        # "lichess_db_standard_rated_2019-03.pgn.zst",
        # "lichess_db_standard_rated_2019-04.pgn.zst",
        # "lichess_db_standard_rated_2019-05.pgn.zst",
        # "lichess_db_standard_rated_2019-06.pgn.zst",
        # "lichess_db_standard_rated_2019-07.pgn.zst",
        # "lichess_db_standard_rated_2019-08.pgn.zst",
        # "lichess_db_standard_rated_2019-09.pgn.zst",
        # "lichess_db_standard_rated_2019-10.pgn.zst",
        # "lichess_db_standard_rated_2019-11.pgn.zst",
        # "lichess_db_standard_rated_2019-12.pgn.zst",
    ]


    input_files = [

        PROJECT_ROOT
        / "data"
        / "raw"
        / filename

        for filename
        in MONTHLY_FILES
    ]


    # =========================================================
    # OUTPUT
    # =========================================================

    detailed_output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_detailed.jsonl"
    )


    aggregated_output_file = (

        PROJECT_ROOT
        / "data"
        / "results"
        / "test_dataset_aggregated.json"
    )


    # =========================================================
    # RUN
    # =========================================================

    collector = (
        TestDatasetCollector()
    )


    collector.run(

        candidate_path=
            candidate_file,

        input_paths=
            input_files,

        detailed_output_path=
            detailed_output_file,

        aggregated_output_path=
            aggregated_output_file
    )