from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import chess
import torch
from torch.amp import autocast

from maia3.dataset import (
    get_historical_tokens,
    get_legal_moves_mask,
    tokenize_board,
)

from maia3.uci import (
    Maia3UCIEngine,
    parse_args,
)

from maia3.utils import (
    mirror_move,
)


# =============================================================
# MAIA-3 ADAPTER
# =============================================================

class Maia3Adapter:
    """
    Provides Maia-3 move probabilities for ALL legal moves.

    Rating is supplied directly to Maia-3.

    This adapter performs no Stockfish analysis and accesses
    no JSON files.
    """

    def __init__(
        self,
        model_alias: str = "maia3-5m",
        device: Optional[str] = None
    ) -> None:

        # -----------------------------------------------------
        # Automatically use GPU when PyTorch can access one.
        # -----------------------------------------------------

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        args = [
            "--model",
            model_alias,

            "--device",
            device,

            "--temperature",
            "1.0",

            "--top-p",
            "1.0"
        ]


        if device == "cpu":

            args.append(
                "--no-use-amp"
            )


        # -----------------------------------------------------
        # Let Maia-3 create the correct model configuration.
        # -----------------------------------------------------

        config = (
            parse_args(
                args
            )
        )


        backend = (
            Maia3UCIEngine(
                config
            )
        )


        # Loads the checkpoint once.
        # On first use Maia-3 may download it.
        backend.ensure_model_loaded()


        self.model_alias = (
            model_alias
        )

        self.device = (
            device
        )

        self.config = (
            config
        )

        self.backend = (
            backend
        )

        self.model = (
            backend.model
        )

        self.all_moves_dict = (
            backend.all_moves_dict
        )


    # =========================================================
    # HISTORY
    # =========================================================

    def initial_history(
        self,
        board: chess.Board
    ) -> Deque[torch.Tensor]:
        """
        Creates Maia history beginning with the known root FEN.

        Earlier game history is unavailable in the aggregated
        dataset and is therefore padded internally by Maia-3.
        """

        history = deque(
            maxlen=
                self.config.history
        )


        history.append(
            tokenize_board(
                board
            )
        )


        return history


    def history_after_move(
        self,
        history: Deque[torch.Tensor],
        board_after_move: chess.Board
    ) -> Deque[torch.Tensor]:
        """
        Returns a COPY of the history with the newly reached
        position appended.
        """

        new_history = deque(
            history,
            maxlen=
                self.config.history
        )


        new_history.append(
            tokenize_board(
                board_after_move
            )
        )


        return new_history


    # =========================================================
    # MOVE PROBABILITIES
    # =========================================================

    def get_move_probabilities(
        self,
        board: chess.Board,
        self_elo: int,
        opponent_elo: int,
        history: Optional[
            Deque[torch.Tensor]
        ] = None
    ) -> Dict[chess.Move, float]:
        """
        Returns:

            {
                chess.Move: probability,
                ...
            }

        for ALL legal moves.

        self_elo:
            rating of player to move

        opponent_elo:
            opponent rating
        """

        if board.is_game_over(
            claim_draw=False
        ):

            return {}


        self._validate_elo(
            self_elo
        )

        self._validate_elo(
            opponent_elo
        )


        if history is None:

            history = (
                self.initial_history(
                    board
                )
            )


        # =====================================================
        # LEGAL MOVE MASK
        # =====================================================

        legal_mask = (

            get_legal_moves_mask(
                board,
                self.all_moves_dict
            )

            .to(
                self.device
            )
        )


        # =====================================================
        # POSITION TOKENS
        # =====================================================

        tokens = (

            get_historical_tokens(

                history,

                self.config,

                base=0.0,
                inc=0.0,

                clk_left_before=0.0,
                clk_ponder=0.0
            )

            .unsqueeze(0)

            .to(
                self.device
            )
        )


        self_elos = torch.tensor(

            [
                self_elo
            ],

            dtype=torch.long,

            device=
                self.device
        )


        opponent_elos = torch.tensor(

            [
                opponent_elo
            ],

            dtype=torch.long,

            device=
                self.device
        )


        # =====================================================
        # ONE MAIA INFERENCE
        # =====================================================

        with torch.no_grad():


            with autocast(

                "cuda",

                enabled=(

                    self.config.use_amp

                    and self.device.startswith(
                        "cuda"
                    )
                )
            ):


                (
                    move_logits,
                    _,
                    _
                ) = (

                    self.model(

                        tokens,

                        self_elos,

                        opponent_elos
                    )
                )


        # =====================================================
        # REMOVE ILLEGAL MOVES
        # =====================================================

        logits = (

            move_logits[0]

            .float()

            .masked_fill(

                ~legal_mask,

                float(
                    "-inf"
                )
            )
        )


        probabilities = (

            torch.softmax(
                logits,
                dim=-1
            )
        )


        # =====================================================
        # MAP MODEL OUTPUT BACK TO CHESS MOVES
        # =====================================================

        result = {}


        for move in board.legal_moves:


            if board.turn == chess.WHITE:

                model_move = (
                    move.uci()
                )


            else:

                model_move = (
                    mirror_move(
                        move.uci()
                    )
                )


            move_index = (

                self.all_moves_dict.get(
                    model_move
                )
            )


            if move_index is None:

                raise RuntimeError(
                    "Maia-3 move vocabulary does not contain "
                    "legal move: "
                    + move.uci()
                )


            result[
                move
            ] = float(

                probabilities[
                    move_index
                ].item()
            )


        # =====================================================
        # NUMERICAL NORMALIZATION
        # =====================================================

        total_probability = sum(
            result.values()
        )


        if total_probability <= 0:

            raise RuntimeError(
                "Maia-3 returned zero probability mass."
            )


        return {

            move:
                probability
                / total_probability

            for (
                move,
                probability
            ) in result.items()
        }


    # =========================================================
    # ELO VALIDATION
    # =========================================================

    @staticmethod
    def _validate_elo(
        elo: int
    ) -> None:

        if not (
            0
            <= elo
            <= 5000
        ):

            raise ValueError(
                "Maia Elo must be between 0 and 5000."
            )