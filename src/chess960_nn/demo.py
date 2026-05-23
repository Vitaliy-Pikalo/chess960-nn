"""Interactive demo helpers: play-vs-engine + watch-vs-stockfish.

This module wraps the existing model/MCTS/Stockfish primitives behind a tiny
in-process session API used by the dashboard's demo views. Sessions live in
memory; the demo is intended for a single local user.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import chess
import torch

from .encoding import decode_move
from .mcts import MCTS, MCTSConfig
from .model import Chess960Net, ModelConfig
from .stockfish import (
    StockfishOptions,
    StockfishPlayer,
    download_stockfish,
)
from .train import load_checkpoint


# ============================================================
# Engine: lazy-loaded NN + MCTS wrapper
# ============================================================


@dataclass
class EngineConfig:
    checkpoint: Path
    device: str = "cuda"
    n_simulations: int = 100
    c_puct: float = 2.0


class Engine:
    """Loads the network once and provides MCTS-backed move selection.

    Thread-safe: a single lock serializes MCTS runs so the underlying network
    isn't used concurrently.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            print("[demo] CUDA not available; falling back to CPU.")
            device_str = "cpu"
        else:
            device_str = cfg.device
        self.device = torch.device(device_str)
        self.net = Chess960Net(ModelConfig())
        load_checkpoint(cfg.checkpoint, net=self.net, map_location=str(self.device))
        self.net.to(self.device)
        self.net.eval()
        self._lock = threading.Lock()

    def choose_move(
        self,
        board: chess.Board,
        n_simulations: int | None = None,
    ) -> tuple[chess.Move | None, float]:
        """Pick the engine's move for the current position.

        Returns ``(move, root_value)`` where ``root_value`` is the MCTS root
        Q-value from the side-to-move's perspective (range [-1, 1]).
        """
        with self._lock:
            mcts_cfg = MCTSConfig(
                n_simulations=n_simulations or self.cfg.n_simulations,
                c_puct=self.cfg.c_puct,
                root_dirichlet_eps=0.0,
            )
            mcts = MCTS(self.net, mcts_cfg, device=self.device)
            root = mcts.search(board)
            # Argmax of visit counts.
            best_child_action = None
            best_visits = -1
            for action, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_child_action = action
            if best_child_action is None:
                return None, 0.0
            move = decode_move(best_child_action, board)
            return move, float(root.value())


# ============================================================
# Play-vs-engine session
# ============================================================


@dataclass
class PlayGame:
    game_id: str
    board: chess.Board
    user_color: chess.Color
    starting_position_index: int
    move_history_san: list[str] = field(default_factory=list)
    result: str | None = None  # "1-0" / "0-1" / "1/2-1/2" / None
    termination: str | None = None

    def public_dict(self) -> dict:
        # Legal moves are sent to the client so it can highlight destinations
        # when the user picks up a piece (chess.com-style dot overlay). For
        # chess960 castling, python-chess emits the "king takes rook" form.
        legal_uci = [m.uci() for m in self.board.legal_moves]
        return {
            "game_id": self.game_id,
            "fen": self.board.fen(),
            "side_to_move": "white" if self.board.turn == chess.WHITE else "black",
            "user_color": "white" if self.user_color == chess.WHITE else "black",
            "starting_position_index": self.starting_position_index,
            "move_history_san": list(self.move_history_san),
            "is_game_over": self.board.is_game_over(claim_draw=True),
            "result": self.result,
            "termination": self.termination,
            "in_check": self.board.is_check(),
            "legal_moves": legal_uci,
        }


class PlayGameRegistry:
    def __init__(self) -> None:
        self._games: dict[str, PlayGame] = {}
        self._lock = threading.Lock()

    def new_game(
        self,
        user_color: chess.Color,
        starting_position_index: int,
    ) -> PlayGame:
        board = chess.Board(chess960=True)
        board.set_chess960_pos(starting_position_index)
        game = PlayGame(
            game_id=uuid.uuid4().hex[:8],
            board=board,
            user_color=user_color,
            starting_position_index=starting_position_index,
        )
        with self._lock:
            self._games[game.game_id] = game
        return game

    def get(self, game_id: str) -> PlayGame | None:
        with self._lock:
            return self._games.get(game_id)

    def delete(self, game_id: str) -> None:
        with self._lock:
            self._games.pop(game_id, None)


def finalize_if_terminal(game: PlayGame) -> None:
    """Set result/termination on the game if the position is terminal."""
    if game.result is not None:
        return
    outcome = game.board.outcome(claim_draw=True)
    if outcome is not None:
        game.result = outcome.result()
        game.termination = outcome.termination.name


# ============================================================
# Watch (NN vs Stockfish) session
# ============================================================


@dataclass
class WatchMove:
    ply: int
    san: str
    uci: str
    fen_after: str
    side_that_moved: str  # "white" or "black"
    actor: str  # "nn" or "stockfish"
    elapsed_ms: int


@dataclass
class WatchMatch:
    match_id: str
    starting_position_index: int
    nn_plays_white: bool
    skill_level: int
    n_simulations: int
    movetime_s: float
    moves: list[WatchMove] = field(default_factory=list)
    finished: bool = False
    result: str | None = None
    error: str | None = None
    starting_fen: str = ""

    def init_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "fen": self.starting_fen,
            "starting_position_index": self.starting_position_index,
            "nn_plays_white": self.nn_plays_white,
            "skill_level": self.skill_level,
            "n_simulations": self.n_simulations,
        }


class WatchMatchRegistry:
    def __init__(self) -> None:
        self._matches: dict[str, WatchMatch] = {}
        self._lock = threading.Lock()

    def add(self, m: WatchMatch) -> None:
        with self._lock:
            self._matches[m.match_id] = m

    def get(self, mid: str) -> WatchMatch | None:
        with self._lock:
            return self._matches.get(mid)


def run_watch_match(
    match: WatchMatch,
    engine: Engine,
    stockfish_path: Path,
    *,
    max_plies: int = 250,
    on_move: Callable[[WatchMove], None] | None = None,
) -> None:
    """Play a single NN-vs-Stockfish game synchronously.

    Calls ``on_move`` after each move is appended to ``match.moves``.
    """
    try:
        board = chess.Board(chess960=True)
        board.set_chess960_pos(match.starting_position_index)
        opts = StockfishOptions(
            skill=match.skill_level, movetime_s=match.movetime_s
        )
        with StockfishPlayer(stockfish_path, opts) as sf:
            ply = 0
            while ply < max_plies and not board.is_game_over(claim_draw=True):
                nn_to_move = (board.turn == chess.WHITE) == match.nn_plays_white
                t0 = time.monotonic()
                if nn_to_move:
                    move, _ = engine.choose_move(
                        board, n_simulations=match.n_simulations
                    )
                    actor = "nn"
                else:
                    move = sf.play(board)
                    actor = "stockfish"
                if move is None or move not in board.legal_moves:
                    match.error = f"illegal move produced by {actor}"
                    break
                san = board.san(move)
                side_that_moved = "white" if board.turn == chess.WHITE else "black"
                board.push(move)
                wm = WatchMove(
                    ply=ply,
                    san=san,
                    uci=move.uci(),
                    fen_after=board.fen(),
                    side_that_moved=side_that_moved,
                    actor=actor,
                    elapsed_ms=int((time.monotonic() - t0) * 1000),
                )
                match.moves.append(wm)
                if on_move is not None:
                    on_move(wm)
                ply += 1
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                match.result = outcome.result()
            else:
                match.result = "*"  # unresolved (hit max_plies)
    except Exception as e:  # noqa: BLE001
        match.error = f"{type(e).__name__}: {e}"
    finally:
        match.finished = True


# ============================================================
# Convenience: serialize a WatchMove for SSE
# ============================================================


def watchmove_to_json_dict(wm: WatchMove) -> dict:
    return asdict(wm)
