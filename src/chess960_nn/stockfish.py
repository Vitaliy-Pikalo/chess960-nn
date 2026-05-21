"""Stockfish auto-download + UCI wrapper for chess960 eval.

* ``download_stockfish``      – pull + extract the official stockfish binary
* ``StockfishPlayer``         – context-manager UCI wrapper (chess960-enabled)
* ``play_vs_stockfish_game``  – one chess960 game NN vs stockfish
* ``evaluate_vs_stockfish``   – multi-skill-level match harness with Elo estimate
* ``skill_to_elo``            – mapping used for the strength estimate
"""

from __future__ import annotations

import math
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chess960_nn.encoding import decode_move
from chess960_nn.mcts import MCTS, MCTSConfig, terminal_value
from chess960_nn.model import Chess960Net

# ============================================================
# Constants
# ============================================================

# Stable github release for Stockfish 17.1. Update if a newer one is desired.
DEFAULT_STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-windows-x86-64-avx2.zip"
)

# Approximate Stockfish skill -> Elo mapping. Linear interp between 1320 (skill 0)
# and 3190 (skill 20). These are rough; treat estimates as approximate.
_SKILL_ELO_MIN = 1320
_SKILL_ELO_MAX = 3190


def skill_to_elo(skill: int) -> int:
    """Approximate Elo for a given Stockfish skill level (0-20)."""
    skill = max(0, min(20, int(skill)))
    return int(_SKILL_ELO_MIN + (skill / 20.0) * (_SKILL_ELO_MAX - _SKILL_ELO_MIN))


# ============================================================
# Downloader
# ============================================================


def download_stockfish(
    dest_dir: Path,
    *,
    url: str = DEFAULT_STOCKFISH_URL,
    chunk_size: int = 1 << 20,
) -> Path:
    """Download + extract stockfish into ``dest_dir`` and return the .exe path.

    Idempotent: if an existing ``stockfish*.exe`` is already there, return it.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest_dir.rglob("stockfish*.exe"))
    if existing:
        return existing[0]

    zip_path = dest_dir / Path(url).name
    print(f"Downloading {url} -> {zip_path} ...")
    with urllib.request.urlopen(url) as resp, open(zip_path, "wb") as f:  # noqa: S310
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    exes = sorted(dest_dir.rglob("stockfish*.exe"))
    if not exes:
        raise RuntimeError(
            f"Downloaded zip but no stockfish*.exe found under {dest_dir}"
        )
    return exes[0]


# ============================================================
# UCI wrapper
# ============================================================


@dataclass
class StockfishOptions:
    skill: int = 10
    threads: int = 1
    hash_mb: int = 64
    # Time / nodes / depth per move. Only one needs to be set (priority: time
    # > nodes > depth). Default: 100ms per move.
    movetime_s: float | None = 0.1
    nodes: int | None = None
    depth: int | None = None


class StockfishPlayer:
    """Single Stockfish engine subprocess configured for Chess960."""

    def __init__(self, binary_path: Path, options: StockfishOptions | None = None):
        self.options = options or StockfishOptions()
        self.engine = chess.engine.SimpleEngine.popen_uci(str(binary_path))
        self._configure()

    def _configure(self) -> None:
        # NOTE: python-chess auto-manages ``UCI_Chess960`` based on the board's
        # ``chess960`` attribute when ``engine.play(board, ...)`` is called.
        # Setting it explicitly here raises EngineError, so we omit it.
        cfg: dict[str, object] = {
            "Skill Level": int(self.options.skill),
            "Threads": int(self.options.threads),
            "Hash": int(self.options.hash_mb),
        }
        # Some old SF versions don't have Skill Level. Send what they support.
        supported = {k: v for k, v in cfg.items() if k in self.engine.options}
        self.engine.configure(supported)

    def play(self, board: chess.Board) -> chess.Move | None:
        if self.options.movetime_s is not None:
            limit = chess.engine.Limit(time=float(self.options.movetime_s))
        elif self.options.nodes is not None:
            limit = chess.engine.Limit(nodes=int(self.options.nodes))
        elif self.options.depth is not None:
            limit = chess.engine.Limit(depth=int(self.options.depth))
        else:
            limit = chess.engine.Limit(time=0.1)
        result = self.engine.play(board, limit)
        return result.move

    def close(self) -> None:
        try:
            self.engine.quit()
        except chess.engine.EngineError:
            pass

    def __enter__(self) -> StockfishPlayer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def stockfish_player(binary_path: Path, options: StockfishOptions | None = None):
    """Context manager that yields a configured StockfishPlayer."""
    p = StockfishPlayer(binary_path, options)
    try:
        yield p
    finally:
        p.close()


# ============================================================
# Single game NN vs Stockfish
# ============================================================


def play_vs_stockfish_game(
    net: Chess960Net,
    stockfish: StockfishPlayer,
    *,
    sp_index: int,
    nn_plays_white: bool,
    n_simulations: int = 200,
    c_puct: float = 2.0,
    max_plies: int = 300,
    device: str = "cuda",
) -> int:
    """Play one chess960 game. Returns outcome from white POV (+1/-1/0)."""
    board = chess.Board(chess960=True)
    board.set_chess960_pos(sp_index)

    mcts_cfg = MCTSConfig(
        n_simulations=n_simulations, c_puct=c_puct, root_dirichlet_eps=0.0
    )
    mcts = MCTS(net, mcts_cfg, device=device)

    for _ in range(max_plies):
        if terminal_value(board) is not None or board.is_game_over(claim_draw=True):
            break
        nn_to_move = (board.turn == chess.WHITE) == nn_plays_white
        if nn_to_move:
            root = mcts.search(board)
            action = mcts.best_action(root, temperature=0)
            move = decode_move(action, board)
        else:
            move = stockfish.play(board)
        if move is None or move not in board.legal_moves:
            # Defensive: end the game as a loss for the side that produced the
            # illegal move (rare). Treat as a draw to keep things sane here.
            return 0
        board.push(move)

    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    return 0


# ============================================================
# Multi-skill-level evaluation
# ============================================================


def _elo_from_score(score: float, opp_elo: int, *, eps: float = 1e-3) -> int:
    """Estimate Elo from a match score using the standard logistic relation.

    score = 1 / (1 + 10 ** ((opp_elo - my_elo) / 400))
    => my_elo - opp_elo = -400 * log10(1/score - 1)
    """
    s = max(eps, min(1.0 - eps, score))
    elo_diff = -400.0 * math.log10(1.0 / s - 1.0)
    return int(round(opp_elo + elo_diff))


def evaluate_vs_stockfish(
    net: Chess960Net,
    binary_path: Path,
    *,
    skill_levels: list[int] | None = None,
    n_games_per_level: int = 10,
    n_simulations: int = 200,
    movetime_s: float = 0.1,
    max_plies: int = 300,
    threads: int = 1,
    seed: int = 0,
    device: str = "cuda",
    on_game_done: callable | None = None,
) -> dict[int, dict[str, float | int]]:
    """Run matches at multiple skill levels. Returns per-level results dict.

    Each level: ``{"wins": N, "losses": N, "draws": N, "score": float,
                    "games": N, "sf_elo": int, "nn_elo_estimate": int}``.
    """
    if skill_levels is None:
        skill_levels = [0, 5, 10]

    rng = np.random.default_rng(seed)
    results: dict[int, dict[str, float | int]] = {}

    for skill in skill_levels:
        opts = StockfishOptions(
            skill=skill, threads=threads, movetime_s=movetime_s
        )
        with stockfish_player(binary_path, opts) as sf:
            wins = losses = draws = 0
            for game_idx in range(n_games_per_level):
                sp_index = int(rng.integers(0, 960))
                nn_plays_white = (game_idx % 2 == 0)
                outcome_white = play_vs_stockfish_game(
                    net, sf,
                    sp_index=sp_index,
                    nn_plays_white=nn_plays_white,
                    n_simulations=n_simulations,
                    max_plies=max_plies,
                    device=device,
                )
                if outcome_white == 0:
                    draws += 1
                elif (
                    (outcome_white == 1 and nn_plays_white)
                    or (outcome_white == -1 and not nn_plays_white)
                ):
                    wins += 1
                else:
                    losses += 1
                if on_game_done is not None:
                    on_game_done(skill, game_idx, wins, losses, draws)

        n = n_games_per_level
        score = (wins + 0.5 * draws) / n if n > 0 else 0.5
        sf_elo = skill_to_elo(skill)
        results[skill] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": score,
            "games": n,
            "sf_elo": sf_elo,
            "nn_elo_estimate": _elo_from_score(score, sf_elo),
        }

    return results
