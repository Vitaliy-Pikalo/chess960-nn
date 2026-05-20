"""Self-play game generation for AlphaZero-style RL.

Workflow:
    1. Sample a random Chess960 starting position (SP 0..959).
    2. Play to game over, using MCTS with Dirichlet noise at the root.
    3. Temperature schedule: T=1 for the first ``temperature_threshold`` plies
       to encourage move diversity, T=0 afterwards for sharp play.
    4. Record (state, MCTS visit distribution, side-to-move) at every ply.
    5. Once the game ends, fill in the value target for each position as the
       outcome from that ply's side-to-move POV.

Resulting shards have the same schema as the supervised-pretrain shards
(``states``, ``actions``, ``values``) PLUS an extra ``policies`` array of
shape ``(N, ACTION_SPACE_SIZE)``. The pretrain training loop ignores the
policies column; the RL trainer (Phase 8) will consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chess960_nn.encoding import (
    ACTION_SPACE_SIZE,
    NUM_BOARD_PLANES,
    decode_move,
    encode_board,
)
from chess960_nn.mcts import MCTS, MCTSConfig, terminal_value
from chess960_nn.model import Chess960Net

# ============================================================
# Config
# ============================================================


@dataclass
class SelfPlayConfig:
    """Hyperparameters for self-play game generation."""

    n_simulations: int = 200
    c_puct: float = 2.0

    # Dirichlet noise at the root for exploration
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Temperature schedule
    temperature_threshold: int = 30  # number of opening plies with T=high
    temperature_high: float = 1.0
    temperature_low: float = 0.0

    # Safety net: cap game length
    max_plies: int = 400

    # Resign: if STM Q drops below this for N consecutive plies, resign.
    # Set to None to disable resignation.
    resign_threshold: float | None = -0.9
    resign_consecutive: int = 10


# ============================================================
# Game record
# ============================================================


@dataclass
class GameRecord:
    """One self-play game's training data."""

    states: np.ndarray   # (N, NUM_BOARD_PLANES, 8, 8) float16
    policies: np.ndarray  # (N, ACTION_SPACE_SIZE) float32 - MCTS visit dists
    values: np.ndarray   # (N,) int8 - +1 / 0 / -1 from STM POV
    actions: np.ndarray  # (N,) int32 - action actually played (the move taken)
    outcome: int         # game result from white POV: +1, -1, 0
    plies: int           # number of moves played
    sp_index: int        # which 960 SP was used
    resigned: bool       # whether the game ended via resignation


# ============================================================
# Single game
# ============================================================


def play_game(
    net: Chess960Net,
    *,
    cfg: SelfPlayConfig | None = None,
    sp_index: int | None = None,
    rng: np.random.Generator | None = None,
    device: str = "cuda",
) -> GameRecord:
    """Play one full self-play game and return its training data.

    Args:
        net: policy/value network (already on the right device).
        cfg: self-play hyperparameters.
        sp_index: Chess960 starting position 0..959. If None, sampled uniformly.
        rng: numpy RNG for reproducibility.
        device: torch device for MCTS evaluation.
    """
    cfg = cfg or SelfPlayConfig()
    rng = rng if rng is not None else np.random.default_rng()
    if sp_index is None:
        sp_index = int(rng.integers(0, 960))

    board = chess.Board(chess960=True)
    board.set_chess960_pos(sp_index)

    mcts_cfg = MCTSConfig(
        n_simulations=cfg.n_simulations,
        c_puct=cfg.c_puct,
        root_dirichlet_alpha=cfg.dirichlet_alpha,
        root_dirichlet_eps=cfg.dirichlet_eps,
    )
    mcts = MCTS(net, mcts_cfg, device=device)

    states_buf: list[np.ndarray] = []
    policies_buf: list[np.ndarray] = []
    actions_buf: list[int] = []
    to_play_buf: list[bool] = []  # True if white-to-move at that ply

    resigned = False
    resigning_color: chess.Color | None = None
    losing_streak = 0

    for ply in range(cfg.max_plies):
        if terminal_value(board) is not None:
            break
        if board.is_game_over(claim_draw=True):
            break

        root = mcts.search(board)

        # Record state + policy target (visit distribution, always T=1 for the
        # *target*; only the *action selection* uses the temperature schedule).
        state = encode_board(board).astype(np.float16)
        policy = mcts.visit_distribution(root, temperature=1.0).astype(np.float32)

        # Resignation check (based on root Q from STM POV)
        if cfg.resign_threshold is not None:
            root_q = root.value()  # already from STM POV
            if root_q < cfg.resign_threshold:
                losing_streak += 1
                if losing_streak >= cfg.resign_consecutive:
                    resigned = True
                    resigning_color = board.turn
                    break
            else:
                losing_streak = 0

        # Action selection
        temp = (
            cfg.temperature_high
            if ply < cfg.temperature_threshold
            else cfg.temperature_low
        )
        action = mcts.best_action(root, temperature=temp, rng=rng)
        move = decode_move(action, board)
        if move is None:
            # Defensive: should never happen because mcts only expands legal moves
            break

        states_buf.append(state)
        policies_buf.append(policy)
        actions_buf.append(action)
        to_play_buf.append(board.turn == chess.WHITE)

        board.push(move)

    # Determine outcome from white POV
    if resigned:
        outcome_white = -1 if resigning_color == chess.WHITE else 1
    else:
        outcome_white = _outcome_white_pov(board)

    # Fill per-ply value targets from each ply's STM POV
    values = np.array(
        [outcome_white if w else -outcome_white for w in to_play_buf],
        dtype=np.int8,
    )

    if states_buf:
        states_np = np.stack(states_buf)
        policies_np = np.stack(policies_buf)
        actions_np = np.asarray(actions_buf, dtype=np.int32)
    else:
        states_np = np.zeros((0, NUM_BOARD_PLANES, 8, 8), dtype=np.float16)
        policies_np = np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32)
        actions_np = np.zeros((0,), dtype=np.int32)

    return GameRecord(
        states=states_np,
        policies=policies_np,
        values=values,
        actions=actions_np,
        outcome=outcome_white,
        plies=len(states_buf),
        sp_index=sp_index,
        resigned=resigned,
    )


def _outcome_white_pov(board: chess.Board) -> int:
    """+1 white wins, -1 black wins, 0 draw (including non-terminal cap-out)."""
    if board.is_checkmate():
        # Side-to-move was mated -> opposite color wins
        return -1 if board.turn == chess.WHITE else 1
    # Stalemate, insufficient material, 50-move, repetition, or max-plies => draw
    return 0


# ============================================================
# Shard writer (RL data, schema-compatible with pretrain + extra policies)
# ============================================================


def write_selfplay_shard(path: Path, games: list[GameRecord]) -> dict[str, int]:
    """Concatenate game records and write a single .npz shard.

    Returns counters: positions and games written.
    """
    if not games:
        raise ValueError("Refusing to write empty self-play shard")

    states_list = [g.states for g in games if g.plies > 0]
    policies_list = [g.policies for g in games if g.plies > 0]
    values_list = [g.values for g in games if g.plies > 0]
    actions_list = [g.actions for g in games if g.plies > 0]

    states = np.concatenate(states_list, axis=0) if states_list else np.zeros(
        (0, NUM_BOARD_PLANES, 8, 8), dtype=np.float16
    )
    policies = np.concatenate(policies_list, axis=0) if policies_list else np.zeros(
        (0, ACTION_SPACE_SIZE), dtype=np.float32
    )
    values = np.concatenate(values_list, axis=0) if values_list else np.zeros(
        (0,), dtype=np.int8
    )
    actions = np.concatenate(actions_list, axis=0) if actions_list else np.zeros(
        (0,), dtype=np.int32
    )

    np.savez_compressed(
        path,
        states=states,
        policies=policies,
        values=values,
        actions=actions,
    )
    return {"positions": int(values.shape[0]), "games": len(games)}
