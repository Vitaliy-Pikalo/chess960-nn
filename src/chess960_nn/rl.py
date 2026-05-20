"""AlphaZero-style RL building blocks.

* ``RLShardDataset``  – yields ``(state, policy_target, value)`` from
  self-play shards. Policy target is the MCTS visit distribution (soft),
  unlike pretrain which used one-hot played moves.
* ``rl_train_one_epoch`` – one pass over an RL loader, using KL-style policy
  loss (cross-entropy with soft target) + MSE value loss.
* ``play_match_game`` – play one chess960 game between two networks (no
  Dirichlet noise, temperature 0, deterministic).
* ``play_match`` – run a match of N games with colour alternation, return
  the score (1=win, 0.5=draw, 0=loss) from net_a's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chess960_nn.encoding import (
    decode_move,
)
from chess960_nn.mcts import MCTS, MCTSConfig, terminal_value
from chess960_nn.model import Chess960Net

# ============================================================
# RL dataset (self-play shard format)
# ============================================================


class RLShardDataset(IterableDataset):
    """Streams (state, policy_target, value) from self-play shards.

    Self-play shards contain ``states``, ``policies`` (MCTS visit dist over
    all 4672 actions), ``values``, and ``actions``. RL training uses the
    policy distribution as the target, not the played action.
    """

    def __init__(self, shard_dir: Path | str, *, shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No self-play shards in {self.shard_dir}")
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        self._total = 0
        for shard in self.shards:
            with np.load(shard) as d:
                self._total += int(d["values"].shape[0])

    def __len__(self) -> int:
        return self._total

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shards = list(self.shards)
            worker_id = 0
        else:
            shards = self.shards[worker.id :: worker.num_workers]
            worker_id = worker.id

        rng = np.random.default_rng(self.seed + self._epoch * 1009 + worker_id)
        if self.shuffle:
            rng.shuffle(shards)  # type: ignore[arg-type]

        for shard_path in shards:
            with np.load(shard_path) as d:
                states = d["states"]
                policies = d["policies"]
                values = d["values"]
            n = int(values.shape[0])
            if n == 0:
                continue
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            for idx in order:
                yield (
                    torch.from_numpy(states[idx]).float(),
                    torch.from_numpy(policies[idx]).float(),
                    torch.tensor(float(values[idx]), dtype=torch.float32),
                )


# ============================================================
# RL training step
# ============================================================


def rl_train_one_epoch(
    net: Chess960Net,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: GradScaler | None,
    *,
    device: torch.device,
    value_loss_weight: float = 1.0,
    log_every: int = 50,
) -> dict[str, float]:
    """One pass over the RL loader. Returns averaged metrics."""
    net.train()
    total_n = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for step, (states, policy_targets, values) in enumerate(loader):
        states = states.to(device, non_blocking=True)
        policy_targets = policy_targets.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                policy_logits, value_pred = net(states)
                log_probs = F.log_softmax(policy_logits.float(), dim=-1)
                policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(value_pred, values)
                loss = policy_loss + value_loss_weight * value_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = net(states)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(value_pred, values)
            loss = policy_loss + value_loss_weight * value_loss
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        n = states.size(0)
        total_n += n
        total_loss += float(loss.item()) * n
        total_policy_loss += float(policy_loss.item()) * n
        total_value_loss += float(value_loss.item()) * n

        if step % log_every == 0:
            print(
                f"  step {step:>5d}  "
                f"loss={total_loss/max(1, total_n):.4f}  "
                f"pol={total_policy_loss/max(1, total_n):.4f}  "
                f"val={total_value_loss/max(1, total_n):.4f}"
            )

    if total_n == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "samples": 0}
    return {
        "loss": total_loss / total_n,
        "policy_loss": total_policy_loss / total_n,
        "value_loss": total_value_loss / total_n,
        "samples": total_n,
    }


# ============================================================
# Match play: candidate vs incumbent
# ============================================================


@dataclass
class MatchConfig:
    """Hyperparameters for a head-to-head match."""

    n_games: int = 20
    n_simulations: int = 100
    c_puct: float = 2.0
    max_plies: int = 400


def play_match_game(
    net_white: Chess960Net,
    net_black: Chess960Net,
    *,
    cfg: MatchConfig,
    sp_index: int,
    device_white: str = "cuda",
    device_black: str = "cuda",
) -> int:
    """Play one game between two networks. Returns outcome from white POV."""
    board = chess.Board(chess960=True)
    board.set_chess960_pos(sp_index)

    mcts_cfg = MCTSConfig(
        n_simulations=cfg.n_simulations,
        c_puct=cfg.c_puct,
        root_dirichlet_eps=0.0,  # deterministic for evaluation
    )
    mcts_w = MCTS(net_white, mcts_cfg, device=device_white)
    mcts_b = MCTS(net_black, mcts_cfg, device=device_black)

    for _ in range(cfg.max_plies):
        if terminal_value(board) is not None or board.is_game_over(claim_draw=True):
            break
        mcts = mcts_w if board.turn == chess.WHITE else mcts_b
        root = mcts.search(board)
        action = mcts.best_action(root, temperature=0)
        move = decode_move(action, board)
        if move is None:
            break
        board.push(move)

    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    return 0  # stalemate / insufficient / cap / etc.


def play_match(
    net_a: Chess960Net,
    net_b: Chess960Net,
    *,
    cfg: MatchConfig | None = None,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, float | int]:
    """Run a match. Returns ``{"score": float in [0,1], "wins": int, "losses": int, "draws": int, "games": int}``.

    Score is net_a's: 1 per win, 0.5 per draw, 0 per loss.
    Colours alternate to remove first-mover bias.
    """
    cfg = cfg or MatchConfig()
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    for game_idx in range(cfg.n_games):
        sp_index = int(rng.integers(0, 960))
        if game_idx % 2 == 0:
            white, black = net_a, net_b
            a_is_white = True
        else:
            white, black = net_b, net_a
            a_is_white = False
        outcome_white = play_match_game(
            white, black, cfg=cfg, sp_index=sp_index,
            device_white=device, device_black=device,
        )
        if outcome_white == 0:
            draws += 1
        elif (outcome_white == 1 and a_is_white) or (outcome_white == -1 and not a_is_white):
            wins += 1
        else:
            losses += 1
    n = cfg.n_games
    score = (wins + 0.5 * draws) / n if n > 0 else 0.5
    return {
        "score": score,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": n,
    }
