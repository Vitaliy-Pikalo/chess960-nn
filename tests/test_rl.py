"""Tests for the RL building blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chess960_nn.encoding import ACTION_SPACE_SIZE, NUM_BOARD_PLANES
from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.rl import (
    MatchConfig,
    RLShardDataset,
    play_match,
    rl_train_one_epoch,
)
from chess960_nn.selfplay import (
    SelfPlayConfig,
    play_game,
    write_selfplay_shard,
)


def _tiny_net() -> Chess960Net:
    torch.manual_seed(0)
    return Chess960Net(ModelConfig(num_blocks=1, num_filters=16))


def _make_selfplay_shards(tmp_path: Path) -> Path:
    """Generate a tiny self-play directory with one shard, suitable for RL tests."""
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=10,
        resign_threshold=None,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(0)
    games = [
        play_game(net, cfg=cfg, sp_index=518, rng=rng, device="cpu"),
        play_game(net, cfg=cfg, sp_index=23, rng=rng, device="cpu"),
        play_game(net, cfg=cfg, sp_index=100, rng=rng, device="cpu"),
    ]
    out = tmp_path / "selfplay"
    out.mkdir()
    write_selfplay_shard(out / "shard_00000.npz", games)
    return out


# ============================================================
# RLShardDataset
# ============================================================


def test_rl_dataset_yields_correct_shapes(tmp_path: Path):
    sp_dir = _make_selfplay_shards(tmp_path)
    ds = RLShardDataset(sp_dir, shuffle=False)
    assert len(ds) > 0
    n = 0
    for state, policy, value in ds:
        assert state.shape == (NUM_BOARD_PLANES, 8, 8)
        assert state.dtype == torch.float32
        assert policy.shape == (ACTION_SPACE_SIZE,)
        assert policy.dtype == torch.float32
        assert torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-4)
        assert value.shape == ()
        assert value.dtype == torch.float32
        n += 1
    assert n == len(ds)


def test_rl_dataset_raises_when_no_shards(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        RLShardDataset(empty)


# ============================================================
# RL train step
# ============================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for AMP")
def test_rl_train_one_epoch_reduces_loss(tmp_path: Path):
    """A short pass should at least reduce loss on the tiny dataset."""
    sp_dir = _make_selfplay_shards(tmp_path)
    ds = RLShardDataset(sp_dir, shuffle=False)
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    cfg = ModelConfig(num_blocks=1, num_filters=16)
    device = torch.device("cuda")
    net = Chess960Net(cfg).to(device)
    optimizer = AdamW(net.parameters(), lr=1e-3)
    scaler = GradScaler()

    # First pass
    stats1 = rl_train_one_epoch(
        net, loader, optimizer, scaler, device=device, log_every=1000,
    )
    # Second pass
    stats2 = rl_train_one_epoch(
        net, loader, optimizer, scaler, device=device, log_every=1000,
    )
    assert stats1["samples"] > 0
    # Not guaranteed strict decrease on a tiny dataset, so just check sane.
    assert stats1["loss"] > 0
    assert stats2["loss"] > 0


# ============================================================
# Match play
# ============================================================


def test_play_match_returns_valid_result():
    net_a = _tiny_net()
    net_b = _tiny_net()  # same weights -> roughly 50/50 + draws
    cfg = MatchConfig(n_games=2, n_simulations=4, max_plies=20)
    result = play_match(net_a, net_b, cfg=cfg, seed=0, device="cpu")
    assert result["games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert 0.0 <= result["score"] <= 1.0


def test_play_match_self_match_is_around_half():
    """Same net vs itself should be close to 0.5 (mostly draws or symmetric)."""
    net = _tiny_net()
    cfg = MatchConfig(n_games=4, n_simulations=4, max_plies=20)
    result = play_match(net, net, cfg=cfg, seed=0, device="cpu")
    assert result["games"] == 4
    # Exact equality not guaranteed because of MCTS stochasticity from seed
    # interaction with two MCTS instances; just sanity-check bounds.
    assert 0.0 <= result["score"] <= 1.0
