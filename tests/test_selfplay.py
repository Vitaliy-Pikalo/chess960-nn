"""Tests for self-play game generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess960_nn.encoding import ACTION_SPACE_SIZE, NUM_BOARD_PLANES
from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.selfplay import (
    SelfPlayConfig,
    play_game,
    write_selfplay_shard,
)


def _tiny_net() -> Chess960Net:
    torch.manual_seed(0)
    return Chess960Net(ModelConfig(num_blocks=1, num_filters=16))


def test_play_game_returns_valid_record():
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=8,
        max_plies=20,
        resign_threshold=None,
        temperature_threshold=5,
        dirichlet_eps=0.25,
    )
    rng = np.random.default_rng(0)
    rec = play_game(net, cfg=cfg, sp_index=518, rng=rng, device="cpu")

    assert rec.plies <= cfg.max_plies
    assert rec.plies > 0
    assert rec.states.shape == (rec.plies, NUM_BOARD_PLANES, 8, 8)
    assert rec.states.dtype == np.float16
    assert rec.policies.shape == (rec.plies, ACTION_SPACE_SIZE)
    assert rec.policies.dtype == np.float32
    assert rec.actions.shape == (rec.plies,)
    assert rec.values.shape == (rec.plies,)
    assert rec.values.dtype == np.int8
    # Each policy row sums to ~1 (it's a probability distribution)
    sums = rec.policies.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)
    # Outcome is -1, 0, or +1
    assert rec.outcome in (-1, 0, 1)
    # SP index is what we requested
    assert rec.sp_index == 518


def test_play_game_value_targets_are_stm_relative():
    """The value at each ply equals outcome from that ply's STM POV.

    We force max_plies low so the game ends in a draw (outcome=0), then check
    every value is 0.
    """
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=6,
        resign_threshold=None,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(1)
    rec = play_game(net, cfg=cfg, sp_index=518, rng=rng, device="cpu")
    # With max_plies=6 the game won't end naturally -> outcome=0 (draw cap)
    assert rec.outcome == 0
    assert (rec.values == 0).all()


def test_play_game_random_sp_when_none():
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=5,
        resign_threshold=None,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(42)
    rec1 = play_game(net, cfg=cfg, sp_index=None, rng=rng, device="cpu")
    rec2 = play_game(net, cfg=cfg, sp_index=None, rng=rng, device="cpu")
    # Different starts almost surely -> different SP indices
    assert 0 <= rec1.sp_index < 960
    assert 0 <= rec2.sp_index < 960


def test_play_game_resignation_never_triggers_with_extreme_threshold():
    """With a threshold below the value range, resignation cannot fire."""
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=20,
        # Q is in [-1, +1]; -2.0 is below that range so Q < -2.0 is never true.
        resign_threshold=-2.0,
        resign_consecutive=1,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(0)
    rec = play_game(net, cfg=cfg, sp_index=518, rng=rng, device="cpu")
    assert not rec.resigned


def test_play_game_resignation_always_triggers_with_loose_threshold():
    """With a threshold above the value range, resignation fires immediately."""
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=100,
        # Q is in [-1, +1]; +2.0 is above so Q < +2.0 is always true.
        resign_threshold=2.0,
        resign_consecutive=1,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(0)
    rec = play_game(net, cfg=cfg, sp_index=518, rng=rng, device="cpu")
    assert rec.resigned


def test_write_selfplay_shard_roundtrip(tmp_path: Path):
    net = _tiny_net()
    cfg = SelfPlayConfig(
        n_simulations=4,
        max_plies=8,
        resign_threshold=None,
        temperature_threshold=0,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(0)
    games = [
        play_game(net, cfg=cfg, sp_index=i, rng=rng, device="cpu")
        for i in (518, 23, 100)
    ]
    shard_path = tmp_path / "shard_00000.npz"
    stats = write_selfplay_shard(shard_path, games)
    assert shard_path.exists()
    assert stats["games"] == 3
    assert stats["positions"] > 0

    # Round-trip: re-load and check shapes/sums
    with np.load(shard_path) as d:
        states = d["states"]
        policies = d["policies"]
        values = d["values"]
        actions = d["actions"]
    n = sum(g.plies for g in games)
    assert states.shape == (n, NUM_BOARD_PLANES, 8, 8)
    assert policies.shape == (n, ACTION_SPACE_SIZE)
    assert values.shape == (n,)
    assert actions.shape == (n,)


def test_write_empty_shard_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_selfplay_shard(tmp_path / "empty.npz", [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_play_game_on_gpu_runs():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg).cuda()
    sp_cfg = SelfPlayConfig(
        n_simulations=8,
        max_plies=10,
        resign_threshold=None,
        temperature_threshold=2,
        dirichlet_eps=0.25,
    )
    rng = np.random.default_rng(0)
    rec = play_game(net, cfg=sp_cfg, sp_index=518, rng=rng, device="cuda")
    assert rec.plies > 0
