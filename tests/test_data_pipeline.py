"""Tests for the data pipeline using a small embedded synthetic PGN."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from chess960_nn.data.dataset import ChessShardDataset, InMemoryChessDataset
from chess960_nn.data.pipeline import (
    GameFilter,
    build_dataset,
    iter_games_from_pgn,
    iter_training_tuples,
    lichess_960_url,
    read_shard,
    write_shard,
)
from chess960_nn.encoding import ACTION_SPACE_SIZE, NUM_BOARD_PLANES

# ============================================================
# Synthetic PGN fixtures
# ============================================================

# Three games:
#   - G1: a clean Chess960 game, both players 2200+, decisive
#   - G2: standard variant (NOT Chess960) - should be filtered out
#   - G3: a Chess960 game but both players below threshold - filtered out
SYNTHETIC_PGN = """[Event "Synthetic 1"]
[Site "?"]
[Date "2025.01.01"]
[Round "?"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[Variant "Chess960"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
[SetUp "1"]
[WhiteElo "2300"]
[BlackElo "2250"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O Bd6 6. d4 exd4 7. Nxd4 Nf6
8. Nc3 O-O 9. Bg5 h6 10. Bh4 1-0

[Event "Standard, not 960"]
[Site "?"]
[Date "2025.01.02"]
[White "Charlie"]
[Black "Dave"]
[Result "0-1"]
[WhiteElo "2400"]
[BlackElo "2400"]

1. d4 d5 2. c4 c6 3. Nc3 Nf6 4. e3 e6 5. Nf3 Nbd7 6. Bd3 Bd6 7. O-O O-O
8. e4 dxe4 9. Nxe4 Nxe4 10. Bxe4 0-1

[Event "Low rated 960"]
[Site "?"]
[Date "2025.01.03"]
[White "Eve"]
[Black "Frank"]
[Result "1/2-1/2"]
[Variant "Chess960"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
[SetUp "1"]
[WhiteElo "1200"]
[BlackElo "1300"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Na5 10. Bc2 1/2-1/2
"""


# ============================================================
# Filtering
# ============================================================


def test_filter_keeps_only_960_with_min_rating():
    stream = io.StringIO(SYNTHETIC_PGN)
    games = list(iter_games_from_pgn(stream, GameFilter(min_rating=1800)))
    # Only G1 should pass: it's Chess960 and both players >= 1800.
    assert len(games) == 1
    assert games[0].headers["Event"] == "Synthetic 1"
    assert games[0].headers["Variant"] == "Chess960"


def test_filter_min_rating_strictly_enforced():
    stream = io.StringIO(SYNTHETIC_PGN)
    games = list(iter_games_from_pgn(stream, GameFilter(min_rating=2400)))
    # Nobody qualifies at 2400 (and Charlie/Dave's standard game is filtered anyway).
    assert len(games) == 0


def test_filter_no_variant_constraint_lets_standard_through():
    stream = io.StringIO(SYNTHETIC_PGN)
    games = list(
        iter_games_from_pgn(
            stream,
            GameFilter(min_rating=1800, require_variant=None),
        )
    )
    # G1 (2300/2250) + G2 (2400/2400) both qualify; G3 too low.
    assert len(games) == 2


def test_filter_short_game_rejected():
    stream = io.StringIO(SYNTHETIC_PGN)
    games = list(iter_games_from_pgn(stream, GameFilter(min_rating=1800, min_moves=100)))
    assert len(games) == 0


# ============================================================
# Tensorisation
# ============================================================


def test_iter_training_tuples_shape_and_value():
    stream = io.StringIO(SYNTHETIC_PGN)
    game = next(iter_games_from_pgn(stream, GameFilter(min_rating=1800)))
    tuples = list(iter_training_tuples(game))

    assert len(tuples) > 0
    for t in tuples:
        assert t.state.shape == (NUM_BOARD_PLANES, 8, 8)
        assert t.state.dtype == np.float16
        assert 0 <= t.action < ACTION_SPACE_SIZE
        assert t.value in (-1, 0, 1)

    # G1 is 1-0 (white wins). Position 0 is white-to-move -> value +1.
    # Position 1 is black-to-move -> value -1.
    assert tuples[0].value == 1
    assert tuples[1].value == -1


def test_iter_training_tuples_skips_games_without_result():
    """Games with Result=* yield no tuples (need a value target)."""
    pgn_no_result = """[Event "Aborted"]
[White "A"]
[Black "B"]
[Result "*"]
[Variant "Chess960"]
[WhiteElo "2200"]
[BlackElo "2200"]

1. e4 e5 *
"""
    stream = io.StringIO(pgn_no_result)
    # Need allow_no_result=True or the filter drops it; either way no tuples.
    games = list(iter_games_from_pgn(stream, GameFilter(allow_no_result=True, min_moves=2)))
    assert len(games) == 1
    assert list(iter_training_tuples(games[0])) == []


# ============================================================
# Shard write / read roundtrip
# ============================================================


def test_write_read_shard_roundtrip(tmp_path: Path):
    stream = io.StringIO(SYNTHETIC_PGN)
    game = next(iter_games_from_pgn(stream, GameFilter(min_rating=1800)))
    tuples = list(iter_training_tuples(game))
    shard_path = tmp_path / "shard_00000.npz"

    write_shard(shard_path, tuples)
    data = read_shard(shard_path)

    n = len(tuples)
    assert data["states"].shape == (n, NUM_BOARD_PLANES, 8, 8)
    assert data["states"].dtype == np.float16
    assert data["actions"].shape == (n,)
    assert data["actions"].dtype == np.int32
    assert data["values"].shape == (n,)
    assert data["values"].dtype == np.int8

    # Round-trip content
    for i, t in enumerate(tuples):
        np.testing.assert_array_equal(data["states"][i], t.state)
        assert int(data["actions"][i]) == t.action
        assert int(data["values"][i]) == t.value


def test_write_empty_shard_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "empty.npz", [])


# ============================================================
# build_dataset end-to-end (using a temp PGN file)
# ============================================================


def test_build_dataset_end_to_end(tmp_path: Path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SYNTHETIC_PGN, encoding="utf-8")
    out_dir = tmp_path / "out"

    stats = build_dataset(
        pgn_path,
        out_dir,
        filt=GameFilter(min_rating=1800),
        shard_size=5,  # small shard so we get multiple
    )

    assert stats["games"] == 1
    assert stats["positions"] > 0
    assert stats["shards"] >= 1

    shard_files = sorted(out_dir.glob("shard_*.npz"))
    assert len(shard_files) == stats["shards"]


def test_build_dataset_respects_max_positions(tmp_path: Path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SYNTHETIC_PGN, encoding="utf-8")
    out_dir = tmp_path / "out"

    stats = build_dataset(
        pgn_path,
        out_dir,
        filt=GameFilter(min_rating=1800),
        shard_size=100,
        max_positions=5,
    )
    assert stats["positions"] <= 5


# ============================================================
# Dataset classes
# ============================================================


def _make_shards(tmp_path: Path) -> Path:
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SYNTHETIC_PGN, encoding="utf-8")
    out_dir = tmp_path / "shards"
    build_dataset(
        pgn_path,
        out_dir,
        filt=GameFilter(min_rating=1800),
        shard_size=5,  # multiple small shards
    )
    return out_dir


def test_chess_shard_dataset_indexing(tmp_path: Path):
    out_dir = _make_shards(tmp_path)
    ds = ChessShardDataset(out_dir)
    assert len(ds) > 0

    # Sequential pass
    seen = 0
    for i in range(len(ds)):
        state, action, value = ds[i]
        assert state.shape == (NUM_BOARD_PLANES, 8, 8)
        assert state.dtype.is_floating_point
        assert isinstance(action, int)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert value in (-1.0, 0.0, 1.0)
        seen += 1
    assert seen == len(ds)


def test_in_memory_dataset_indexing(tmp_path: Path):
    out_dir = _make_shards(tmp_path)
    ds = InMemoryChessDataset(out_dir)
    assert len(ds) > 0

    state, action, value = ds[0]
    assert state.shape == (NUM_BOARD_PLANES, 8, 8)
    assert state.dtype.is_floating_point
    assert isinstance(action, int)
    assert value in (-1.0, 0.0, 1.0)


def test_datasets_agree(tmp_path: Path):
    """The two Dataset implementations must yield identical items at every index."""
    out_dir = _make_shards(tmp_path)
    a = ChessShardDataset(out_dir)
    b = InMemoryChessDataset(out_dir)
    assert len(a) == len(b)
    for i in range(len(a)):
        sa, aa, va = a[i]
        sb, ab, vb = b[i]
        assert (sa == sb).all()
        assert aa == ab
        assert va == vb


def test_dataset_raises_when_no_shards(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ChessShardDataset(empty)
    with pytest.raises(FileNotFoundError):
        InMemoryChessDataset(empty)


# ============================================================
# Misc
# ============================================================


def test_lichess_url_format():
    url = lichess_960_url("2025-04")
    assert url.endswith("lichess_db_chess960_rated_2025-04.pgn.zst")
    assert url.startswith("https://")
