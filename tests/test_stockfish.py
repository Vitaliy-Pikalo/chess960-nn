"""Tests for the Stockfish wrapper. Tests that need a real binary are
auto-skipped if it isn't downloaded yet."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import chess
import pytest

from chess960_nn.stockfish import (
    StockfishOptions,
    StockfishPlayer,
    _elo_from_score,
    skill_to_elo,
    stockfish_player,
)


def _find_stockfish() -> Path | None:
    # Look in common locations the project might have downloaded it.
    candidates = []
    env = os.environ.get("STOCKFISH_BIN")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("bin/stockfish").glob("**/stockfish*.exe"),
        Path("./bin/stockfish").glob("**/stockfish*.exe"),
    ])
    # Flatten globs
    flat: list[Path] = []
    for c in candidates:
        if isinstance(c, Path):
            if c.exists():
                flat.append(c)
        else:
            flat.extend(c)
    # Also a stockfish on PATH
    which = shutil.which("stockfish")
    if which:
        flat.append(Path(which))
    return flat[0] if flat else None


_SF_PATH = _find_stockfish()
_SF_REQUIRED = pytest.mark.skipif(
    _SF_PATH is None,
    reason="stockfish.exe not found (run DOWNLOAD stockfish.ps1 first)",
)


# ============================================================
# Pure-Python helpers (no binary needed)
# ============================================================


def test_skill_to_elo_endpoints():
    assert skill_to_elo(0) == 1320
    assert skill_to_elo(20) == 3190


def test_skill_to_elo_monotonic():
    last = -1
    for s in range(21):
        cur = skill_to_elo(s)
        assert cur > last
        last = cur


def test_skill_to_elo_clamps():
    assert skill_to_elo(-5) == 1320
    assert skill_to_elo(100) == 3190


def test_elo_from_score_symmetric_around_half():
    # 50% -> equal Elo to opponent
    assert _elo_from_score(0.5, 2000) == 2000
    # Higher score -> higher Elo
    assert _elo_from_score(0.75, 2000) > 2000
    # Lower score -> lower Elo
    assert _elo_from_score(0.25, 2000) < 2000


def test_elo_from_score_clamps_extremes():
    """0% and 100% would blow up the logistic; we clamp to avoid -/+inf."""
    low = _elo_from_score(0.0, 2000)
    high = _elo_from_score(1.0, 2000)
    assert isinstance(low, int)
    assert isinstance(high, int)
    assert low < 2000 < high


# ============================================================
# Binary-required tests
# ============================================================


@_SF_REQUIRED
def test_stockfish_starts_and_responds_uci():
    with stockfish_player(_SF_PATH, StockfishOptions(skill=0, movetime_s=0.05)) as sf:
        board = chess.Board(chess960=True)
        board.set_chess960_pos(518)
        move = sf.play(board)
        assert move in board.legal_moves


@_SF_REQUIRED
def test_stockfish_handles_chess960_castle_position():
    """Stockfish should generate a legal move from a non-standard 960 SP."""
    with stockfish_player(_SF_PATH, StockfishOptions(skill=0, movetime_s=0.05)) as sf:
        board = chess.Board(chess960=True)
        board.set_chess960_pos(42)
        move = sf.play(board)
        assert move in board.legal_moves


@_SF_REQUIRED
def test_stockfish_player_context_manager_closes_cleanly():
    sf = StockfishPlayer(_SF_PATH, StockfishOptions(skill=0, movetime_s=0.05))
    sf.close()
    # Closing again should not raise
    sf.close()
