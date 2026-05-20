"""Lichess Chess960 data pipeline.

Pipeline stages:
    1. (Optional) Download a monthly Lichess Chess960 PGN dump.
    2. Stream-parse games out of the PGN file (transparently handles ``.zst``,
       ``.bz2``, plain text).
    3. Filter by variant + rating + minimum number of moves.
    4. Tensorise each ``(board, move)`` into ``(state, action, value)`` tuples
       using the encoding module.
    5. Write to numbered ``.npz`` shards in an output directory for later
       random access by the training Dataset.

The module is designed to stream so it works on multi-GB PGN dumps without
loading them all into memory.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from chess960_nn.encoding import (
    NUM_BOARD_PLANES,
    encode_board,
    encode_move,
)

# ============================================================
# Filter + tuple types
# ============================================================


@dataclass
class GameFilter:
    """Criteria a game must satisfy to be included in the dataset."""

    min_rating: int = 1800
    """Minimum of WhiteElo and BlackElo. Lower bound for both players."""

    min_moves: int = 10
    """Discard games shorter than this (bot games, abandons, premove losses)."""

    require_variant: str | None = "Chess960"
    """If set, only games with this Variant header are kept. None = accept any."""

    allow_no_result: bool = False
    """If False, drop games whose Result is ``*`` (unfinished)."""


@dataclass
class TrainingTuple:
    """One training datapoint: position + the move played + game outcome."""

    state: np.ndarray  # (NUM_BOARD_PLANES, 8, 8) float16
    action: int  # action index in [0, 4672)
    value: int  # +1 / 0 / -1 from side-to-move's perspective


# ============================================================
# PGN parsing + filtering
# ============================================================


def iter_games_from_pgn(
    stream: IO[str], filt: GameFilter | None = None
) -> Iterator[chess.pgn.Game]:
    """Yield games from ``stream`` that pass ``filt``.

    Robust to malformed games: any individual parse error is skipped.
    """
    filt = filt or GameFilter()
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except (ValueError, RuntimeError):
            # Malformed game header / unparseable section - skip to next.
            continue
        if game is None:
            return
        if _passes_filter(game, filt):
            yield game


def _passes_filter(game: chess.pgn.Game, filt: GameFilter) -> bool:
    h = game.headers

    if filt.require_variant is not None and h.get("Variant") != filt.require_variant:
        return False

    try:
        white_elo = int(h.get("WhiteElo", "0"))
        black_elo = int(h.get("BlackElo", "0"))
    except ValueError:
        return False
    if min(white_elo, black_elo) < filt.min_rating:
        return False

    result = h.get("Result", "*")
    if not filt.allow_no_result and result == "*":
        return False
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        return False

    # Count plies (walk mainline) until we hit the threshold.
    n_plies = 0
    node = game
    while node.variations:
        node = node.variations[0]
        n_plies += 1
        if n_plies >= filt.min_moves:
            break
    if n_plies < filt.min_moves:
        return False

    return True


def _game_outcome_white_pov(game: chess.pgn.Game) -> int | None:
    """+1 white wins, -1 black wins, 0 draw, None if no result."""
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    return None


# ============================================================
# Tensorisation
# ============================================================


def iter_training_tuples(game: chess.pgn.Game) -> Iterator[TrainingTuple]:
    """Yield ``TrainingTuple`` for each move in ``game``'s mainline.

    Skips games with no decisive/draw result (we need a value target).
    The board is always recreated in chess960 mode so castle moves use the
    king-to-rook convention that ``encoding`` expects.
    """
    outcome = _game_outcome_white_pov(game)
    if outcome is None:
        return

    # Always re-create the board in chess960 mode so castling moves are
    # encoded as king-takes-rook (the only convention encoding.py understands).
    start_fen = game.headers.get("FEN") or chess.STARTING_FEN
    board = chess.Board(start_fen, chess960=True)

    node = game
    while node.variations:
        node = node.variations[0]
        move = node.move
        if move is None:
            return

        # Encode FROM the current STM perspective.
        state = encode_board(board).astype(np.float16)
        try:
            action = encode_move(move, board)
        except ValueError:
            # Skip games containing un-encodable moves. Shouldn't happen for
            # real chess960 games, but better safe than crash mid-shard.
            return

        value_stm = outcome if board.turn == chess.WHITE else -outcome

        yield TrainingTuple(state=state, action=action, value=value_stm)
        board.push(move)


# ============================================================
# Shard I/O
# ============================================================


def write_shard(path: Path, tuples: list[TrainingTuple]) -> None:
    """Save a list of training tuples as a compressed npz shard.

    Schema (all numpy arrays, length N):
        states:  (N, NUM_BOARD_PLANES, 8, 8) float16
        actions: (N,) int32
        values:  (N,) int8
    """
    if not tuples:
        raise ValueError("Refusing to write an empty shard")
    states = np.stack([t.state for t in tuples])
    assert states.shape[1:] == (NUM_BOARD_PLANES, 8, 8)
    actions = np.fromiter((t.action for t in tuples), dtype=np.int32, count=len(tuples))
    values = np.fromiter((t.value for t in tuples), dtype=np.int8, count=len(tuples))
    np.savez_compressed(path, states=states, actions=actions, values=values)


def read_shard(path: Path) -> dict[str, np.ndarray]:
    """Load a shard back into a dict of arrays (compressed npz)."""
    with np.load(path) as d:
        return {
            "states": d["states"],
            "actions": d["actions"],
            "values": d["values"],
        }


# ============================================================
# Compressed reader
# ============================================================


def open_pgn_stream(path: Path) -> IO[str]:
    """Open ``path`` as a text stream, transparently decompressing ``.zst``/``.bz2``."""
    suffix = path.suffix.lower()
    if suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Install zstandard to read .pgn.zst files: uv add zstandard"
            ) from exc
        dctx = zstd.ZstdDecompressor()
        raw = open(path, "rb")
        return io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8", errors="replace")
    if suffix == ".bz2":
        import bz2

        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


# ============================================================
# Lichess download
# ============================================================


LICHESS_960_BASE = "https://database.lichess.org/chess960"


def lichess_960_url(month: str, base_url: str = LICHESS_960_BASE) -> str:
    """Return the canonical URL for a Lichess Chess960 monthly dump.

    ``month`` is in ``YYYY-MM`` format (e.g. ``"2025-04"``).
    """
    return f"{base_url}/lichess_db_chess960_rated_{month}.pgn.zst"


def download_lichess_960(
    month: str,
    dest_dir: Path,
    *,
    base_url: str = LICHESS_960_BASE,
    chunk_size: int = 1 << 20,
) -> Path:
    """Download a Lichess Chess960 monthly dump if not already present.

    Returns the path to the downloaded ``.pgn.zst`` file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = lichess_960_url(month, base_url=base_url)
    target = dest_dir / Path(url).name

    if target.exists() and target.stat().st_size > 0:
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - trusted source
        total = int(resp.headers.get("Content-Length", "0"))
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as pbar,
        ):
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(target)
    return target


# ============================================================
# Top-level pipeline
# ============================================================


def build_dataset(
    pgn_path: Path,
    out_dir: Path,
    *,
    filt: GameFilter | None = None,
    shard_size: int = 100_000,
    max_positions: int | None = None,
) -> dict[str, int]:
    """End-to-end: stream PGN, filter, tensorise, write npz shards.

    Returns a stats dict ``{"games": int, "positions": int, "shards": int}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer: list[TrainingTuple] = []
    n_games = 0
    n_positions = 0

    with open_pgn_stream(pgn_path) as stream:
        for game in tqdm(iter_games_from_pgn(stream, filt), desc="games", unit="game"):
            n_games += 1
            for tup in iter_training_tuples(game):
                buffer.append(tup)
                n_positions += 1
                if len(buffer) >= shard_size:
                    write_shard(out_dir / f"shard_{shard_idx:05d}.npz", buffer)
                    shard_idx += 1
                    buffer.clear()
                if max_positions is not None and n_positions >= max_positions:
                    break
            if max_positions is not None and n_positions >= max_positions:
                break

    if buffer:
        write_shard(out_dir / f"shard_{shard_idx:05d}.npz", buffer)
        shard_idx += 1
        buffer.clear()

    return {"games": n_games, "positions": n_positions, "shards": shard_idx}
