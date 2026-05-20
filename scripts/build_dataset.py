"""CLI wrapper for the Lichess Chess960 data pipeline.

Examples:
    # Download the April 2025 dump and build shards from it
    uv run python scripts/build_dataset.py --month 2025-04 \\
        --raw-dir datasets/raw --out-dir datasets/cached/2025-04 \\
        --min-rating 1900

    # Use a PGN file you already have
    uv run python scripts/build_dataset.py \\
        --pgn path/to/games.pgn.zst --out-dir datasets/cached/mine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess960_nn.data.pipeline import (
    GameFilter,
    build_dataset,
    download_lichess_960,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--month",
        type=str,
        help="YYYY-MM Lichess Chess960 monthly dump to download.",
    )
    src.add_argument(
        "--pgn",
        type=Path,
        help="Path to an existing PGN (.pgn / .pgn.zst / .pgn.bz2).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw"),
        help="Where to download (or look up) the PGN file. Default: datasets/raw",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write sharded .npz files.",
    )
    p.add_argument("--min-rating", type=int, default=1800)
    p.add_argument("--min-moves", type=int, default=10)
    p.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Positions per shard. Smaller = more files, faster random access.",
    )
    p.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap total positions written. Useful for quick smoke tests.",
    )
    args = p.parse_args(argv)

    if args.month is not None:
        pgn_path = download_lichess_960(args.month, args.raw_dir)
    else:
        pgn_path = args.pgn
        if not pgn_path.exists():
            print(f"PGN not found: {pgn_path}", file=sys.stderr)
            return 2

    filt = GameFilter(min_rating=args.min_rating, min_moves=args.min_moves)
    stats = build_dataset(
        pgn_path,
        args.out_dir,
        filt=filt,
        shard_size=args.shard_size,
        max_positions=args.max_positions,
    )
    print()
    print(f"Games kept:      {stats['games']:>10,}")
    print(f"Positions:       {stats['positions']:>10,}")
    print(f"Shards written:  {stats['shards']:>10,}")
    print(f"Output:          {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
