"""Generate self-play games using a trained checkpoint.

Example:
    uv run python scripts/selfplay.py \\
        --checkpoint runs/pretrain-002/checkpoints/last.pt \\
        --out-dir runs/selfplay-001 \\
        --n-games 50 --simulations 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.selfplay import (
    SelfPlayConfig,
    play_game,
    write_selfplay_shard,
)
from chess960_nn.train import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-games", type=int, default=50)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=2.0)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--games-per-shard", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=192)
    p.add_argument(
        "--no-resign",
        action="store_true",
        help="Disable resignation (play out every game to completion).",
    )
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = ModelConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    net = Chess960Net(model_cfg).to(device)
    payload = load_checkpoint(args.checkpoint, net=net, map_location=str(device))
    print(
        f"Loaded {args.checkpoint}  step={payload.get('step')}  "
        f"epoch={payload.get('epoch')}"
    )
    net.eval()

    sp_cfg = SelfPlayConfig(
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        max_plies=args.max_plies,
        resign_threshold=None if args.no_resign else -0.9,
    )

    rng = np.random.default_rng(args.seed)

    n_white_wins = n_black_wins = n_draws = n_resigns = 0
    total_plies = 0
    shard_idx = 0
    buffer: list = []
    t0 = time.time()

    pbar = tqdm(range(args.n_games), desc="self-play", unit="game")
    for game_idx in pbar:
        rec = play_game(net, cfg=sp_cfg, rng=rng, device=str(device))
        buffer.append(rec)
        total_plies += rec.plies
        if rec.outcome == 1:
            n_white_wins += 1
        elif rec.outcome == -1:
            n_black_wins += 1
        else:
            n_draws += 1
        if rec.resigned:
            n_resigns += 1
        avg_plies = total_plies / (game_idx + 1)
        pbar.set_postfix(
            W=n_white_wins, L=n_black_wins, D=n_draws,
            resign=n_resigns, avg_plies=f"{avg_plies:.0f}",
        )

        if len(buffer) >= args.games_per_shard:
            shard_path = args.out_dir / f"shard_{shard_idx:05d}.npz"
            write_selfplay_shard(shard_path, buffer)
            shard_idx += 1
            buffer.clear()

    if buffer:
        shard_path = args.out_dir / f"shard_{shard_idx:05d}.npz"
        write_selfplay_shard(shard_path, buffer)
        shard_idx += 1
        buffer.clear()

    elapsed = time.time() - t0
    print()
    print(f"Played:       {args.n_games:>6,} games in {elapsed/60:.1f} min "
          f"({args.n_games / elapsed * 60:.2f} games/min)")
    print(f"  White wins: {n_white_wins:>6,}")
    print(f"  Black wins: {n_black_wins:>6,}")
    print(f"  Draws:      {n_draws:>6,}")
    print(f"  Resigns:    {n_resigns:>6,}")
    print(f"Positions:    {total_plies:>6,}")
    print(f"Shards:       {shard_idx:>6,}  -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
