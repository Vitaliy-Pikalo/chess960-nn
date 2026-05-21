"""Evaluate the trained net against Stockfish at multiple skill levels.

Example:
    uv run python scripts/eval_vs_stockfish.py \\
        --checkpoint runs/pretrain-002/checkpoints/last.pt \\
        --stockfish bin/stockfish/stockfish-windows-x86-64-avx2.exe \\
        --skill-levels 0,5,10 --games-per-level 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.stockfish import (
    download_stockfish,
    evaluate_vs_stockfish,
)
from chess960_nn.train import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--stockfish",
        type=Path,
        default=None,
        help="Path to stockfish.exe. If omitted, auto-downloads into bin/stockfish.",
    )
    p.add_argument(
        "--skill-levels",
        type=str,
        default="0,5,10",
        help="Comma-separated stockfish skill levels (0-20).",
    )
    p.add_argument("--games-per-level", type=int, default=6)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--movetime-s", type=float, default=0.1,
                   help="Stockfish thinking time per move (seconds).")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the results JSON.")
    args = p.parse_args(argv)

    # Stockfish binary
    if args.stockfish is None:
        sf_dir = Path("bin/stockfish")
        sf_path = download_stockfish(sf_dir)
        print(f"Using stockfish: {sf_path}")
    else:
        sf_path = args.stockfish
        if not sf_path.exists():
            raise SystemExit(f"Stockfish binary not found: {sf_path}")

    skills = [int(s.strip()) for s in args.skill_levels.split(",") if s.strip()]
    print(f"Skill levels: {skills}  Games per level: {args.games_per_level}")

    # Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_cfg = ModelConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    net = Chess960Net(model_cfg).to(device)
    load_checkpoint(args.checkpoint, net=net, map_location=str(device))
    net.eval()
    print(f"Loaded {args.checkpoint}  params={net.num_parameters():,}")
    print()

    def on_game(skill, game_idx, wins, losses, draws):
        print(
            f"  [skill {skill:>2d}] game {game_idx+1:>2d}/{args.games_per_level}  "
            f"running W/L/D = {wins}/{losses}/{draws}"
        )

    t0 = time.time()
    results = evaluate_vs_stockfish(
        net,
        sf_path,
        skill_levels=skills,
        n_games_per_level=args.games_per_level,
        n_simulations=args.simulations,
        movetime_s=args.movetime_s,
        max_plies=args.max_plies,
        seed=args.seed,
        device=str(device),
        on_game_done=on_game,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Eval done in {elapsed/60:.1f} min")
    print()
    print(f"{'skill':>6} {'SF Elo':>7} {'W':>3} {'L':>3} {'D':>3} {'score':>7} {'est. NN Elo':>13}")
    for skill, r in results.items():
        print(
            f"{skill:>6} {r['sf_elo']:>7} {r['wins']:>3} {r['losses']:>3} {r['draws']:>3} "
            f"{r['score']:>7.3f} {r['nn_elo_estimate']:>13}"
        )
    print()
    # Weighted average elo estimate (each level equally weighted, ignoring 0/100% scores)
    valid = [r for r in results.values()
             if 0.0 < r["score"] < 1.0 and r["games"] > 0]
    if valid:
        avg = sum(r["nn_elo_estimate"] for r in valid) / len(valid)
        print(f"Average NN Elo estimate (excluding sweeps): {avg:.0f}")
    print()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({str(k): v for k, v in results.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
