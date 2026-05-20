"""Run ONE AlphaZero RL iteration: self-play -> train -> gating eval.

Workflow:
    1. Use the current best checkpoint to generate N self-play games.
    2. Load a candidate net (warm-started from the same checkpoint).
       Train it on the new self-play shards.
    3. Play a match between candidate and current best at low simulations.
       Promote the candidate if win-rate >= --promote-threshold; else discard.

Example:
    uv run python scripts/rl_iteration.py \\
        --current runs/pretrain-002/checkpoints/last.pt \\
        --out-dir runs/rl-iter-001 \\
        --selfplay-games 20 --train-steps 500 --eval-games 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.rl import (
    MatchConfig,
    RLShardDataset,
    play_match,
)
from chess960_nn.selfplay import (
    SelfPlayConfig,
    play_game,
    write_selfplay_shard,
)
from chess960_nn.train import load_checkpoint, save_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", type=Path, required=True,
                   help="Current best checkpoint .pt")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write self-play data, candidate ckpt, and metadata.")
    p.add_argument("--selfplay-games", type=int, default=20)
    p.add_argument("--selfplay-sims", type=int, default=100)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--train-steps", type=int, default=500,
                   help="Number of optimisation steps over the new self-play data.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--promote-threshold", type=float, default=0.55)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp_dir = args.out_dir / "selfplay"
    sp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_cfg = ModelConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)

    # -------- Stage 1: self-play with current net --------
    print()
    print("=" * 60)
    print("STAGE 1 / 3 : Self-play")
    print("=" * 60)
    current_net = Chess960Net(model_cfg).to(device)
    load_checkpoint(args.current, net=current_net, map_location=str(device))
    current_net.eval()

    sp_cfg = SelfPlayConfig(
        n_simulations=args.selfplay_sims,
        max_plies=args.max_plies,
        resign_threshold=-0.9,
    )
    rng = np.random.default_rng(args.seed)
    games = []
    t0 = time.time()
    wins = losses = draws = resigns = 0
    plies_total = 0
    for game_idx in range(args.selfplay_games):
        rec = play_game(current_net, cfg=sp_cfg, rng=rng, device=str(device))
        games.append(rec)
        plies_total += rec.plies
        if rec.outcome == 1:
            wins += 1
        elif rec.outcome == -1:
            losses += 1
        else:
            draws += 1
        if rec.resigned:
            resigns += 1
        elapsed = time.time() - t0
        outcome_letter = "W" if rec.outcome == 1 else "L" if rec.outcome == -1 else "D"
        print(
            f"  game {game_idx+1:>3d}/{args.selfplay_games}  "
            f"outcome={outcome_letter}  "
            f"plies={rec.plies:>3d}  "
            f"resigned={rec.resigned}  "
            f"avg={plies_total/(game_idx+1):.0f}p  "
            f"t={elapsed/60:.1f}m"
        )

    if games:
        write_selfplay_shard(sp_dir / "shard_00000.npz", games)
    print(
        f"Self-play done: W={wins} L={losses} D={draws} resign={resigns}  "
        f"positions={plies_total}  time={(time.time()-t0)/60:.1f} min"
    )

    # -------- Stage 2: train candidate --------
    print()
    print("=" * 60)
    print("STAGE 2 / 3 : Train candidate net")
    print("=" * 60)
    candidate_net = Chess960Net(model_cfg).to(device)
    load_checkpoint(args.current, net=candidate_net, map_location=str(device))
    optimizer = AdamW(
        candidate_net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler() if device.type == "cuda" else None

    rl_ds = RLShardDataset(sp_dir, shuffle=True, seed=args.seed)
    print(f"Training positions: {len(rl_ds):,}")
    loader = DataLoader(
        rl_ds, batch_size=args.batch_size, num_workers=0, pin_memory=True
    )

    # Cap training at requested steps. Easiest: limit iterations of the loader.
    # We do this by wrapping the loader in a finite iterator.
    t0 = time.time()
    train_stats = _train_with_step_cap(
        candidate_net, loader, optimizer, scaler, device,
        n_steps=args.train_steps,
    )
    print(
        f"Train done: loss={train_stats['loss']:.4f}  "
        f"pol={train_stats['policy_loss']:.4f}  "
        f"val={train_stats['value_loss']:.4f}  "
        f"steps={train_stats['steps']}  "
        f"time={(time.time()-t0)/60:.1f} min"
    )

    cand_path = ckpt_dir / "candidate.pt"
    save_checkpoint(
        cand_path, candidate_net, optimizer,
        step=train_stats["steps"], epoch=0,
        extra={"selfplay_games": args.selfplay_games, "train_stats": train_stats},
    )
    print(f"Saved candidate -> {cand_path}")

    # -------- Stage 3: gating match --------
    print()
    print("=" * 60)
    print("STAGE 3 / 3 : Gating match (candidate vs current)")
    print("=" * 60)
    candidate_net.eval()
    match_cfg = MatchConfig(
        n_games=args.eval_games,
        n_simulations=args.eval_sims,
        max_plies=args.max_plies,
    )
    t0 = time.time()
    result = play_match(
        candidate_net, current_net, cfg=match_cfg, seed=args.seed, device=str(device)
    )
    print(
        f"Match: {result['wins']}W / {result['losses']}L / {result['draws']}D  "
        f"score={result['score']:.3f}  "
        f"time={(time.time()-t0)/60:.1f} min"
    )

    promoted = result["score"] >= args.promote_threshold
    if promoted:
        promo_path = ckpt_dir / "best.pt"
        shutil.copy2(cand_path, promo_path)
        print(f"PROMOTED -> {promo_path}")
    else:
        print(
            f"NOT promoted (score {result['score']:.3f} < "
            f"threshold {args.promote_threshold})"
        )

    # Metadata
    meta = {
        "selfplay": {
            "games": args.selfplay_games, "W": wins, "L": losses, "D": draws,
            "resigns": resigns, "positions": plies_total,
        },
        "train": train_stats,
        "match": result,
        "promoted": promoted,
        "promote_threshold": args.promote_threshold,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return 0


def _train_with_step_cap(
    net, loader, optimizer, scaler, device, *, n_steps,
) -> dict[str, float]:
    """Run rl_train_one_epoch logic but stop after ``n_steps`` optimiser steps."""
    import torch.nn.functional as F  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415 - local import to avoid top-level clutter
    from torch.amp import autocast  # noqa: PLC0415

    net.train()
    total_n = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    step = 0
    # Loop until step cap; loader may need to be re-iterated.
    while step < n_steps:
        for states, policy_targets, values in loader:
            if step >= n_steps:
                break
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
                    loss = policy_loss + value_loss
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
                loss = policy_loss + value_loss
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            n = states.size(0)
            total_n += n
            total_loss += float(loss.item()) * n
            total_policy_loss += float(policy_loss.item()) * n
            total_value_loss += float(value_loss.item()) * n
            step += 1

            if step % 50 == 0 or step == 1:
                print(
                    f"  step {step:>5d}/{n_steps}  "
                    f"loss={total_loss/max(1,total_n):.4f}  "
                    f"pol={total_policy_loss/max(1,total_n):.4f}  "
                    f"val={total_value_loss/max(1,total_n):.4f}"
                )
        # If loader is exhausted before step cap, iterate again
        if step >= n_steps:
            break

    return {
        "loss": total_loss / max(1, total_n),
        "policy_loss": total_policy_loss / max(1, total_n),
        "value_loss": total_value_loss / max(1, total_n),
        "samples": total_n,
        "steps": step,
    }


if __name__ == "__main__":
    raise SystemExit(main())
