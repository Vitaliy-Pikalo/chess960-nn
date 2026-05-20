"""Supervised pretrain CLI.

Example:
    uv run python scripts/pretrain.py \\
        --shard-dir datasets/cached/2026-04 \\
        --out-dir runs/pretrain-001 \\
        --epochs 5 --batch-size 512

The script auto-splits the last 5%% of shards out as a validation set unless
``--val-shard-dir`` is given explicitly.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chess960_nn.data.dataset import StreamingShardDataset
from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.train import (
    MetricsLogger,
    TrainConfig,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--val-shard-dir", type=Path, default=None)
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="If --val-shard-dir not given, hold out this fraction "
                        "of the train shards as val.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-split train/val by symlinking/copying shard subsets into
    # the out_dir, OR by directly subsetting the StreamingShardDataset's
    # internal shard list. We do the latter — no disk duplication.
    all_shards = sorted(args.shard_dir.glob("shard_*.npz"))
    if not all_shards:
        raise SystemExit(f"No shards in {args.shard_dir}")

    if args.val_shard_dir is not None:
        train_shards = all_shards
        val_shards = sorted(args.val_shard_dir.glob("shard_*.npz"))
        if not val_shards:
            raise SystemExit(f"No val shards in {args.val_shard_dir}")
    else:
        n_val = max(1, int(len(all_shards) * args.val_fraction))
        # Hold out the LAST shards as val. They come from later in the PGN file,
        # i.e. later games, so train/val are time-correlated but not duplicated.
        train_shards = all_shards[:-n_val]
        val_shards = all_shards[-n_val:]

    print(f"Train shards: {len(train_shards)}  Val shards: {len(val_shards)}")

    train_ds = StreamingShardDataset(args.shard_dir, shuffle=True, seed=args.seed)
    train_ds.shards = train_shards  # subset in place
    train_ds._total = _count_positions(train_shards)  # type: ignore[attr-defined]

    val_ds = StreamingShardDataset(args.shard_dir, shuffle=False, seed=args.seed)
    val_ds.shards = val_shards
    val_ds._total = _count_positions(val_shards)  # type: ignore[attr-defined]

    print(f"Train positions: {len(train_ds):,}  Val positions: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        # Windows multiprocessing with PyTorch is fragile; bump timeout so a
        # slow worker doesn't get killed for being late.
        timeout=120 if args.num_workers > 0 else 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=0,  # main-process eval avoids silent worker crashes
        pin_memory=True,
    )

    # tqdm needs to know the per-epoch batch count; stash it on the loader.
    est_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    train_loader._estimated_batches = est_batches  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  AMP: {not args.no_amp and device.type == 'cuda'}")

    model_cfg = ModelConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    net = Chess960Net(model_cfg).to(device)
    print(f"Model params: {net.num_parameters():,}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if (not args.no_amp and device.type == "cuda") else None

    cfg = TrainConfig(
        shard_dir=args.shard_dir,
        out_dir=args.out_dir,
        val_shard_dir=args.val_shard_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device=device.type,
        use_amp=scaler is not None,
        log_every=args.log_every,
        resume_from=args.resume_from,
        run_name=args.out_dir.name,
        model=model_cfg,
    )

    start_step = 0
    start_epoch = 0
    if args.resume_from is not None:
        payload = load_checkpoint(args.resume_from, net=net, optimizer=optimizer, map_location=str(device))
        start_step = payload.get("step", 0)
        start_epoch = payload.get("epoch", 0) + 1
        print(f"Resumed from {args.resume_from}: step={start_step} epoch={start_epoch}")

    metrics = MetricsLogger(args.out_dir / "metrics.jsonl")
    metrics.log({
        "phase": "init",
        "params": net.num_parameters(),
        "train_positions": len(train_ds),
        "val_positions": len(val_ds),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    })

    total_steps = est_batches * args.epochs

    t0 = time.time()
    step = start_step
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        train_stats, step = train_one_epoch(
            net=net,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            step_start=step,
            total_steps=total_steps,
            cfg=cfg,
            metrics=metrics,
            epoch_idx=epoch,
        )
        print(
            f"[epoch {epoch + 1}/{args.epochs}] train: "
            f"loss={train_stats['loss']:.4f}  "
            f"pol={train_stats['policy_loss']:.4f}  "
            f"val={train_stats['value_loss']:.4f}  "
            f"top1={train_stats['top1_acc']:.4f}"
        )
        metrics.log({"phase": "train_epoch_end", "epoch": epoch, "step": step, **train_stats})

        # Save checkpoint FIRST so we never lose epoch progress if eval fails.
        ck_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(ck_path, net, optimizer, step=step, epoch=epoch, extra={"train": train_stats})
        last_path = ckpt_dir / "last.pt"
        shutil.copy2(ck_path, last_path)
        print(f"saved checkpoint: {ck_path}")

        # Eval is best-effort - if it fails, log and continue. Training keeps going.
        try:
            val_stats = evaluate(net, val_loader, device=device, cfg=cfg)
            print(
                f"[epoch {epoch + 1}/{args.epochs}]   val: "
                f"loss={val_stats['loss']:.4f}  "
                f"pol={val_stats['policy_loss']:.4f}  "
                f"val={val_stats['value_loss']:.4f}  "
                f"top1={val_stats['top1_acc']:.4f}"
            )
            metrics.log({"phase": "val_epoch_end", "epoch": epoch, "step": step, **val_stats})
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: eval failed: {type(e).__name__}: {e}")
            metrics.log({"phase": "val_epoch_failed", "epoch": epoch, "step": step,
                         "error": f"{type(e).__name__}: {e}"})

    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    return 0


def _count_positions(shards: list[Path]) -> int:
    import numpy as np
    total = 0
    for s in shards:
        with np.load(s) as d:
            total += int(d["actions"].shape[0])
    return total


if __name__ == "__main__":
    raise SystemExit(main())
