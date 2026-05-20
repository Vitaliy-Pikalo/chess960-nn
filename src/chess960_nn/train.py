"""Training loops: supervised pretrain (this phase) + RL self-play (later).

Loss = cross-entropy(policy_logits, played_action) + MSE(value_pred, outcome).
We use AdamW + cosine LR schedule + fp16 mixed precision via torch.amp.

During training we:
- print a tqdm progress bar with running averages
- print a console line every ``log_every`` steps
- append a JSONL metrics record to ``runs/<name>/metrics.jsonl`` periodically
- save a checkpoint to ``runs/<name>/checkpoints/`` after each epoch
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess960_nn.model import Chess960Net, ModelConfig

# ============================================================
# Config
# ============================================================


@dataclass
class TrainConfig:
    """Hyperparameters and run config for supervised pretrain."""

    shard_dir: Path
    out_dir: Path
    val_shard_dir: Path | None = None

    # Optimisation
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000

    # System
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"
    use_amp: bool = True

    # Logging / IO
    log_every: int = 50
    save_every_epochs: int = 1
    resume_from: Path | None = None
    run_name: str = "pretrain"

    # Loss weighting
    value_loss_weight: float = 1.0

    # Set at runtime
    model: ModelConfig = field(default_factory=ModelConfig)


# ============================================================
# Metric logger
# ============================================================


class MetricsLogger:
    """Append-only JSONL writer for training metrics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # touch so consumers can tail before training starts
        self.path.touch(exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ============================================================
# Checkpointing
# ============================================================


def save_checkpoint(
    path: Path,
    net: Chess960Net,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "model_cfg": asdict(net.cfg),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    net: Chess960Net | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if net is not None:
        net.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


# ============================================================
# LR schedule
# ============================================================


def lr_at_step(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Linear warmup then cosine decay to 1/10th of base_lr.

    Step indexing: ``step`` runs from 0 to ``total - 1``.
    - At step 0: lr = base_lr / warmup
    - At step warmup-1: lr = base_lr (peak)
    - At step total-1: lr = base_lr / 10 (min) exactly
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * 0.1
    return min_lr + (base_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ============================================================
# Train / eval steps
# ============================================================


def train_one_epoch(
    net: Chess960Net,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    *,
    device: torch.device,
    step_start: int,
    total_steps: int,
    cfg: TrainConfig,
    metrics: MetricsLogger | None = None,
    epoch_idx: int = 0,
) -> tuple[dict[str, float], int]:
    """Run one pass over ``loader``. Returns (epoch_metrics, last_step)."""
    net.train()
    running = _RunningStats()
    step = step_start

    desc = f"epoch {epoch_idx + 1}/{cfg.epochs}"
    pbar = tqdm(
        loader,
        total=getattr(loader, "_estimated_batches", None),
        desc=desc,
        leave=True,
    )

    for states, actions, values in pbar:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        # python float -> default_collate makes float64; cast to float32 for AMP-friendly loss
        values = values.to(device, non_blocking=True).float()

        lr = lr_at_step(step, cfg.warmup_steps, total_steps, cfg.lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                policy_logits, value_pred = net(states)
                policy_loss = nn.functional.cross_entropy(policy_logits, actions)
                value_loss = nn.functional.mse_loss(value_pred, values)
                loss = policy_loss + cfg.value_loss_weight * value_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = net(states)
            policy_loss = nn.functional.cross_entropy(policy_logits, actions)
            value_loss = nn.functional.mse_loss(value_pred, values)
            loss = policy_loss + cfg.value_loss_weight * value_loss
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            preds = policy_logits.argmax(-1)
            correct = (preds == actions).sum().item()
        n = actions.size(0)

        running.update(
            n=n,
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            correct=correct,
        )

        if step % cfg.log_every == 0:
            stats = running.snapshot()
            pbar.set_postfix(
                loss=f"{stats['loss']:.3f}",
                pol=f"{stats['policy_loss']:.3f}",
                val=f"{stats['value_loss']:.3f}",
                top1=f"{stats['top1_acc']:.3f}",
                lr=f"{lr:.1e}",
            )
            if metrics is not None:
                metrics.log({
                    "phase": "train",
                    "epoch": epoch_idx,
                    "step": step,
                    "lr": lr,
                    **stats,
                })

        step += 1

    return running.snapshot(), step


def evaluate(
    net: Chess960Net,
    loader: DataLoader,
    *,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    net.eval()
    running = _RunningStats()
    pbar = tqdm(loader, desc="eval", leave=False)
    with torch.no_grad():
        for states, actions, values in pbar:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).float()

            with autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
                policy_logits, value_pred = net(states)
                policy_loss = nn.functional.cross_entropy(policy_logits, actions)
                value_loss = nn.functional.mse_loss(value_pred, values)
                loss = policy_loss + cfg.value_loss_weight * value_loss

            preds = policy_logits.argmax(-1)
            correct = (preds == actions).sum().item()
            n = actions.size(0)

            running.update(
                n=n,
                loss=loss.item(),
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                correct=correct,
            )

    return running.snapshot()


# ============================================================
# Running stats
# ============================================================


class _RunningStats:
    def __init__(self):
        self.n = 0
        self.loss_sum = 0.0
        self.policy_sum = 0.0
        self.value_sum = 0.0
        self.correct = 0

    def update(
        self,
        *,
        n: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        correct: int,
    ) -> None:
        self.n += n
        self.loss_sum += loss * n
        self.policy_sum += policy_loss * n
        self.value_sum += value_loss * n
        self.correct += correct

    def snapshot(self) -> dict[str, float]:
        n = max(self.n, 1)
        return {
            "loss": self.loss_sum / n,
            "policy_loss": self.policy_sum / n,
            "value_loss": self.value_sum / n,
            "top1_acc": self.correct / n,
            "samples": self.n,
        }
