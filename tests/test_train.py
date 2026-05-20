"""Tests for the training module.

Covers:
- LR schedule (warmup + cosine)
- Checkpoint save/load roundtrip
- Overfit-on-tiny-batch sanity check (loss should drop in a few steps)
- MetricsLogger appends valid JSONL
- StreamingShardDataset yields the right total when iterated
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chess960_nn.data.dataset import StreamingShardDataset
from chess960_nn.data.pipeline import GameFilter, build_dataset
from chess960_nn.encoding import ACTION_SPACE_SIZE, NUM_BOARD_PLANES
from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.train import (
    MetricsLogger,
    TrainConfig,
    load_checkpoint,
    lr_at_step,
    save_checkpoint,
    train_one_epoch,
)

SYNTHETIC_PGN = """[Event "T1"]
[Site "?"]
[Date "2025.01.01"]
[White "A"]
[Black "B"]
[Result "1-0"]
[Variant "Chess960"]
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
[SetUp "1"]
[WhiteElo "2200"]
[BlackElo "2200"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O Bd6 6. d4 exd4
7. Nxd4 Nf6 8. Nc3 O-O 9. Bg5 h6 10. Bh4 1-0
"""


# ============================================================
# LR schedule
# ============================================================


def test_lr_schedule_warmup_and_decay():
    base = 1e-3
    warmup = 10
    total = 110
    # Step 0 -> first warmup step, lr = base/warmup
    assert lr_at_step(0, warmup, total, base) == pytest.approx(base / warmup)
    # Step warmup-1 -> last warmup step, lr = base
    assert lr_at_step(warmup - 1, warmup, total, base) == pytest.approx(base)
    # Step at end of training -> lr = base/10 (min)
    end = lr_at_step(total - 1, warmup, total, base)
    assert end == pytest.approx(base * 0.1, abs=1e-9)


# ============================================================
# Checkpointing
# ============================================================


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = ModelConfig(num_blocks=1, num_filters=16)
    net = Chess960Net(cfg)
    optimizer = AdamW(net.parameters(), lr=1e-3)

    # Modify some weights so we know loading is doing something.
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(0.1)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, optimizer, step=42, epoch=2)

    # Reset and reload
    net2 = Chess960Net(cfg)
    opt2 = AdamW(net2.parameters(), lr=1e-3)
    payload = load_checkpoint(path, net=net2, optimizer=opt2, map_location="cpu")

    assert payload["step"] == 42
    assert payload["epoch"] == 2
    for p1, p2 in zip(net.parameters(), net2.parameters(), strict=True):
        assert torch.allclose(p1, p2)


# ============================================================
# MetricsLogger
# ============================================================


def test_metrics_logger_appends_jsonl(tmp_path: Path):
    log = MetricsLogger(tmp_path / "m.jsonl")
    log.log({"step": 1, "loss": 0.5})
    log.log({"step": 2, "loss": 0.4})

    lines = (tmp_path / "m.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["step"] == 1
    assert rec2["loss"] == 0.4
    # Both records have a 'time' field
    assert "time" in rec1 and "time" in rec2


# ============================================================
# Streaming dataset
# ============================================================


def _build_shards(tmp_path: Path) -> Path:
    pgn_path = tmp_path / "g.pgn"
    pgn_path.write_text(SYNTHETIC_PGN, encoding="utf-8")
    out_dir = tmp_path / "shards"
    build_dataset(
        pgn_path,
        out_dir,
        filt=GameFilter(min_rating=1800),
        shard_size=5,
    )
    return out_dir


def test_streaming_dataset_yields_all_samples(tmp_path: Path):
    out_dir = _build_shards(tmp_path)
    ds = StreamingShardDataset(out_dir, shuffle=False)
    total = len(ds)
    count = 0
    for state, action, value in ds:
        assert state.shape == (NUM_BOARD_PLANES, 8, 8)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert value in (-1.0, 0.0, 1.0)
        count += 1
    assert count == total


def test_streaming_dataset_shuffles_across_epochs(tmp_path: Path):
    out_dir = _build_shards(tmp_path)
    ds = StreamingShardDataset(out_dir, shuffle=True, seed=0)
    ds.set_epoch(0)
    actions_e0 = [a for _, a, _ in ds]
    ds.set_epoch(1)
    actions_e1 = [a for _, a, _ in ds]
    # Same set of actions, different order (probably).
    assert sorted(actions_e0) == sorted(actions_e1)
    # Not guaranteed to differ but extremely unlikely on a meaningful sample
    assert actions_e0 != actions_e1 or len(actions_e0) < 10


# ============================================================
# Overfit-on-tiny-batch (sanity check the training loop trains)
# ============================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for AMP path")
def test_overfit_tiny_batch_reduces_loss(tmp_path: Path):
    """Loss should drop sharply when the same batch is shown repeatedly."""
    torch.manual_seed(0)
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    device = torch.device("cuda")
    net = Chess960Net(cfg).to(device)

    # Random fixed batch
    n = 16
    states = torch.randn(n, NUM_BOARD_PLANES, 8, 8, device=device)
    actions = torch.randint(0, ACTION_SPACE_SIZE, (n,), device=device)
    values = torch.empty(n, device=device).uniform_(-1, 1)

    optimizer = AdamW(net.parameters(), lr=1e-2)
    scaler = GradScaler()

    initial_loss = None
    final_loss = None
    for step in range(50):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            policy_logits, value_pred = net(states)
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, actions)
            value_loss = torch.nn.functional.mse_loss(value_pred, values)
            loss = policy_loss + value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"loss did not drop enough: initial={initial_loss:.3f}, final={final_loss:.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_train_one_epoch_runs_end_to_end(tmp_path: Path):
    """A single pass over a tiny dataset completes without error."""
    out_dir = _build_shards(tmp_path)
    ds = StreamingShardDataset(out_dir, shuffle=False)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    loader._estimated_batches = (len(ds) + 3) // 4  # type: ignore[attr-defined]

    cfg = TrainConfig(
        shard_dir=out_dir,
        out_dir=tmp_path / "run",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        warmup_steps=2,
        num_workers=0,
        log_every=1,
        run_name="test",
        model=ModelConfig(num_blocks=1, num_filters=16),
    )

    device = torch.device("cuda")
    net = Chess960Net(cfg.model).to(device)
    optimizer = AdamW(net.parameters(), lr=cfg.lr)
    scaler = GradScaler()

    metrics = MetricsLogger(tmp_path / "m.jsonl")
    stats, last_step = train_one_epoch(
        net=net,
        loader=loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        step_start=0,
        total_steps=loader._estimated_batches,  # type: ignore[attr-defined]
        cfg=cfg,
        metrics=metrics,
        epoch_idx=0,
    )
    assert "loss" in stats and stats["samples"] > 0
    assert last_step >= 1
