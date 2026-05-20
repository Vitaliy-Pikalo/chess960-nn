"""Tests for the dashboard API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chess960_nn.dashboard import create_app


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """Build a couple of fake run dirs in tmp_path."""
    # Pretrain-style run
    pre = tmp_path / "pretrain-test"
    (pre / "checkpoints").mkdir(parents=True)
    (pre / "checkpoints" / "last.pt").write_bytes(b"\0")
    (pre / "metrics.jsonl").write_text(
        '{"phase": "init", "params": 100, "train_positions": 1000, "batch_size": 32}\n'
        '{"phase": "train", "epoch": 0, "step": 0, "lr": 1e-3, "loss": 8.5, '
        '"policy_loss": 8.4, "value_loss": 0.1, "top1_acc": 0.001, "samples": 32}\n'
        '{"phase": "train", "epoch": 0, "step": 100, "lr": 1e-3, "loss": 3.0, '
        '"policy_loss": 2.9, "value_loss": 0.1, "top1_acc": 0.35, "samples": 3200}\n'
        '{"phase": "val_epoch_end", "epoch": 0, "step": 100, "loss": 3.2, '
        '"policy_loss": 3.0, "value_loss": 0.2, "top1_acc": 0.30, "samples": 500}\n',
        encoding="utf-8",
    )

    # RL-style run
    rl = tmp_path / "rl-iter-test"
    (rl / "checkpoints").mkdir(parents=True)
    (rl / "selfplay").mkdir()
    (rl / "selfplay" / "shard_00000.npz").write_bytes(b"\0")
    (rl / "summary.json").write_text(
        json.dumps({
            "selfplay": {"games": 10, "W": 5, "L": 3, "D": 2, "resigns": 0, "positions": 800},
            "train": {"loss": 1.8, "policy_loss": 1.7, "value_loss": 0.1, "steps": 200, "samples": 800},
            "match": {"score": 0.6, "wins": 5, "losses": 3, "draws": 2, "games": 10},
            "promoted": True,
            "promote_threshold": 0.55,
        }),
        encoding="utf-8",
    )

    # Empty run dir (should classify as "unknown")
    (tmp_path / "empty-run").mkdir()
    return tmp_path


@pytest.fixture
def client(runs_dir: Path, tmp_path: Path) -> TestClient:
    # Create a fake web dir with index.html
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    app = create_app(runs_dir=runs_dir, web_dir=web_dir)
    return TestClient(app)


def test_index_serves_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.text.lower()


def test_list_runs(client: TestClient):
    r = client.get("/api/runs")
    assert r.status_code == 200
    data = r.json()
    names = {x["name"] for x in data}
    assert {"pretrain-test", "rl-iter-test", "empty-run"} <= names

    pre = next(x for x in data if x["name"] == "pretrain-test")
    assert pre["type"] == "pretrain"
    assert pre["has_metrics"] is True
    assert pre["n_metric_lines"] == 4
    assert pre["n_checkpoints"] == 1

    rl = next(x for x in data if x["name"] == "rl-iter-test")
    assert rl["type"] == "rl"
    assert rl["has_summary"] is True
    assert rl["n_selfplay_shards"] == 1

    empty = next(x for x in data if x["name"] == "empty-run")
    assert empty["type"] == "unknown"


def test_get_metrics(client: TestClient):
    r = client.get("/api/runs/pretrain-test/metrics")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 4
    assert data[0]["phase"] == "init"
    assert data[1]["phase"] == "train"


def test_get_metrics_missing_returns_empty(client: TestClient):
    r = client.get("/api/runs/rl-iter-test/metrics")
    assert r.status_code == 200
    assert r.json() == []


def test_get_summary(client: TestClient):
    r = client.get("/api/runs/rl-iter-test/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["promoted"] is True
    assert data["match"]["score"] == 0.6


def test_get_summary_missing_returns_empty(client: TestClient):
    r = client.get("/api/runs/pretrain-test/summary")
    assert r.status_code == 200
    assert r.json() == {}


def test_get_info(client: TestClient):
    r = client.get("/api/runs/pretrain-test/info")
    assert r.status_code == 200
    info = r.json()
    assert info["type"] == "pretrain"
    assert info["n_metric_lines"] == 4


def test_path_traversal_rejected(client: TestClient):
    # Try to climb out of runs_dir
    r = client.get("/api/runs/..%2Fweb/info")
    # FastAPI may URL-decode and pass the literal name; either 400 (rejected)
    # or 404 (not found) are acceptable outcomes - we just don't want a 200.
    assert r.status_code != 200


def test_missing_run_returns_404(client: TestClient):
    r = client.get("/api/runs/nonexistent-run/info")
    assert r.status_code == 404
