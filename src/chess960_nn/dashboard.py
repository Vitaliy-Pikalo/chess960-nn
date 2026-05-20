"""Live training dashboard.

FastAPI app that exposes:
- ``GET /``                              – serves ``web/index.html``
- ``GET /api/runs``                      – list run dirs with detected type
- ``GET /api/runs/{name}/metrics``       – JSONL contents as array
- ``GET /api/runs/{name}/summary``       – ``summary.json`` if present
- ``GET /api/runs/{name}/info``          – combined metadata + counts

Launch:
    uv run python scripts/dashboard.py --runs-dir runs --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse


def create_app(runs_dir: Path, web_dir: Path) -> FastAPI:
    """Build a FastAPI app rooted at ``runs_dir`` with static files in ``web_dir``."""
    runs_dir = Path(runs_dir).resolve()
    web_dir = Path(web_dir).resolve()
    app = FastAPI(title="chess960-nn dashboard", version="0.1.0")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        return JSONResponse(_list_runs(runs_dir))

    @app.get("/api/runs/{name}/metrics")
    def get_metrics(name: str) -> JSONResponse:
        path = _safe_run_path(runs_dir, name) / "metrics.jsonl"
        if not path.exists():
            return JSONResponse([])
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return JSONResponse(records)

    @app.get("/api/runs/{name}/summary")
    def get_summary(name: str) -> JSONResponse:
        path = _safe_run_path(runs_dir, name) / "summary.json"
        if not path.exists():
            return JSONResponse({})
        try:
            return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return JSONResponse({"error": "summary.json is not valid JSON"})

    @app.get("/api/runs/{name}/info")
    def get_info(name: str) -> JSONResponse:
        run_path = _safe_run_path(runs_dir, name)
        info = _describe_run(run_path)
        return JSONResponse(info)

    return app


# ============================================================
# Helpers
# ============================================================


def _safe_run_path(runs_dir: Path, name: str) -> Path:
    """Resolve a run directory by name, refusing path traversal."""
    candidate = (runs_dir / name).resolve()
    try:
        candidate.relative_to(runs_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid run name") from exc
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"run not found: {name}")
    return candidate


def _list_runs(runs_dir: Path) -> list[dict[str, str | int | bool]]:
    if not runs_dir.is_dir():
        return []
    out: list[dict[str, str | int | bool]] = []
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        info = _describe_run(sub)
        info["name"] = sub.name
        out.append(info)
    return out


def _describe_run(run_path: Path) -> dict[str, str | int | bool]:
    """Classify a run directory and gather counts."""
    metrics_path = run_path / "metrics.jsonl"
    summary_path = run_path / "summary.json"
    ckpt_dir = run_path / "checkpoints"
    selfplay_dir = run_path / "selfplay"

    has_metrics = metrics_path.exists() and metrics_path.stat().st_size > 0
    has_summary = summary_path.exists()
    has_checkpoints = ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pt"))
    has_selfplay_shards = selfplay_dir.is_dir() and any(
        selfplay_dir.glob("shard_*.npz")
    )

    if has_summary and has_selfplay_shards:
        run_type = "rl"
    elif has_metrics:
        run_type = "pretrain"
    else:
        run_type = "unknown"

    n_metric_lines = 0
    if has_metrics:
        with metrics_path.open(encoding="utf-8") as f:
            for _ in f:
                n_metric_lines += 1

    n_checkpoints = sum(1 for _ in ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else 0
    n_selfplay_shards = (
        sum(1 for _ in selfplay_dir.glob("shard_*.npz"))
        if selfplay_dir.is_dir()
        else 0
    )

    return {
        "type": run_type,
        "has_metrics": has_metrics,
        "has_summary": has_summary,
        "has_checkpoints": has_checkpoints,
        "has_selfplay_shards": has_selfplay_shards,
        "n_metric_lines": n_metric_lines,
        "n_checkpoints": n_checkpoints,
        "n_selfplay_shards": n_selfplay_shards,
    }
