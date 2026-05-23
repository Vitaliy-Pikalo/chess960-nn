"""Live training dashboard + interactive demo.

FastAPI app that exposes:

Training metrics (existing):
- ``GET /``                              – serves ``web/index.html``
- ``GET /api/runs``                      – list run dirs with detected type
- ``GET /api/runs/{name}/metrics``       – JSONL contents as array
- ``GET /api/runs/{name}/summary``       – ``summary.json`` if present
- ``GET /api/runs/{name}/info``          – combined metadata + counts

Play-vs-engine:
- ``POST /api/play/new_game``            – start a new game (optionally pick color, sp960 index)
- ``POST /api/play/move``                – submit a UCI move; server replies with engine's move
- ``POST /api/play/resign``              – end the game in resignation

Watch (NN vs Stockfish):
- ``POST /api/watch/new_match``          – start a NN-vs-Stockfish game (skill level configurable)
- ``GET  /api/watch/{match_id}/stream``  – server-sent events stream of moves

Engine introspection:
- ``GET /api/engine/info``               – which checkpoint is loaded, device, sim count

Launch:
    uv run python scripts/dashboard.py --runs-dir runs --port 8000 \\
        --checkpoint runs/rl-loop-001/final_best.pt
"""

from __future__ import annotations

import json
import queue
import random
import threading
import uuid
from pathlib import Path
from typing import Any

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .demo import (
    Engine,
    EngineConfig,
    PlayGameRegistry,
    WatchMatch,
    WatchMatchRegistry,
    finalize_if_terminal,
    run_watch_match,
    watchmove_to_json_dict,
)
from .stockfish import download_stockfish


# ============================================================
# Public app factory
# ============================================================


def create_app(
    runs_dir: Path,
    web_dir: Path,
    checkpoint: Path | None = None,
    stockfish_path: Path | None = None,
    n_simulations: int = 100,
    device: str = "cuda",
) -> FastAPI:
    """Build a FastAPI app rooted at ``runs_dir`` with static files in ``web_dir``.

    If ``checkpoint`` is given, the play + watch demo endpoints are enabled.
    """
    runs_dir = Path(runs_dir).resolve()
    web_dir = Path(web_dir).resolve()
    app = FastAPI(title="chess960-nn dashboard", version="0.2.0")

    state = _DemoState(
        checkpoint=checkpoint,
        stockfish_path=stockfish_path,
        n_simulations=n_simulations,
        device=device,
    )

    # ---------- training metrics ----------

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

    # ---------- engine info ----------

    @app.get("/api/engine/info")
    def engine_info() -> JSONResponse:
        return JSONResponse({
            "checkpoint": str(state.checkpoint) if state.checkpoint else None,
            "demo_enabled": state.demo_enabled,
            "n_simulations": state.n_simulations,
            "device": state.device,
            "engine_loaded": state.engine is not None,
        })

    # ---------- play vs engine ----------

    @app.post("/api/play/new_game")
    def play_new_game(req: NewGameReq) -> JSONResponse:
        if not state.demo_enabled:
            raise HTTPException(503, "Demo disabled: no checkpoint configured")
        engine = state.get_engine()

        # Resolve color
        color_str = req.user_color.lower()
        if color_str == "random":
            color = chess.WHITE if random.random() < 0.5 else chess.BLACK
        elif color_str == "white":
            color = chess.WHITE
        elif color_str == "black":
            color = chess.BLACK
        else:
            raise HTTPException(400, "user_color must be 'white', 'black', or 'random'")

        # Resolve starting position (518 = standard chess starting position)
        sp = req.starting_position_index if req.starting_position_index is not None else 518
        if sp < 0 or sp > 959:
            raise HTTPException(400, "starting_position_index must be in [0, 959]")

        game = state.play_games.new_game(
            user_color=color, starting_position_index=sp
        )

        # If engine moves first, generate that move
        engine_first_move = None
        engine_first_san = None
        if game.board.turn != color:
            move, _ = engine.choose_move(game.board, n_simulations=state.n_simulations)
            if move is not None and move in game.board.legal_moves:
                san = game.board.san(move)
                game.board.push(move)
                game.move_history_san.append(san)
                engine_first_move = move.uci()
                engine_first_san = san

        resp = game.public_dict()
        resp["engine_move_uci"] = engine_first_move
        resp["engine_move_san"] = engine_first_san
        return JSONResponse(resp)

    @app.post("/api/play/move")
    def play_move(req: PlayMoveReq) -> JSONResponse:
        if not state.demo_enabled:
            raise HTTPException(503, "Demo disabled")
        game = state.play_games.get(req.game_id)
        if game is None:
            raise HTTPException(404, "Game not found")
        if game.result is not None:
            raise HTTPException(400, "Game is already over")
        if game.board.turn != game.user_color:
            raise HTTPException(400, "Not your turn")

        # Parse move (accept lowercase, strip whitespace)
        uci = req.uci.strip().lower()
        try:
            move = chess.Move.from_uci(uci)
        except (chess.InvalidMoveError, ValueError) as e:
            raise HTTPException(400, f"Invalid UCI: {e}") from e
        # Translate standard-notation castling (e.g. e1g1) to the chess960
        # king-takes-rook form expected by python-chess in chess960 mode.
        move = _translate_castle_if_needed(game.board, move)
        if move not in game.board.legal_moves:
            raise HTTPException(400, f"Illegal move: {uci}")

        # User move
        user_san = game.board.san(move)
        game.board.push(move)
        game.move_history_san.append(user_san)
        finalize_if_terminal(game)

        engine_move_uci = None
        engine_move_san = None
        if game.result is None:
            engine = state.get_engine()
            # Allow client-side override of MCTS sim count for the "fast / normal
            # / strong" speed selector. Clamp to a safe range.
            sims = state.n_simulations
            if req.n_simulations is not None:
                sims = max(20, min(800, int(req.n_simulations)))
            eng_move, _ = engine.choose_move(game.board, n_simulations=sims)
            if eng_move is None or eng_move not in game.board.legal_moves:
                raise HTTPException(500, "Engine produced no legal move")
            engine_move_san = game.board.san(eng_move)
            game.board.push(eng_move)
            game.move_history_san.append(engine_move_san)
            engine_move_uci = eng_move.uci()
            finalize_if_terminal(game)

        resp = game.public_dict()
        resp["user_move_uci"] = uci
        resp["user_move_san"] = user_san
        resp["engine_move_uci"] = engine_move_uci
        resp["engine_move_san"] = engine_move_san
        return JSONResponse(resp)

    @app.post("/api/play/resign")
    def play_resign(req: ResignReq) -> JSONResponse:
        game = state.play_games.get(req.game_id)
        if game is None:
            raise HTTPException(404, "Game not found")
        if game.result is None:
            game.result = "0-1" if game.user_color == chess.WHITE else "1-0"
            game.termination = "RESIGNATION"
        return JSONResponse(game.public_dict())

    # ---------- watch (NN vs Stockfish) ----------

    @app.post("/api/watch/new_match")
    def watch_new_match(req: NewMatchReq) -> JSONResponse:
        if not state.demo_enabled:
            raise HTTPException(503, "Demo disabled")
        engine = state.get_engine()
        sf_path = state.ensure_stockfish()

        sp = req.starting_position_index if req.starting_position_index is not None else 518
        if sp < 0 or sp > 959:
            raise HTTPException(400, "starting_position_index must be in [0, 959]")
        if req.skill_level < 0 or req.skill_level > 20:
            raise HTTPException(400, "skill_level must be in [0, 20]")
        nn_white = (
            req.nn_plays_white
            if req.nn_plays_white is not None
            else random.random() < 0.5
        )

        board = chess.Board(chess960=True)
        board.set_chess960_pos(sp)

        match = WatchMatch(
            match_id=uuid.uuid4().hex[:8],
            starting_position_index=sp,
            nn_plays_white=nn_white,
            skill_level=req.skill_level,
            n_simulations=state.n_simulations,
            movetime_s=req.movetime_s,
            starting_fen=board.fen(),
        )
        state.watch_matches.add(match)

        move_q: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                run_watch_match(
                    match, engine, sf_path,
                    on_move=lambda wm: move_q.put(wm),
                )
            finally:
                move_q.put(None)

        # Attach queue to match for SSE handler
        state.watch_queues[match.match_id] = move_q
        threading.Thread(target=worker, daemon=True).start()

        return JSONResponse(match.init_dict())

    @app.get("/api/watch/{match_id}/stream")
    def watch_stream(match_id: str) -> StreamingResponse:
        match = state.watch_matches.get(match_id)
        if match is None:
            raise HTTPException(404, "Match not found")
        q = state.watch_queues.get(match_id)
        if q is None:
            raise HTTPException(404, "Match queue not found")

        def gen():
            yield _sse("init", match.init_dict())
            while True:
                try:
                    item = q.get(timeout=180)
                except queue.Empty:
                    yield _sse("error", {"error": "stream timeout"})
                    return
                if item is None:
                    yield _sse("end", {
                        "result": match.result,
                        "error": match.error,
                        "total_plies": len(match.moves),
                    })
                    return
                yield _sse("move", watchmove_to_json_dict(item))

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# ============================================================
# Request models
# ============================================================


class NewGameReq(BaseModel):
    user_color: str = Field(default="white", description="'white', 'black', or 'random'")
    starting_position_index: int | None = Field(
        default=None, description="0-959 chess960 SP index; default 518 (standard)"
    )


class PlayMoveReq(BaseModel):
    game_id: str
    uci: str
    n_simulations: int | None = Field(
        default=None,
        description="Optional override for MCTS sims on this move (fast/normal/strong selector).",
    )


class ResignReq(BaseModel):
    game_id: str


class NewMatchReq(BaseModel):
    skill_level: int = 5
    nn_plays_white: bool | None = None
    starting_position_index: int | None = None
    movetime_s: float = 0.3


# ============================================================
# Internal state container
# ============================================================


class _DemoState:
    def __init__(
        self,
        checkpoint: Path | None,
        stockfish_path: Path | None,
        n_simulations: int,
        device: str,
    ):
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.stockfish_path = Path(stockfish_path).resolve() if stockfish_path else None
        self.n_simulations = n_simulations
        self.device = device
        self.engine: Engine | None = None
        self._engine_lock = threading.Lock()
        self.play_games = PlayGameRegistry()
        self.watch_matches = WatchMatchRegistry()
        self.watch_queues: dict[str, queue.Queue] = {}

    @property
    def demo_enabled(self) -> bool:
        return self.checkpoint is not None and self.checkpoint.exists()

    def get_engine(self) -> Engine:
        if not self.demo_enabled:
            raise HTTPException(503, "Demo disabled: no checkpoint configured")
        with self._engine_lock:
            if self.engine is None:
                print(f"[demo] loading engine from {self.checkpoint} ...")
                self.engine = Engine(EngineConfig(
                    checkpoint=self.checkpoint,
                    device=self.device,
                    n_simulations=self.n_simulations,
                ))
                print(f"[demo] engine loaded on device={self.engine.device}")
            return self.engine

    def ensure_stockfish(self) -> Path:
        if self.stockfish_path is not None and self.stockfish_path.exists():
            return self.stockfish_path
        print("[demo] locating/downloading stockfish ...")
        self.stockfish_path = download_stockfish(Path("bin/stockfish"))
        print(f"[demo] using stockfish: {self.stockfish_path}")
        return self.stockfish_path


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _translate_castle_if_needed(board: chess.Board, move: chess.Move) -> chess.Move:
    """Map a standard-notation castle (e.g. ``e1g1``) to the chess960
    king-takes-rook form (e.g. ``e1h1``) used by python-chess in chess960 mode.

    If the move is not a king move, or if no matching castling move is legal,
    returns the input move unchanged.
    """
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.KING:
        return move
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    # Castling stays on the same rank and moves the king at least 2 files.
    if from_rank != to_rank or abs(from_file - to_file) < 2:
        return move
    direction = 1 if to_file > from_file else -1
    for legal in board.legal_moves:
        if legal.from_square != move.from_square:
            continue
        if not board.is_castling(legal):
            continue
        # In chess960 mode, the castling target is the rook's square.
        legal_target_file = chess.square_file(legal.to_square)
        if (legal_target_file - from_file) * direction > 0:
            return legal
    return move


# ============================================================
# Helpers (training runs)
# ============================================================


def _safe_run_path(runs_dir: Path, name: str) -> Path:
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
