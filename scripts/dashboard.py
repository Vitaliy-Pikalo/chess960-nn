"""Launch the live training dashboard + interactive demo.

Examples:
    # Metrics only (training viz):
    uv run python scripts/dashboard.py --runs-dir runs --port 8000

    # With play + watch demo enabled:
    uv run python scripts/dashboard.py \\
        --runs-dir runs --port 8000 \\
        --checkpoint runs/rl-loop-001/final_best.pt

Then open http://localhost:8000/ in your browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from chess960_nn.dashboard import create_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"),
                   help="Directory containing run subdirs (default: ./runs)")
    p.add_argument("--web-dir", type=Path, default=Path("web"),
                   help="Directory containing index.html (default: ./web)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pt checkpoint to enable play/watch demo. "
                        "If omitted, demo endpoints return 503 and only "
                        "training viz works.")
    p.add_argument("--stockfish", type=Path, default=None,
                   help="Path to stockfish.exe. If omitted, auto-downloaded into bin/stockfish.")
    p.add_argument("--n-simulations", type=int, default=100,
                   help="MCTS simulations per move for the demo (default: 100).")
    p.add_argument("--device", default="cuda",
                   help="torch device for the NN (default: cuda; falls back to cpu).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    runs_dir = args.runs_dir.resolve()
    web_dir = args.web_dir.resolve()

    if not runs_dir.is_dir():
        print(f"WARNING: --runs-dir {runs_dir} does not exist; creating it.")
        runs_dir.mkdir(parents=True, exist_ok=True)
    if not (web_dir / "index.html").exists():
        raise SystemExit(f"index.html not found in {web_dir}")

    if args.checkpoint is not None and not args.checkpoint.exists():
        raise SystemExit(f"--checkpoint not found: {args.checkpoint}")

    app = create_app(
        runs_dir=runs_dir,
        web_dir=web_dir,
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        stockfish_path=args.stockfish.resolve() if args.stockfish else None,
        n_simulations=args.n_simulations,
        device=args.device,
    )
    print(f"Dashboard runs from: {runs_dir}")
    if args.checkpoint:
        print(f"Demo checkpoint:     {args.checkpoint}")
        print(f"MCTS simulations:    {args.n_simulations}")
    else:
        print("Demo:                disabled (no --checkpoint given)")
    print(f"Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
