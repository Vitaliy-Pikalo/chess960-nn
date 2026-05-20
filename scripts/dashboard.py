"""Launch the live training dashboard.

Example:
    uv run python scripts/dashboard.py --runs-dir runs --port 8000
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

    app = create_app(runs_dir=runs_dir, web_dir=web_dir)
    print(f"Dashboard runs from: {runs_dir}")
    print(f"Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
