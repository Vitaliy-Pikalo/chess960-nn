# chess960-nn

An AlphaZero-style neural network engine for **Chess960** (Fischer Random chess).

- Supervised pretrain on Lichess master games → ~1600 Elo vs Stockfish (100-game eval).
- AlphaZero-style RL self-play loop on top, with gated promotion.
- Runs on a single NVIDIA GPU (built on an RTX 3060 Ti, 8 GB VRAM).
- Interactive demo site: play the engine, watch it face Stockfish, browse training metrics.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/Vitaliy-Pikalo/chess960-nn)

## docs

- **[`EXPLAINER.md`](./EXPLAINER.md)** — plain-language walkthrough of how the engine works and what was built, step by step. Start here if you're new to the project.
- **[`SUMMARY.md`](./SUMMARY.md)** — bottom-line results, wall times, what worked / didn't, and honest assessment vs the goal.

## interactive demo

Three views:

1. **Play vs Engine** — full Chess960 board, click-to-move, the engine replies with MCTS-backed inference (~1-2 s/move on a 3060 Ti at 100 sims).
2. **Watch vs Stockfish** — live NN-vs-Stockfish game streamed move-by-move over server-sent events.
3. **Training** — historical loss curves, accuracy, and RL iteration summaries.

### Run it locally

```powershell
# Right-click DEMO launcher.ps1 -> "Run with PowerShell"
# Or directly:
uv run python scripts/dashboard.py \
    --runs-dir runs \
    --checkpoint runs/rl-loop-001/final_best.pt \
    --port 8000
```

Then open <http://127.0.0.1:8000/>.

### Hosting the public-facing site

The static **Overview** tab is deployable to any static host (Netlify, Vercel, GitHub Pages, etc.). The `netlify.toml` at the project root sets `publish = "web"` and ships sensible security headers.

The interactive **Play / Watch** tabs require the Python backend (FastAPI + PyTorch + Stockfish). When deployed without a backend, those tabs detect the missing engine and show a "static preview — clone for live demo" banner.

## quickstart

```powershell
# install
uv sync

# gpu check
uv run python scripts/check_cuda.py

# tests + lint
uv run pytest
uv run ruff check src tests scripts
```

See [`EXPLAINER.md` section 6](./EXPLAINER.md#6-how-to-reproduce) for the full reproduction recipe (dataset → pretrain → rl loop → eval).

## status

Phases 1-12 complete. Final Stockfish Elo: see [`SUMMARY.md`](./SUMMARY.md).
