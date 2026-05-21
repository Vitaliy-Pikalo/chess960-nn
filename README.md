# chess960-nn

an alphazero-style neural network engine for **chess960** (fischer random chess).

- supervised pretrain on lichess master games → ~1600-1800 elo vs stockfish.
- alphazero-style rl self-play loop on top, with gated promotion.
- runs on a single nvidia gpu (built on an rtx 3060 ti, 8gb vram).

## docs

- **[`EXPLAINER.md`](./EXPLAINER.md)** — plain-language walkthrough of how the engine works and what was built, step by step. start here if you're new to the project.
- **[`SUMMARY.md`](./SUMMARY.md)** — bottom-line results, wall times, what worked / didn't, and honest assessment vs the goal.

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

see [`EXPLAINER.md` section 6](./EXPLAINER.md#6-how-to-reproduce) for the full reproduction recipe (dataset → pretrain → rl loop → eval).

## status

phases 1-12 complete. final stockfish elo: see `SUMMARY.md` (post 100-game eval).
