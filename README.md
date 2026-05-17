# chess960-nn

Chess960 (Fischer Random) neural network engine. AlphaZero-style hybrid: supervised pretrain on Lichess master games, then RL self-play.

**Status:** work in progress. Phase 1: scaffold complete.

## Quickstart

```powershell
uv run python scripts/check_cuda.py
uv run pytest
uv run ruff check src tests scripts
```
