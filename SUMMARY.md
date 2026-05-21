# chess960-nn — project summary

## tl;dr

built an alphazero-style chess960 engine on an rtx 3060 ti. pretrain on lichess masters got the net to ~1600-1800 elo vs stockfish skill levels. ran 5 iterations of rl self-play on top; only iteration 0 promoted past the gate. final elo estimate vs stockfish: **[FINAL ELO TBD — pending sf-eval-final.json]**.

goal was 2000+ elo (to beat ~2000-rated humans). honest assessment: **likely short of that target**, and the noise on small-sample stockfish evals is large enough that earlier "+200 elo" claims were within noise. see [final eval](#final-evaluation) for the defensible number.

## goal & approach

- **target:** chess960 engine strong enough to beat ~2000-rated club players.
- **hardware:** rtx 3060 ti (8gb vram), windows 11, python + pytorch + uv.
- **architecture:** alphazero-style residual conv net (10 blocks, 192 filters, ~6.7m params) + mcts.
- **pipeline:**
  1. **supervised pretrain** on lichess master games (chess960-relevant openings + general positions).
  2. **rl self-play loop** — each iteration: self-play games → train on (state, policy, value) → gated match vs current champion. promote if score ≥ 0.55.
  3. **stockfish eval** every 2 iterations to anchor elo.

## results

### baseline (after pretrain)

run: `runs/stockfish-eval-001.json` (6 games per skill level)

| stockfish skill | sf elo | score | nn elo est |
|---|---|---|---|
| 0 | 1320 | 0.833 (5W/1L/0D) | 1600 |
| 5 | 1787 | 0.250 (0W/3L/3D) | 1596 |
| 10 | 2255 | 0.083 (0W/5L/1D) | 1838 |
| **avg** | | | **~1678** |

### rl loop (5 iterations)

run: `runs/rl-loop-001/`

| iter | self-play (W/L/D) | match score | promoted | wall time |
|---|---|---|---|---|
| 0 | 11/14/5 | 0.60 (4W/2L/4D) | ✅ yes | 36 min |
| 1 | 16/12/2 | 0.20 (1W/7L/2D) | ❌ | 29 min |
| 2 | 8/16/6 | 0.50 (2W/2L/6D) | ❌ | 33 min |
| 3 | 11/11/8 | 0.40 (2W/4L/4D) | ❌ | 7h (anomaly) |
| 4 | 14/14/2 | 0.45 (3W/4L/3D) | ❌ | 36 min |

→ **final champion = iter-0 checkpoint.** later iterations could not beat it by enough margin to promote.

### intermediate stockfish evals (4 games per skill, very noisy)

| | baseline (pretrain) | after iter 1 (iter-0 best) | after iter 3 (iter-0 best) |
|---|---|---|---|
| vs skill 0 (1320) | 0.833 (n=6) | 1.000 (n=4) | 0.500 (n=4) |
| vs skill 5 (1787) | 0.250 (n=6) | 0.625 (n=4) | 0.125 (n=4) |
| vs skill 10 (2255) | 0.083 (n=6) | 0.000 (n=4) | 0.125 (n=4) |

→ note the wild swings between "after iter 1" and "after iter 3" — **same checkpoint**, just different game samples. 4 games per cell is statistically useless.

### final evaluation

run: `runs/sf-eval-final.json` (100 games — 20 per skill level × 5 skill levels)

**[TBD — to be filled in once `STOCKFISH eval FINAL.ps1` finishes]**

| stockfish skill | sf elo | score (n=20) | nn elo est | 95% CI |
|---|---|---|---|---|
| 0 | 1320 | — | — | — |
| 3 | ~1600 | — | — | — |
| 5 | 1787 | — | — | — |
| 7 | ~2000 | — | — | — |
| 10 | 2255 | — | — | — |
| **best estimate** | | | **TBD** | |

## wall time

| phase | wall time |
|---|---|
| supervised pretrain (5 epochs, 6.4m positions) | ~1h 39min |
| baseline stockfish eval | ~5 min |
| rl loop (5 iters incl. eval) | ~9h (incl. ~7h iter-3 anomaly) |
| rl loop minus anomaly | ~2h 50min |
| final stockfish eval | TBD (~60-90 min expected) |

iter 3 took ~7 hours instead of the usual ~35 min — most likely the machine slept or the process stalled. did not affect correctness, just wall time. duplicate iter-4 entry in `metrics.jsonl` suggests a crash + resume during iter 4 as well.

## what worked

- **uv + pytorch + python-chess** stack — zero friction, fast iteration.
- **`UCI_Chess960` auto-managed** by python-chess once we let it handle it (don't set it manually).
- **gated promotion** caught real regressions: iter 1 self-play looked positive (16W/12L/2D) but the gating match showed it was actually worse vs iter 0. without the gate we'd have shipped a regression.
- **separate pretrain checkpoint as starting point** for rl was the right call — pretrain gave a strong prior, rl just couldn't push it further with the budget we had.

## what didn't

- **rl loop produced no net improvement past iter 0.** iters 1-4 all failed to clear the 0.55 promotion gate. likely causes:
  - 30 self-play games per iter is too few — not enough data per training step.
  - 100 mcts simulations per move is on the low end; alphazero-style improvement typically needs 400-800 sims and thousands of games per iter.
  - 800 train steps per iter may be over-fitting to the small self-play buffer.
- **stockfish elo estimates were extremely noisy.** 4 games per skill level produced ±400 elo swings on identical checkpoints (see iter-1 vs iter-3 evals above). single-sample elo claims should not be trusted at this game count.
- **iter 3 stalled for hours** — no monitoring/heartbeat in the loop driver.
- **chess960 starting position diversity** means a single bad opening line can dominate a small eval batch — we should sample positions more aggressively or use balanced opening pairs (each position played from both sides).

## honest assessment vs the 2000 elo goal

even taking the most optimistic intermediate number (~1876 from sf-eval-after-iter-001), we're ~125 elo short of the stated goal. and that number was within the noise floor — the same checkpoint scored 1449 on a different sample. the defensible number is whatever sf-eval-final.json says with 100 games behind it.

**realistic expectation:** the engine is strong enough to give a moderately strong club player a real game, but probably not consistently beat a true 2000.

## if i kept going

ordered by expected impact:

1. **more self-play volume per iter.** 100-200 games per iter, not 30. with the 3060 ti this is mostly a wall-time problem, not a memory one.
2. **bigger mcts budget.** 400 sims/move minimum for the training games.
3. **balanced opening pairs in eval.** each chess960 start position played twice (once from each side) to control for opening variance.
4. **heartbeat/wall-clock watchdog** in `rl_loop.py` to detect stalls (e.g., iter 3).
5. **separate "fast eval" net** (smaller, fewer sims) for the gating match so we can run 30+ gating games cheaply instead of 10.
6. **policy temperature annealing** during self-play — start hot for exploration, cool for the gating match.

## files of record

- `runs/pretrain-002/checkpoints/last.pt` — supervised baseline (~1678 elo avg vs sf skills 0/5/10).
- `runs/rl-loop-001/final_best.pt` — final champion (= iter-0 best, never beaten by later iters).
- `runs/rl-loop-001/metrics.jsonl` — per-phase metrics for every iteration.
- `runs/stockfish-eval-001.json` — baseline stockfish eval.
- `runs/sf-eval-final.json` — final 100-game stockfish eval (pending).

## reproducibility

```powershell
# pretrain (or use existing runs/pretrain-002/)
uv run python scripts/pretrain.py --config configs/pretrain.yaml

# rl loop
.\"RL loop full.ps1"

# final eval
.\"STOCKFISH eval FINAL.ps1"
```
