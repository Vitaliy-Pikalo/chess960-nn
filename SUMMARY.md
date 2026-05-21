# chess960-nn

**End-to-end AlphaZero-style neural chess engine for Chess960 (Fischer Random), built from scratch on a single consumer GPU.**

Final strength: **~1600 Elo** vs. Stockfish, measured over 100 games across 5 calibrated skill levels.

[Project explainer](./EXPLAINER.md) · [GitHub repo](https://github.com/Vitaliy-Pikalo/chess960-nn) · Stack: Python · PyTorch · CUDA · `uv` · `python-chess` · Stockfish UCI

---

## What I built

A complete, reproducible AlphaZero-style training pipeline targeting Chess960:

- **Custom board encoding** (8×8 plane stack) supporting all 960 Chess960 starting positions and full move space.
- **Residual convolutional network** (10 blocks, 192 filters, 6.7M parameters) with dual policy + value heads.
- **Monte Carlo Tree Search** with PUCT selection, integrated with the network for move selection and training-target generation.
- **Supervised pretraining pipeline** on 6.4M positions from Lichess master games.
- **Reinforcement learning self-play loop** with self-play data generation, gradient updates, and **gated promotion** (new networks must beat the current champion before replacing it).
- **Stockfish-based evaluation harness** with calibrated skill levels for absolute Elo measurement.
- **Resumable multi-iteration driver** with crash recovery, per-iteration checkpoints, and JSON metrics streaming.
- **Live training dashboard** for real-time loss/score monitoring.

All on an RTX 3060 Ti (8 GB VRAM), Windows 11, single workstation.

---

## Results

### Final evaluation (100 games, 20 per skill level)

| Stockfish skill | Stockfish Elo | Score (n=20) | NN Elo estimate |
|---|---|---|---|
| 0  | 1320 | 0.900 (17W / 1L / 2D)  | 1702 |
| 3  | 1600 | 0.450 (8W / 10L / 2D)  | 1565 |
| 5  | 1787 | 0.300 (3W / 11L / 6D)  | 1640 |
| 7  | 1974 | 0.150 (1W / 15L / 4D)  | 1673 |
| 10 | 2255 | 0.025 (0W / 19L / 1D)  | 1619 |

- **Cross-over (~50% score) against Stockfish skill 3 → ~1600 Elo.**
- Mean across all 5 levels: 1640 Elo. Curve is smooth and monotonic, indicating a consistent measurement rather than a single bad batch.

### Baseline (supervised pretraining only, 6 games per skill level)

| Skill | Score | NN Elo estimate |
|---|---|---|
| 0  | 0.833 | 1600 |
| 5  | 0.250 | 1596 |
| 10 | 0.083 | 1838 |

Baseline mean (skills 0/5/10): 1678. Final mean on same skills: 1654. **Net change from RL: within measurement noise.** The strength of the engine is attributable to supervised pretraining; the RL loop did not extend it under the compute budget used.

### Why an earlier intermediate estimate of 1876 Elo was discarded

Mid-loop Stockfish evaluations used only 4 games per skill level. The *same* checkpoint produced score swings of 1.0 → 0.5 at skill 0 and 0.625 → 0.125 at skill 5 across two evaluations — roughly ±400 Elo of measurement noise. The 100-game final eval was commissioned specifically to control this, and it confirmed the smaller samples were unreliable.

---

## Engineering decisions that worked

| Decision | Why it mattered |
|---|---|
| **Gated promotion** (≥0.55 score vs. current champion) | Caught a regression at iteration 1 (1W/7L/2D) that self-play stats had hidden. Without the gate the loop would have shipped a weaker network. |
| **Scaling up evaluation when noise was suspected** | Moved from 4-game to 20-game-per-level evaluation after observing implausible Elo swings. Updated the public claim from 1876 → 1600 once the data demanded it. |
| **Per-iteration JSON metrics + resumable state** | Loop survived a mid-iteration crash during iter 4 and a ~7-hour iter-3 stall without data loss. |
| **Separate pretrain and RL stages with isolated checkpoint directories** | Made it trivial to pin the supervised baseline as the absolute reference point. |
| **Strict `.gitignore` for checkpoints, datasets, and binaries** | Repo stays ~20 KB of code/docs; no 140 MB Stockfish binary or 80 MB `.pt` files in history. |

---

## Honest assessment vs. the original goal

Target was **2000+ Elo** to compete with strong club players. Actual: **~1600 Elo**. Roughly 400 Elo short.

The bottleneck is identifiable and is not architectural:

- **Self-play volume** — 30 games per iteration is far below the thousands used in AlphaZero-class training. The RL loop never saw enough diverse data to improve on the pretrain prior.
- **MCTS simulation budget** — 100 sims/move yields noisy policy targets. AlphaZero used 800.
- **Gating sample size** — 10 games per match is too noisy to reliably detect small improvements; good iterations may have failed to promote.

The architecture (network depth/width, MCTS, evaluation methodology) is sound. The gap to 2000 Elo is a compute-budget problem, not a design problem.

---

## What I would do differently

In priority order by expected impact:

1. **10× the self-play volume per iteration** (300+ games instead of 30).
2. **4× MCTS simulations per move** (400+ instead of 100).
3. **Bigger gating matches** (40+ games) or a faster eval network for the gating loop.
4. **Balanced opening pairs** in evaluation — each Chess960 starting position played from both sides to control for opening-line variance.
5. **Heartbeat / wall-clock watchdog** on the RL loop driver to catch the iter-3-style stall (~7 h instead of ~35 min).
6. **Policy temperature annealing** during self-play for exploration → exploitation balance.

---

## Wall time

| Phase | Time |
|---|---|
| Supervised pretrain (5 epochs, 6.4 M positions) | ~1 h 40 min |
| RL loop (5 iterations) | ~9 h (includes a ~7 h iter-3 stall) |
| RL loop excluding the stall | ~2 h 50 min |
| Final Stockfish evaluation (100 games) | ~50 min |
| **Total active compute** | **~5 h 20 min** |

---

## Tech stack

**Languages / runtime:** Python 3.11, PyTorch (CUDA 12), `uv` for dependency + venv management.
**Chess infrastructure:** `python-chess` for rules + UCI; Stockfish 16 (AVX2 build) as the absolute Elo yardstick.
**Tooling:** `pytest`, `ruff`, PowerShell launchers, JSON Lines metrics streaming, lightweight web dashboard.
**Hardware:** Single RTX 3060 Ti (8 GB), Windows 11.

---

## Reproducibility

Full pipeline runs from a clean checkout:

```powershell
uv sync                                      # install deps
uv run python scripts/check_cuda.py          # GPU sanity check
uv run python scripts/build_dataset.py       # Lichess master-game corpus
uv run python scripts/pretrain.py            # supervised pretrain
.\"RL loop full.ps1"                         # 5-iteration RL loop
.\"STOCKFISH eval FINAL.ps1"                 # final 100-game eval
```

Every script accepts `--help`. Run-level outputs land in `runs/` with per-iteration checkpoints, JSON metrics, and self-play data.

---

## Repository layout

```
chess960-nn/
├── src/chess960_nn/    # board encoding, model, MCTS, self-play, training, eval, Stockfish UCI
├── scripts/            # pretrain, RL iteration, RL loop, Stockfish eval entry points
├── runs/               # checkpoints + metrics (gitignored)
├── tests/              # unit tests
├── EXPLAINER.md        # plain-language project walkthrough
├── SUMMARY.md          # this file
└── README.md
```

See [EXPLAINER.md](./EXPLAINER.md) for a step-by-step walkthrough of how each component works, the order of the build phases, and a glossary for non-chess-engine readers.
