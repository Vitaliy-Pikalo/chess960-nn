# chess960-nn — how it works, step by step

a plain-language walkthrough of this project, written so someone who hasn't built a chess engine before can follow along. if you want the bottom-line results, see [`SUMMARY.md`](./SUMMARY.md). this file is the "how" and "why".

---

## 1. what is this project?

a neural-network chess engine that plays **chess960** (also called fischer random chess). the engine learns by:

1. first studying human master games (supervised learning).
2. then playing millions of moves against itself, improving as it goes (reinforcement learning).

the design is based on **alphazero** — the deepmind system that taught itself to play chess, go, and shogi from scratch. ours is a smaller, single-gpu version.

### what is chess960?

regular chess has one starting position. **chess960 has 960 different starting positions** — the back-rank pieces (rooks, knights, bishops, queen, king) get shuffled at the start of each game, with two rules:

- the king must be between the two rooks (so castling still makes sense).
- the two bishops must be on opposite-colored squares.

this makes opening theory ("memorize 20 moves of the najdorf") basically useless — you have to actually understand chess from move 1. that makes it interesting for human players and a good test for an engine.

### why a neural net?

traditional engines (stockfish, leela) are extremely strong but took decades of hand-engineering or massive compute. an alphazero-style net learns positions and moves from data, no chess knowledge hard-coded except the rules. it's a fun project because you can build the whole stack yourself.

---

## 2. the big picture (one diagram in words)

think of the engine as three things working together:

```
       +-----------------+
       |   neural net    |   <-- looks at a position, outputs:
       | (the "brain")   |       1. probability for each move
       +-----------------+       2. who's winning (-1..+1)
              ^                
              |
       +-----------------+
       |     mcts        |   <-- runs many "what if" simulations
       | (the "planner") |       using the net's guesses,
       +-----------------+       picks the move that holds up best
              ^                
              |
       +-----------------+
       |  self-play /    |   <-- the engine plays itself,
       |  training loop  |       saves the games, trains the net
       +-----------------+       on its own results
```

every time the engine "thinks", it's the net + mcts. every time it "learns", it's the training loop using games the engine just played.

---

## 3. how each piece works

### 3.1 board encoding (`src/chess960_nn/encoding.py`)

the neural net can't read a chess board directly. it reads numbers. we convert each position into a stack of **8×8 planes** (think: layers of an 8×8 image, one layer per "thing we want to tell the net"):

- one plane per piece type per color (12 planes: white pawn, white knight, ... black king).
- planes for castling rights, side to move, etc.

the result is a 3d array (planes × 8 × 8). the net learns to read these like an image.

**moves** get encoded as integers too — there's a fixed list of possible moves ("from square 12 to square 28", "promote pawn at 56 to queen", etc.), and the net outputs a probability for each one.

### 3.2 the network (`src/chess960_nn/model.py`)

a **residual convolutional network**:

- input: the board planes.
- 10 "residual blocks" (a standard image-net building block — convolutions + skip connections).
- 192 filters per layer (channel count).
- **two heads** at the end:
  - **policy head** — outputs a probability per legal move ("how good does each move look?").
  - **value head** — outputs a single number from -1 to +1 ("am i winning?").
- about **6.7 million parameters** total. small by modern standards; fits easily on the 8gb 3060 ti.

### 3.3 mcts — monte carlo tree search (`src/chess960_nn/mcts.py`)

the net's raw output isn't strong enough on its own. **mcts** is a search algorithm that takes the net's guesses and refines them by simulating many possible futures.

simplified loop:

1. start at the current position. ask the net: "what moves are promising?".
2. pick a promising move, advance the board.
3. recurse — go a few moves deep, asking the net at each step.
4. once you've gone far enough (or hit game over), backpropagate the value estimate back up the tree.
5. repeat hundreds of times.
6. the move that got searched the most at the root = the move we play.

the magic: even with a mediocre net, mcts can play surprisingly well because it explores. and the search results become better training data for the net (see 3.5).

in this project we use **100-200 mcts simulations per move** depending on the script. alphazero used 800+; we're constrained by gpu time.

### 3.4 self-play (`src/chess960_nn/selfplay.py`)

the engine plays full games against itself. for each move it makes, it records:

- the position (as planes).
- the mcts visit counts at the root (a "policy target" — which moves the search liked).
- once the game ends, who won (-1 / 0 / +1) — this becomes the "value target".

after 30 games we have ~2000 positions to train on, each with a policy target and a value target.

### 3.5 training (`src/chess960_nn/train.py`)

standard supervised learning, just on data the engine generated itself:

- show the net a position.
- ask it to predict the policy and value.
- compare against the targets (mcts visits + game outcome).
- backprop, update weights.

repeat for many batches. the net gets a bit better at predicting which moves mcts will like and who's winning.

**the bootstrapping insight from alphazero:** the policy target is mcts's output, and mcts uses the net. so as the net improves, mcts gets better targets, which trains the net to be even better. that's the self-improving loop.

### 3.6 gated promotion (`src/chess960_nn/eval.py`)

a freshly-trained net could be worse than the previous one. so before we let it become the "champion" we run a **gating match**:

- new net plays 10 games against the current champion.
- if it scores ≥ 0.55 (i.e. wins more than it loses), it becomes the new champion.
- otherwise we discard it and try again from the current champion.

this catches regressions. **it was the single most useful safety net in this project.** without it, iteration 1 of our rl loop would have replaced a decent champion with a worse one.

### 3.7 stockfish eval (`src/chess960_nn/stockfish.py`, `scripts/eval_vs_stockfish.py`)

we need an absolute yardstick for "how strong is our engine, in elo terms?". we use **stockfish** (the world's strongest open-source chess engine) at limited skill levels.

stockfish has skill levels 0-20. each skill level has a rough elo equivalent (skill 0 ≈ 1320, skill 5 ≈ 1787, skill 10 ≈ 2255, skill 20 is super-grandmaster). we play our net against stockfish at several skill levels and estimate elo from the score.

**caveat:** with only ~4-6 games per skill level, the noise is huge (~±400 elo). small evals only tell you very roughly where you are. for the final number we run 100 games.

---

## 4. step-by-step: what i actually did

phases roughly in chronological order.

### phase 1 — scaffold

set up the project skeleton: `pyproject.toml`, package layout, `uv` for dependency management, basic config system, tests, lint, ci-style checks.

### phase 2 — board encoding

implemented the planes-from-board encoding and tested it on every chess960 starting position to make sure castling rights, en passant, etc. round-trip correctly.

### phase 3 — model

implemented the residual conv net with policy + value heads. unit tests for shapes, forward pass, gpu placement.

### phase 4 — mcts

implemented monte-carlo tree search with the puct selection rule (alphazero-style). tested against trivial positions (mate-in-1, mate-in-2) to make sure it actually finds the right move.

### phase 5 — self-play

wired the net + mcts into a self-play loop that plays games and emits training data. tested by playing a few games and inspecting the output.

### phase 6 — rl loop infra

wrote the multi-iteration driver: self-play → train → gating match → maybe promote. saves checkpoints, metrics, can resume after a crash.

### phase 7 — dashboard

simple local web dashboard (`src/chess960_nn/dashboard.py`, `web/`) for watching training metrics in real time.

### phase 8 — pretrain pipeline

built the supervised pretraining pipeline:

- downloaded lichess master-games databases (millions of games).
- filtered for high-rated games.
- generated training positions with the human's move as the target ("the master played e4 here — try to predict e4").
- trained the net for 5 epochs on 6.4 million positions.

**result:** after pretrain, the net plays at roughly 1600-1800 elo vs stockfish.

### phase 9 — stockfish eval

implemented the uci wrapper around stockfish so we can talk to it programmatically. wrote the eval harness that plays n games at each skill level and computes an elo estimate.

### phase 10 — rl loop infra (refined)

hardened the rl loop: better logging, resumability, the multi-iteration powershell launcher (`RL loop full.ps1`).

### phase 11 — actually run the rl loop

ran 5 iterations of self-play rl on top of the pretrained net.

**config used:**
- 30 self-play games per iteration
- 100 mcts simulations per move
- 800 training steps per iteration
- 10 games per gating match
- stockfish eval every 2 iterations

**outcome:** only iteration 0 promoted past the gate. iterations 1-4 all played close but couldn't beat iter 0 by enough margin (≥0.55).

### phase 12 — writeup + push

what you're reading now, plus `SUMMARY.md` and pushing to github.

---

## 5. file map

```
chess960-nn/
├── src/chess960_nn/
│   ├── encoding.py     <- board <-> planes
│   ├── model.py        <- the neural network
│   ├── mcts.py         <- monte-carlo tree search
│   ├── selfplay.py     <- play games against itself
│   ├── train.py        <- training loop / checkpoints
│   ├── rl.py           <- rl-specific dataset utilities
│   ├── eval.py         <- gating match (net vs net)
│   ├── stockfish.py    <- uci wrapper + stockfish eval
│   ├── dashboard.py    <- live metrics web UI
│   └── data/           <- dataset loaders
│
├── scripts/
│   ├── build_dataset.py     <- prepare pretrain dataset
│   ├── pretrain.py          <- supervised pretrain entry point
│   ├── selfplay.py          <- run a self-play session (standalone)
│   ├── rl_iteration.py      <- one iteration of the rl loop
│   ├── rl_loop.py           <- multi-iter rl driver
│   ├── eval_vs_stockfish.py <- the final yardstick
│   └── check_cuda.py        <- gpu sanity check
│
├── runs/
│   ├── pretrain-002/        <- the supervised baseline
│   ├── rl-loop-001/         <- the rl run (5 iters)
│   │   ├── iter-000/        <- iter checkpoints + selfplay games
│   │   ├── ...
│   │   ├── final_best.pt    <- the final champion (= iter-0 best)
│   │   ├── metrics.jsonl    <- everything that happened
│   │   ├── sf-eval-*.json   <- intermediate stockfish checks
│   │   └── loop_state.json  <- resume state
│   ├── stockfish-eval-001.json   <- baseline eval after pretrain
│   └── sf-eval-final.json        <- final 100-game eval
│
├── tests/                  <- unit tests
├── web/                    <- dashboard frontend
├── README.md
├── SUMMARY.md              <- bottom-line results
└── EXPLAINER.md            <- this file
```

(launchers like `RL loop full.ps1` and `STOCKFISH eval FINAL.ps1` live one level up, in `C:\Users\pikal\Downloads\chess bot\`, because the user runs them via right-click → "run with powershell".)

---

## 6. how to reproduce

prerequisites:
- windows 11 (the launchers are powershell; on linux/mac you'd run the python commands directly).
- python 3.11+ via [uv](https://github.com/astral-sh/uv).
- nvidia gpu with cuda 12+ (3060 ti or better; 8gb vram is enough).
- ~50gb free disk (datasets + checkpoints).

steps:

```powershell
# 1. install deps
uv sync

# 2. sanity check the gpu
uv run python scripts/check_cuda.py

# 3. prepare the pretrain dataset (downloads lichess + processes it)
uv run python scripts/build_dataset.py

# 4. pretrain
uv run python scripts/pretrain.py

# 5. baseline stockfish check
# (right-click "STOCKFISH eval demo.ps1" -> run with powershell)

# 6. rl loop
# (right-click "RL loop full.ps1" -> run with powershell)

# 7. final eval
# (right-click "STOCKFISH eval FINAL.ps1" -> run with powershell)
```

every script accepts `--help` to see its options.

---

## 7. what i'd do differently

honest list, in priority order:

1. **way more self-play per iteration.** 30 games is too few. alphazero used thousands; even a few hundred would have given the rl loop a real chance.
2. **more mcts simulations.** 100/move is on the low end. 400+ would produce better policy targets.
3. **bigger / smarter gating sample.** 10 games is also noisy — a small skill difference can hide under sampling variance, so good iterations might not promote.
4. **balanced opening pairs in eval.** chess960 has 960 starting positions; one bad opening line can dominate a small batch. play each position from both sides.
5. **a watchdog on the rl loop.** iteration 3 stalled for ~7 hours instead of the usual 35 minutes (probably the machine slept). a heartbeat check would catch that.
6. **better elo measurement.** ~4-6 games per skill level produced ±400 elo swings on the *same* checkpoint. final eval bumps this to 20 games per level, which still isn't tournament-grade but is enough to make defensible claims.

most of these are budget problems, not design problems. the architecture is fine; it just needs more compute time.

---

## 8. glossary

- **alphazero** — the deepmind system this project is modeled on. self-play rl with mcts + a neural net.
- **chess960 / fischer random** — chess variant with 960 randomized starting positions.
- **elo** — chess rating system. ~1000 = beginner, 1500 = club player, 2000 = strong club player, 2500 = master, 2800+ = top grandmaster.
- **mcts** — monte carlo tree search. simulates many possible futures to refine the net's move probabilities.
- **policy head / value head** — the two outputs of the net. policy = "which move?", value = "who's winning?".
- **promotion gate** — the rule "only replace the champion if the new net beats it by a meaningful margin".
- **self-play** — the engine plays games against itself to generate training data.
- **stockfish** — the world's strongest open-source chess engine. we use it as our elo yardstick.
- **uci** — universal chess interface, the text protocol engines speak.
