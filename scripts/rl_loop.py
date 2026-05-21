"""Multi-iteration AlphaZero RL loop with periodic Stockfish eval.

Runs ``--n-iterations`` of (self-play -> train candidate -> gating match) by
shelling out to ``scripts/rl_iteration.py`` per iteration. Tracks the current
"best" checkpoint, promotes the candidate whenever it beats the threshold,
and inserts a Stockfish strength check every ``--stockfish-every`` iters.

Resume-on-restart: if you close the window mid-run, re-launching with the
same ``--out-dir`` continues at the last completed iteration.

Example:
    uv run python scripts/rl_loop.py \\
        --starting-ckpt runs/pretrain-002/checkpoints/last.pt \\
        --out-dir runs/rl-loop-001 \\
        --n-iterations 5 --selfplay-games 30 --train-steps 800 \\
        --eval-games 10 --stockfish-every 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--starting-ckpt", type=Path, required=True,
                   help="Initial checkpoint (e.g. pretrain output)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-iterations", type=int, default=5)

    # Per-iter RL config (passed straight through to rl_iteration.py)
    p.add_argument("--selfplay-games", type=int, default=30)
    p.add_argument("--selfplay-sims", type=int, default=100)
    p.add_argument("--train-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--promote-threshold", type=float, default=0.55)

    # Stockfish strength check
    p.add_argument("--stockfish-every", type=int, default=2,
                   help="Run Stockfish eval every N iterations (0 = never)")
    p.add_argument("--stockfish-bin", type=Path, default=None,
                   help="Path to stockfish.exe; auto-downloads if omitted")
    p.add_argument("--stockfish-skills", type=str, default="0,5,10")
    p.add_argument("--stockfish-games", type=int, default=4,
                   help="Games per skill level during the interleaved check")

    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load / init state -----
    state_path = args.out_dir / "loop_state.json"
    metrics_path = args.out_dir / "metrics.jsonl"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_iter = int(state["next_iter"])
        current_best = Path(state["current_best"])
        print(f"Resuming at iter {start_iter} with current_best={current_best}")
    else:
        start_iter = 0
        current_best = args.starting_ckpt.resolve()
        # Seed the metrics log with config
        _append_metric(metrics_path, {
            "phase": "loop_init",
            "starting_ckpt": str(current_best),
            "config": {
                "n_iterations": args.n_iterations,
                "selfplay_games": args.selfplay_games,
                "selfplay_sims": args.selfplay_sims,
                "train_steps": args.train_steps,
                "eval_games": args.eval_games,
                "stockfish_every": args.stockfish_every,
            },
        })
        _save_state(state_path, next_iter=0, current_best=current_best)

    t_loop_start = time.time()

    for iter_idx in range(start_iter, args.n_iterations):
        print()
        print("#" * 70)
        print(f"#  ITER {iter_idx + 1}/{args.n_iterations}   "
              f"current_best={current_best.name}")
        print("#" * 70)
        iter_dir = args.out_dir / f"iter-{iter_idx:03d}"

        t0 = time.time()
        rc = _run_rl_iteration(args, current_best, iter_dir)
        elapsed = time.time() - t0
        if rc != 0:
            print(f"ERROR: rl_iteration.py exited {rc}. Aborting loop.")
            return rc

        # Read the iter's summary.json
        summary = json.loads((iter_dir / "summary.json").read_text(encoding="utf-8"))
        promoted = bool(summary.get("promoted", False))
        _append_metric(metrics_path, {
            "phase": "iter",
            "iter": iter_idx,
            "promoted": promoted,
            "elapsed_sec": elapsed,
            "summary": summary,
        })

        # Promote if won
        if promoted:
            promo_src = iter_dir / "checkpoints" / "best.pt"
            current_best = promo_src.resolve()
            print(f"-> promoted. current_best is now {current_best}")
        else:
            print(f"-> not promoted (score {summary['match']['score']:.3f}). "
                  f"current_best unchanged.")

        _save_state(state_path, next_iter=iter_idx + 1, current_best=current_best)

        # Optional Stockfish strength check
        is_eval_iter = (
            args.stockfish_every > 0
            and (iter_idx + 1) % args.stockfish_every == 0
        )
        if is_eval_iter:
            print()
            print("--- stockfish strength check ---")
            t_sf = time.time()
            sf_result = _run_stockfish_eval(args, current_best, iter_idx, args.out_dir)
            sf_elapsed = time.time() - t_sf
            if sf_result is not None:
                _append_metric(metrics_path, {
                    "phase": "stockfish_eval",
                    "iter": iter_idx,
                    "elapsed_sec": sf_elapsed,
                    "results": sf_result,
                })
                print(_format_sf_summary(sf_result))

        total_min = (time.time() - t_loop_start) / 60.0
        print(f"-- iter done in {elapsed/60:.1f} min, total {total_min:.1f} min --")

    print()
    print("=" * 70)
    print(f"Loop complete in {(time.time() - t_loop_start)/60:.1f} min")
    print(f"Final best: {current_best}")
    # Convenience copy
    final = args.out_dir / "final_best.pt"
    shutil.copy2(current_best, final)
    print(f"Copied -> {final}")
    return 0


# ============================================================
# Helpers
# ============================================================


def _save_state(path: Path, *, next_iter: int, current_best: Path) -> None:
    path.write_text(
        json.dumps({"next_iter": next_iter, "current_best": str(current_best)}, indent=2),
        encoding="utf-8",
    )


def _append_metric(path: Path, record: dict) -> None:
    record = {"time": time.time(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_rl_iteration(args, current_best: Path, iter_dir: Path) -> int:
    """Shell out to scripts/rl_iteration.py for one iteration."""
    cmd = [
        sys.executable, "scripts/rl_iteration.py",
        "--current", str(current_best),
        "--out-dir", str(iter_dir),
        "--selfplay-games", str(args.selfplay_games),
        "--selfplay-sims", str(args.selfplay_sims),
        "--train-steps", str(args.train_steps),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--eval-games", str(args.eval_games),
        "--eval-sims", str(args.eval_sims),
        "--max-plies", str(args.max_plies),
        "--promote-threshold", str(args.promote_threshold),
    ]
    print("  +", " ".join(cmd))
    return subprocess.call(cmd)


def _run_stockfish_eval(args, current_best: Path, iter_idx: int, out_dir: Path):
    """Run a quick stockfish eval; returns the parsed results dict or None on failure."""
    sf_out = out_dir / f"sf-eval-after-iter-{iter_idx:03d}.json"
    cmd = [
        sys.executable, "scripts/eval_vs_stockfish.py",
        "--checkpoint", str(current_best),
        "--skill-levels", args.stockfish_skills,
        "--games-per-level", str(args.stockfish_games),
        "--simulations", str(args.selfplay_sims),
        "--movetime-s", "0.1",
        "--max-plies", str(args.max_plies),
        "--out", str(sf_out),
    ]
    if args.stockfish_bin is not None:
        cmd += ["--stockfish", str(args.stockfish_bin)]
    print("  +", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 or not sf_out.exists():
        print(f"WARNING: stockfish eval returned {rc}; skipping.")
        return None
    return json.loads(sf_out.read_text(encoding="utf-8"))


def _format_sf_summary(results: dict) -> str:
    lines = ["  skill  SF_elo    W  L  D   score    est. NN Elo"]
    elos = []
    for skill_str, r in results.items():
        lines.append(
            f"  {int(skill_str):>5d}  {r['sf_elo']:>6d}  {r['wins']:>2d} {r['losses']:>2d} "
            f"{r['draws']:>2d}  {r['score']:>6.3f}  {r['nn_elo_estimate']:>9d}"
        )
        if 0.0 < r["score"] < 1.0:
            elos.append(r["nn_elo_estimate"])
    if elos:
        lines.append(f"  Avg NN Elo estimate (excluding sweeps): {sum(elos)/len(elos):.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
