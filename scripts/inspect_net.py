"""Load a trained checkpoint and show what MCTS picks on a few positions.

Example:
    uv run python scripts/inspect_net.py \\
        --checkpoint runs/pretrain-002/checkpoints/last.pt \\
        --simulations 400
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import torch

from chess960_nn.encoding import decode_move
from chess960_nn.mcts import MCTS, MCTSConfig
from chess960_nn.model import Chess960Net, ModelConfig
from chess960_nn.train import load_checkpoint

# A handful of test positions: opening (SP 518), midgame, an endgame tactic.
POSITIONS: list[tuple[str, str]] = [
    (
        "Chess960 standard start (SP 518)",
        chess.STARTING_FEN,
    ),
    (
        "After 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez setup, 960 castling)",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ),
    (
        "Mate in 1: White to play Qh8#",
        "k7/8/K7/8/8/8/8/7Q w - - 0 1",
    ),
    (
        "Random 960 SP 23 starting position",
        None,  # special: set_chess960_pos(23)
    ),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=5, help="Show top-K candidate moves.")
    p.add_argument("--num-blocks", type=int, default=10)
    p.add_argument("--num-filters", type=int, default=192)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = ModelConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    net = Chess960Net(model_cfg).to(device)
    payload = load_checkpoint(args.checkpoint, net=net, map_location=str(device))
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  step={payload.get('step')}  epoch={payload.get('epoch')}")
    print(f"  Model params: {net.num_parameters():,}")
    print()

    mcts_cfg = MCTSConfig(
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        root_dirichlet_eps=0.0,
    )
    mcts = MCTS(net, mcts_cfg, device=device)

    for label, fen in POSITIONS:
        if fen is None:
            board = chess.Board(chess960=True)
            board.set_chess960_pos(23)
        else:
            board = chess.Board(fen, chess960=True)
        print(f"=== {label} ===")
        print(board)
        print(f"FEN: {board.fen()}")
        print(f"Side to move: {'white' if board.turn else 'black'}")

        # Raw policy/value (no search)
        priors, value = mcts.evaluate(board)
        print(f"Raw net value (STM POV): {value:+.3f}")

        t0 = time.time()
        root = mcts.search(board)
        elapsed = time.time() - t0
        print(f"MCTS: {args.simulations} sims in {elapsed:.2f}s "
              f"({args.simulations / elapsed:.1f} sims/s)")

        # Show top-K children by visit count
        children = sorted(
            root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True
        )
        total = sum(c.visit_count for _, c in children) or 1
        print(f"Top {min(args.top_k, len(children))} moves:")
        for action, child in children[: args.top_k]:
            move = decode_move(action, board)
            if move is None:
                continue
            san = board.san(move)
            pct = 100.0 * child.visit_count / total
            q = -child.value()
            print(f"  {san:<8s}  visits={child.visit_count:>4d} ({pct:5.1f}%)  "
                  f"Q={q:+.3f}  P={child.prior:.3f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
