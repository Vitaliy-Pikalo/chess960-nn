"""Tests for MCTS."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess960_nn.encoding import ACTION_SPACE_SIZE, decode_move
from chess960_nn.mcts import (
    MCTS,
    MCTSConfig,
    Node,
    softmax_with_mask,
    terminal_value,
)
from chess960_nn.model import Chess960Net, ModelConfig

# ============================================================
# Small fixtures
# ============================================================


def _tiny_net() -> Chess960Net:
    """A small randomly-initialised net so tests don't depend on training."""
    torch.manual_seed(0)
    return Chess960Net(ModelConfig(num_blocks=1, num_filters=16))


# ============================================================
# Helpers
# ============================================================


def test_terminal_value_checkmate():
    # Fool's mate: 1.f3 e5 2.g4 Qh4#
    board = chess.Board(chess960=True)
    board.push_uci("f2f3")
    board.push_uci("e7e5")
    board.push_uci("g2g4")
    board.push_uci("d8h4")
    assert board.is_checkmate()
    assert terminal_value(board) == -1.0


def test_terminal_value_stalemate():
    # Classic stalemate
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", chess960=True)
    assert board.is_stalemate()
    assert terminal_value(board) == 0.0


def test_terminal_value_none_for_ongoing():
    board = chess.Board(chess960=True)
    assert terminal_value(board) is None


def test_softmax_with_mask_basic():
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask = np.array([True, False, True, False])
    p = softmax_with_mask(logits, mask)
    assert np.isclose(p.sum(), 1.0)
    assert p[1] == 0 and p[3] == 0
    # p[2] (logit 3) should be larger than p[0] (logit 1)
    assert p[2] > p[0]


def test_softmax_with_mask_no_legal_returns_zeros():
    logits = np.zeros(4, dtype=np.float32)
    mask = np.array([False] * 4)
    p = softmax_with_mask(logits, mask)
    assert np.all(p == 0)


# ============================================================
# Node + backup
# ============================================================


def test_node_value_zero_before_visits():
    n = Node(prior=0.5)
    assert n.value() == 0.0
    assert not n.expanded()


def test_backup_alternates_sign():
    leaf = Node()
    mid = Node()
    root = Node()
    path = [root, mid, leaf]
    MCTS._backup(path, value=0.7)
    # Leaf gets +0.7, mid gets -0.7, root gets +0.7
    assert leaf.value() == pytest.approx(0.7)
    assert mid.value() == pytest.approx(-0.7)
    assert root.value() == pytest.approx(0.7)


# ============================================================
# Search
# ============================================================


def test_search_records_n_simulations_at_root():
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=20, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board(chess960=True)
    root = mcts.search(board)
    # Root visit count == number of simulations.
    assert root.visit_count == 20
    # Sum of child visit counts == n_simulations (root expanded once for free).
    assert sum(c.visit_count for c in root.children.values()) == 20


def test_search_expands_only_legal_children():
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=5, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board(chess960=True)
    root = mcts.search(board)
    n_legal = board.legal_moves.count()
    # Children correspond 1-to-1 with legal moves at the root.
    assert len(root.children) == n_legal


def test_visit_distribution_sums_to_one():
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=10, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board(chess960=True)
    root = mcts.search(board)
    dist = mcts.visit_distribution(root, temperature=1.0)
    assert dist.shape == (ACTION_SPACE_SIZE,)
    assert np.isclose(dist.sum(), 1.0)


def test_visit_distribution_temperature_zero_is_onehot():
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=10, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board(chess960=True)
    root = mcts.search(board)
    dist = mcts.visit_distribution(root, temperature=0)
    assert np.isclose(dist.sum(), 1.0)
    assert (dist > 0).sum() == 1


# ============================================================
# Correctness: find mate in 1
# ============================================================


def test_mcts_finds_mate_in_one():
    """Even with a random net, MCTS should pick *some* mating move.

    Position: White Q on h1, White K on a6, Black K on a8. There are several
    mates available here (Qh8#, Qb7#, etc.) so we just assert that whichever
    move MCTS picks delivers checkmate.
    """
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=200, root_dirichlet_eps=0.0, c_puct=2.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board("k7/8/K7/8/8/8/8/7Q w - - 0 1", chess960=True)
    root = mcts.search(board)
    action = mcts.best_action(root, temperature=0)
    move = decode_move(action, board)
    assert move is not None
    board.push(move)
    assert board.is_checkmate(), (
        f"MCTS picked {move} which is not checkmate. "
        f"Board:\n{board}"
    )


# ============================================================
# Terminal handling
# ============================================================


def test_search_on_already_checkmated_position():
    """If the root is already checkmated, search should still return cleanly."""
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=5, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    # Fool's mate position, black is checkmated
    board = chess.Board(chess960=True)
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(uci)
    assert board.is_checkmate()
    # search should not crash; root has no legal children
    root = mcts.search(board)
    # When position is terminal the expand call will still happen but priors
    # will be all zero -> no children. We don't care about visit count here,
    # just that we don't crash.
    assert isinstance(root, Node)


# ============================================================
# Chess960 castling reachable through MCTS
# ============================================================


def test_mcts_can_select_chess960_castling():
    """Set up a position where kingside castling is legal in 960 form and
    confirm MCTS can pick it (or at least decode it without crashing)."""
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=30, root_dirichlet_eps=0.0)
    mcts = MCTS(net, cfg, device="cpu")
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    # Manoeuvre to a position where O-O is legal
    for uci in ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5"]:
        board.push_uci(uci)
    assert chess.Move.from_uci("e1h1") in board.legal_moves
    root = mcts.search(board)
    # The castling action should appear among children since it's a legal move.
    found = False
    for action in root.children:
        move = decode_move(action, board)
        if move == chess.Move.from_uci("e1h1"):
            found = True
            break
    assert found, "Kingside castle not present among MCTS children"


# ============================================================
# Determinism
# ============================================================


def test_search_is_deterministic_without_dirichlet():
    """Same net, same board, no Dirichlet noise -> same visit distribution."""
    net = _tiny_net()
    cfg = MCTSConfig(n_simulations=30, root_dirichlet_eps=0.0)
    board = chess.Board(chess960=True)

    mcts1 = MCTS(net, cfg, device="cpu")
    root1 = mcts1.search(board)
    counts1 = sorted(
        (a, c.visit_count) for a, c in root1.children.items()
    )

    mcts2 = MCTS(net, cfg, device="cpu")
    root2 = mcts2.search(board)
    counts2 = sorted(
        (a, c.visit_count) for a, c in root2.children.items()
    )

    assert counts1 == counts2
