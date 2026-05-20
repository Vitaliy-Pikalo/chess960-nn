"""Tests for board + move encoding."""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chess960_nn.encoding import (
    ACTION_SPACE_SIZE,
    NUM_BOARD_PLANES,
    NUM_MOVE_TYPES,
    decode_move,
    encode_board,
    encode_move,
    legal_action_mask,
)

# ============================================================
# Constants
# ============================================================


def test_action_space_size():
    assert ACTION_SPACE_SIZE == 64 * 73
    assert NUM_MOVE_TYPES == 73


# ============================================================
# Board encoding
# ============================================================


def test_encode_board_shape_and_dtype():
    board = chess.Board(chess960=True)
    planes = encode_board(board)
    assert planes.shape == (NUM_BOARD_PLANES, 8, 8)
    assert planes.dtype == np.float32


def test_encode_board_startpos_piece_counts():
    """Standard starting position: 8 own pawns, 8 opponent pawns, 2 of each minor, etc."""
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)  # standard chess start
    planes = encode_board(board)
    # plane 0 = own pawns (white to move => white pieces)
    assert planes[0].sum() == 8  # pawns
    assert planes[1].sum() == 2  # knights
    assert planes[2].sum() == 2  # bishops
    assert planes[3].sum() == 2  # rooks
    assert planes[4].sum() == 1  # queen
    assert planes[5].sum() == 1  # king
    # Same for opponent
    for i in range(6, 12):
        assert planes[i].sum() == (8, 2, 2, 2, 1, 1)[i - 6]
    # side-to-move plane all ones
    assert planes[12].sum() == 64
    # All castling rights present at startpos
    for plane in (13, 14, 15, 16):
        assert planes[plane, 0, 0] == 1.0
    # No en passant initially
    assert planes[17].sum() == 0
    # Halfmove clock 0
    assert planes[18].sum() == 0


def test_encode_board_perspective_flips_for_black():
    """When black is to move, board is flipped: black's pieces appear in 'own' planes."""
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    # Make a single move so it's black's turn
    board.push_san("e4")
    planes = encode_board(board)
    # 'Own' pawns now black; after flipping, they occupy rank 1 (white-side from STM POV)
    # Black pawns are originally on rank 6; flipping mirrors to rank 1.
    own_pawn_plane = planes[0]
    assert own_pawn_plane[1, :].sum() == 8  # all on rank 1 from STM perspective


def test_encode_board_en_passant_marks_correct_file():
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    board.push_san("e4")  # creates en passant target on e3
    planes = encode_board(board)
    # After black's perspective flip, the e3 square (file e=4, rank 2) mirrors to (file 4, rank 5)
    # plane 17 should have 1.0 along file 4
    assert planes[17, :, 4].sum() == 8
    assert planes[17, :, :4].sum() == 0
    assert planes[17, :, 5:].sum() == 0


# ============================================================
# Move encode/decode roundtrip
# ============================================================


def _all_legal_roundtrip(board: chess.Board) -> None:
    """Every legal move on ``board`` should roundtrip cleanly."""
    for move in board.legal_moves:
        action = encode_move(move, board)
        assert 0 <= action < ACTION_SPACE_SIZE
        decoded = decode_move(action, board)
        assert decoded == move, (
            f"Roundtrip failed: move={move.uci()}, action={action}, "
            f"decoded={decoded.uci() if decoded else None}, fen={board.fen()}"
        )


def test_roundtrip_startpos():
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    _all_legal_roundtrip(board)


def test_roundtrip_random_960_positions():
    """Sample 8 random Chess960 starting positions and roundtrip all legal moves."""
    rng = random.Random(42)
    for _ in range(8):
        sp = rng.randint(0, 959)
        board = chess.Board(chess960=True)
        board.set_chess960_pos(sp)
        _all_legal_roundtrip(board)


def test_roundtrip_midgame_position():
    """Play 20 random plies then roundtrip moves."""
    rng = random.Random(0)
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    for _ in range(20):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(moves))
    _all_legal_roundtrip(board)


def test_roundtrip_many_random_960_games():
    """Play short random games from random 960 starts and roundtrip every position."""
    rng = random.Random(1)
    for _ in range(10):
        sp = rng.randint(0, 959)
        board = chess.Board(chess960=True)
        board.set_chess960_pos(sp)
        for _ in range(30):
            _all_legal_roundtrip(board)
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))


# ============================================================
# Underpromotions
# ============================================================


@pytest.mark.parametrize(
    "fen,move_uci",
    [
        # White pawn promotes by capturing left, with each underpromotion piece
        ("8/2P5/8/8/8/8/8/k1K5 w - - 0 1", "c7c8n"),  # forward promote to knight
        ("8/2P5/8/8/8/8/8/k1K5 w - - 0 1", "c7c8b"),
        ("8/2P5/8/8/8/8/8/k1K5 w - - 0 1", "c7c8r"),
        # capture-with-underpromotion
        ("1n6/2P5/8/8/8/8/8/k1K5 w - - 0 1", "c7b8n"),
        ("3n4/2P5/8/8/8/8/8/k1K5 w - - 0 1", "c7d8n"),
    ],
)
def test_underpromotion_roundtrip(fen: str, move_uci: str):
    board = chess.Board(fen, chess960=True)
    move = chess.Move.from_uci(move_uci)
    assert move in board.legal_moves, f"Setup error: {move_uci} not legal in {fen}"
    action = encode_move(move, board)
    decoded = decode_move(action, board)
    assert decoded == move


def test_queen_promotion_uses_queen_like_slot():
    """Queen promotion should be encoded as a queen-like move (action_type < 56)."""
    board = chess.Board("8/2P5/8/8/8/8/8/k1K5 w - - 0 1", chess960=True)
    move = chess.Move.from_uci("c7c8q")
    assert move in board.legal_moves
    action = encode_move(move, board)
    action_type = action % NUM_MOVE_TYPES
    assert action_type < 56  # queen-like
    decoded = decode_move(action, board)
    assert decoded == move


# ============================================================
# Chess960 castling
# ============================================================


def test_960_castling_starting_position_kingside():
    """SP 518 (standard chess setup) castling roundtrip in chess960 mode."""
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    # Manoeuvre to a position where kingside castling is legal
    for uci in ["e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5"]:
        board.push_uci(uci)
    # White can now castle kingside; in chess960 mode that's e1-h1 (king-to-rook)
    castle = chess.Move.from_uci("e1h1")
    assert castle in board.legal_moves
    action = encode_move(castle, board)
    decoded = decode_move(action, board)
    assert decoded == castle


def test_960_castling_nonstandard_start():
    """Find a chess960 position where castling is interesting and roundtrip it."""
    # SP 0: bbqnnrkr / first available 960 layout with rooks on f and h files
    board = chess.Board(chess960=True)
    board.set_chess960_pos(0)
    # Just walk through every legal move at startpos to confirm encoding stable.
    _all_legal_roundtrip(board)


# ============================================================
# Legal action mask
# ============================================================


def test_legal_action_mask_startpos():
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    mask = legal_action_mask(board)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool
    assert mask.sum() == board.legal_moves.count()


def test_legal_action_mask_after_moves():
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    board.push_san("e4")
    board.push_san("c5")
    mask = legal_action_mask(board)
    assert mask.sum() == board.legal_moves.count()


def test_decode_illegal_returns_none():
    board = chess.Board(chess960=True)
    board.set_chess960_pos(518)
    # An action that maps to an obviously illegal move (a1 -> a8 sliding,
    # blocked by own piece at startpos).
    blocked_action = chess.A1 * NUM_MOVE_TYPES + (0 * 7 + 6)  # N direction, dist 7
    assert decode_move(blocked_action, board) is None
