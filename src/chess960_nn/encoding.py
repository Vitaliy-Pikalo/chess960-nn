"""Board encoding + move representation for Chess960.

AlphaZero-style:
- Board state -> (19, 8, 8) float32 tensor, perspective-encoded for side-to-move.
- Moves <-> action indices in [0, 4672). 64 source squares x 73 move types:
    * 0-55: queen-like sliding moves (8 directions x 7 distances)
    * 56-63: knight moves (8 jumps)
    * 64-72: underpromotions (3 promotion pieces x 3 forward directions)

Chess960 castling fits naturally: python-chess in chess960 mode represents
castling as Move(king_sq, rook_sq), which is just a king move along the rank
in our encoding. No special case needed for encode/decode.
"""

from __future__ import annotations

import chess
import numpy as np

# ============================================================
# Constants
# ============================================================

NUM_SQUARES: int = 64
NUM_MOVE_TYPES: int = 73
ACTION_SPACE_SIZE: int = NUM_SQUARES * NUM_MOVE_TYPES  # 4672

NUM_BOARD_PLANES: int = 19

# Queen-like move directions as (drank, dfile). Order is fixed and must match
# in both encode and decode paths.
QUEEN_DIRECTIONS: list[tuple[int, int]] = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

# Knight L-shape offsets as (drank, dfile).
KNIGHT_OFFSETS: list[tuple[int, int]] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

# Underpromotion: forward by one rank with file delta in {-1, 0, 1}.
# Promotion targets (queen is encoded as a queen-like move).
UNDERPROMO_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_FILE_DELTAS: list[int] = [-1, 0, 1]

# Index lookups
_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}
_KNIGHT_INDEX = {d: i for i, d in enumerate(KNIGHT_OFFSETS)}


# ============================================================
# Board encoding
# ============================================================


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as a ``(NUM_BOARD_PLANES, 8, 8)`` float32 array.

    Perspective-encoded: when it is black to move, the board is flipped
    vertically AND the piece colours are swapped, so the network always
    sees its own pieces on ranks 0-3 and its opponent's on ranks 4-7.

    Plane layout:
        0-5    Own pieces (P, N, B, R, Q, K)
        6-11   Opponent pieces (P, N, B, R, Q, K)
        12     Side-to-move (constant 1.0 - kept so the net can learn from it)
        13     Own kingside castling right
        14     Own queenside castling right
        15     Opponent kingside castling right
        16     Opponent queenside castling right
        17     En passant target file (1.0 along that file's column, else 0)
        18     Halfmove clock, normalised by 100
    """
    planes = np.zeros((NUM_BOARD_PLANES, 8, 8), dtype=np.float32)

    flipped = not board.turn  # True iff black to move
    own_color = chess.BLACK if flipped else chess.WHITE
    opp_color = chess.WHITE if flipped else chess.BLACK

    # Pieces
    for piece_type in range(1, 7):  # 1 .. 6 = P, N, B, R, Q, K
        # Own
        for sq in board.pieces(piece_type, own_color):
            r, f = _square_to_rf(sq, flipped)
            planes[piece_type - 1, r, f] = 1.0
        # Opponent
        for sq in board.pieces(piece_type, opp_color):
            r, f = _square_to_rf(sq, flipped)
            planes[6 + piece_type - 1, r, f] = 1.0

    # Side-to-move plane (constant)
    planes[12, :, :] = 1.0

    # Castling rights (perspective)
    planes[13, :, :] = float(board.has_kingside_castling_rights(own_color))
    planes[14, :, :] = float(board.has_queenside_castling_rights(own_color))
    planes[15, :, :] = float(board.has_kingside_castling_rights(opp_color))
    planes[16, :, :] = float(board.has_queenside_castling_rights(opp_color))

    # En passant file
    if board.ep_square is not None:
        ep = board.ep_square
        if flipped:
            ep = chess.square_mirror(ep)
        planes[17, :, chess.square_file(ep)] = 1.0

    # Halfmove clock (capped at 100 then normalised)
    planes[18, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


def _square_to_rf(square: int, flipped: bool) -> tuple[int, int]:
    """Convert a python-chess square to (rank, file), flipping rank if needed."""
    if flipped:
        square = chess.square_mirror(square)
    return chess.square_rank(square), chess.square_file(square)


# ============================================================
# Move <-> action index
# ============================================================


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """Encode a chess.Move into an action index in ``[0, ACTION_SPACE_SIZE)``.

    The move must be a legal move on ``board``. Raises ``ValueError`` if the
    move shape is unsupported (it shouldn't happen for any legal chess move).
    """
    flipped = not board.turn

    from_sq = move.from_square
    to_sq = move.to_square
    if flipped:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    drank = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    dfile = chess.square_file(to_sq) - chess.square_file(from_sq)

    # Underpromotion (queen promotion falls through to queen-like move)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        if move.promotion not in UNDERPROMO_PIECES:
            raise ValueError(f"Unsupported promotion piece: {move.promotion}")
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        if dfile not in UNDERPROMO_FILE_DELTAS:
            raise ValueError(
                f"Invalid underpromotion file delta {dfile} for move {move.uci()}"
            )
        dir_idx = UNDERPROMO_FILE_DELTAS.index(dfile)
        action_type = 64 + dir_idx * len(UNDERPROMO_PIECES) + piece_idx
        return from_sq * NUM_MOVE_TYPES + action_type

    # Queen-like sliding move
    action_type = _try_encode_queen(drank, dfile)
    if action_type is not None:
        return from_sq * NUM_MOVE_TYPES + action_type

    # Knight move
    action_type = _try_encode_knight(drank, dfile)
    if action_type is not None:
        return from_sq * NUM_MOVE_TYPES + action_type

    raise ValueError(
        f"Cannot encode move {move.uci()} "
        f"(drank={drank}, dfile={dfile})"
    )


def decode_move(action: int, board: chess.Board) -> chess.Move | None:
    """Decode an action index to a legal ``chess.Move`` on ``board``.

    Returns ``None`` if the action does not correspond to any legal move
    on the given board.
    """
    if not 0 <= action < ACTION_SPACE_SIZE:
        return None

    flipped = not board.turn

    from_sq_p = action // NUM_MOVE_TYPES
    action_type = action % NUM_MOVE_TYPES

    from_rank = chess.square_rank(from_sq_p)
    from_file = chess.square_file(from_sq_p)

    promotion: int | None = None

    if action_type < 56:
        dir_idx, dist = divmod(action_type, 7)
        dist += 1
        ur, uf = QUEEN_DIRECTIONS[dir_idx]
        drank, dfile = ur * dist, uf * dist
    elif action_type < 64:
        drank, dfile = KNIGHT_OFFSETS[action_type - 56]
    else:
        up = action_type - 64
        dir_idx, piece_idx = divmod(up, len(UNDERPROMO_PIECES))
        dfile = UNDERPROMO_FILE_DELTAS[dir_idx]
        drank = 1  # always forward from STM perspective
        promotion = UNDERPROMO_PIECES[piece_idx]

    to_rank = from_rank + drank
    to_file = from_file + dfile
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None

    to_sq_p = chess.square(to_file, to_rank)

    if flipped:
        from_sq = chess.square_mirror(from_sq_p)
        to_sq = chess.square_mirror(to_sq_p)
    else:
        from_sq, to_sq = from_sq_p, to_sq_p

    # Queen promotion: if a pawn lands on the back rank without an explicit
    # promotion piece, it's a queen promotion. (Underpromotions already have
    # ``promotion`` set above.)
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            to_r = chess.square_rank(to_sq)
            if to_r == 0 or to_r == 7:
                promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return a ``(ACTION_SPACE_SIZE,)`` bool array marking legal action indices."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move, board)] = True
    return mask


# ============================================================
# Internals
# ============================================================


def _try_encode_queen(drank: int, dfile: int) -> int | None:
    if drank == 0 and dfile == 0:
        return None
    if drank != 0 and dfile != 0 and abs(drank) != abs(dfile):
        return None
    dist = max(abs(drank), abs(dfile))
    if dist > 7:
        return None
    ur = 0 if drank == 0 else (1 if drank > 0 else -1)
    uf = 0 if dfile == 0 else (1 if dfile > 0 else -1)
    dir_idx = _QUEEN_DIR_INDEX.get((ur, uf))
    if dir_idx is None:
        return None
    return dir_idx * 7 + (dist - 1)


def _try_encode_knight(drank: int, dfile: int) -> int | None:
    knight_idx = _KNIGHT_INDEX.get((drank, dfile))
    if knight_idx is None:
        return None
    return 56 + knight_idx
