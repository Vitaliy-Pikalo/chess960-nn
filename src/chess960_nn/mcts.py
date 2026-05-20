"""Monte Carlo Tree Search with PUCT for Chess960.

Single-leaf MCTS (one net inference per simulation). Batched inference can be
added later as an optimisation if self-play throughput becomes a bottleneck.

Algorithm (per simulation, AlphaZero-style):

    1. SELECT: descend from root, at each node pick the child that maximises
       PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
       Q(s, a) is from the parent's perspective; we negate child.value().
    2. EXPAND: at a leaf node (not yet expanded, non-terminal), run the network
       to get policy priors + value. Create a child per legal action.
    3. BACKUP: propagate the leaf value up the path, flipping sign at every
       step (opponent's POV).

Terminal positions short-circuit expansion and return their deterministic
value (-1 for checkmate against side-to-move, 0 for stalemate/insufficient
material/50-move rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np
import torch

from chess960_nn.encoding import (
    ACTION_SPACE_SIZE,
    decode_move,
    encode_board,
    legal_action_mask,
)
from chess960_nn.model import Chess960Net

# ============================================================
# Config
# ============================================================


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS."""

    n_simulations: int = 400
    c_puct: float = 2.0
    # Add Dirichlet noise to root priors. Used during self-play for exploration;
    # set eps=0.0 for evaluation / deterministic play.
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_eps: float = 0.0


# ============================================================
# Tree node
# ============================================================


class Node:
    """A single node in the search tree.

    Holds aggregated statistics + a sparse map from action index -> child node.
    Children are created lazily on expansion.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float = 0.0):
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, Node] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Mean value, from this node's side-to-move's POV."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ============================================================
# Helpers
# ============================================================


def terminal_value(board: chess.Board) -> float | None:
    """Return STM-POV value if the board is terminal, else None."""
    if board.is_checkmate():
        return -1.0  # side-to-move just got mated
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
    ):
        return 0.0
    return None


def softmax_with_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a masked support.

    Args:
        logits: shape ``(ACTION_SPACE_SIZE,)`` raw logits.
        mask:   shape ``(ACTION_SPACE_SIZE,)`` bool, True for legal actions.

    Returns:
        Array of shape ``(ACTION_SPACE_SIZE,)``; sum is 1 over legal actions,
        0 elsewhere. If no actions are legal returns all zeros.
    """
    masked = np.where(mask, logits, -1e30)
    m = masked.max()
    e = np.exp(masked - m)
    e = np.where(mask, e, 0.0)
    s = e.sum()
    if s <= 0:
        return e
    return e / s


# ============================================================
# Search
# ============================================================


class MCTS:
    """PUCT-style MCTS bound to a single network."""

    def __init__(
        self,
        net: Chess960Net,
        cfg: MCTSConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        self.net = net
        self.cfg = cfg or MCTSConfig()
        self.device = torch.device(device)
        self.net.eval()
        self.net.to(self.device)

    # ----- public API -----

    def search(self, board: chess.Board) -> Node:
        """Run ``n_simulations`` from ``board`` and return the root node."""
        # Always operate on a chess960 board so castling moves use king-to-rook.
        if not board.chess960:
            board = chess.Board(board.fen(), chess960=True)

        root = Node()
        self._expand(root, board)
        self._maybe_add_root_noise(root)

        for _ in range(self.cfg.n_simulations):
            self._simulate(root, board.copy())
        return root

    def visit_distribution(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        """Return a probability vector of length ``ACTION_SPACE_SIZE`` from visits.

        ``temperature=0`` → one-hot on the argmax visit count.
        ``temperature>0`` → softmax over ``N(s,a) ** (1/T)``.
        """
        counts = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        if temperature == 0 or counts.sum() == 0:
            policy = np.zeros_like(counts)
            policy[counts.argmax()] = 1.0
            return policy
        weighted = counts ** (1.0 / temperature)
        total = weighted.sum()
        if total == 0:
            policy = np.zeros_like(counts)
            policy[counts.argmax()] = 1.0
            return policy
        return weighted / total

    def best_action(
        self,
        root: Node,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Pick an action: argmax (T=0) or sample from visit distribution."""
        dist = self.visit_distribution(root, temperature)
        if temperature == 0:
            return int(dist.argmax())
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(len(dist), p=dist))

    # ----- inference -----

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Run net forward on a single position.

        Returns:
            priors: ``(ACTION_SPACE_SIZE,)`` softmax over legal actions only.
            value:  scalar in ``[-1, 1]`` from STM POV.
        """
        state = encode_board(board)
        x = torch.from_numpy(state).unsqueeze(0).to(self.device)
        # Cast to float16 if model is on GPU (autocast equivalent for inference)
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                policy_logits, value = self.net(x)
            policy_logits = policy_logits.float()
            value = value.float()
        else:
            policy_logits, value = self.net(x)
        logits_np = policy_logits.squeeze(0).cpu().numpy()
        v = float(value.item())
        mask = legal_action_mask(board)
        priors = softmax_with_mask(logits_np, mask)
        return priors, v

    # ----- internals -----

    def _expand(self, node: Node, board: chess.Board) -> float:
        priors, value = self.evaluate(board)
        for action_idx in np.nonzero(priors > 0)[0]:
            node.children[int(action_idx)] = Node(prior=float(priors[action_idx]))
        return value

    def _maybe_add_root_noise(self, root: Node) -> None:
        if self.cfg.root_dirichlet_eps <= 0 or not root.children:
            return
        actions = list(root.children.keys())
        alphas = [self.cfg.root_dirichlet_alpha] * len(actions)
        noise = np.random.default_rng().dirichlet(alphas)
        eps = self.cfg.root_dirichlet_eps
        for a, n in zip(actions, noise, strict=True):
            child = root.children[a]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    def _select_child(self, node: Node) -> tuple[int, Node]:
        """Argmax PUCT score across children. Assumes node is expanded."""
        sqrt_total_n = math.sqrt(max(1, node.visit_count))
        c = self.cfg.c_puct
        best_score = -float("inf")
        best_action = -1
        best_child: Node | None = None
        for action, child in node.children.items():
            q = -child.value()  # negate: child.value is from opp POV
            u = c * child.prior * sqrt_total_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None  # node is expanded -> at least one child
        return best_action, best_child

    def _simulate(self, root: Node, board: chess.Board) -> None:
        """One simulation: select to a leaf, expand or evaluate terminal, backup."""
        path: list[Node] = [root]
        node = root

        # SELECTION: walk down expanded subtree until we hit an unexpanded node
        # or a terminal position.
        while node.expanded():
            term = terminal_value(board)
            if term is not None:
                self._backup(path, term)
                return
            action, child = self._select_child(node)
            move = decode_move(action, board)
            if move is None:
                # Shouldn't happen because priors only cover legal actions, but
                # guard against any encoding drift.
                self._backup(path, 0.0)
                return
            board.push(move)
            node = child
            path.append(node)

        # EVALUATION (terminal short-circuit or net forward + expand)
        term = terminal_value(board)
        if term is not None:
            value = term
        else:
            value = self._expand(node, board)

        self._backup(path, value)

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """Walk path leaf -> root, flipping sign each level (alternating POV)."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
