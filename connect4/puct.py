"""PUCT: MCTS steered by the network instead of random rollouts.

Differences from mcts.py:

- no rollout, the value comes from one evaluator call
- the policy prior decides what to explore, so all children get made at once
- every node's stats are from the point of view of whoever moves there, so the
  value flips sign on the way back up

Child selection score, from the parent's side:

    -child.Q + c_puct * child.prior * sqrt(parent.visits) / (1 + child.visits)

The minus is that sign flip. The evaluator is just a callable taking
(board, player) and returning (priors, value), which makes it easy to swap in
stubs for tests or a cached version later.
"""

from typing import Callable

import numpy as np

from connect4.board import Board, COLS
from connect4.encoding import legal_move_mask, mask_and_normalise
from connect4.mcts import other
from connect4.network import Connect4Net, predict

# Higher spreads visits wider.
C_PUCT = 1.5

# Noise added at the root during self-play. Without it every game comes out the
# same and the training set stops growing.
DIRICHLET_ALPHA = 1.0
DIRICHLET_WEIGHT = 0.25

Evaluator = Callable[[Board, str], tuple[np.ndarray, float]]


class Node:
    """A position in the tree.

    Stats are from the point of view of player_to_move, so Q > 0 means good for
    whoever moves here. prior is the parent policy's probability for the move
    that led here; the root's is unused.
    """

    __slots__ = (
        "board",
        "player_to_move",
        "parent",
        "move",
        "prior",
        "children",
        "visits",
        "value_sum",
    )

    def __init__(
        self,
        board: Board,
        player_to_move: str,
        parent: "Node | None" = None,
        move: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: dict[int, "Node"] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    @property
    def q(self) -> float:
        """Average value here. Unvisited returns 0, not infinity like UCB1 -
        the prior term is what gets new children explored."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, c_puct: float = C_PUCT) -> float:
        """Score used to pick this child, from the parent's side."""
        exploration = (
            c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        )
        return -self.q + exploration


def network_evaluator(net: Connect4Net) -> Evaluator:
    """Wrap a network so the search can call it."""

    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        return predict(net, board, player)

    return evaluate


def caching_evaluator(evaluator: Evaluator, capacity: int = 200_000) -> Evaluator:
    """Remember evaluations, keyed on the position and whose turn it is.

    Helps when searches run one at a time and can't be batched, since successive
    moves in a game re-search a lot of the same tree. Only valid for one set of
    weights - make a new one when the network changes.
    """
    cache: dict[tuple, tuple[np.ndarray, float]] = {}

    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        key = (tuple(map(tuple, board.grid)), player)
        hit = cache.get(key)
        if hit is not None:
            return hit

        result = evaluator(board, player)
        if len(cache) < capacity:
            cache[key] = result
        return result

    return evaluate


def uniform_evaluator(board: Board, player: str) -> tuple[np.ndarray, float]:
    """No network: flat priors and a neutral value. Still plays reasonably,
    because wins found inside the tree are exact either way."""
    return np.full(COLS, 1.0 / COLS, dtype=np.float32), 0.0


def terminal_value(board: Board, player_to_move: str) -> float:
    """Value of a finished position for whoever is to move.

    Almost always 0 or -1: whoever just moved is the one who won, so the player
    on the move is the loser. The +1 case is there for completeness.
    """
    winner = board.winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == player_to_move else -1.0


def select_child(node: Node, c_puct: float = C_PUCT) -> Node:
    """Child with the best PUCT score."""
    return max(node.children.values(), key=lambda child: child.puct_score(c_puct))


def expand_with(node: Node, priors: np.ndarray) -> None:
    """Make all the legal children at once, with their priors.

    Separate from expand() so a caller that already has the evaluation from a
    batch can use it without calling the evaluator again.
    """
    priors = mask_and_normalise(np.asarray(priors), legal_move_mask(node.board))
    opponent = other(node.player_to_move)

    for col in node.board.available_moves():
        child_board = node.board.copy()
        child_board.make_move(col, node.player_to_move)
        node.children[col] = Node(
            board=child_board,
            player_to_move=opponent,
            parent=node,
            move=col,
            prior=float(priors[col]),
        )


def expand(node: Node, evaluator: Evaluator) -> float:
    """Evaluate the node, make its children, return the value for backup()."""
    priors, value = evaluator(node.board, node.player_to_move)
    expand_with(node, priors)
    return value


def backup(node: Node | None, value: float) -> None:
    """Walk up to the root adding a visit and the value at each step.

    The sign flips every level, since whose turn it is alternates. Miss that and
    the search confidently picks losing moves.
    """
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value
        node = node.parent


def add_dirichlet_noise(
    root: Node,
    alpha: float = DIRICHLET_ALPHA,
    weight: float = DIRICHLET_WEIGHT,
    rng: np.random.Generator | None = None,
) -> None:
    """Mix noise into the root priors. Self-play only - it is there for variety
    in the training data, and would just weaken the agent during evaluation."""
    rng = rng if rng is not None else np.random.default_rng()
    columns = list(root.children)
    noise = rng.dirichlet([alpha] * len(columns))

    for col, sample in zip(columns, noise):
        child = root.children[col]
        child.prior = (1.0 - weight) * child.prior + weight * float(sample)


def run_search(
    board: Board,
    player: str,
    evaluator: Evaluator = uniform_evaluator,
    simulations: int = 200,
    c_puct: float = C_PUCT,
    add_noise: bool = False,
    rng: np.random.Generator | None = None,
) -> Node:
    """Run the search and return the root.

    The root is expanded first so noise has real priors to mix into. That
    expansion is not backed up, so root.visits ends up equal to `simulations`.
    """
    root = Node(board.copy(), player)

    if not root.board.is_terminal():
        expand(root, evaluator)
        if add_noise:
            add_dirichlet_noise(root, rng=rng)

    for _ in range(simulations):
        node = root

        # descend
        while node.is_expanded and not node.board.is_terminal():
            node = select_child(node, c_puct)

        # exact value if the game is over there, otherwise ask the evaluator
        if node.board.is_terminal():
            value = terminal_value(node.board, node.player_to_move)
        else:
            value = expand(node, evaluator)

        backup(node, value)

    return root


def visit_counts(root: Node) -> np.ndarray:
    """Visit count per column, 0 for illegal ones."""
    counts = np.zeros(COLS, dtype=np.float32)
    for col, child in root.children.items():
        counts[col] = child.visits
    return counts


def policy_target(root: Node, temperature: float = 1.0) -> np.ndarray:
    """Visit counts normalised into a distribution - the training target.

    The network suggested the priors, the search checked them, and this is what
    came out. It is better than what the network started with, which is what
    there is to learn from. temperature 0 gives one-hot.
    """
    counts = visit_counts(root)
    if counts.sum() == 0:
        raise ValueError("cannot build a policy target from an unvisited root")

    if temperature == 0:
        target = np.zeros_like(counts)
        target[int(counts.argmax())] = 1.0
        return target

    scaled = counts ** (1.0 / temperature)
    return (scaled / scaled.sum()).astype(np.float32)


def select_move(
    root: Node,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Pick a move from the visit counts.

    temperature 0 takes the most visited. 1 samples in proportion, which is how
    self-play games end up different from each other.
    """
    if not root.children:
        raise ValueError("no legal moves at the root")

    if temperature == 0:
        return max(root.children.values(), key=lambda child: child.visits).move

    rng = rng if rng is not None else np.random.default_rng()
    probabilities = policy_target(root, temperature)
    return int(rng.choice(COLS, p=probabilities))


def puct_move(
    board: Board,
    player: str,
    evaluator: Evaluator = uniform_evaluator,
    simulations: int = 200,
    c_puct: float = C_PUCT,
) -> int | None:
    """Best column, or None if the board is full."""
    root = run_search(board, player, evaluator, simulations, c_puct)
    if not root.children:
        return None
    return select_move(root, temperature=0.0)


class Search:
    """A search that stops and waits whenever it needs an evaluation.

    run_search() calls the evaluator itself, one position at a time, which wastes
    most of the GPU. This hands the leaf back instead, so a caller can run lots of
    games at once and evaluate all their leaves in one batch.

    Only one leaf is outstanding per search, so there is no need for virtual loss -
    the parallelism is across games, not inside one tree.

        search = Search(board, player)
        while search.simulations_done < n:
            leaf = search.pending_leaf()   # None if it hit a finished position
            if leaf is not None:
                search.resolve(priors, value)
    """

    def __init__(self, board: Board, player: str, c_puct: float = C_PUCT) -> None:
        self.root = Node(board.copy(), player)
        self.c_puct = c_puct
        self.simulations_done = 0
        self._pending: Node | None = None
        self._root_expanded = False

    @property
    def finished(self) -> bool:
        """Root is already a finished game, so there is nothing to search."""
        return self.root.board.is_terminal()

    def pending_leaf(self) -> Node | None:
        """Next position needing an evaluation, or None if this simulation
        finished without one (it hit the end of a game)."""
        if self._pending is not None:
            raise RuntimeError("a leaf is already awaiting resolve()")

        if self.finished:
            self.root.visits += 1
            self.root.value_sum += terminal_value(
                self.root.board, self.root.player_to_move
            )
            self.simulations_done += 1
            return None

        # Root gets expanded before any simulation counts, same as run_search.
        if not self._root_expanded:
            self._pending = self.root
            return self.root

        node = self.root
        while node.is_expanded and not node.board.is_terminal():
            node = select_child(node, self.c_puct)

        if node.board.is_terminal():
            backup(node, terminal_value(node.board, node.player_to_move))
            self.simulations_done += 1
            return None

        self._pending = node
        return node

    def resolve(self, priors: np.ndarray, value: float) -> None:
        """Hand back an evaluation for the leaf we are waiting on."""
        node = self._pending
        if node is None:
            raise RuntimeError("resolve() called with no leaf outstanding")

        expand_with(node, priors)
        self._pending = None

        if node is self.root and not self._root_expanded:
            # Setup, not a simulation, so no backup.
            self._root_expanded = True
            return

        backup(node, value)
        self.simulations_done += 1

    def add_noise(
        self,
        alpha: float = DIRICHLET_ALPHA,
        weight: float = DIRICHLET_WEIGHT,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Add root noise. The root has to be expanded first."""
        if not self._root_expanded:
            raise RuntimeError("expand the root before adding noise")
        add_dirichlet_noise(self.root, alpha, weight, rng)
