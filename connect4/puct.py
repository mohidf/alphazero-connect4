"""PUCT search: MCTS guided by a policy prior and a value estimate.

Three things change from the vanilla MCTS in mcts.py:

* **No rollout.** A leaf's value comes from one evaluator call instead of 42
  random moves. That is the whole efficiency win.
* **Priors steer exploration.** UCB1 knew only visit counts, so an unvisited node
  had to score infinity. PUCT weights exploration by the policy's prior, so all
  children are created at once and ranked from the start.
* **Values are signed by perspective.** Every node's statistics are stored from
  the perspective of the player to move *at that node*, so the value flips sign
  at every level of the backup.

The evaluator is a plain callable, ``(board, player) -> (priors, value)``, rather
than a network. The search does not care where the numbers come from, which makes
it testable with stubs and lets a batched or cached evaluator drop in later.
Use network_evaluator() to wrap a Connect4Net.

Selection score for a child, from the parent's point of view:

    -child.Q  +  c_puct * child.prior * sqrt(parent.visits) / (1 + child.visits)

The leading minus is the sign flip: a child's Q is stored from the child's
mover's perspective, and that is the opponent of the parent's mover.
"""

from typing import Callable

import numpy as np

from connect4.board import Board, COLS
from connect4.encoding import legal_move_mask, mask_and_normalise
from connect4.mcts import other
from connect4.network import Connect4Net, predict

# Exploration strength. Higher trusts the prior less and spreads visits wider.
C_PUCT = 1.5

# Root exploration noise. AlphaZero scales alpha roughly as 10 / (legal moves),
# which lands near 1.0 for Connect-4's seven columns. Without this, self-play
# collapses into replaying the same game and the training set stops growing.
DIRICHLET_ALPHA = 1.0
DIRICHLET_WEIGHT = 0.25

# (priors over COLS, value in [-1, 1] for the player to move)
Evaluator = Callable[[Board, str], tuple[np.ndarray, float]]


class Node:
    """One position in the search tree.

    `visits` and `value_sum` are stored from the perspective of
    `player_to_move` — the player about to move *at this node*. So Q > 0 means
    "good for whoever is on the move here", which is the same convention the
    network's value head uses.

    `prior` is P(s, a) for the move that leads here, taken from the parent's
    policy. The root's prior is unused.
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
        """Mean value from this node's mover's perspective; 0 when unvisited.

        Returning 0 for an unvisited node makes PUCT treat it as even rather than
        as infinitely attractive. The prior term is what gets it explored, which
        is the whole reason UCB1's "infinity for unvisited" is not needed here.
        """
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, c_puct: float = C_PUCT) -> float:
        """Selection score for this node, from its *parent's* perspective."""
        exploration = (
            c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        )
        return -self.q + exploration


def network_evaluator(net: Connect4Net) -> Evaluator:
    """Adapt a Connect4Net to the Evaluator interface."""

    def evaluate(board: Board, player: str) -> tuple[np.ndarray, float]:
        return predict(net, board, player)

    return evaluate


def caching_evaluator(evaluator: Evaluator, capacity: int = 200_000) -> Evaluator:
    """Memoise an evaluator on (position, player to move).

    Worth it wherever searches run one at a time and cannot be batched — arena
    games especially, where consecutive moves re-search overlapping subtrees.
    Positions also repeat within a single tree via different move orders.

    The cache is only valid for one fixed network: build a new one whenever the
    weights change, or it will serve a previous network's opinions.
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
    """A network-free evaluator: flat priors, neutral value.

    Useful as a baseline — PUCT with this still plays respectably, because
    terminal values inside the tree are exact regardless of the evaluator.
    """
    return np.full(COLS, 1.0 / COLS, dtype=np.float32), 0.0


def terminal_value(board: Board, player_to_move: str) -> float:
    """Exact value of a finished position, from `player_to_move`'s perspective.

    In practice this always returns 0.0 or -1.0: a win is created by the move
    that just happened, so the player now on the move is the one who lost. The
    +1.0 branch is kept for correctness rather than reachability.
    """
    winner = board.winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == player_to_move else -1.0


def select_child(node: Node, c_puct: float = C_PUCT) -> Node:
    """Return the child with the highest PUCT score."""
    return max(node.children.values(), key=lambda child: child.puct_score(c_puct))


def expand_with(node: Node, priors: np.ndarray) -> None:
    """Create every legal child of `node`, carrying masked-and-normalised priors.

    Split out from expand() so a caller that already has the evaluation — because
    it batched several positions together — can apply it without calling an
    evaluator again.

    Unlike vanilla MCTS this creates all children at once: the priors rank them,
    so there is no need to try them one at a time.
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
    """Evaluate `node`, create its children, and return the value.

    The returned value is from `node`'s mover's perspective, ready for backup().
    """
    priors, value = evaluator(node.board, node.player_to_move)
    expand_with(node, priors)
    return value


def backup(node: Node | None, value: float) -> None:
    """Add one visit and `value` at each node from `node` up to the root.

    `value` flips sign at every level: it arrives from the perspective of the
    leaf's mover, and each step up the tree changes whose turn it is. Dropping
    the flip gives a search that confidently prefers losing moves — the same bug
    as crediting the wrong player in vanilla MCTS backpropagate().
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
    """Mix Dirichlet noise into the root's priors, in place.

    Only at the root, and only during self-play: it is there to force variety
    into the training set, not to improve play. Applying it during evaluation
    would just make the agent weaker.
    """
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
    """Run `simulations` PUCT iterations and return the root.

    The root is expanded before the loop so noise can be applied to real priors;
    that expansion is not backed up, so root.visits ends up exactly
    `simulations`. A terminal root is left unexpanded and gains no children.
    """
    root = Node(board.copy(), player)

    if not root.board.is_terminal():
        expand(root, evaluator)
        if add_noise:
            add_dirichlet_noise(root, rng=rng)

    for _ in range(simulations):
        node = root

        # SELECT: descend while the node has children to choose between.
        while node.is_expanded and not node.board.is_terminal():
            node = select_child(node, c_puct)

        # EVALUATE: exact value at a finished position, evaluator otherwise.
        if node.board.is_terminal():
            value = terminal_value(node.board, node.player_to_move)
        else:
            value = expand(node, evaluator)

        backup(node, value)

    return root


def visit_counts(root: Node) -> np.ndarray:
    """Raw visit counts as a float32 array of shape (COLS,); 0 for illegal moves."""
    counts = np.zeros(COLS, dtype=np.float32)
    for col, child in root.children.items():
        counts[col] = child.visits
    return counts


def policy_target(root: Node, temperature: float = 1.0) -> np.ndarray:
    """Normalised visit counts — the search's improved policy, i.e. the pi target.

    This is policy improvement by planning: the network proposed `prior`, the
    search spent its budget checking, and the resulting visit distribution is
    better than what the network started with. Training toward it is what makes
    the loop learn.

    temperature 0 puts all mass on the most-visited move.
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
    """Choose a move from the root's visit counts.

    temperature 0 takes the most-visited move — correct for evaluation and for
    the later part of a self-play game. temperature 1 samples proportional to
    visits, which is what generates variety in the opening.
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
    """Return the column PUCT favours, or None if there are no legal moves."""
    root = run_search(board, player, evaluator, simulations, c_puct)
    if not root.children:
        return None
    return select_move(root, temperature=0.0)


class Search:
    """A PUCT search that can be paused whenever it needs an evaluation.

    run_search() calls the evaluator itself, which forces one network call per
    simulation — a batch of one, where kernel-launch overhead dwarfs the actual
    arithmetic. This class inverts that: it descends to a leaf, hands it back, and
    waits. A driver can therefore run many searches side by side and evaluate all
    of their pending leaves in a single batched forward pass.

    Each Search has at most one leaf outstanding at a time, so no virtual loss is
    needed — the parallelism is across games, not within a single tree.

    Usage:
        search = Search(board, player)
        leaf = search.pending_leaf()          # root, needing evaluation
        search.resolve(priors, value)
        while search.simulations_done < n:
            leaf = search.pending_leaf()      # None if it finished on a terminal
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
        """True when the root is terminal, so no search is possible at all."""
        return self.root.board.is_terminal()

    def pending_leaf(self) -> Node | None:
        """Advance to the next position needing evaluation, or None if none does.

        Returns None when the simulation resolved without an evaluation — either
        it ended on a terminal position, or the root itself is terminal.
        """
        if self._pending is not None:
            raise RuntimeError("a leaf is already awaiting resolve()")

        if self.finished:
            self.root.visits += 1
            self.root.value_sum += terminal_value(
                self.root.board, self.root.player_to_move
            )
            self.simulations_done += 1
            return None

        # The root is expanded before any simulation is counted, matching
        # run_search, so root.visits ends up equal to the simulation count.
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
        """Apply an evaluation to the outstanding leaf."""
        node = self._pending
        if node is None:
            raise RuntimeError("resolve() called with no leaf outstanding")

        expand_with(node, priors)
        self._pending = None

        if node is self.root and not self._root_expanded:
            # Root expansion is setup, not a simulation: it is not backed up.
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
        """Apply root Dirichlet noise. Only valid once the root is expanded."""
        if not self._root_expanded:
            raise RuntimeError("expand the root before adding noise")
        add_dirichlet_noise(self.root, alpha, weight, rng)
