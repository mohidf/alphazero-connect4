"""Vanilla Monte Carlo Tree Search for Connect-4: random rollouts, no network.

Structurally the same four phases as the Tic-Tac-Toe version — select, expand,
simulate, backpropagate — with one thing to respect throughout: every board a
node holds must come from Board.copy(), never Board(list(grid)). See the note in
board.py for why both halves of that matter.

Expect this to beat a random player easily and lose to depth-6 alpha-beta.
Random rollouts over 42 moves are a poor proxy for good play, and are nearly
blind to forced tactics. That gap is what Phase 4's learned value head closes.
"""

import math
import random

from connect4.board import Board, PLAYER_R, PLAYER_Y

# sqrt(2) is the standard UCB1 constant for rewards scaled to [0, 1], which is
# what the win/draw/loss encoding below produces.
EXPLORATION = math.sqrt(2)

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0


def other(player: str) -> str:
    """Return the opposing player."""
    return PLAYER_Y if player == PLAYER_R else PLAYER_R


class Node:
    """One position in the search tree.

    `player_to_move` is whose turn it is *from* this node. The move that created
    this node was therefore played by other(player_to_move) — which is the
    player whose result backpropagate() must credit here.
    """

    def __init__(
        self,
        board: Board,
        player_to_move: str,
        parent: "Node | None" = None,
        move: int | None = None,
    ) -> None:
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move                       # column played to reach this node
        self.children: list["Node"] = []
        self.untried_moves: list[int] = board.available_moves()
        self.visits = 0                        # n_i
        self.wins = 0.0                        # w_i, in reward units above

    def is_fully_expanded(self) -> bool:
        """True when every legal move from here has a child."""
        return len(self.untried_moves) == 0

    def ucb1(self, exploration: float = EXPLORATION) -> float:
        """UCB1 score, used to choose among a parent's children during selection.

        An unvisited node must score infinitely high so it gets tried before any
        visited sibling is revisited.
        """
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term

    def best_child(self) -> "Node":
        """Return the child with the highest UCB1 score."""
        return max(self.children, key=lambda child: child.ucb1())


def select(node: Node) -> Node:
    """Descend from `node` via best_child() while it is fully expanded and
    non-terminal. Return the node where descent stopped."""
    while node.is_fully_expanded() and not node.board.is_terminal():
        node = node.best_child()
    return node


def expand(node: Node) -> Node:
    """Take one untried move from `node`, attach a child for it, return the child.

    The child's board must be node.board.copy() with the move applied — a fresh
    Board(grid) would lose last_move and report no winner.
    """
    if not node.untried_moves:
        raise ValueError("Cannot expand a fully expanded node")
    move = random.choice(node.untried_moves)
    node.untried_moves.remove(move)
    new_board = node.board.copy()
    new_board.make_move(move, node.player_to_move)
    child = Node(
        board=new_board,
        player_to_move=other(node.player_to_move),
        parent=node,
        move=move,
    )
    node.children.append(child)
    return child


def rollout(board: Board, player_to_move: str) -> str | None:
    """Play uniformly random moves from a copy of `board` until terminal.

    Returns the winning player, or None for a draw. Must not mutate `board`.
    """
    rollout_board = board.copy()
    while not rollout_board.is_terminal():
        moves = rollout_board.available_moves()
        if not moves:
            break
        move = random.choice(moves)
        rollout_board.make_move(move, player_to_move)
        player_to_move = other(player_to_move)
    return rollout_board.winner()


def backpropagate(node: Node | None, winner: str | None) -> None:
    """Walk from `node` to the root, adding one visit and the reward at each step.

    The reward at each node is from the perspective of the player who *moved into*
    it — other(node.player_to_move) — not the player to move from it. Getting
    this backwards produces a search that reliably prefers losing moves.
    """
    while node is not None:
        node.visits += 1
        if winner is None:
            node.wins += DRAW_REWARD
        elif winner == other(node.player_to_move):
            node.wins += WIN_REWARD
        else:
            node.wins += LOSS_REWARD
        node = node.parent


def build_tree(board: Board, player: str, simulations: int) -> Node:
    """Run `simulations` MCTS iterations from `board` and return the root node.

    Split out from mcts_move() so the tree itself can be inspected — visit
    distributions, whether terminal nodes stayed unexpanded, whether the root
    saw every simulation. mcts_move() is then just "build a tree, read off the
    most-visited child".
    """
    # Copy at the root so the caller's board can never be touched, whatever the
    # tree does with its own copies.
    root = Node(board=board.copy(), player_to_move=player)

    for _ in range(simulations):
        node = select(root)

        # Guard on the position, NOT on is_fully_expanded(). A won Connect-4
        # board usually still has legal moves, so a terminal node is never
        # "fully expanded" — expanding it would play on past the win and make
        # winner() report None for a decided game.
        if not node.board.is_terminal():
            node = expand(node)

        # A terminal node needs no rollout: the loop inside rollout() exits
        # immediately and returns the winner already on the board.
        winner = rollout(node.board, node.player_to_move)
        backpropagate(node, winner)

    return root


def mcts_move(board: Board, player: str, simulations: int = 1000) -> int | None:
    """Return the column MCTS favours after `simulations` iterations.

    Final choice is by visit count, not by win rate: visits are the robust
    statistic, since a child with one lucky win has a 100% rate on one sample.

    Returns None if the position has no legal moves.
    """
    root = build_tree(board, player, simulations)
    if not root.children:
        return None
    return max(root.children, key=lambda child: child.visits).move
