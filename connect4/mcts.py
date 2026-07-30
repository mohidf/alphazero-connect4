"""Monte Carlo Tree Search with random rollouts. No network involved.

The usual four steps: select, expand, simulate, backpropagate. Node boards come
from Board.copy() so they keep their move history.
"""

import math
import random

from connect4.board import Board, PLAYER_R, PLAYER_Y

# Standard UCB1 constant for rewards in [0, 1].
EXPLORATION = math.sqrt(2)

WIN_REWARD = 1.0
DRAW_REWARD = 0.5
LOSS_REWARD = 0.0


def other(player: str) -> str:
    return PLAYER_Y if player == PLAYER_R else PLAYER_R


class Node:
    """A position in the tree.

    player_to_move is whose turn it is here, so the move that got us here was
    played by the *other* one. That's who backpropagate() credits.
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
        self.move = move
        self.children: list["Node"] = []
        self.untried_moves: list[int] = board.available_moves()
        self.visits = 0
        self.wins = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def ucb1(self, exploration: float = EXPLORATION) -> float:
        # Unvisited scores infinity so it gets tried before anything is revisited.
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term

    def best_child(self) -> "Node":
        return max(self.children, key=lambda child: child.ucb1())


def select(node: Node) -> Node:
    """Walk down while there's nothing new to try and the game isn't over."""
    while node.is_fully_expanded() and not node.board.is_terminal():
        node = node.best_child()
    return node


def expand(node: Node) -> Node:
    """Try one of the untried moves and hang a new child off it."""
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
    """Play random moves on a copy until the game ends. Returns the winner."""
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
    """Add a visit and the result to every node back up to the root.

    Each node is scored for whoever moved *into* it. Get this backwards and the
    search happily picks losing moves.
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
    """Run the search and hand back the root, so the tree can be inspected."""
    root = Node(board=board.copy(), player_to_move=player)

    for _ in range(simulations):
        node = select(root)

        # Check the position, not is_fully_expanded(). A won board usually still
        # has legal moves, so expanding here would play on past the win.
        if not node.board.is_terminal():
            node = expand(node)

        # On a terminal node rollout() returns straight away with the winner.
        winner = rollout(node.board, node.player_to_move)
        backpropagate(node, winner)

    return root


def mcts_move(board: Board, player: str, simulations: int = 1000) -> int | None:
    """Pick a move. Uses visit count, not win rate - one lucky win off a single
    visit would otherwise look like a 100% move."""
    root = build_tree(board, player, simulations)
    if not root.children:
        return None
    return max(root.children, key=lambda child: child.visits).move
