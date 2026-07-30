"""Minimax with a depth limit.

The full tree is about 4.5e12 nodes, so the search stops at a fixed depth and
uses evaluate() to guess the value there. R maximizes, Y minimizes.
"""

import operator

from connect4.board import Board, PLAYER_R, PLAYER_Y
from connect4.evaluate import evaluate

# Has to be much larger than anything evaluate() can return, or the search would
# rather have a nice-looking position than an actual win.
WIN_SCORE = 1_000_000

# Stand-ins for +/-infinity that stay ints.
WORST_FOR_R = -2 * WIN_SCORE
WORST_FOR_Y = 2 * WIN_SCORE


def score(board: Board, depth: int) -> int:
    """Score a finished board. `depth` is the depth *left*, so quicker wins
    score higher and the engine doesn't dawdle."""
    winner = board.winner()
    if winner == PLAYER_R:
        return WIN_SCORE + depth
    if winner == PLAYER_Y:
        return -(WIN_SCORE + depth)
    return 0


# Node counter, for comparing against alpha-beta. Both count the same thing:
# one per call, before the base cases.
_nodes = 0


def reset_nodes() -> None:
    global _nodes
    _nodes = 0


def nodes_visited() -> int:
    return _nodes


def minimax(board: Board, depth: int, is_maximizing: bool) -> int:
    """Estimated value of the position. Positive favours R.

    Terminal check goes before the depth check, or a win sitting exactly at the
    depth limit gets scored by the heuristic instead.
    """
    global _nodes
    _nodes += 1

    if board.is_terminal():
        return score(board, depth)
    if depth == 0:
        return evaluate(board)

    # Don't name a local `score` here - it shadows the function above and Python
    # decides that at compile time, so the call up there breaks.
    if is_maximizing:
        best = WORST_FOR_R
        for col in board.available_moves():
            board.make_move(col, PLAYER_R)
            value = minimax(board, depth - 1, False)
            board.undo_move(col)
            best = max(best, value)
    else:
        best = WORST_FOR_Y
        for col in board.available_moves():
            board.make_move(col, PLAYER_Y)
            value = minimax(board, depth - 1, True)
            board.undo_move(col)
            best = min(best, value)
    return best


def best_move(board: Board, depth: int, is_maximizing: bool) -> tuple[int | None, int]:
    """Best (column, value) from this position. Ties go to the lowest column."""
    moves = board.available_moves()
    if not moves:
        return None, 0

    player = PLAYER_R if is_maximizing else PLAYER_Y
    is_better = operator.gt if is_maximizing else operator.lt
    best_col = None
    best_score = WORST_FOR_R if is_maximizing else WORST_FOR_Y

    for col in moves:
        board.make_move(col, player)
        value = minimax(board, depth - 1, not is_maximizing)
        board.undo_move(col)
        if is_better(value, best_score):
            best_score = value
            best_col = col

    return best_col, best_score
