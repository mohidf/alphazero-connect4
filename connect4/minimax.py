"""Depth-limited minimax for Connect-4.

The Tic-Tac-Toe version searched to terminal and returned the true game value.
Connect-4's tree is ~4.5e12 nodes, so this one stops at a depth limit and asks
evaluate() to guess the value of the position it stopped at. The result is an
estimate, and it's only as good as the heuristic at the leaves.

Convention: PLAYER_R maximizes, PLAYER_Y minimizes.
"""

import operator

from connect4.board import Board, PLAYER_R, PLAYER_Y
from connect4.evaluate import evaluate

# A win must outrank every heuristic value, or the search would prefer a
# good-looking position to an actual win. evaluate() is bounded by MAX_EVAL,
# which is two orders of magnitude below this.
WIN_SCORE = 1_000_000

# Every score lies in [-(WIN_SCORE + depth), +(WIN_SCORE + depth)], so these are
# beyond any reachable value and serve as +/-infinity while staying ints.
WORST_FOR_R = -2 * WIN_SCORE
WORST_FOR_Y = 2 * WIN_SCORE


def score(board: Board, depth: int) -> int:
    """Return the score of a terminal board. Assumes board.is_terminal().

    `depth` is the search depth *remaining*, so a win found sooner arrives with
    more left over and scores higher. That's what makes the engine prefer a win
    in 1 over a win in 7, and a loss in 7 over a loss in 1.
    """
    winner = board.winner()
    if winner == PLAYER_R:
        return WIN_SCORE + depth
    if winner == PLAYER_Y:
        return -(WIN_SCORE + depth)
    return 0


def minimax(board: Board, depth: int, is_maximizing: bool) -> int:
    """Return the estimated value of `board`. Positive favours R.

    The terminal check must come before the depth check: a position that is both
    a win and at the horizon has to return the win score, not the heuristic.
    """
    if board.is_terminal():
        return score(board, depth)
    if depth == 0:
        return evaluate(board)

    # Locals must not be named `score` — that would shadow the function above
    # and, because Python binds locals at compile time, break the call on
    # line 46 with UnboundLocalError.
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
    """Return the (column, value) this search considers best from `board`.

    Ties go to whichever column `available_moves()` yields first, so the choice
    among equal moves is currently left-to-right.
    """
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
