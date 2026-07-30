"""Minimax plus alpha-beta pruning and move ordering.

Returns the same value as plain minimax, though not always the same move, since
pruning changes which tied moves ever get looked at.

Move order matters a lot here: with good ordering alpha-beta is O(b^(d/2)), with
bad ordering it's O(b^d) and no better than plain minimax.
"""

import operator

from connect4.board import Board, COLS, PLAYER_R, PLAYER_Y
from connect4.evaluate import evaluate
from connect4.minimax import WIN_SCORE, WORST_FOR_R, WORST_FOR_Y, score

# Middle columns first. They're usually the better moves, so trying them early
# tightens the window sooner and cuts more branches.
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

_nodes = 0


def reset_nodes() -> None:
    global _nodes
    _nodes = 0


def nodes_visited() -> int:
    return _nodes


def ordered_moves(board: Board, order: list[int] = COLUMN_ORDER) -> list[int]:
    """Legal columns, in `order`. Pass list(range(COLS)) for left-to-right."""
    available = board.available_moves()
    return [col for col in order if col in available]


def alphabeta(
    board: Board,
    depth: int,
    is_maximizing: bool,
    alpha: int = WORST_FOR_R,
    beta: int = WORST_FOR_Y,
    order: list[int] = COLUMN_ORDER,
) -> int:
    """Estimated value of the position. Positive favours R.

    `order` gets passed down so children use the same ordering, otherwise the
    left-to-right baseline isn't actually left-to-right below the root.

    The break comes after undo_move, or pruned branches leave the board dirty.
    """
    global _nodes
    _nodes += 1

    if board.is_terminal():
        return score(board, depth)
    if depth == 0:
        return evaluate(board)

    if is_maximizing:
        best = WORST_FOR_R
        for col in ordered_moves(board, order):
            board.make_move(col, PLAYER_R)
            value = alphabeta(board, depth - 1, False, alpha, beta, order)
            board.undo_move(col)
            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
    else:
        best = WORST_FOR_Y
        for col in ordered_moves(board, order):
            board.make_move(col, PLAYER_Y)
            value = alphabeta(board, depth - 1, True, alpha, beta, order)
            board.undo_move(col)
            best = min(best, value)
            beta = min(beta, best)
            if beta <= alpha:
                break
    return best


def best_move(
    board: Board,
    depth: int,
    is_maximizing: bool,
    order: list[int] = COLUMN_ORDER,
) -> tuple[int | None, int]:
    """Best (column, value) from this position.

    Carries alpha/beta across the children instead of restarting the window each
    time, which is where most of the pruning at the root comes from. Doesn't
    touch the node counter, so it can be compared with minimax.best_move().
    """
    moves = ordered_moves(board, order)
    if not moves:
        return None, 0

    player = PLAYER_R if is_maximizing else PLAYER_Y
    is_better = operator.gt if is_maximizing else operator.lt
    best_col = None
    best_score = WORST_FOR_R if is_maximizing else WORST_FOR_Y
    alpha, beta = WORST_FOR_R, WORST_FOR_Y

    for col in moves:
        board.make_move(col, player)
        value = alphabeta(board, depth - 1, not is_maximizing, alpha, beta, order)
        board.undo_move(col)

        if is_better(value, best_score):
            best_score = value
            best_col = col

        # A child that can't beat the best so far might return a loose bound,
        # but the strict comparison above means it can't get picked anyway.
        if is_maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)

    return best_col, best_score
