"""Depth-limited minimax with alpha-beta pruning, plus move ordering.

Same contract as minimax(): returns the estimated value of a position, positive
favouring R. Provably returns the same *value* as plain minimax at the root —
but not necessarily the same move, since ties are broken by iteration order and
pruning changes which ties are ever seen.

Alpha-beta's payoff depends on move order:
    perfect ordering    -> O(b^(d/2))
    worst-case ordering -> O(b^d), no better than plain minimax
Hence COLUMN_ORDER below. It is the cheapest large win available in this phase.
"""

import operator

from connect4.board import Board, COLS, PLAYER_R, PLAYER_Y
from connect4.evaluate import evaluate
from connect4.minimax import WIN_SCORE, WORST_FOR_R, WORST_FOR_Y, score

# Centre-out. Central columns sit in more windows and tend to be stronger, so
# trying them first tightens alpha/beta earlier and cuts more subtrees.
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Nodes entered by the last top-level search. Reset by reset_nodes(); compared
# against minimax's counter to prove the pruning is real.
_nodes = 0


def reset_nodes() -> None:
    """Zero the node counter. Call before each top-level search."""
    global _nodes
    _nodes = 0


def nodes_visited() -> int:
    """Nodes entered since the last reset_nodes()."""
    return _nodes


def ordered_moves(board: Board, order: list[int] = COLUMN_ORDER) -> list[int]:
    """Return the legal columns of `board`, sequenced by `order`.

    A permutation of board.available_moves() — same columns, different order.
    Pass order=list(range(COLS)) for plain left-to-right, which is the baseline
    the ordering comparison measures against.

    Walks `order` and filters, rather than sorting by rank: this runs at every
    node, so it stays O(COLS) with no per-comparison index lookups.
    """
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
    """Return the estimated value of `board`. Positive favours R.

    Base cases are identical to minimax(), and the ordering rule is the same:
    terminal before depth.

    `order` is threaded into every recursive call. Dropping it would let children
    revert to the default centre-first ordering, which would quietly make the
    left-to-right baseline a lie and invalidate the ordering comparison.

    Note the `break` sits after undo_move: an early exit still has to restore
    the board, and the pruned paths are the hardest ones to notice corruption on.
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
    """Return the (column, value) this search considers best from `board`.

    Worth threading alpha/beta through this loop too rather than calling
    alphabeta() with fresh bounds each time: after the first child returns, its
    value is a valid alpha (or beta) for the remaining children, and skipping
    that gives up most of the pruning at the root.

    Does NOT reset the node counter — the caller owns that boundary, so the same
    measurement applies to minimax.best_move() and this one.
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

        # Narrow the window for the remaining children. A child that can't beat
        # the best so far may return an inexact bound, but it also can't be
        # selected, since the comparison above is strict.
        if is_maximizing:
            alpha = max(alpha, best_score)
        else:
            beta = min(beta, best_score)

    return best_col, best_score
