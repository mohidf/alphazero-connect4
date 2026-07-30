"""Tests for alpha-beta and move ordering.

Three separate things: that pruning gives the same answer as plain minimax, that
it visits fewer nodes, and that centre-first ordering visits fewer still.

Same value, not necessarily the same move - when moves tie, pruning changes which
one gets reported, so only forced positions are checked by column.
"""

import pytest

from connect4.board import Board, COLS, ROWS
from connect4 import minimax as mm
from connect4 import alphabeta as ab
from connect4.alphabeta import COLUMN_ORDER, ordered_moves, alphabeta
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y
from tests.test_minimax import (
    r_has_won,
    y_has_won,
    r_can_win_in_one,
    y_can_win_in_one,
)

LEFT_TO_RIGHT = list(range(COLS))

# Varied in density and symmetry. The empty board is the one most
# likely to expose a tie-breaking difference, and the denser ones stress the
# depth cutoff.
AGREEMENT_POSITIONS = [
    Board(),
    play([(3, R)]),
    play([(3, R), (3, Y)]),
    play([(0, R), (1, Y), (2, R), (3, Y), (4, R)]),
    play([(3, R), (3, Y), (4, R), (2, Y), (5, R), (1, Y)]),
    r_can_win_in_one(),
    y_can_win_in_one(),
]


def count_minimax(board: Board, depth: int, is_maximizing: bool) -> int:
    """Run plain minimax and return the number of nodes it entered."""
    mm.reset_nodes()
    mm.minimax(board, depth, is_maximizing)
    return mm.nodes_visited()


def count_alphabeta(
    board: Board, depth: int, is_maximizing: bool, order: list[int] = COLUMN_ORDER
) -> int:
    """Run alpha-beta and return the number of nodes it entered."""
    ab.reset_nodes()
    alphabeta(board, depth, is_maximizing, order=order)
    return ab.nodes_visited()


def full_column_board(col: int) -> Board:
    """A board with `col` filled to the top."""
    board = Board()
    for i in range(ROWS):
        board.make_move(col, R if i % 2 == 0 else Y)
    return board


# --------------------------------------------------------------------------
# move ordering
# --------------------------------------------------------------------------

def test_column_order_is_a_permutation_of_all_columns():
    """COLUMN_ORDER must name every column exactly once - otherwise the search silently
    never considers some legal moves."""
    assert sorted(COLUMN_ORDER) == LEFT_TO_RIGHT


def test_ordered_moves_is_a_permutation_of_available_moves():
    """Same columns as available_moves(), different sequence."""
    for board in [Board(), play([(3, R), (3, Y)]), full_column_board(3)]:
        assert sorted(ordered_moves(board)) == board.available_moves()


def test_ordered_moves_puts_centre_first_on_an_empty_board():
    """All seven legal, so the result is COLUMN_ORDER itself."""
    assert ordered_moves(Board()) == COLUMN_ORDER


def test_ordered_moves_skips_full_columns():
    """Fill column 3 and it drops out while the rest keep centre-out order."""
    board = full_column_board(3)
    assert 3 not in ordered_moves(board)
    assert ordered_moves(board) == [col for col in COLUMN_ORDER if col != 3]


def test_left_to_right_order_matches_available_moves():
    """order=LEFT_TO_RIGHT must reproduce available_moves() exactly - this is the baseline
    the ordering comparison depends on being honest."""
    for board in [Board(), full_column_board(0), play([(2, R), (2, Y)])]:
        assert ordered_moves(board, LEFT_TO_RIGHT) == board.available_moves()


# --------------------------------------------------------------------------
# correctness: same value as plain minimax
# --------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_agrees_with_minimax_on_value(depth):
    """The core correctness claim, swept over positions and depths."""
    for board in AGREEMENT_POSITIONS:
        for is_maximizing in (True, False):
            expected = mm.minimax(board, depth, is_maximizing)
            actual = alphabeta(board, depth, is_maximizing)
            assert actual == expected


@pytest.mark.parametrize("depth", [2, 4])
def test_agrees_with_minimax_under_left_to_right_order(depth):
    """Same claim with ordering held identical to minimax's, which isolates the pruning
    logic from the ordering change."""
    for board in AGREEMENT_POSITIONS:
        expected = mm.minimax(board, depth, True)
        actual = alphabeta(board, depth, True, order=LEFT_TO_RIGHT)
        assert actual == expected


def test_takes_immediate_win():
    """Unique best move, so the column is assertable, not just the value."""
    col, value = ab.best_move(r_can_win_in_one(), 1, True)
    assert col == 3
    assert value >= mm.WIN_SCORE


def test_blocks_immediate_loss():
    """Also a unique best move."""
    col, _ = ab.best_move(y_can_win_in_one(), 2, True)
    assert col == 3


def test_terminal_and_draw_scoring_match_minimax():
    """Terminal boards must not be affected by pruning at all."""
    for board in [r_has_won(), y_has_won(), play_alternating(DRAW_SEQUENCE)]:
        for depth in (0, 3):
            assert alphabeta(board, depth, True) == mm.minimax(board, depth, True)


# --------------------------------------------------------------------------
# effectiveness: pruning
# --------------------------------------------------------------------------

def test_pruning_visits_strictly_fewer_nodes():
    """Same position, same depth, same ordering - only the algorithm differs."""
    plain = count_minimax(Board(), 4, True)
    pruned = count_alphabeta(Board(), 4, True, order=LEFT_TO_RIGHT)
    assert pruned < plain


def test_node_counter_resets():
    """Two identical searches must report the same count."""
    first = count_alphabeta(Board(), 3, True)
    second = count_alphabeta(Board(), 3, True)
    assert first == second

    first_mm = count_minimax(Board(), 3, True)
    second_mm = count_minimax(Board(), 3, True)
    assert first_mm == second_mm


def test_pruning_advantage_grows_with_depth():
    """The saving should widen as depth increases - the b^(d/2) vs b^d gap."""
    def ratio(depth: int) -> float:
        plain = count_minimax(Board(), depth, True)
        pruned = count_alphabeta(Board(), depth, True, order=LEFT_TO_RIGHT)
        return plain / pruned

    assert ratio(4) > ratio(2)


# --------------------------------------------------------------------------
# effectiveness: move ordering
# --------------------------------------------------------------------------

def test_centre_first_ordering_visits_fewer_nodes():
    """Same algorithm, same depth, same position - only the order differs."""
    centre_first = count_alphabeta(Board(), 5, True, order=COLUMN_ORDER)
    left_to_right = count_alphabeta(Board(), 5, True, order=LEFT_TO_RIGHT)
    assert centre_first < left_to_right


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def test_search_leaves_board_unchanged():
    """Pruning breaks out of loops early - every early exit must still undo."""
    for order in (COLUMN_ORDER, LEFT_TO_RIGHT):
        board = play([(3, R), (3, Y), (4, R)])
        grid_before = [row[:] for row in board.grid]
        count_before = board.move_count

        alphabeta(board, 5, False, order=order)

        assert board.grid == grid_before
        assert board.move_count == count_before


def test_best_move_leaves_board_unchanged():
    """Same contract for the root wrapper, which threads its own alpha/beta."""
    board = play([(3, R), (3, Y)])
    grid_before = [row[:] for row in board.grid]

    ab.best_move(board, 4, True)

    assert board.grid == grid_before
    assert board.move_count == 2


def test_best_move_returns_a_legal_column():
    board = play([(3, R), (3, Y)])
    col, _ = ab.best_move(board, 3, True)
    assert col in board.available_moves()


def test_best_move_on_full_board_returns_none():
    board = play_alternating(DRAW_SEQUENCE)
    assert board.available_moves() == []
    col, _ = ab.best_move(board, 3, True)
    assert col is None


def test_best_move_agrees_with_minimax_on_forced_positions():
    """Where the best move is unique, both searches must name the same column."""
    for board, expected in [(r_can_win_in_one(), 3), (y_can_win_in_one(), 3)]:
        assert mm.best_move(board, 2, True)[0] == expected
        assert ab.best_move(board, 2, True)[0] == expected
