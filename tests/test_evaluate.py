"""Tests for the scoring function.

There's no single right number for a position, so these check properties that
hold whatever the weights are: symmetry, sign, ordering, and the bound that keeps
WIN_SCORE on top.
"""

import pytest

from connect4.board import Board, EMPTY, PLAYER_R, PLAYER_Y, ROWS, COLS, CONNECT
from connect4.evaluate import (
    WINDOWS,
    WINDOW_SCORE,
    COLUMN_BONUS,
    MAX_EVAL,
    score_window,
    evaluate,
)
from connect4.minimax import WIN_SCORE
from tests.test_board import play, DRAW_SEQUENCE, R, Y


def mirror_colours(board: Board) -> Board:
    """Return a copy of `board` with every R swapped for Y and vice versa."""
    swap = {PLAYER_R: PLAYER_Y, PLAYER_Y: PLAYER_R, EMPTY: EMPTY}
    return Board([[swap[cell] for cell in row] for row in board.grid])


def window_of(r_count: int, y_count: int) -> list[str]:
    """Build a window with the given piece counts, padded with EMPTY."""
    return (
        [PLAYER_R] * r_count
        + [PLAYER_Y] * y_count
        + [EMPTY] * (CONNECT - r_count - y_count)
    )


# --------------------------------------------------------------------------
# window enumeration
# --------------------------------------------------------------------------

def test_there_are_69_windows():
    """24 horizontal + 21 vertical + 12 + 12 diagonal."""
    assert len(WINDOWS) == 69


def test_every_window_has_four_distinct_cells_on_the_board():
    """No duplicates within a window, and every (row, col) in range."""
    for window in WINDOWS:
        assert len(window) == CONNECT
        assert len(set(window)) == CONNECT
        for row, col in window:
            assert 0 <= row < ROWS
            assert 0 <= col < COLS


def test_windows_are_unique():
    """No window is enumerated twice."""
    assert len(set(WINDOWS)) == len(WINDOWS)


def test_all_four_orientations_are_present():
    """Derive each window's direction from its first two cells; all four of horizontal,
    vertical, and both diagonals must appear."""
    directions = set()
    for (r0, c0), (r1, c1), *_ in WINDOWS:
        directions.add((r1 - r0, c1 - c0))
    assert directions == {(0, 1), (1, 0), (1, 1), (1, -1)}


# --------------------------------------------------------------------------
# score_window
# --------------------------------------------------------------------------

def test_empty_window_scores_zero():
    assert score_window([EMPTY] * CONNECT) == 0


def test_window_with_only_r_scores_positive():
    """And more R pieces scores strictly higher than fewer."""
    one = score_window(window_of(1, 0))
    two = score_window(window_of(2, 0))
    three = score_window(window_of(3, 0))
    assert 0 < one < two < three


def test_window_with_only_y_scores_negative():
    """Exact mirror of the R case."""
    for count in (1, 2, 3):
        assert score_window(window_of(0, count)) == -score_window(window_of(count, 0))
    assert score_window(window_of(0, 3)) < 0


def test_mixed_window_is_dead():
    """A window holding both colours scores 0 - nobody can ever complete it."""
    assert score_window([PLAYER_R, PLAYER_R, PLAYER_R, PLAYER_Y]) == 0
    assert score_window([PLAYER_R, PLAYER_Y, PLAYER_R, EMPTY]) == 0
    assert score_window([PLAYER_Y, PLAYER_R, EMPTY, PLAYER_Y]) == 0


def test_completed_line_raises():
    """score_window rejects four of a colour rather than scoring it."""
    with pytest.raises(ValueError):
        score_window([PLAYER_R] * CONNECT)
    with pytest.raises(ValueError):
        score_window([PLAYER_Y] * CONNECT)


# --------------------------------------------------------------------------
# evaluate - structural properties
# --------------------------------------------------------------------------

def test_empty_board_scores_zero():
    """Every window empty, no pieces to award a column bonus for."""
    assert evaluate(Board()) == 0


def test_colour_swap_negates_the_score():
    """evaluate(mirror_colours(b)) == -evaluate(b), for several positions."""
    for board in [
        Board(),
        play([(0, R), (1, Y), (2, R)]),
        play([(0, R), (1, Y), (2, R), (3, Y), (4, R)]),
        play([(3, R), (3, Y), (4, R), (2, Y), (5, R)]),
    ]:
        assert evaluate(mirror_colours(board)) == -evaluate(board)


def test_centre_column_beats_edge_column():
    """One R in column 3 scores higher than one R in column 0."""
    assert evaluate(play([(3, R)])) > evaluate(play([(0, R)]))


def test_column_bonus_is_actually_applied():
    """COLUMN_BONUS must contribute, not sit unused."""
    board = play([(3, R)])
    window_only = sum(
        score_window([board.grid[row][col] for row, col in window])
        for window in WINDOWS
    )
    assert evaluate(board) == window_only + COLUMN_BONUS[3]


def test_r_advantage_scores_positive():
    """R has a live three and Y has nothing in reach."""
    board = play([(0, R), (6, Y), (1, R), (6, Y), (2, R)])
    assert board.winner() is None
    assert evaluate(board) > 0


def test_blocked_three_scores_below_live_three():
    """R with three in a window Y has plugged must score below R with three in an open
    window."""
    live = play([(0, R), (1, R), (2, R)])
    blocked = play([(0, R), (1, R), (2, R), (3, Y)])
    assert evaluate(live) > evaluate(blocked)


# --------------------------------------------------------------------------
# the bound that keeps WIN_SCORE dominant
# --------------------------------------------------------------------------

def test_eval_is_bounded_by_max_eval():
    """|evaluate(board)| <= MAX_EVAL across many positions."""
    board = Board()
    for i, col in enumerate(DRAW_SEQUENCE):
        board.make_move(col, R if i % 2 == 0 else Y)
        if not board.is_terminal():
            assert abs(evaluate(board)) <= MAX_EVAL


def test_max_eval_is_far_below_win_score():
    """The gap that stops the search preferring a pretty position to a win."""
    assert MAX_EVAL * 100 <= WIN_SCORE


def test_theoretical_max_eval_fits_the_bound():
    """The worst case the weights permit must also fit under MAX_EVAL."""
    worst = len(WINDOWS) * WINDOW_SCORE[CONNECT - 1] + ROWS * sum(COLUMN_BONUS)
    assert worst <= MAX_EVAL
