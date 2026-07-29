"""Tests for the heuristic evaluation function.

evaluate() is a judgement call, not a specification — there's no single correct
number for a position. So these tests pin down *structural properties* rather
than literal scores: symmetry, sign, ordering, and the magnitude bound that
keeps WIN_SCORE dominant. Those hold no matter how the weights get tuned.
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
    """Return a copy of `board` with every R swapped for Y and vice versa.

    Built by hand rather than by replaying moves, because the swapped position
    has the opposite player to move and isn't reachable by the same sequence.
    Safe here only because evaluate() never reads last_move — unlike winner().
    """
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
    """No duplicates within a window, and every (row, col) in range.

    Catches the off-by-one that lets a run of four hang off an edge.
    """
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
    """Derive each window's direction from its first two cells; all four of
    horizontal, vertical, and both diagonals must appear.

    len(WINDOWS) == 69 alone wouldn't catch a duplicated orientation paired
    with a missing one.
    """
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
    """A window holding both colours scores 0 — nobody can ever complete it.

    3-R-1-Y is the tempting one to score highly, and must not be.
    """
    assert score_window([PLAYER_R, PLAYER_R, PLAYER_R, PLAYER_Y]) == 0
    assert score_window([PLAYER_R, PLAYER_Y, PLAYER_R, EMPTY]) == 0
    assert score_window([PLAYER_Y, PLAYER_R, EMPTY, PLAYER_Y]) == 0


def test_completed_line_raises():
    """score_window rejects four of a colour rather than scoring it.

    That's a terminal line, which score() in minimax.py owns. Returning a
    large-but-finite number here would undercut WIN_SCORE's dominance.
    """
    with pytest.raises(ValueError):
        score_window([PLAYER_R] * CONNECT)
    with pytest.raises(ValueError):
        score_window([PLAYER_Y] * CONNECT)


# --------------------------------------------------------------------------
# evaluate — structural properties
# --------------------------------------------------------------------------

def test_empty_board_scores_zero():
    """Every window empty, no pieces to award a column bonus for."""
    assert evaluate(Board()) == 0


def test_colour_swap_negates_the_score():
    """evaluate(mirror_colours(b)) == -evaluate(b), for several positions.

    The single most valuable eval test: catches sign errors, asymmetric window
    scoring, and a column bonus applied to the wrong player.
    """
    for board in [
        Board(),
        play([(0, R), (1, Y), (2, R)]),
        play([(0, R), (1, Y), (2, R), (3, Y), (4, R)]),
        play([(3, R), (3, Y), (4, R), (2, Y), (5, R)]),
    ]:
        assert evaluate(mirror_colours(board)) == -evaluate(board)


def test_centre_column_beats_edge_column():
    """One R in column 3 scores higher than one R in column 0.

    Holds via COLUMN_BONUS and also via window count, so the margin is wide.
    """
    assert evaluate(play([(3, R)])) > evaluate(play([(0, R)]))


def test_column_bonus_is_actually_applied():
    """COLUMN_BONUS must contribute, not sit unused.

    Comparing two columns can't prove this: window count and column bonus both
    rise toward the centre, so any pair separated by one is separated by the
    other. Instead, compute the window-only total here and require evaluate()
    to exceed it by exactly the bonus — which pins the magnitude and the sign.
    """
    board = play([(3, R)])
    window_only = sum(
        score_window([board.grid[row][col] for row, col in window])
        for window in WINDOWS
    )
    assert evaluate(board) == window_only + COLUMN_BONUS[3]


def test_r_advantage_scores_positive():
    """R has a live three and Y has nothing in reach.

    Note this deliberately does NOT reuse r_can_win_in_one() from
    test_minimax: that position stacks Y directly on top of R, giving Y a live
    three of its own, and it evaluates slightly negative. It's built to be
    tactically winnable in one move, not positionally good.
    """
    board = play([(0, R), (6, Y), (1, R), (6, Y), (2, R)])
    assert board.winner() is None
    assert evaluate(board) > 0


def test_blocked_three_scores_below_live_three():
    """R with three in a window Y has plugged must score below R with three in
    an open window. This is the whole point of killing mixed windows."""
    live = play([(0, R), (1, R), (2, R)])
    blocked = play([(0, R), (1, R), (2, R), (3, Y)])
    assert evaluate(live) > evaluate(blocked)


# --------------------------------------------------------------------------
# the bound that keeps WIN_SCORE dominant
# --------------------------------------------------------------------------

def test_eval_is_bounded_by_max_eval():
    """|evaluate(board)| <= MAX_EVAL across many positions.

    Sweeps the 42 boards reachable by playing DRAW_SEQUENCE one move at a time —
    real positions of increasing density. The final board is terminal, and
    evaluate() is only defined on non-terminal positions, so it's skipped.
    """
    board = Board()
    for i, col in enumerate(DRAW_SEQUENCE):
        board.make_move(col, R if i % 2 == 0 else Y)
        if not board.is_terminal():
            assert abs(evaluate(board)) <= MAX_EVAL


def test_max_eval_is_far_below_win_score():
    """The gap that stops the search preferring a pretty position to a win.

    A pure constants check — no board involved. It fails the moment someone
    tunes WINDOW_SCORE upward without revisiting MAX_EVAL.
    """
    assert MAX_EVAL * 100 <= WIN_SCORE


def test_theoretical_max_eval_fits_the_bound():
    """The worst case the weights permit must also fit under MAX_EVAL.

    Stronger than the sweep above, which only samples one game's worth of
    positions: every window at maximum plus every cell's column bonus.
    """
    worst = len(WINDOWS) * WINDOW_SCORE[CONNECT - 1] + ROWS * sum(COLUMN_BONUS)
    assert worst <= MAX_EVAL
