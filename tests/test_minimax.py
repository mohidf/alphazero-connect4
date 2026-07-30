"""Tests for depth-limited minimax.

evaluate() is stubbed to 0 for most of these, so the only thing the search can
find is a forced win or loss. That keeps the search separate from the quality of
the heuristic.
"""

from connect4.board import Board
from connect4.evaluate import evaluate
from connect4.minimax import minimax, best_move, score, WIN_SCORE
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y


# --------------------------------------------------------------------------
# shared positions
# --------------------------------------------------------------------------

def r_has_won() -> Board:
    """R has four stacked in column 0."""
    return play([(0, R), (1, Y), (0, R), (1, Y), (0, R), (1, Y), (0, R)])


def y_has_won() -> Board:
    """Y has four stacked in column 0."""
    return play([(0, Y), (1, R), (0, Y), (1, R), (0, Y), (1, R), (0, Y)])


def r_can_win_in_one() -> Board:
    """R holds (5,0) (5,1) (5,2); playing column 3 completes the bottom row."""
    return play([(0, R), (0, Y), (1, R), (1, Y), (2, R), (2, Y)])


def y_can_win_in_one() -> Board:
    """Mirror image: Y threatens (5,3), so R to move must block column 3."""
    return play([(0, Y), (0, R), (1, Y), (1, R), (2, Y), (2, R)])


# --------------------------------------------------------------------------
# terminal scoring
# --------------------------------------------------------------------------

def test_score_of_r_win_is_positive():
    board = r_has_won()
    assert score(board, 0) == WIN_SCORE
    assert score(board, 3) == WIN_SCORE + 3


def test_score_of_y_win_is_negative():
    board = y_has_won()
    assert score(board, 0) == -WIN_SCORE
    assert score(board, 3) == -(WIN_SCORE + 3)


def test_score_of_draw_is_zero():
    board = play_alternating(DRAW_SEQUENCE)
    assert board.is_terminal()
    assert board.winner() is None
    assert score(board, 4) == 0


def test_faster_win_scores_higher():
    """More depth remaining means the win was reached sooner, so it scores higher."""
    assert score(r_has_won(), 5) > score(r_has_won(), 2)


def test_slower_loss_scores_higher():
    """The mirror property: R prefers to delay defeat."""
    assert score(y_has_won(), 2) > score(y_has_won(), 5)


# --------------------------------------------------------------------------
# the search finds forced wins (holds while evaluate() returns 0)
# --------------------------------------------------------------------------

def test_takes_immediate_win():
    board = r_can_win_in_one()
    col, value = best_move(board, 1, True)
    assert col == 3
    assert value >= WIN_SCORE


def test_blocks_immediate_loss():
    """R must play column 3 or Y completes the bottom row next move."""
    board = y_can_win_in_one()
    col, _ = best_move(board, 2, True)
    assert col == 3


def test_prefers_winning_now_over_winning_later():
    """Searching deeper must still take the win at the shallowest ply."""
    board = r_can_win_in_one()
    depth = 5
    col, value = best_move(board, depth, True)
    assert col == 3
    assert value == WIN_SCORE + depth - 1


# --------------------------------------------------------------------------
# search invariants
# --------------------------------------------------------------------------

def test_search_leaves_board_unchanged():
    """A full search must undo everything it does."""
    board = play([(3, R), (3, Y), (4, R)])
    grid_before = [row[:] for row in board.grid]
    count_before = board.move_count

    minimax(board, 4, False)

    assert board.grid == grid_before
    assert board.move_count == count_before


def test_best_move_returns_a_legal_column():
    board = play([(3, R), (3, Y)])
    col, _ = best_move(board, 3, True)
    assert col in board.available_moves()


def test_best_move_on_full_board_returns_none():
    board = play_alternating(DRAW_SEQUENCE)
    assert board.available_moves() == []
    col, _ = best_move(board, 3, True)
    assert col is None


def test_depth_zero_returns_the_heuristic():
    """At depth 0 on a non-terminal board, minimax returns evaluate() and does not recurse."""
    board = play([(3, R), (3, Y)])
    assert not board.is_terminal()
    assert minimax(board, 0, True) == evaluate(board)


def test_terminal_beats_depth_limit():
    """A board that is BOTH terminal and at depth 0 must return the win score, not the
    heuristic - the base-case ordering trap."""
    board = r_has_won()
    assert minimax(board, 0, False) == WIN_SCORE
    assert minimax(board, 0, True) == WIN_SCORE
