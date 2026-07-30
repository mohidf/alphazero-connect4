"""Tests for the board itself.

Positions are built with play()/play_alternating() rather than by writing to the
grid - a hand-made grid can hold impossible states and has no move history, which
winner() needs.
"""

import pytest

from connect4.board import (
    Board,
    EMPTY,
    PLAYER_R,
    PLAYER_Y,
    ROWS,
    COLS,
)

R = PLAYER_R
Y = PLAYER_Y

# A 42-move column order that fills the board with strict R/Y alternation and
# never completes a four-in-a-row - not at the end, and not at any point along
# the way, so the last-move win check stays valid throughout.
# Found by backtracking search; obvious fill orders (left-to-right, column by
# column) all produce accidental wins long before the board fills.
DRAW_SEQUENCE = [
    4, 1, 6, 2, 1, 0, 1, 4, 4, 2, 4, 1, 3, 6,
    3, 3, 1, 0, 4, 6, 6, 1, 2, 3, 2, 3, 2, 2,
    6, 5, 5, 0, 5, 4, 6, 5, 5, 5, 0, 0, 3, 0,
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def play(moves: list[tuple[int, str]]) -> Board:
    """Build a board from explicit (column, player) pairs."""
    board = Board()
    for col, player in moves:
        board.make_move(col, player)
    return board


def play_alternating(cols: list[int], first: str = PLAYER_R) -> Board:
    """Build a board by alternating players over a list of columns."""
    other = PLAYER_Y if first == PLAYER_R else PLAYER_R
    return play([
        (col, first if i % 2 == 0 else other)
        for i, col in enumerate(cols)
    ])


# --------------------------------------------------------------------------
# board setup
# --------------------------------------------------------------------------

def test_new_board_is_empty():
    """Every cell is EMPTY, move_count is 0."""
    board = Board()
    for row in board.grid:
        for cell in row:
            assert cell == EMPTY
    assert board.move_count == 0


def test_new_board_has_no_winner():
    """winner() must handle last_move being None."""
    board = Board()
    assert board.winner() is None


# --------------------------------------------------------------------------
# gravity and move mechanics
# --------------------------------------------------------------------------

def test_first_piece_lands_on_bottom_row():
    """make_move on an empty column returns ROWS - 1 and fills grid[ROWS-1][col]."""
    board = Board()
    row = board.make_move(0, PLAYER_R)
    assert row == ROWS - 1
    assert board.grid[row][0] == PLAYER_R
    assert board.move_count == 1


def test_pieces_stack_upward():
    """Second piece in the same column lands one row above the first."""
    board = Board()
    board.make_move(0, PLAYER_R)
    row = board.make_move(0, PLAYER_Y)
    assert row == ROWS - 2
    assert board.grid[row][0] == PLAYER_Y
    assert board.move_count == 2


def test_pieces_in_other_columns_are_independent():
    """Filling column 0 must not affect where a piece lands in column 6."""
    board = Board()
    board.make_move(0, PLAYER_R)
    row = board.make_move(6, PLAYER_Y)
    assert row == ROWS - 1
    assert board.grid[row][6] == PLAYER_Y
    assert board.move_count == 2


def test_available_moves_on_empty_board():
    """All seven columns are playable."""
    board = Board()
    assert board.available_moves() == list(range(COLS))


def test_available_moves_still_seven_after_one_move():
    """A column with room left stays playable - unlike Tic-Tac-Toe cells."""
    board = Board()
    board.make_move(0, PLAYER_R)
    # Column 0 still has ROWS - 1 empty cells above the piece, so it stays legal.
    # (In Tic-Tac-Toe the played cell would leave the list; here it takes six.)
    assert board.available_moves() == list(range(COLS))


def test_full_column_drops_out_of_available_moves():
    """After ROWS pieces, the column disappears from the legal move list."""
    board = Board()
    for _ in range(ROWS):
        board.make_move(0, PLAYER_R)
    assert 0 not in board.available_moves()


def test_move_into_full_column_raises():
    """Dropping into a full column raises ValueError."""
    board = Board()
    for _ in range(ROWS):
        board.make_move(0, PLAYER_R)
    with pytest.raises(ValueError):
        board.make_move(0, PLAYER_Y)


def test_move_out_of_bounds_raises():
    """Column -1 and column COLS both raise ValueError, not IndexError."""
    board = Board()
    with pytest.raises(ValueError):
        board.make_move(-1, PLAYER_R)
    with pytest.raises(ValueError):
        board.make_move(COLS, PLAYER_R)


# --------------------------------------------------------------------------
# undo
# --------------------------------------------------------------------------

def test_undo_removes_topmost_piece_only():
    """Undoing a column with two pieces leaves the lower one untouched."""
    board = Board()
    board.make_move(0, PLAYER_R)
    board.make_move(0, PLAYER_Y)
    board.undo_move(0)
    assert board.grid[ROWS - 1][0] == PLAYER_R
    assert board.grid[ROWS - 2][0] == EMPTY
    assert board.move_count == 1
    


def test_undo_restores_board_exactly():
    """make/undo round trip: grid and move_count identical to before."""
    board = Board()
    before = [row[:] for row in board.grid]

    # A sequence with stacking, so undo has to pick the right cell in a column
    # that holds more than one piece.
    seq = [(3, R), (3, Y), (4, R), (2, Y), (3, R), (0, Y)]
    for col, player in seq:
        board.make_move(col, player)

    # Reverse order: this is how search unwinds. Undoing forwards would pass
    # here while still being wrong.
    for col, _ in reversed(seq):
        board.undo_move(col)

    assert board.grid == before
    assert board.move_count == 0


def test_undo_empty_column_raises():
    """Undoing a column that was never played raises ValueError."""
    board = Board()
    with pytest.raises(ValueError):
        board.undo_move(0)


# --------------------------------------------------------------------------
# win detection - one test per direction
# --------------------------------------------------------------------------

def test_vertical_win():
    """Four stacked in one column."""
    moves = [(0, R), (1, Y), (0, R), (1, Y), (0, R), (1, Y), (0, R)]
    board = play(moves)
    assert board.winner() == R


def test_horizontal_win_at_left_edge():
    """Cols 0-3."""
    moves = [(0, R), (0, Y), (1, R), (1, Y), (2, R), (2, Y), (3, R)]
    board = play(moves)
    assert board.winner() == R


def test_horizontal_win_at_right_edge():
    """Cols 3-6."""
    moves = [(3, R), (3, Y), (4, R), (4, Y), (5, R), (5, Y), (6, R)]
    board = play(moves)
    assert board.winner() == R


def test_diagonal_win_ascending():
    """The '/' diagonal: R at (5,0) (4,1) (3,2) (2,3)."""
    moves = [
        (0, R),                          # (5,0)
        (1, Y), (1, R),                  # (4,1)
        (2, Y), (2, Y), (2, R),          # (3,2)
        (3, Y), (3, Y), (3, Y), (3, R),  # (2,3)
    ]
    board = play(moves)
    assert board.winner() == R


def test_diagonal_win_descending():
    """The '\\' diagonal: R at (2,0) (3,1) (4,2) (5,3)."""
    moves = [
        (0, Y), (0, Y), (0, Y), (0, R),  # (2,0)
        (1, Y), (1, Y), (1, R),          # (3,1)
        (2, Y), (2, R),                  # (4,2)
        (3, R),                          # (5,3)
    ]
    board = play(moves)
    assert board.winner() == R


# --------------------------------------------------------------------------
# win detection - things that must NOT count as wins
# --------------------------------------------------------------------------

def test_three_in_a_row_is_not_a_win():
    """The +1 boundary: three must return None, four must return the player."""
    moves = [(0, R), (1, R), (2, R)]
    board = play(moves)
    assert board.winner() is None
    board.make_move(3, R)
    assert board.winner() == R


def test_mixed_colours_do_not_combine():
    """R Y R Y across the bottom row is not a win for anyone."""
    moves = [(0, R), (1, Y), (2, R), (3, Y)]
    board = play(moves)
    assert board.winner() is None


def test_no_wrap_around_between_rows():
    """R on row 5 cols 5-6 plus R on row 4 cols 0-1 is four pieces, not a line."""
    moves = [
        (5, R), (6, R),                  # (5,5) (5,6) - right end of the bottom row
        (0, Y), (1, Y),                  # filler, to lift the next two to row 4
        (0, R), (1, R),                  # (4,0) (4,1) - left end of the row above
    ]
    board = play(moves)
    assert board.grid[ROWS - 1][COLS - 1] == R
    assert board.grid[ROWS - 2][0] == R
    assert board.winner() is None


# --------------------------------------------------------------------------
# terminal states
# --------------------------------------------------------------------------

def test_new_board_is_not_terminal():
    """A new board is not a win or a draw."""
    board = Board()
    assert not board.is_terminal()


def test_win_is_terminal():
    """A board with a winner is terminal, even if not full."""
    moves = [(0, R), (1, Y), (0, R), (1, Y), (0, R), (1, Y), (0, R)]
    board = play(moves)
    assert board.is_terminal()


def test_is_full_only_at_42_pieces():
    """41 pieces is not full; 42 is."""
    board = play_alternating(DRAW_SEQUENCE[:-1])
    assert board.move_count == ROWS * COLS - 1
    assert not board.is_full()

    # Move 42 is index 41, which is Y's turn - keep the alternation intact so
    # the position stays one a real game could reach.
    board.make_move(DRAW_SEQUENCE[-1], PLAYER_Y)
    assert board.is_full()


def test_full_board_without_winner_is_a_draw():
    """A full board with no four-in-a-row is terminal with no winner."""
    board = play_alternating(DRAW_SEQUENCE)
    assert board.is_full()
    assert board.winner() is None
    assert board.is_terminal()

