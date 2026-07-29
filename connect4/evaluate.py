"""Heuristic evaluation of a non-terminal Connect-4 position.

This is the function the neural network replaces in Phase 4. Same slot, same
contract: take a position, return a number saying how good it looks for R.

Sign convention matches minimax: **positive favours R, negative favours Y.**
"""

from connect4.board import Board, EMPTY, PLAYER_R, PLAYER_Y, ROWS, COLS, CONNECT

# Window scores by how many friendly pieces it contains (and no enemy pieces).
# Index = piece count, so WINDOW_SCORE[3] is a window one move from winning.
# Steeply increasing: a live three is worth far more than two live twos.
WINDOW_SCORE = [0, 1, 10, 50]

# Pieces near the middle sit in more windows, so they're structurally worth more.
# Index = column.
COLUMN_BONUS = [0, 1, 2, 3, 2, 1, 0]

# Loose upper bound on |evaluate()|, used to sanity-check that WIN_SCORE in
# minimax.py dominates every heuristic value. See test_eval_is_bounded.
MAX_EVAL = 10_000


def _build_windows() -> list[tuple[tuple[int, int], ...]]:
    """Enumerate every group of CONNECT cells in a line, as (row, col) tuples.

    Four orientations — horizontal, vertical, and both diagonals. For each,
    every starting cell where the full run of 4 still fits on the board.

    Sanity check: a 6x7 board has exactly **69** of these:

        horizontal   6 rows x 4 starts = 24
        vertical     7 cols x 3 starts = 21
        diagonal \\   3 rows x 4 starts = 12
        diagonal /   3 rows x 4 starts = 12

    If you get a different number, an off-by-one in one of the ranges is the
    usual cause.
    """
    windows: list[tuple[tuple[int, int], ...]] = []

    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple((row, col + i) for i in range(CONNECT)))

    # Vertical
    for col in range(COLS):
        for row in range(ROWS - CONNECT + 1):
            windows.append(tuple((row + i, col) for i in range(CONNECT)))

    # Diagonal \
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple((row + i, col + i) for i in range(CONNECT)))

    # Diagonal /
    for row in range(ROWS - CONNECT + 1):
        for col in range(CONNECT - 1, COLS):
            windows.append(tuple((row + i, col - i) for i in range(CONNECT)))

    return windows


# Built once at import. Rebuilding this per evaluate() call would dominate
# runtime at a few hundred thousand calls per search.
WINDOWS: list[tuple[tuple[int, int], ...]] = _build_windows()


def score_window(cells: list[str]) -> int:
    """Score a single window of CONNECT cells from R's perspective.

    - only R pieces  -> +WINDOW_SCORE[count]
    - only Y pieces  -> -WINDOW_SCORE[count]
    - both colours   -> 0, the window is dead and nobody can ever complete it
    - all empty      -> 0

    Raises ValueError on a window that is already four of one colour. That's a
    terminal position, which score() handles — reaching here means the caller
    evaluated a finished game, and returning a merely-large number for it would
    quietly undermine WIN_SCORE's dominance.
    """
    r_count = cells.count(PLAYER_R)
    y_count = cells.count(PLAYER_Y)

    if r_count == CONNECT or y_count == CONNECT:
        raise ValueError(
            "score_window called on a completed line; evaluate() is only valid "
            "on non-terminal positions"
        )

    if y_count == 0:
        return WINDOW_SCORE[r_count]
    if r_count == 0:
        return -WINDOW_SCORE[y_count]
    return 0  # both colours present — nobody can ever complete this window


def evaluate(board: Board) -> int:
    """Estimate the value of a non-terminal position. Positive favours R.

    Sum score_window over every window in WINDOWS, then add the column bonus
    for each placed piece (positive for R, negative for Y).

    Must never return a value near WIN_SCORE — see MAX_EVAL above.
    """
    total = 0

    for window in WINDOWS:
        total += score_window([board.grid[row][col] for row, col in window])

    for row in board.grid:
        for col, cell in enumerate(row):
            if cell == PLAYER_R:
                total += COLUMN_BONUS[col]
            elif cell == PLAYER_Y:
                total -= COLUMN_BONUS[col]

    return total
