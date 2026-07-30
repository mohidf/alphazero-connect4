"""Hand-written scoring function for a position. Positive is good for R."""

from connect4.board import Board, EMPTY, PLAYER_R, PLAYER_Y, ROWS, COLS, CONNECT

# Score for a group of 4 cells holding this many of one colour and none of the
# other. Steep, because a three is much closer to winning than two twos.
WINDOW_SCORE = [0, 1, 10, 50]

# Middle columns appear in more groups, so pieces there are worth more.
COLUMN_BONUS = [0, 1, 2, 3, 2, 1, 0]

# Upper bound on |evaluate()|. WIN_SCORE in minimax.py has to stay well above it.
MAX_EVAL = 10_000


def _build_windows() -> list[tuple[tuple[int, int], ...]]:
    """Every group of 4 cells in a line. There are 69 on a 6x7 board.

    24 horizontal + 21 vertical + 12 + 12 diagonal. If the count is off, one of
    the ranges below is wrong.
    """
    windows: list[tuple[tuple[int, int], ...]] = []

    for row in range(ROWS):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple((row, col + i) for i in range(CONNECT)))

    for col in range(COLS):
        for row in range(ROWS - CONNECT + 1):
            windows.append(tuple((row + i, col) for i in range(CONNECT)))

    for row in range(ROWS - CONNECT + 1):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple((row + i, col + i) for i in range(CONNECT)))

    for row in range(ROWS - CONNECT + 1):
        for col in range(CONNECT - 1, COLS):
            windows.append(tuple((row + i, col - i) for i in range(CONNECT)))

    return windows


# Built once; evaluate() is called far too often to rebuild it each time.
WINDOWS: list[tuple[tuple[int, int], ...]] = _build_windows()


def score_window(cells: list[str]) -> int:
    """Score one group of 4 cells. Mixed colours are dead and score 0.

    Raises on four of a colour: that's a finished game, which minimax scores
    separately.
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
    return 0


def evaluate(board: Board) -> int:
    """Score a position that isn't finished. Positive favours R."""
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
