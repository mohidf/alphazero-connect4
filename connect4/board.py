"""6x7 Connect-4 board mechanics: state, column-drop moves, and terminal detection.

Board layout
------------
2D grid, indexed ``grid[row][col]``.

- ``ROWS = 6``, ``COLS = 7``
- **Row 0 is the TOP, row 5 is the BOTTOM.** Gravity pulls pieces toward row 5,
  so a piece dropped into an empty column comes to rest at ``grid[5][col]``.
- An **action is a column index 0-6**, never a cell. The landing row is derived.

        col:  0   1   2   3   4   5   6
    row 0     .   .   .   .   .   .   .     <- top (last to fill)
    row 1     .   .   .   .   .   .   .
    row 2     .   .   .   .   .   .   .
    row 3     .   .   .   .   .   .   .
    row 4     .   .   .   .   .   .   .
    row 5     .   .   .   .   .   .   .     <- bottom (pieces land here first)
"""

EMPTY = " "
PLAYER_R = "R"
PLAYER_Y = "Y"

ROWS = 6
COLS = 7
CONNECT = 4

# The four directions a win can run in, as (row_delta, col_delta).
# Only four (not eight) because each line is checked in *both* directions from
# the last-placed piece, so e.g. (0, +1) also covers (0, -1).
DIRECTIONS = [
    (0, 1),    # horizontal  -
    (1, 0),    # vertical    |
    (1, 1),    # diagonal    \
    (1, -1),   # diagonal    /
]


class Board:
    def __init__(self, grid: list[list[str]] | None = None) -> None:
        """Create a board, defaulting to an empty 6x7 grid.

        Careful with the default: ``[[EMPTY] * COLS] * ROWS`` creates six
        references to the *same* row list, so writing one cell writes six.
        """
        self.grid = grid if grid is not None else [[EMPTY] * COLS for _ in range(ROWS)]

        # Win detection only ever needs to look at lines through the piece that
        # was just placed, so `make_move` should record it here for `winner()`.
        self.last_move: tuple[int, int] | None = None   # (row, col)
        self.last_player: str | None = None

        # Derived from the grid, not assumed to be 0, so that boards constructed
        # from a literal grid (i.e. every test position) report is_full correctly.
        self.move_count = sum(
            1 for row in self.grid for cell in row if cell != EMPTY
        )

    def available_moves(self) -> list[int]:
        """Return the column indices that still have room, e.g. ``[0, 1, 3, 6]``.

        A column is playable iff its *top* cell is empty.
        """
        return [col for col in range(COLS) if self.grid[0][col] == EMPTY]

    def make_move(self, col: int, player: str) -> int:
        """Drop `player`'s piece into `col` and return the row it landed on.

        Raises ValueError if the column is full or out of range.
        Should also update `last_move`, `last_player`, and `move_count`.

        (Returning the landed row is one of the two options we discussed — the
        other is returning None and having `undo_move` re-derive it. If you
        prefer that, change this signature to `-> None`.)
        """
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of bounds (0-{COLS - 1})")

        # Scanning bottom-up, the first empty cell is where gravity drops the
        # piece. Finding none means the column is full — one loop, both jobs.
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == EMPTY:
                self.grid[row][col] = player
                self.last_move = (row, col)
                self.last_player = player
                self.move_count += 1
                return row

        raise ValueError(f"Column {col} is full")

    def undo_move(self, col: int) -> None:
        """Remove the topmost piece from `col`, restoring the previous state.

        Used heavily by search: make_move -> recurse -> undo_move must leave the
        board byte-for-byte identical to before.
        """
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of bounds (0-{COLS - 1})")

        # Mirror image of make_move: scanning top-down, the first non-empty cell
        # is the most recently dropped piece. Finding none means empty column.
        for row in range(ROWS):
            if self.grid[row][col] != EMPTY:
                self.grid[row][col] = EMPTY
                # Can't restore the *previous* last_move — the board doesn't
                # remember it. Clearing is the honest option: a stale read then
                # fails loudly instead of reporting a win for a removed piece.
                self.last_move = None
                self.last_player = None
                self.move_count -= 1
                return 

        raise ValueError(f"Column {col} is empty")

    def winner(self) -> str | None:
        """Return PLAYER_R, PLAYER_Y, or None if there's no winner yet."""
        if self.last_move is None or self.last_player is None:
            return None

        row, col = self.last_move
        player = self.last_player

        # Check for a win in all four directions.
        for d_row, d_col in DIRECTIONS:
            count = 1  # Count the last-placed piece itself.
            # Check in the positive direction.
            r, c = row + d_row, col + d_col
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r += d_row
                c += d_col
            # Check in the negative direction.
            r, c = row - d_row, col - d_col
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r -= d_row
                c -= d_col
            if count >= CONNECT:
                return player

        return None

    def is_full(self) -> bool:
        """Return True if the board is full (no empty cells), else False."""
        return self.move_count == ROWS * COLS

    def is_terminal(self) -> bool:
        """Return True if the game is over (win or draw), else False."""
        return self.winner() is not None or self.is_full()

    def __str__(self) -> str:
        """Render the board for debugging, bottom row last, with column numbers."""
        lines = []
        for row in self.grid:
            lines.append("|" + "|".join(row) + "|")
        lines.append(" " + " ".join(str(col) for col in range(COLS)))
        return "\n".join(lines)     

