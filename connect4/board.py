"""Connect 4 board: 6 rows, 7 columns.

grid[row][col], where row 0 is the TOP and row 5 is the bottom. Pieces fall
toward row 5. Moves are column indices; the row is worked out from gravity.
"""

EMPTY = " "
PLAYER_R = "R"
PLAYER_Y = "Y"

ROWS = 6
COLS = 7
CONNECT = 4

# (row_delta, col_delta). Four, not eight, because each line is walked both ways.
DIRECTIONS = [
    (0, 1),    # -
    (1, 0),    # |
    (1, 1),    # \
    (1, -1),   # /
]


class Board:
    def __init__(self, grid: list[list[str]] | None = None) -> None:
        # Comprehension, not [[EMPTY] * COLS] * ROWS, which shares one row list.
        self.grid = grid if grid is not None else [[EMPTY] * COLS for _ in range(ROWS)]

        # winner() only looks at lines through the last piece played.
        self.last_move: tuple[int, int] | None = None   # (row, col)
        self.last_player: str | None = None

        # Counted rather than set to 0, so a board built from a grid is correct.
        self.move_count = sum(
            1 for row in self.grid for cell in row if cell != EMPTY
        )

    def copy(self) -> "Board":
        """Independent copy, including move history.

        The rows have to be copied too, and last_move has to come along or the
        copy reports no winner on a won board.
        """
        board = Board([row[:] for row in self.grid])
        board.last_move = self.last_move
        board.last_player = self.last_player
        board.move_count = self.move_count
        return board

    def available_moves(self) -> list[int]:
        """Columns that still have room."""
        return [col for col in range(COLS) if self.grid[0][col] == EMPTY]

    def make_move(self, col: int, player: str) -> int:
        """Drop a piece into `col`, return the row it landed on."""
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of bounds (0-{COLS - 1})")

        # Bottom-up: first empty cell is where it lands. None means full.
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == EMPTY:
                self.grid[row][col] = player
                self.last_move = (row, col)
                self.last_player = player
                self.move_count += 1
                return row

        raise ValueError(f"Column {col} is full")

    def undo_move(self, col: int) -> None:
        """Take the top piece out of `col`. Must exactly undo make_move."""
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of bounds (0-{COLS - 1})")

        # Top-down: first occupied cell is the most recent piece.
        for row in range(ROWS):
            if self.grid[row][col] != EMPTY:
                self.grid[row][col] = EMPTY
                # The board doesn't remember the move before this one, so clear
                # rather than restore.
                self.last_move = None
                self.last_player = None
                self.move_count -= 1
                return

        raise ValueError(f"Column {col} is empty")

    def winner(self) -> str | None:
        """Winning player, or None."""
        if self.last_move is None or self.last_player is None:
            return None

        row, col = self.last_move
        player = self.last_player

        for d_row, d_col in DIRECTIONS:
            count = 1  # the piece itself
            r, c = row + d_row, col + d_col
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r += d_row
                c += d_col
            r, c = row - d_row, col - d_col
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r -= d_row
                c -= d_col
            if count >= CONNECT:
                return player

        return None

    def is_full(self) -> bool:
        return self.move_count == ROWS * COLS

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()

    def __str__(self) -> str:
        lines = ["|" + "|".join(row) + "|" for row in self.grid]
        lines.append(" " + " ".join(str(col) for col in range(COLS)))
        return "\n".join(lines)
