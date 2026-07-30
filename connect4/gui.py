"""A tkinter Connect 4 window - you against the bot.

    python -m connect4.gui
    python -m connect4.gui --opponent alphabeta --depth 6
    python -m connect4.gui --checkpoint checkpoints/big/best.pt --simulations 800

Hover over a column to see where the piece would land, click to drop it.

The board is drawn with Pillow rather than with canvas shapes, because Tk's canvas
has no anti-aliasing and its circles come out visibly jagged. Everything is drawn
at 4x and scaled down, which smooths the edges. The pieces are rendered once at
startup and pasted, so a redraw is 42 paste calls rather than 42 resampled circles.

The bot runs in a worker thread. tkinter is single threaded, so doing the search
on the main thread freezes the window for as long as it takes - which looks like a
crash. The worker never touches a widget; it hands the move back with after().
"""

import argparse
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from connect4.board import (
    Board, COLS, CONNECT, DIRECTIONS, EMPTY, PLAYER_R, PLAYER_Y, ROWS,
)
from connect4.mcts import other

CELL = 84
MARGIN = 18
RADIUS = CELL // 2 - 9
CORNER = 22

# Everything is drawn this many times larger and then scaled down. 4 is enough
# that the stepping disappears; higher just costs startup time.
SUPERSAMPLE = 4

# Soft palette on white. The board is a pale blue rather than the usual navy, and
# the pieces are muted so nothing on screen is fully saturated.
PAGE = "#ffffff"
BOARD = "#d6e0f2"
BOARD_EDGE = "#c3d1ea"
HOLE = "#f7f9fc"
HOLE_EDGE = "#e3e9f4"

RED = "#e8827c"
RED_EDGE = "#d96f68"
RED_GHOST = "#f7d5d2"

YELLOW = "#f0cc6e"
YELLOW_EDGE = "#e0b953"
YELLOW_GHOST = "#faedcc"

INK = "#3d4756"
MUTED = "#8b93a1"

WIN_BG = "#e4f3e8"
WIN_EDGE = "#9ccfae"
LOSS_BG = "#fbe9e7"
LOSS_EDGE = "#e8b3ae"
DRAW_BG = "#eef0f4"
DRAW_EDGE = "#c8cdd6"

BUTTON = "#eef1f7"
BUTTON_HOVER = "#e2e7f1"
BUTTON_ACTIVE = "#d7deec"
BUTTON_TEXT = "#4a5464"

COLOURS = {PLAYER_R: (RED, RED_EDGE), PLAYER_Y: (YELLOW, YELLOW_EDGE)}
GHOSTS = {PLAYER_R: RED_GHOST, PLAYER_Y: YELLOW_GHOST}
NAMES = {PLAYER_R: "Red", PLAYER_Y: "Yellow"}

FONT = ("Segoe UI", 11)
FONT_BUTTON = ("Segoe UI Semibold", 10)

DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")


def column_at(x: int) -> int | None:
    """Which column an x pixel is over, or None if outside the board."""
    col = (x - MARGIN) // CELL
    if 0 <= col < COLS:
        return int(col)
    return None


def landing_row(board: Board, col: int) -> int | None:
    """Row a piece dropped into `col` would come to rest on, or None if full."""
    for row in range(ROWS - 1, -1, -1):
        if board.grid[row][col] == EMPTY:
            return row
    return None


def winning_cells(board: Board) -> list[tuple[int, int]] | None:
    """The four cells that won, or None. Same walk as Board.winner(), but it
    keeps the squares so they can be marked on screen."""
    if board.last_move is None or board.last_player is None:
        return None

    row, col = board.last_move
    player = board.last_player

    for d_row, d_col in DIRECTIONS:
        line = [(row, col)]
        for step in (1, -1):
            r, c = row + d_row * step, col + d_col * step
            while 0 <= r < ROWS and 0 <= c < COLS and board.grid[r][c] == player:
                line.append((r, c))
                r += d_row * step
                c += d_col * step
        if len(line) >= CONNECT:
            # All of them, not just four - a five in a row should light up whole.
            return sorted(line)

    return None


def _smooth(draw_at_scale, size: tuple[int, int]) -> Image.Image:
    """Draw at SUPERSAMPLE times the size, then scale down to smooth the edges.

    Pillow's shapes aren't anti-aliased either, but LANCZOS resampling on the way
    down gives the same result.
    """
    big = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)
    image = Image.new("RGBA", big, (0, 0, 0, 0))
    draw_at_scale(ImageDraw.Draw(image), SUPERSAMPLE)
    return image.resize(size, Image.LANCZOS)


def make_disc(diameter: int, fill: str, outline: str, width: int = 2) -> Image.Image:
    """One anti-aliased circle, with transparent corners so it can be pasted."""
    def render(draw, scale):
        inset = scale // 2
        draw.ellipse(
            [inset, inset, diameter * scale - inset - 1, diameter * scale - inset - 1],
            fill=fill, outline=outline, width=width * scale,
        )

    return _smooth(render, (diameter, diameter))


def make_board(width: int, height: int) -> Image.Image:
    """The rounded board panel the holes sit in."""
    def render(draw, scale):
        draw.rounded_rectangle(
            [0, 0, width * scale - 1, height * scale - 1],
            radius=CORNER * scale, fill=BOARD, outline=BOARD_EDGE, width=2 * scale,
        )

    return _smooth(render, (width, height))


class SoftButton(tk.Label):
    """A flat button that lights up under the pointer.

    tk.Button on Windows draws a grey 3D border that clashes with everything
    here, so this is a Label with the click and hover behaviour added.
    """

    def __init__(self, parent, text, command, **kwargs):
        super().__init__(
            parent, text=text, font=FONT_BUTTON, fg=BUTTON_TEXT, bg=BUTTON,
            padx=16, pady=8, cursor="hand2", **kwargs
        )
        self.command = command
        self.enabled = True
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _enter(self, _event=None):
        if self.enabled:
            self.config(bg=BUTTON_HOVER)

    def _leave(self, _event=None):
        if self.enabled:
            self.config(bg=BUTTON)

    def _press(self, _event=None):
        if self.enabled:
            self.config(bg=BUTTON_ACTIVE)

    def _release(self, _event=None):
        if self.enabled:
            self.config(bg=BUTTON_HOVER)
            self.command()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.config(
            bg=BUTTON if enabled else PAGE,
            fg=BUTTON_TEXT if enabled else MUTED,
            cursor="hand2" if enabled else "",
        )


def ask_colour(root: tk.Tk) -> str | None:
    """Modal 'which colour?' box. Returns a player, or None if it was closed."""
    dialog = tk.Toplevel(root, bg=PAGE)
    dialog.title("New game")
    dialog.resizable(False, False)
    dialog.transient(root)

    choice: dict[str, str | None] = {"player": None}

    tk.Label(
        dialog, text="Play as", font=("Segoe UI", 13), fg=INK, bg=PAGE, pady=14
    ).pack()

    row = tk.Frame(dialog, bg=PAGE)
    row.pack(padx=24, pady=(0, 8))
    swatches: list[ImageTk.PhotoImage] = []

    def pick(player: str) -> None:
        choice["player"] = player
        dialog.destroy()

    for player, note in ((PLAYER_R, "moves first"), (PLAYER_Y, "moves second")):
        cell = tk.Frame(row, bg=PAGE)
        cell.pack(side="left", padx=10)

        fill, edge = COLOURS[player]
        disc = ImageTk.PhotoImage(make_disc(56, fill, edge))
        swatches.append(disc)  # same reference-keeping problem as the board

        swatch = tk.Label(cell, image=disc, bg=PAGE, cursor="hand2")
        swatch.pack()
        swatch.bind("<Button-1>", lambda _e, p=player: pick(p))

        SoftButton(cell, NAMES[player], lambda p=player: pick(p)).pack(pady=(8, 2))
        tk.Label(cell, text=note, font=("Segoe UI", 9), fg=MUTED, bg=PAGE).pack()

    tk.Frame(dialog, bg=PAGE, height=10).pack()

    # Centre on the parent window.
    dialog.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width() - dialog.winfo_width()) // 2
    y = root.winfo_rooty() + (root.winfo_height() - dialog.winfo_height()) // 3
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    dialog.grab_set()
    root.wait_window(dialog)
    return choice["player"]


def make_agent(opponent: str, checkpoint: Path, depth: int, simulations: int):
    """Return a (board, player) -> column function, and a description of it."""
    if opponent == "random":
        import numpy as np
        from connect4 import arena

        return arena.random_agent(np.random.default_rng()), "random"

    if opponent == "alphabeta":
        from connect4 import arena

        return arena.alphabeta_agent(depth), f"alpha-beta depth {depth}"

    if not Path(checkpoint).exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train one first, or run with "
            f"--opponent alphabeta."
        )

    from connect4.network import default_device, load
    from connect4.puct import caching_evaluator, network_evaluator, run_search, select_move

    net = load(checkpoint, default_device())
    evaluator = caching_evaluator(network_evaluator(net))

    def agent(board: Board, player: str) -> int:
        root = run_search(board, player, evaluator, simulations)
        return select_move(root, temperature=0.0)

    return agent, f"network ({simulations} sims)"


class Game:
    def __init__(self, root: tk.Tk, agent, description: str, human: str) -> None:
        self.root = root
        self.agent = agent
        self.description = description
        self.human = human

        self.board = Board()
        self.player = PLAYER_R
        self.hover: int | None = None
        self.thinking = False
        self.over = False
        self.history: list[int] = []

        width = COLS * CELL + 2 * MARGIN
        height = ROWS * CELL + 2 * MARGIN

        root.title("Connect 4")
        root.resizable(False, False)
        root.configure(bg=PAGE)

        # Rendered once. A redraw then pastes sprites instead of resampling
        # circles, which keeps hover repaints instant.
        diameter = RADIUS * 2
        self.board_image = make_board(COLS * CELL, ROWS * CELL)
        self.sprites = {
            "hole": make_disc(diameter, HOLE, HOLE_EDGE),
            PLAYER_R: make_disc(diameter, RED, RED_EDGE),
            PLAYER_Y: make_disc(diameter, YELLOW, YELLOW_EDGE),
            ("ghost", PLAYER_R): make_disc(diameter, RED_GHOST, RED_EDGE),
            ("ghost", PLAYER_Y): make_disc(diameter, YELLOW_GHOST, YELLOW_EDGE),
            # A heavy ring, so the four that won stand out from the rest.
            ("won", PLAYER_R): make_disc(diameter, RED, INK, width=4),
            ("won", PLAYER_Y): make_disc(diameter, YELLOW, INK, width=4),
        }
        self.photo: ImageTk.PhotoImage | None = None

        self.canvas = tk.Canvas(root, width=width, height=height,
                                highlightthickness=0, bg=PAGE)
        self.canvas.pack(padx=8, pady=(8, 0))
        self.image_id = self.canvas.create_image(MARGIN, MARGIN, anchor="nw")
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        # One message area under the board, in a fixed-height frame so swapping
        # between the small turn text and the big result card doesn't make the
        # window jump.
        message_area = tk.Frame(root, bg=PAGE, height=74, width=width)
        message_area.pack(fill="x")
        message_area.pack_propagate(False)

        self.message = tk.Label(message_area, text="", bg=PAGE, fg=INK)
        self.message.pack(expand=True)

        controls = tk.Frame(root, bg=PAGE)
        controls.pack(pady=(0, 14))
        self.new_button = SoftButton(controls, "New game", self.new_game)
        self.new_button.pack(side="left", padx=5)
        self.undo_button = SoftButton(controls, "Undo", self.undo)
        self.undo_button.pack(side="left", padx=5)

        self.draw()
        self.maybe_bot_move()

    # ---------------------------------------------------------------- drawing

    def draw(self) -> None:
        image = self.board_image.copy()
        ghost_cell = self.ghost_cell()
        offset = CELL // 2 - RADIUS
        won = set(winning_cells(self.board) or ()) if self.over else set()

        for row in range(ROWS):
            for col in range(COLS):
                cell = self.board.grid[row][col]
                if cell != EMPTY:
                    key = ("won", cell) if (row, col) in won else cell
                    sprite = self.sprites[key]
                elif (row, col) == ghost_cell:
                    sprite = self.sprites[("ghost", self.human)]
                else:
                    sprite = self.sprites["hole"]

                image.paste(
                    sprite,
                    (col * CELL + offset, row * CELL + offset),
                    sprite,  # its own alpha, so the corners stay transparent
                )

        # Keep the reference alive - tkinter does not, and the image would vanish.
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfig(self.image_id, image=self.photo)

        self.update_message()
        self.new_button.set_enabled(not self.thinking)
        self.undo_button.set_enabled(bool(self.history) and not self.thinking)

    def update_message(self) -> None:
        """The line under the board. Plain during play, a result card at the end."""
        if not self.over:
            self.message.config(
                text=self.status_text(), font=FONT, fg=INK, bg=PAGE,
                padx=0, pady=0, highlightthickness=0,
            )
            return

        winner = self.board.winner()
        if winner is None:
            text, background, edge = "It's a draw", DRAW_BG, DRAW_EDGE
        elif winner == self.human:
            text, background, edge = "You win!", WIN_BG, WIN_EDGE
        else:
            text, background, edge = "The bot wins", LOSS_BG, LOSS_EDGE

        self.message.config(
            text=text, font=("Segoe UI Semibold", 19), fg=INK, bg=background,
            padx=30, pady=12, highlightthickness=2,
            highlightbackground=edge, highlightcolor=edge,
        )

    def ghost_cell(self) -> tuple[int, int] | None:
        """Where the hovered piece would land, if it's the human's turn."""
        if self.hover is None or self.over or self.thinking:
            return None
        if self.player != self.human:
            return None
        row = landing_row(self.board, self.hover)
        return None if row is None else (row, self.hover)

    def status_text(self) -> str:
        if self.thinking:
            return f"Thinking... ({self.description})"
        return f"Your turn - you are {NAMES[self.human]}."

    # ----------------------------------------------------------------- events

    def on_motion(self, event: tk.Event) -> None:
        col = column_at(event.x)
        if col != self.hover:
            self.hover = col
            self.draw()

    def on_leave(self, _event: tk.Event) -> None:
        if self.hover is not None:
            self.hover = None
            self.draw()

    def on_click(self, event: tk.Event) -> None:
        if self.over or self.thinking or self.player != self.human:
            return
        col = column_at(event.x)
        if col is None or landing_row(self.board, col) is None:
            return
        self.play(col)
        self.maybe_bot_move()

    def new_game(self) -> None:
        if self.thinking:
            return
        chosen = ask_colour(self.root)
        if chosen is None:
            return

        self.human = chosen
        self.board = Board()
        self.player = PLAYER_R
        self.history.clear()
        self.hover = None
        self.over = False
        self.draw()
        self.maybe_bot_move()

    def undo(self) -> None:
        """Take back a full round, so it's the human's turn again."""
        if self.thinking or not self.history:
            return
        for _ in range(min(2, len(self.history))):
            self.board.undo_move(self.history.pop())
            self.player = other(self.player)
        self.over = False
        self.draw()

    # ------------------------------------------------------------------- play

    def play(self, col: int) -> None:
        self.board.make_move(col, self.player)
        self.history.append(col)
        self.player = other(self.player)
        self.over = self.board.is_terminal()
        self.draw()

    def maybe_bot_move(self) -> None:
        if self.over or self.player == self.human:
            return

        self.thinking = True
        self.hover = None
        self.draw()

        # A copy, because the worker must not touch the board the UI is drawing.
        board = self.board.copy()
        player = self.player

        def work() -> None:
            col = self.agent(board, player)
            # Back to the main thread; tkinter cannot be called from here.
            self.root.after(0, self.finish_bot_move, col)

        threading.Thread(target=work, daemon=True).start()

    def finish_bot_move(self, col: int) -> None:
        self.thinking = False
        self.play(col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect 4 against the bot.")
    parser.add_argument("--opponent", choices=("net", "alphabeta", "random"), default="net")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--second", action="store_true", help="let the bot move first")
    args = parser.parse_args()

    agent, description = make_agent(
        args.opponent, args.checkpoint, args.depth, args.simulations
    )

    root = tk.Tk()
    Game(root, agent, description, human=PLAYER_Y if args.second else PLAYER_R)
    root.mainloop()


if __name__ == "__main__":
    main()
