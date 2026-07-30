"""Play against any of the agents from the terminal.

    python -m connect4.play                      # the best checkpoint
    python -m connect4.play --opponent alphabeta --depth 6
    python -m connect4.play --opponent random
    python -m connect4.play --second             # let it move first

Against the network it prints what the search was thinking - how the visits were
spread across the columns and what it thinks of its position. Evenly spread visits
means it has no real opinion.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # So this works when run as a plain file, not just with -m.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from connect4.board import Board, COLS, PLAYER_R, PLAYER_Y
from connect4.mcts import other
from connect4 import alphabeta as ab
from connect4.network import default_device, load
from connect4.puct import (
    C_PUCT,
    caching_evaluator,
    network_evaluator,
    run_search,
    select_move,
    visit_counts,
)

DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")


def render(board: Board, highlight: int | None = None) -> str:
    """Board, with a marker under the last move."""
    lines = [str(board)]
    if highlight is not None:
        lines.append(" " + "  " * highlight + "^")
    return "\n".join(lines)


def ask_for_column(board: Board) -> int | None:
    """Ask for a column. None means quit."""
    legal = board.available_moves()
    while True:
        raw = input(f"Your move {legal} (or 'q' to quit): ").strip().lower()

        if raw in ("q", "quit", "exit"):
            return None
        if not raw.isdigit():
            print("  Enter a column number.")
            continue

        col = int(raw)
        if col not in legal:
            print(f"  Column {col} is not playable. Legal: {legal}")
            continue
        return col


def describe_search(root, value: float) -> str:
    """One line showing where the search spent its time."""
    counts = visit_counts(root)
    total = counts.sum()
    shares = " ".join(
        f"{col}:{counts[col] / total:4.0%}" if counts[col] else f"{col}:   -"
        for col in range(COLS)
    )
    # positive means it likes its position
    return f"  search: {shares}   value {value:+.2f}"


def network_move(net, board: Board, player: str, simulations: int, verbose: bool) -> int:
    evaluator = caching_evaluator(network_evaluator(net))
    root = run_search(board, player, evaluator, simulations)
    if verbose:
        print(describe_search(root, root.q))
    return select_move(root, temperature=0.0)


def play(
    opponent: str = "net",
    checkpoint: Path = DEFAULT_CHECKPOINT,
    depth: int = 6,
    simulations: int = 400,
    human_first: bool = True,
    verbose: bool = True,
    seed: int | None = None,
) -> str | None:
    """Play a game. Returns the winner, or None for a draw."""
    net = None
    if opponent == "net":
        if not Path(checkpoint).exists():
            raise SystemExit(
                f"No checkpoint at {checkpoint}. Train one first "
                f"(python -m connect4.pipeline), or use --opponent alphabeta."
            )
        net = load(checkpoint, default_device())
        print(f"Opponent: network from {checkpoint} at {simulations} simulations")
    elif opponent == "alphabeta":
        print(f"Opponent: alpha-beta at depth {depth}")
    elif opponent == "random":
        print("Opponent: random")
    else:
        raise SystemExit(f"Unknown opponent {opponent!r}")

    rng = np.random.default_rng(seed)
    human = PLAYER_R if human_first else PLAYER_Y
    board = Board()
    player = PLAYER_R
    last: int | None = None

    print(f"You are {human}. {'You' if human_first else 'The agent'} move first.\n")

    while not board.is_terminal():
        print(render(board, last))
        print()

        if player == human:
            col = ask_for_column(board)
            if col is None:
                print("Resigned.")
                return None
        elif opponent == "net":
            col = network_move(net, board, player, simulations, verbose)
            print(f"  agent plays {col}")
        elif opponent == "alphabeta":
            col, score = ab.best_move(board, depth, player == PLAYER_R)
            if verbose:
                # flip so positive always means the agent is happy
                own = score if player == PLAYER_R else -score
                print(f"  agent plays {col}   score {own:+d}")
            else:
                print(f"  agent plays {col}")
        else:
            col = int(rng.choice(board.available_moves()))
            print(f"  agent plays {col}")

        board.make_move(col, player)
        last = col
        player = other(player)
        print()

    print(render(board, last))
    winner = board.winner()
    print()
    if winner is None:
        print("Draw.")
    elif winner == human:
        print("You win.")
    else:
        print("The agent wins.")
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect-4 against an agent.")
    parser.add_argument(
        "--opponent", choices=("net", "alphabeta", "random"), default="net"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--depth", type=int, default=6, help="alpha-beta search depth")
    parser.add_argument(
        "--simulations", type=int, default=400, help="PUCT simulations per move"
    )
    parser.add_argument(
        "--second", action="store_true", help="let the agent move first"
    )
    parser.add_argument("--quiet", action="store_true", help="hide search output")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    play(
        opponent=args.opponent,
        checkpoint=args.checkpoint,
        depth=args.depth,
        simulations=args.simulations,
        human_first=not args.second,
        verbose=not args.quiet,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
