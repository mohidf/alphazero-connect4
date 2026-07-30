"""Error rate against a perfect solver, on the standard Connect 4 test sets.

For each test position the solver is asked to score every legal move, which gives
the set of moves that keep the game-theoretic result. The agent's move is an error
if it isn't in that set - so this measures mistakes, not wins, and a single blunder
shows up even in a game the agent goes on to win.

    python solver_benchmark.py --truth                 # build the cache (once)
    python solver_benchmark.py --dir checkpoints/run1  # run it and plot

Ground truth is cached in external/testsets/ground_truth.json, since it depends
only on the test sets and not on any network.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from connect4 import arena, solver
from connect4.board import Board, COLS, PLAYER_R, PLAYER_Y
from connect4.mcts import other
from connect4.network import default_device, load, predict_batch
from connect4.encoding import legal_move_mask, mask_and_normalise

TRUTH_PATH = solver.TEST_DIR / "ground_truth.json"


def board_from_moves(moves: str) -> tuple[Board, str]:
    """Turn a solver move string (columns 1-7) into a board and whose turn it is."""
    board = Board()
    player = PLAYER_R
    for ch in moves:
        board.make_move(int(ch) - 1, player)
        player = other(player)
    return board, player


def build_ground_truth() -> dict[str, dict[str, list[int]]]:
    """For every test position, the set of moves that preserve the result.

    Solving is cheap with the opening book - about a millisecond a position - so
    all six sets take under a minute.
    """
    truth: dict[str, dict[str, list[int]]] = {}

    for name, label in solver.TEST_SETS.items():
        data = solver.load_test_set(name)

        children, index = [], []
        for moves, score in data:
            for col in solver.legal_columns(moves):
                children.append(moves + str(col))
                index.append((moves, col))

        started = time.perf_counter()
        scores = solver.solve_many(children)
        elapsed = time.perf_counter() - started

        # A move is optimal when the position it leads to, scored for the
        # opponent and negated, matches the position's own score.
        best: dict[str, int] = defaultdict(lambda: -1000)
        child_scores: dict[str, dict[int, int]] = defaultdict(dict)
        for (moves, col), child_score in zip(index, scores):
            value = -child_score
            child_scores[moves][col] = value
            best[moves] = max(best[moves], value)

        optimal = {}
        mismatched = 0
        for moves, score in data:
            if best[moves] != score:
                mismatched += 1
            optimal[moves] = sorted(
                col - 1 for col, v in child_scores[moves].items() if v == best[moves]
            )

        truth[name] = optimal
        print(f"  {label:20} {len(data):>4} positions, {len(children):>5} solves, "
              f"{elapsed:>5.1f}s"
              + (f"   MISMATCHED {mismatched}" if mismatched else ""), flush=True)

    return truth


def load_ground_truth() -> dict[str, dict[str, list[int]]]:
    if not TRUTH_PATH.exists():
        raise SystemExit("no ground truth cache. Run with --truth first.")
    return json.loads(TRUTH_PATH.read_text())


def policy_errors(net, positions: list[str], optimal: dict[str, list[int]],
                  batch: int = 512) -> float:
    """Error rate for the raw policy head, evaluated in batches."""
    wrong = 0
    for start in range(0, len(positions), batch):
        chunk = positions[start:start + batch]
        boards, players = zip(*(board_from_moves(m) for m in chunk))
        priors, _ = predict_batch(net, list(boards), list(players))

        for moves, board, prior in zip(chunk, boards, priors):
            masked = mask_and_normalise(prior, legal_move_mask(board))
            if int(masked.argmax()) not in optimal[moves]:
                wrong += 1

    return 100.0 * wrong / len(positions)


def agent_errors(agent, positions: list[str], optimal: dict[str, list[int]]) -> float:
    """Error rate for any agent that picks a column from a board."""
    wrong = 0
    for moves in positions:
        board, player = board_from_moves(moves)
        if agent(board, player) not in optimal[moves]:
            wrong += 1
    return 100.0 * wrong / len(positions)


def checkpoints(directories: list[Path], points: int) -> list[tuple[int, Path]]:
    """Evenly spread checkpoints, with resumed runs offset so the x-axis is real."""
    import re

    found: list[tuple[int, Path]] = []
    offset = 0
    for directory in directories:
        local = []
        for path in directory.glob("iter*.pt"):
            match = re.fullmatch(r"iter(\d+)\.pt", path.name)
            if match:
                local.append((int(match.group(1)), path))
        if not local:
            raise SystemExit(f"no checkpoints in {directory}")
        local.sort()
        found.extend((i + offset, p) for i, p in local)
        offset += local[-1][0]

    if len(found) <= points:
        return found
    picks = np.linspace(0, len(found) - 1, points).round().astype(int)
    return [found[i] for i in sorted(set(picks))]


def plot(results: list[dict], reference: dict[str, float], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(solver.TEST_SETS)
    figure, axes = plt.subplots(3, 2, figsize=(6, 4), sharex=True)
    order = [0, 3, 1, 4, 5, 2]  # roughly easiest to hardest, down the columns

    iterations = [r["iteration"] for r in results]
    for slot, index in enumerate(order):
        name = names[index]
        ax = axes[slot % 3][slot // 3]
        ax.plot(iterations, [r[name] for r in results], color="tab:blue", linewidth=1)
        ax.axhline(reference[name], color="tab:orange", linewidth=1)
        ax.set_title(solver.TEST_SETS[name], fontsize=9)
        ax.set_ylabel("Error Rate (in %)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", action="store_true", help="build the cache and exit")
    parser.add_argument("--dir", type=Path, action="append", dest="dirs")
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="positions per set (0 = all)")
    parser.add_argument("--reference-depth", type=int, default=5)
    args = parser.parse_args()

    if args.truth:
        print("building ground truth")
        truth = build_ground_truth()
        TRUTH_PATH.write_text(json.dumps(truth))
        print("wrote", TRUTH_PATH)
        return

    truth = load_ground_truth()
    sets = {
        name: (list(optimal)[: args.limit] if args.limit else list(optimal))
        for name, optimal in truth.items()
    }

    print(f"reference: alpha-beta depth {args.reference_depth}")
    reference = {}
    for name, positions in sets.items():
        rate = agent_errors(arena.alphabeta_agent(args.reference_depth), positions, truth[name])
        reference[name] = rate
        print(f"  {solver.TEST_SETS[name]:20} {rate:5.2f}%", flush=True)

    dirs = args.dirs or [Path("checkpoints/overnight2")]
    device = default_device()
    selected = checkpoints(dirs, args.points)
    print(f"\n{len(selected)} checkpoints")

    results = []
    for iteration, path in selected:
        net = load(path, device)
        row = {"iteration": iteration}
        for name, positions in sets.items():
            row[name] = policy_errors(net, positions, truth[name])
        results.append(row)
        summary = "  ".join(f"{row[n]:5.1f}" for n in sets)
        print(f"iter {iteration:>4}  {summary}", flush=True)

    out_dir = dirs[-1]
    (out_dir / "solver_benchmark.json").write_text(
        json.dumps({"reference": reference, "results": results}, indent=2)
    )
    plot(results, reference, out_dir / "solver_error_rates.png")


if __name__ == "__main__":
    main()
