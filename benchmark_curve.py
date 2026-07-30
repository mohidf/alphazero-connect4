"""Win rate against fixed opponents, plotted over training iterations.

Each checkpoint plays two opponents that never change - MCTS with 1000 rollouts,
and minimax at depth 5 - twice over: once using just the policy head with no
search, and once with the full PUCT search.

The gap between those two lines is how much work the search is doing at play
time, as opposed to what the network knows by itself.

    python benchmark_curve.py --dir checkpoints/run1 --games 20
    python benchmark_curve.py --dir checkpoints/run1 --dir checkpoints/run2

Results go to curve.json as it goes, so partial runs can still be plotted.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from connect4 import arena
from connect4.network import default_device, load

OPPONENTS = {
    "MCTS (1000 rollouts)": lambda: arena.mcts_agent(1000),
    "MinMax (depth 5)": lambda: arena.alphabeta_agent(5),
}


def checkpoints(directories: list[Path], points: int) -> list[tuple[int, Path]]:
    """Pick checkpoints to test, spread over the whole history.

    Directories can be chained, since a resumed run starts numbering again at 1
    and would otherwise look like it learned everything from scratch.

    Only promoted iterations get saved, so they're unevenly spaced. Subsampling
    keeps this from taking all day.
    """
    found: list[tuple[int, Path]] = []
    offset = 0
    for directory in directories:
        local = []
        for path in directory.glob("iter*.pt"):
            match = re.fullmatch(r"iter(\d+)\.pt", path.name)
            if match:
                local.append((int(match.group(1)), path))
        if not local:
            raise SystemExit(f"no iter*.pt checkpoints in {directory}")
        local.sort()
        found.extend((iteration + offset, path) for iteration, path in local)
        offset += local[-1][0]

    if len(found) <= points:
        return found

    picks = np.linspace(0, len(found) - 1, points).round().astype(int)
    return [found[i] for i in sorted(set(picks))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, action="append", dest="dirs",
                        help="checkpoint directory; repeat to chain resumed runs in order")
    parser.add_argument("--games", type=int, default=20, help="games per data point")
    parser.add_argument("--points", type=int, default=16, help="checkpoints to sample")
    parser.add_argument("--simulations", type=int, default=400, help="PUCT sims for AlphaZero mode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dirs = args.dirs or [Path("checkpoints/overnight2")]
    device = default_device()
    selected = checkpoints(dirs, args.points)
    out_path = dirs[-1] / "curve.json"

    print(f"{len(selected)} checkpoints x {len(OPPONENTS)} opponents x 2 modes, "
          f"{args.games} games each")

    results = []
    started = time.perf_counter()

    for index, (iteration, path) in enumerate(selected, 1):
        net = load(path, device)
        row = {"iteration": iteration, "checkpoint": path.name}

        modes = {
            "Network Only": arena.policy_agent(net),
            "AlphaZero": arena.network_agent(net, args.simulations),
        }
        for mode_name, agent in modes.items():
            for opponent_name, make_opponent in OPPONENTS.items():
                # same seed everywhere, so every point faces the same openings
                result = arena.play_match(
                    agent,
                    make_opponent(),
                    games=args.games,
                    rng=np.random.default_rng(args.seed),
                )
                row[f"{mode_name} / {opponent_name}"] = {
                    "win_pct": 100.0 * result.wins / result.games,
                    "score": result.score,
                    "w": result.wins, "l": result.losses, "d": result.draws,
                }

        results.append(row)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        elapsed = time.perf_counter() - started
        remaining = elapsed / index * (len(selected) - index)
        summary = "  ".join(
            f"{k.split(' / ')[0][:3]}/{k.split('(')[1][:4]}:{v['win_pct']:5.1f}%"
            for k, v in row.items() if isinstance(v, dict)
        )
        print(f"iter {iteration:>4}  {summary}   [{elapsed/60:.0f}m done, "
              f"~{remaining/60:.0f}m left]", flush=True)

    plot(results, dirs[-1], args.simulations)


def plot(results: list[dict], directory: Path, simulations: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [r["iteration"] for r in results]

    for mode in ("Network Only", "AlphaZero"):
        figure, axes = plt.subplots(figsize=(6, 4))
        for opponent, colour in zip(OPPONENTS, ("tab:blue", "tab:orange")):
            key = f"{mode} / {opponent}"
            axes.plot(
                iterations,
                [r[key]["win_pct"] for r in results],
                color=colour,
                label=f"{mode} / {opponent}",
            )

        axes.set_title("Percentage of Won Games")
        axes.set_xlabel("Iteration number")
        axes.set_ylim(0, 100)
        axes.grid(alpha=0.3)
        axes.legend(loc="lower right", fontsize=8)

        name = mode.lower().replace(" ", "_")
        out = directory / f"curve_{name}.png"
        figure.tight_layout()
        figure.savefig(out, dpi=150)
        plt.close(figure)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
