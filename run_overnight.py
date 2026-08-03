"""Long training run. Start it and leave it.

    python run_overnight.py --hours 8
    python run_overnight.py --hours 10 --tag big --channels 128 --blocks 8
    python run_overnight.py --hours 6 --tag more --resume checkpoints/x/best.pt

Output goes to checkpoints/<tag>/train.log as well as the terminal, so it doesn't
matter if the terminal goes away. Checkpoints land in the same folder.

Defaults are sized for this machine: about 2.5 minutes an iteration, so roughly
170 iterations in 8 hours.
"""

import argparse
import sys
from pathlib import Path

from connect4.pipeline import Config, run


class Tee:
    """Write to the terminal and the log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Long unattended training run.")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--tag", default="overnight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=64, help="trunk width (fresh runs only)")
    parser.add_argument("--blocks", type=int, default=4, help="residual blocks (fresh runs only)")
    parser.add_argument("--simulations", type=int, default=400, help="PUCT sims per self-play move")
    parser.add_argument("--games", type=int, default=64, help="self-play games per iteration")
    parser.add_argument("--parallel", type=int, default=64, help="games in flight (batch size)")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="start from an existing checkpoint instead of a fresh network",
    )
    args = parser.parse_args()

    checkpoint_dir = Path("checkpoints") / args.tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        # Huge on purpose; max_hours is what actually stops it.
        iterations=10_000,
        max_hours=args.hours,

        games_per_iteration=args.games,
        simulations=args.simulations,
        parallel_games=args.parallel,
        channels=args.channels,
        blocks=args.blocks,

        train_steps=250,
        batch_size=256,
        buffer_size=150_000,

        learning_rate=1e-3,
        lr_decay_every=40,
        lr_decay_gamma=0.5,
        lr_min=5e-5,

        eval_games=20,
        eval_simulations=100,
        opening_plies=4,
        benchmark_depth=4,
        benchmark_every=5,        # it's slow, and every 5th is enough

        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
        seed=args.seed,
    )

    log_path = checkpoint_dir / "train.log"
    with open(log_path, "a", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            print(f"=== run '{args.tag}': {args.hours}h budget, seed {args.seed} ===")
            if args.resume:
                print(f"    resuming from {args.resume}")
            print(f"    logging to {log_path}")
            reports = run(config)
        finally:
            sys.stdout = original

    if not reports:
        print("no iterations completed")
        return

    scores = [r.vs_incumbent.score for r in reports if r.vs_incumbent is not None]
    benchmarks = [r for r in reports if r.vs_benchmark is not None]

    print()
    print(f"finished {len(reports)} iterations in {reports[-1].elapsed / 3600:.1f}h")
    if scores:
        # One 20-game match is mostly noise, but the mean over a whole run is not:
        # the standard error shrinks by sqrt(len(scores)). Above 0.5 means the
        # average training step made the network better.
        mean = sum(scores) / len(scores)
        better = sum(s > 0.5 for s in scores)
        print(f"mean score vs previous: {mean:.3f}  ({better}/{len(scores)} above 0.5)")
    print(f"policy loss: {reports[0].loss_policy:.4f} -> {reports[-1].loss_policy:.4f}")
    if benchmarks:
        print("vs alpha-beta depth", config.benchmark_depth)
        for r in benchmarks[:: max(1, len(benchmarks) // 10)]:
            print(f"  iter {r.iteration:>4}: {r.vs_benchmark}")
        print(f"  final    : {benchmarks[-1].vs_benchmark}")
    print(f"\nbest network: {checkpoint_dir / 'best.pt'}")
    print(f"play it: python -m connect4.play --checkpoint {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
