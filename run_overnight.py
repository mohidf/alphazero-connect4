"""Overnight training run. Launch and leave it.

    python run_overnight.py                  # ~8 hours
    python run_overnight.py --hours 4
    python run_overnight.py --hours 8 --tag second-attempt

Everything is logged to `checkpoints/<tag>/train.log` as well as stdout, so the run
survives losing the terminal. Checkpoints land in the same directory: one per
promotion, plus `best.pt` written when the run finishes or the time budget expires.

Settings are calibrated to this machine (RTX 4050, ~300k-parameter network):

    self-play        2.65 s/game at 400 simulations, 64 games in flight
    arena vs prev    ~50 s for 20 games at 100 simulations, cached
    arena vs ab-d4   ~27 s for 20 games

which works out to roughly 3.5 minutes per iteration, so about 140 iterations in
8 hours and ~380k training positions. The previous 10-iteration run produced 16k.
"""

import argparse
import sys
from pathlib import Path

from connect4.pipeline import Config, run


class Tee:
    """Write to several streams at once, so the log and the console agree."""

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
    args = parser.parse_args()

    checkpoint_dir = Path("checkpoints") / args.tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        # A ceiling far above what the time budget allows: max_hours is what
        # actually ends the run, and it ends it cleanly with best.pt written.
        iterations=10_000,
        max_hours=args.hours,

        games_per_iteration=64,
        simulations=400,          # 4x the short run; this is what search quality buys
        parallel_games=64,        # every game in flight, so batches stay full

        train_steps=250,
        batch_size=256,
        buffer_size=150_000,      # ~55 iterations of history

        learning_rate=1e-3,
        lr_decay_every=40,
        lr_decay_gamma=0.5,
        lr_min=5e-5,

        eval_games=20,
        eval_simulations=100,
        opening_plies=4,
        benchmark_depth=4,
        benchmark_every=5,        # the expensive one; every 5th iteration is plenty

        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
    )

    log_path = checkpoint_dir / "train.log"
    with open(log_path, "a", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            print(f"=== run '{args.tag}': {args.hours}h budget, seed {args.seed} ===")
            print(f"    logging to {log_path}")
            reports = run(config)
        finally:
            sys.stdout = original

    if not reports:
        print("no iterations completed")
        return

    promoted = sum(r.promoted for r in reports)
    benchmarks = [r for r in reports if r.vs_benchmark is not None]

    print()
    print(f"finished {len(reports)} iterations in {reports[-1].elapsed / 3600:.1f}h")
    print(f"promotions: {promoted}/{len(reports)}")
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
