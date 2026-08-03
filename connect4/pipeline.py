"""The main loop: play games, train on them, check it improved, repeat.

    each iteration:
        self-play with the current network
        add those positions to the buffer
        train on a sample of the buffer
        play the new network against the old one, for the record
        keep the new one either way

There used to be a promotion gate here: the new network replaced the old one only
if it scored above 0.55 in a 20-game match. The idea was that training can make
things worse, and a worse network generating the next batch of games would slide
the whole thing backwards.

It was thrown out because 20 games can't tell those cases apart. The standard
error on a 20-game score is about 0.11, so a 0.55 bar sits under half an SE above
even - an equally strong network cleared it about a third of the time, and a
genuinely better one failed it about a third of the time. In the 'big' run only 25
of 78 iterations promoted, and a rejection costs more than the training step: the
next iteration's self-play comes from the old network again, so the data stops
improving too. AlphaGo Zero gated this way; AlphaZero dropped it and always took
the latest network, which is what this does now.

The match against the previous network is still played and still logged. It just
doesn't decide anything - it's there to show whether training steps are helping.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from connect4 import arena
from connect4.network import (
    BLOCKS,
    CHANNELS,
    Connect4Net,
    count_parameters,
    default_device,
    load,
    save,
)
from connect4.selfplay import generate_games
from connect4.train import (
    BATCH_SIZE,
    BUFFER_SIZE,
    LEARNING_RATE,
    ReplayBuffer,
    make_optimizer,
    train_epoch,
)

@dataclass
class Config:
    iterations: int = 10
    games_per_iteration: int = 24
    simulations: int = 120
    # Only used for a fresh run; a resumed one gets its shape from the file.
    channels: int = CHANNELS
    blocks: int = BLOCKS
    train_steps: int = 100
    batch_size: int = BATCH_SIZE
    # Wall-clock limit. For an unattended run this is what actually ends it, and
    # it ends cleanly with best.pt written.
    max_hours: float | None = None
    buffer_size: int = BUFFER_SIZE
    # Step decay. A fixed rate flattens out after a hundred-odd iterations.
    learning_rate: float = LEARNING_RATE
    lr_decay_every: int = 40
    lr_decay_gamma: float = 0.5
    lr_min: float = 5e-5
    parallel_games: int = 24
    eval_games: int = 20
    # Arena searches can't be batched, so a smaller budget here stops evaluation
    # eating the whole loop. Both sides get the same number, so it stays fair.
    eval_simulations: int = 60
    # Without this a match is really just two games repeated. See arena.py.
    opening_plies: int = 4
    # Every iteration is kept now, so numbered checkpoints are written on a
    # cadence instead of on every promotion - otherwise a long run at 128
    # channels leaves gigabytes behind. best.pt is still written every time.
    checkpoint_every: int = 5
    benchmark_depth: int = 4
    benchmark_every: int = 1
    checkpoint_dir: Path = Path("checkpoints")
    # Carry on from an existing network. The replay buffer isn't saved, so the
    # first few iterations refill it, but the weights are the expensive part.
    resume_from: Path | None = None
    seed: int = 0


def learning_rate_for(iteration: int, config: "Config") -> float:
    """Learning rate for this iteration."""
    decays = (iteration - 1) // config.lr_decay_every
    return max(config.lr_min, config.learning_rate * config.lr_decay_gamma**decays)


@dataclass
class IterationReport:
    iteration: int
    samples: int
    buffer: int
    loss_total: float
    loss_policy: float
    loss_value: float
    vs_incumbent: arena.MatchResult | None = None
    vs_benchmark: arena.MatchResult | None = None
    learning_rate: float = LEARNING_RATE
    elapsed: float = 0.0
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [
            f"iter {self.iteration:>2}",
            f"samples {self.samples:>5}",
            f"buffer {self.buffer:>6}",
            f"loss {self.loss_total:.4f} (p {self.loss_policy:.4f} v {self.loss_value:.4f})",
            f"lr {self.learning_rate:.1e}",
            f"{self.elapsed / 60:.0f}m",
        ]
        if self.vs_incumbent is not None:
            parts.append(f"vs prev {self.vs_incumbent}")
        if self.vs_benchmark is not None:
            parts.append(f"vs ab {self.vs_benchmark}")
        return "  |  ".join(parts)


def run(config: Config | None = None, verbose: bool = True) -> list[IterationReport]:
    """Run the loop. Returns one report per iteration."""
    config = config or Config()
    device = default_device()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if config.resume_from is not None:
        best = load(config.resume_from, device)
        if verbose:
            print(f"resuming from {config.resume_from}", flush=True)
    else:
        best = Connect4Net(channels=config.channels, blocks=config.blocks).to(device).eval()
        if verbose:
            print(
                f"fresh network: {config.channels} channels, {config.blocks} blocks, "
                f"{count_parameters(best):,} parameters",
                flush=True,
            )
        save(best, config.checkpoint_dir / "iter000.pt")

    buffer = ReplayBuffer(capacity=config.buffer_size)
    reports: list[IterationReport] = []
    started = time.monotonic()
    deadline = None if config.max_hours is None else started + config.max_hours * 3600
    stop_file = config.checkpoint_dir / "STOP"

    try:
        for iteration in range(1, config.iterations + 1):
            if deadline is not None and time.monotonic() >= deadline:
                if verbose:
                    print(f"time budget reached after {iteration - 1} iterations", flush=True)
                break

            # Drop a file called STOP in the checkpoint folder to end the run
            # after this iteration. Ctrl+C does the same thing.
            if stop_file.exists():
                if verbose:
                    print(f"stop file found after {iteration - 1} iterations", flush=True)
                stop_file.unlink()
                break

            # self-play
            samples = generate_games(
                best,
                games=config.games_per_iteration,
                simulations=config.simulations,
                parallel=config.parallel_games,
                add_noise=True,
                rng=rng,
            )
            buffer.extend(samples)

            # train a copy of the current best
            # (shape from the incumbent, not the config, in case we resumed)
            challenger = Connect4Net(
                channels=best.channels, blocks=best.blocks_count
            ).to(device)
            challenger.load_state_dict(best.state_dict())
            lr = learning_rate_for(iteration, config)
            optimizer = make_optimizer(challenger, learning_rate=lr)
            loss = train_epoch(
                challenger,
                optimizer,
                buffer,
                device,
                steps=config.train_steps,
                batch_size=config.batch_size,
                rng=rng,
            )

            report = IterationReport(
                iteration=iteration,
                samples=len(samples),
                buffer=len(buffer),
                loss_total=loss.total,
                loss_policy=loss.policy,
                loss_value=loss.value,
                learning_rate=lr,
                elapsed=time.monotonic() - started,
            )

            # Played against the outgoing network, so it has to happen before the
            # swap below. Reported only - it doesn't decide anything any more.
            report.vs_incumbent = arena.play_match(
                arena.network_agent(challenger, config.eval_simulations),
                arena.network_agent(best, config.eval_simulations),
                games=config.eval_games,
                rng=rng,
                opening_plies=config.opening_plies,
            )

            best = challenger.eval()
            # best.pt every time, not just at the end, so a crash or a lost
            # terminal can't cost hours of training.
            save(best, config.checkpoint_dir / "best.pt")
            if iteration % config.checkpoint_every == 0:
                save(best, config.checkpoint_dir / f"iter{iteration:03d}.pt")

            # fixed opponent, so progress is comparable across iterations
            if iteration % config.benchmark_every == 0:
                report.vs_benchmark = arena.play_match(
                    arena.network_agent(best, config.eval_simulations),
                    arena.alphabeta_agent(config.benchmark_depth),
                    games=config.eval_games,
                    rng=rng,
                    opening_plies=config.opening_plies,
                )

            reports.append(report)
            if verbose:
                print(report, flush=True)

    except KeyboardInterrupt:
        if verbose:
            print("\ninterrupted - saving the best network before exit", flush=True)

    # On every exit path, including the interrupt above.
    save(best, config.checkpoint_dir / "best.pt")
    return reports