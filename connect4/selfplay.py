"""Self-play game generation: turn a network into training data.

Each position visited in a game becomes one training sample:

    state   the canonical encoding of the position
    policy  the search's visit distribution — the *improved* policy
    value   the eventual game result, from that position's mover's perspective

The policy target is the interesting one. The network proposed `prior`, the search
spent its budget checking, and the resulting visit distribution is better than
what the network started with. Training toward it is policy improvement by
planning; the value target closes the loop by grounding it in actual outcomes.

Games are generated **many at a time** so their pending evaluations can be batched
into one forward pass. A batch-of-one forward pass is almost entirely kernel-launch
overhead, so this is the difference between a self-play iteration taking minutes
and taking an hour. The parallelism is across games — each search has at most one
leaf outstanding — so no virtual loss is needed.
"""

from dataclasses import dataclass

import numpy as np

from connect4.board import Board, COLS
from connect4.encoding import encode
from connect4.mcts import other
from connect4.network import Connect4Net, predict_batch
from connect4.puct import (
    C_PUCT,
    DIRICHLET_ALPHA,
    DIRICHLET_WEIGHT,
    Search,
    policy_target,
    select_move,
)

# Plies played at temperature 1 (sampled proportional to visits) before switching
# to temperature 0 (always the most-visited move). Sampling early is what makes
# games diverge; playing greedily later keeps the value targets meaningful.
TEMPERATURE_MOVES = 10

# Games kept in flight together. Larger batches use the GPU better; too large and
# early-finishing games leave the batch half empty.
DEFAULT_PARALLEL_GAMES = 32


@dataclass(frozen=True)
class Sample:
    """One training example."""

    state: np.ndarray      # (2, 6, 7) float32, canonical
    policy: np.ndarray     # (7,) float32, sums to 1
    value: float           # in {-1.0, 0.0, +1.0}, from this position's mover's view


def mirror(sample: Sample) -> Sample:
    """Left-right mirror of a sample.

    Connect-4 is symmetric under horizontal reflection: the mirrored position is
    legal, its value is unchanged, and its policy is the original reversed. Free
    data, and it discourages the network from learning column-specific quirks.
    """
    return Sample(
        state=np.ascontiguousarray(sample.state[:, :, ::-1]),
        policy=np.ascontiguousarray(sample.policy[::-1]),
        value=sample.value,
    )


def augment(samples: list[Sample]) -> list[Sample]:
    """Return `samples` plus their mirror images."""
    return samples + [mirror(s) for s in samples]


def assign_values(
    states: list[np.ndarray],
    policies: list[np.ndarray],
    movers: list[str],
    winner: str | None,
) -> list[Sample]:
    """Attach the game result to every position, from each mover's perspective.

    The same game gives +1 to the winner's positions and -1 to the loser's. Using
    a single fixed perspective instead would teach the network that one colour
    tends to win, which is not a fact about Connect-4 positions.
    """
    samples = []
    for state, policy, mover in zip(states, policies, movers):
        if winner is None:
            value = 0.0
        else:
            value = 1.0 if winner == mover else -1.0
        samples.append(Sample(state=state, policy=policy, value=value))
    return samples


class _GameInProgress:
    """Bookkeeping for one self-play game while it runs alongside others."""

    def __init__(self, first_player: str, rng: np.random.Generator) -> None:
        self.board = Board()
        self.player = first_player
        self.rng = rng
        self.states: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.movers: list[str] = []
        self.search: Search | None = None

    @property
    def done(self) -> bool:
        return self.board.is_terminal()

    def temperature(self) -> float:
        return 1.0 if len(self.movers) < TEMPERATURE_MOVES else 0.0

    def start_search(self, c_puct: float) -> None:
        self.search = Search(self.board, self.player, c_puct)

    def finish_move(self, noise: bool) -> None:
        """Record the position, play the chosen move, and advance the turn."""
        search = self.search
        assert search is not None

        self.states.append(encode(self.board, self.player))
        self.policies.append(policy_target(search.root, temperature=1.0))
        self.movers.append(self.player)

        move = select_move(search.root, self.temperature(), self.rng)
        self.board.make_move(move, self.player)
        self.player = other(self.player)
        self.search = None

    def samples(self) -> list[Sample]:
        return assign_values(
            self.states, self.policies, self.movers, self.board.winner()
        )


def generate_games(
    net: Connect4Net,
    games: int,
    simulations: int = 160,
    c_puct: float = C_PUCT,
    parallel: int = DEFAULT_PARALLEL_GAMES,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    augment_samples: bool = True,
) -> list[Sample]:
    """Play `games` self-play games and return every position as a Sample.

    All in-flight games advance in lockstep: each contributes at most one pending
    leaf per round, and all pending leaves are evaluated in a single batch.
    """
    rng = rng if rng is not None else np.random.default_rng()
    remaining = games
    active: list[_GameInProgress] = []
    collected: list[Sample] = []

    while remaining > 0 or active:
        # Top up the batch.
        while len(active) < parallel and remaining > 0:
            first = "R" if remaining % 2 == 0 else "Y"
            active.append(_GameInProgress(first, rng))
            remaining -= 1

        for game in active:
            if game.search is None:
                game.start_search(c_puct)

        # Run simulations for every active game, batching the evaluations.
        while any(g.search.simulations_done < simulations for g in active):
            leaves, owners = [], []
            for game in active:
                search = game.search
                if search.simulations_done >= simulations:
                    continue
                leaf = search.pending_leaf()
                if leaf is not None:
                    leaves.append(leaf)
                    owners.append(search)

            if not leaves:
                continue  # every pending simulation resolved on a terminal node

            priors, values = predict_batch(
                net,
                [leaf.board for leaf in leaves],
                [leaf.player_to_move for leaf in leaves],
            )
            for i, search in enumerate(owners):
                search.resolve(priors[i], float(values[i]))
                # Noise goes on immediately after the root is expanded, so the
                # very first simulation already explores under it.
                if add_noise and search.root.children and search.simulations_done == 0:
                    search.add_noise(DIRICHLET_ALPHA, DIRICHLET_WEIGHT, rng)

        for game in active:
            game.finish_move(add_noise)

        finished = [g for g in active if g.done]
        for game in finished:
            samples = game.samples()
            collected.extend(augment(samples) if augment_samples else samples)
        active = [g for g in active if not g.done]

    return collected
