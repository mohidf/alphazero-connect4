"""Playing games against itself to make training data.

Every position in a game becomes one sample:

    state   the encoded board
    policy  what the search decided, i.e. its visit counts
    value   who ended up winning, from that position's point of view

Games run many at a time so all their pending evaluations can go through the
network in one batch. Doing them one at a time is mostly launch overhead and
takes roughly five times as long.
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

# Moves sampled rather than played greedily, at the start of each game. This is
# what makes the games differ; after that it plays properly so the results mean
# something.
TEMPERATURE_MOVES = 10

DEFAULT_PARALLEL_GAMES = 32


@dataclass(frozen=True)
class Sample:
    """One training example."""

    state: np.ndarray      # (2, 6, 7)
    policy: np.ndarray     # (7,), sums to 1
    value: float           # -1, 0 or +1


def mirror(sample: Sample) -> Sample:
    """Flip a sample left-right. Connect 4 is symmetric, so the mirrored board
    is just as valid - free extra training data."""
    return Sample(
        state=np.ascontiguousarray(sample.state[:, :, ::-1]),
        policy=np.ascontiguousarray(sample.policy[::-1]),
        value=sample.value,
    )


def augment(samples: list[Sample]) -> list[Sample]:
    """Samples plus their mirrors."""
    return samples + [mirror(s) for s in samples]


def assign_values(
    states: list[np.ndarray],
    policies: list[np.ndarray],
    movers: list[str],
    winner: str | None,
) -> list[Sample]:
    """Label every position with the result, from that player's side.

    +1 on the winner's positions, -1 on the loser's. Labelling from one fixed
    colour instead would just teach the network that red usually wins.
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
    """One game in progress."""

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
        """Save the position, play the move, swap turns."""
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
    """Play `games` games and return every position as a Sample.

    The games step forward together, so their leaves can be evaluated in one
    batch each round.
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
                continue  # they all hit finished positions

            priors, values = predict_batch(
                net,
                [leaf.board for leaf in leaves],
                [leaf.player_to_move for leaf in leaves],
            )
            for i, search in enumerate(owners):
                search.resolve(priors[i], float(values[i]))
                # Noise as soon as the root exists, so it affects everything.
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
