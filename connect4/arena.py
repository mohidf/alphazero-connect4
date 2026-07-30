"""Head-to-head evaluation: is the new network actually better?

Self-play loss going down is not evidence of improvement — the targets move every
iteration, so the loss is measured against a shifting reference. The only honest
check is playing games.

Three opponents, in increasing order of what they tell you:

* **random** — a floor. Failing this means something is broken, not weak.
* **the previous checkpoint** — did this iteration help?
* **alpha-beta** — a fixed, genuinely strong benchmark of known depth. This is the
  one worth watching, and most people building AlphaZero from scratch have no
  equivalent.

Every match alternates who moves first, because Connect-4 is a first-player win
with perfect play and a fixed seat would flatter one side.

**Openings are randomised, and that is not optional.** Greedy PUCT and alpha-beta
are both deterministic, so without it a 12-game match replays the same two games
six times each: the reported score is correct but carries two games' worth of
information, and every promotion decision rests on those two. Each random opening
is played twice with the seats swapped, so both agents meet the same position from
both sides — that pairing is what keeps an unbalanced opening from deciding the
match. Pass opening_plies=0 for a deterministic match when that is what you want.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from connect4.board import Board, PLAYER_R, PLAYER_Y
from connect4.mcts import other
from connect4 import alphabeta as ab
from connect4.network import Connect4Net
from connect4.puct import C_PUCT, network_evaluator, puct_move

# (board, player) -> column
Agent = Callable[[Board, str], int]


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """Points per game, draws counting a half — the usual chess convention."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def __str__(self) -> str:
        return (
            f"W {self.wins} L {self.losses} D {self.draws} "
            f"(score {self.score:.2f})"
        )


def random_agent(rng: np.random.Generator) -> Agent:
    def agent(board: Board, player: str) -> int:
        return int(rng.choice(board.available_moves()))

    return agent


def alphabeta_agent(depth: int) -> Agent:
    def agent(board: Board, player: str) -> int:
        col, _ = ab.best_move(board, depth, player == PLAYER_R)
        return col

    return agent


def network_agent(
    net: Connect4Net, simulations: int = 160, c_puct: float = C_PUCT
) -> Agent:
    """PUCT guided by `net`, played greedily — no noise, no temperature.

    Noise exists to diversify training data; using it here would just make the
    agent weaker and the measurement noisier.
    """
    evaluator = network_evaluator(net)

    def agent(board: Board, player: str) -> int:
        return puct_move(board, player, evaluator, simulations, c_puct)

    return agent


# Plies played at random before the agents take over. Four is safe: the earliest
# possible win is ply seven, so a shorter opening can never be terminal.
DEFAULT_OPENING_PLIES = 4


def random_opening(rng: np.random.Generator, plies: int = DEFAULT_OPENING_PLIES) -> list[int]:
    """Return `plies` random legal columns, played alternately from an empty board.

    Kept under seven plies so the opening can never be a finished game — the
    earliest win in Connect-4 needs four moves by one player, i.e. ply seven.
    """
    if plies >= 7:
        raise ValueError("openings must be shorter than 7 plies to stay non-terminal")

    board = Board()
    player = PLAYER_R
    columns = []
    for _ in range(plies):
        col = int(rng.choice(board.available_moves()))
        columns.append(col)
        board.make_move(col, player)
        player = other(player)
    return columns


def play_game(first: Agent, second: Agent, opening: list[int] | None = None) -> str | None:
    """Play one game, `first` moving first as R. Returns the winner or None.

    `opening` is a list of columns played before either agent is consulted,
    alternating from R.
    """
    board = Board()
    player = PLAYER_R

    for col in opening or ():
        board.make_move(col, player)
        player = other(player)

    while not board.is_terminal():
        agent = first if player == PLAYER_R else second
        board.make_move(agent(board, player), player)
        player = other(player)

    return board.winner()


def play_match(
    challenger: Agent,
    incumbent: Agent,
    games: int,
    rng: np.random.Generator | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
) -> MatchResult:
    """Play `games` games and return the result from `challenger`'s side.

    Games come in pairs: one random opening, played once with the challenger
    moving first and once with the incumbent moving first. So the seats alternate
    *and* both agents face every opening from both sides, which is what makes a
    lopsided opening cancel out instead of deciding the match.

    opening_plies=0 gives a deterministic match — useful for tests, useless for
    measuring two deterministic agents against each other.
    """
    rng = rng if rng is not None else np.random.default_rng()
    wins = losses = draws = 0
    opening: list[int] = []

    for i in range(games):
        challenger_is_first = i % 2 == 0
        if challenger_is_first:
            # New opening for each pair; the second game of the pair reuses it.
            opening = random_opening(rng, opening_plies) if opening_plies else []

        if challenger_is_first:
            winner = play_game(challenger, incumbent, opening)
            challenger_colour = PLAYER_R
        else:
            winner = play_game(incumbent, challenger, opening)
            challenger_colour = PLAYER_Y

        if winner is None:
            draws += 1
        elif winner == challenger_colour:
            wins += 1
        else:
            losses += 1

    return MatchResult(wins=wins, losses=losses, draws=draws)
