"""Playing two agents against each other.

Training loss going down doesn't mean the network got better, because the targets
move every iteration. Actually playing games is the only real check. Opponents,
roughly in order of usefulness: random (a floor), the previous checkpoint, and
alpha-beta at a fixed depth.

Who moves first alternates, since Connect 4 is a first-player win and a fixed seat
would flatter one side.

Openings are randomised, which matters more than it sounds. Both agents are
deterministic, so without it a 12-game match is really the same two games played
six times each - the score looks fine but means almost nothing. Each opening is
played twice with the sides swapped so a lopsided one cancels out. Use
opening_plies=0 if you want a deterministic match.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from connect4.board import Board, PLAYER_R, PLAYER_Y
from connect4.encoding import legal_move_mask, mask_and_normalise
from connect4.mcts import mcts_move, other
from connect4 import alphabeta as ab
from connect4.network import Connect4Net, predict
from connect4.puct import C_PUCT, caching_evaluator, network_evaluator, puct_move

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
        """Points per game, draws worth a half."""
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


def policy_agent(net: Connect4Net) -> Agent:
    """Just the policy head, no search - whichever legal column it likes most.

    Shows what the network itself has learned, as opposed to what the search
    manages to find at play time.
    """
    def agent(board: Board, player: str) -> int:
        priors, _ = predict(net, board, player)
        legal = mask_and_normalise(priors, legal_move_mask(board))
        return int(legal.argmax())

    return agent


def mcts_agent(simulations: int = 1000) -> Agent:
    """Plain MCTS with random rollouts, no network."""
    def agent(board: Board, player: str) -> int:
        return mcts_move(board, player, simulations)

    return agent


def network_agent(
    net: Connect4Net, simulations: int = 160, c_puct: float = C_PUCT
) -> Agent:
    """PUCT with the network, played greedily. No noise - that is only for
    generating varied training data and would just weaken it here.

    Evaluations are cached, since these searches run one at a time and successive
    moves cover a lot of the same tree. The cache is per call so it never
    outlives the weights it came from.
    """
    evaluator = caching_evaluator(network_evaluator(net))

    def agent(board: Board, player: str) -> int:
        return puct_move(board, player, evaluator, simulations, c_puct)

    return agent


# Random moves played before the agents take over. Under 7 is safe, since the
# earliest possible win is move 7.
DEFAULT_OPENING_PLIES = 4


def random_opening(rng: np.random.Generator, plies: int = DEFAULT_OPENING_PLIES) -> list[int]:
    """A few random opening moves, alternating from an empty board."""
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
    """Play a game. `first` is R. `opening` is played before either agent gets
    a turn. Returns the winner, or None for a draw."""
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
    """Play a match, scored from the challenger's side.

    Games come in pairs sharing one random opening, played from both sides, so an
    unfair opening cancels out. opening_plies=0 makes it deterministic, which is
    handy in tests and useless for actual measurement.
    """
    rng = rng if rng is not None else np.random.default_rng()
    wins = losses = draws = 0
    opening: list[int] = []

    for i in range(games):
        challenger_is_first = i % 2 == 0
        if challenger_is_first:
            # new opening per pair, reused by the second game
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
