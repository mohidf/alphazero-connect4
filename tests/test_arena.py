"""Tests for the arena.

The thing most likely to be wrong is which side won, since the challenger swaps
colours between games. The rest is about openings being randomised, without which
a match of two deterministic agents is just the same game repeated.
"""

import numpy as np
import pytest
import torch

from connect4.board import Board, COLS, PLAYER_R, PLAYER_Y
from connect4 import arena
from connect4.arena import (
    DEFAULT_OPENING_PLIES,
    MatchResult,
    alphabeta_agent,
    network_agent,
    play_game,
    play_match,
    random_agent,
    random_opening,
)
from connect4.network import Connect4Net

CPU = torch.device("cpu")


def first_legal_agent(board: Board, player: str) -> int:
    """Deterministic and weak: always the lowest-numbered legal column."""
    return board.available_moves()[0]


def last_legal_agent(board: Board, player: str) -> int:
    return board.available_moves()[-1]


def fixed_column_agent(col: int):
    """Plays `col` while it is legal, else the first legal column."""
    def agent(board: Board, player: str) -> int:
        return col if col in board.available_moves() else board.available_moves()[0]
    return agent


# --------------------------------------------------------------------------
# result bookkeeping
# --------------------------------------------------------------------------

def test_score_counts_a_draw_as_a_half():
    assert MatchResult(wins=1, losses=1, draws=2).score == pytest.approx(0.5)


def test_score_of_a_clean_sweep_is_one():
    assert MatchResult(wins=10, losses=0, draws=0).score == 1.0


def test_score_of_no_games_is_zero_not_a_crash():
    assert MatchResult(0, 0, 0).score == 0.0


def test_games_counts_every_result():
    assert MatchResult(wins=3, losses=4, draws=5).games == 12


# --------------------------------------------------------------------------
# playing a game
# --------------------------------------------------------------------------

def test_play_game_returns_a_winner_or_none():
    assert play_game(first_legal_agent, last_legal_agent) in (PLAYER_R, PLAYER_Y, None)


def test_play_game_runs_to_a_terminal_position():
    """Two column-stacking agents: R fills column 0 and wins vertically."""
    assert play_game(fixed_column_agent(0), fixed_column_agent(1)) == PLAYER_R


def test_the_second_agent_plays_yellow():
    """Mirror of the test above - whoever moves second is Y and should lose here."""
    assert play_game(fixed_column_agent(1), fixed_column_agent(0)) == PLAYER_R


# --------------------------------------------------------------------------
# matches
# --------------------------------------------------------------------------

def test_match_plays_the_requested_number_of_games():
    result = play_match(first_legal_agent, last_legal_agent, games=6, opening_plies=0)
    assert result.games == 6


def test_match_alternates_who_moves_first():
    """Connect 4 is a first-player win with perfect play,
    so a fixed seat would flatter one side."""
    result = play_match(fixed_column_agent(0), fixed_column_agent(1), games=8, opening_plies=0)

    assert result.wins == 4
    assert result.losses == 4
    assert result.score == pytest.approx(0.5)


def test_match_attributes_wins_to_the_challenger_in_both_seats():
    """A strictly stronger challenger must score 1.0 regardless of colour."""
    result = play_match(alphabeta_agent(4), first_legal_agent, games=4, opening_plies=0)
    assert result.score == 1.0
    assert result.losses == 0


def test_a_weaker_challenger_scores_below_a_half():
    result = play_match(first_legal_agent, alphabeta_agent(4), games=4, opening_plies=0)
    assert result.score == 0.0


# --------------------------------------------------------------------------
# agents
# --------------------------------------------------------------------------

def test_random_agent_only_plays_legal_columns():
    rng = np.random.default_rng(0)
    agent = random_agent(rng)
    board = Board()
    for _ in range(10):
        assert agent(board, PLAYER_R) in board.available_moves()


def test_alphabeta_agent_takes_an_immediate_win():
    from tests.test_minimax import r_can_win_in_one

    assert alphabeta_agent(2)(r_can_win_in_one(), PLAYER_R) == 3


def test_alphabeta_agent_plays_both_colours():
    """is_maximizing has to follow the colour, not be hardcoded."""
    from tests.test_minimax import r_can_win_in_one

    assert alphabeta_agent(2)(r_can_win_in_one(), PLAYER_Y) == 3


def test_network_agent_only_plays_legal_columns():
    """Players must alternate: five moves by one colour makes four in a row, the board goes
    terminal, and puct_move() correctly returns None."""
    from connect4.mcts import other

    torch.manual_seed(0)
    net = Connect4Net().to(CPU).eval()
    agent = network_agent(net, simulations=8)

    board = Board()
    player = PLAYER_R
    for _ in range(6):
        assert not board.is_terminal()
        col = agent(board, player)
        assert col in board.available_moves()
        board.make_move(col, player)
        player = other(player)


def test_network_agent_finds_a_forced_win_untrained():
    """Terminal values are exact, so this holds before any training."""
    from tests.test_minimax import r_can_win_in_one

    torch.manual_seed(0)
    net = Connect4Net().to(CPU).eval()

    assert network_agent(net, simulations=60)(r_can_win_in_one(), PLAYER_R) == 3


def test_untrained_network_beats_random():
    """A floor, not a milestone."""
    torch.manual_seed(0)
    net = Connect4Net().to(CPU).eval()

    result = play_match(
        network_agent(net, simulations=40),
        random_agent(np.random.default_rng(0)),
        games=6,
        rng=np.random.default_rng(0),
    )
    assert result.score >= 0.8, result


# --------------------------------------------------------------------------
# randomised openings - what makes a match of deterministic agents informative
# --------------------------------------------------------------------------

def distinct_games(challenger, incumbent, games, rng, opening_plies):
    """Replay a match, recording the move sequence of each game."""
    from connect4.arena import play_game
    from connect4.mcts import other

    seen = set()
    opening = []
    for i in range(games):
        first_turn = i % 2 == 0
        if first_turn:
            opening = random_opening(rng, opening_plies) if opening_plies else []
        a, b = (challenger, incumbent) if first_turn else (incumbent, challenger)

        board = Board()
        player = PLAYER_R
        moves = []
        for col in opening:
            moves.append(col)
            board.make_move(col, player)
            player = other(player)
        while not board.is_terminal():
            col = (a if player == PLAYER_R else b)(board, player)
            moves.append(col)
            board.make_move(col, player)
            player = other(player)
        seen.add((first_turn, tuple(moves)))
    return seen


def test_deterministic_agents_replay_the_same_two_games_without_openings():
    """The flaw this feature exists to fix, pinned so it cannot come back."""
    seen = distinct_games(
        alphabeta_agent(2), first_legal_agent, 12, np.random.default_rng(0), 0
    )
    assert len(seen) == 2


def test_randomised_openings_make_the_games_distinct():
    """Same agents, same game count - now many distinct games."""
    seen = distinct_games(
        alphabeta_agent(2), first_legal_agent, 12, np.random.default_rng(0),
        DEFAULT_OPENING_PLIES,
    )
    assert len(seen) >= 8


def test_random_opening_is_legal_and_never_terminal():
    """The earliest possible win is ply 7, so a shorter opening cannot be a finished game -
    otherwise a match would score openings, not agents."""
    from connect4.mcts import other

    rng = np.random.default_rng(0)
    for plies in range(0, 7):
        for _ in range(20):
            board = Board()
            player = PLAYER_R
            for col in random_opening(rng, plies):
                assert col in board.available_moves()
                board.make_move(col, player)
                player = other(player)
            assert not board.is_terminal()


def test_random_opening_rejects_lengths_that_could_be_terminal():
    with pytest.raises(ValueError):
        random_opening(np.random.default_rng(0), plies=7)


def test_openings_are_paired_across_the_seats():
    """Both agents must face each opening from both sides."""
    seen = distinct_games(
        alphabeta_agent(2), first_legal_agent, 4, np.random.default_rng(1),
        DEFAULT_OPENING_PLIES,
    )
    prefixes = [tuple(moves[:DEFAULT_OPENING_PLIES]) for _, moves in seen]
    assert len(set(prefixes)) == 2          # two openings across four games
    assert all(prefixes.count(p) == 2 for p in set(prefixes))


def test_match_is_reproducible_given_a_seed():
    """A surprising promotion decision has to be re-runnable."""
    kwargs = dict(games=6, opening_plies=DEFAULT_OPENING_PLIES)
    first = play_match(alphabeta_agent(2), first_legal_agent, rng=np.random.default_rng(3), **kwargs)
    second = play_match(alphabeta_agent(2), first_legal_agent, rng=np.random.default_rng(3), **kwargs)

    assert (first.wins, first.losses, first.draws) == (second.wins, second.losses, second.draws)


def test_a_stronger_agent_still_wins_with_random_openings():
    """Randomisation must not swamp a real strength difference."""
    result = play_match(
        alphabeta_agent(4), first_legal_agent, games=8, rng=np.random.default_rng(0)
    )
    assert result.score >= 0.75, result
