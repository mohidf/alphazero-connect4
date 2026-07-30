"""Tests for self-play data generation.

Mostly about the training targets, since a wrong target trains a wrong network
without anything failing: the value sign per position, and the mirroring.
"""

import numpy as np
import pytest
import torch

from connect4.board import Board, COLS, ROWS, PLAYER_R, PLAYER_Y
from connect4.encoding import encode
from connect4.network import Connect4Net
from connect4.selfplay import (
    Sample,
    TEMPERATURE_MOVES,
    assign_values,
    augment,
    generate_games,
    mirror,
)
from tests.test_board import play, R, Y

CPU = torch.device("cpu")


@pytest.fixture
def net() -> Connect4Net:
    torch.manual_seed(0)
    return Connect4Net().to(CPU).eval()


def a_sample() -> Sample:
    """A sample with an asymmetric position and an asymmetric policy."""
    return Sample(
        state=encode(play([(0, R), (1, Y), (0, R)]), PLAYER_Y),
        policy=np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0], dtype=np.float32),
        value=1.0,
    )


# --------------------------------------------------------------------------
# mirror augmentation
# --------------------------------------------------------------------------

def test_mirror_reverses_the_policy():
    sample = a_sample()
    assert np.array_equal(mirror(sample).policy, sample.policy[::-1])


def test_mirror_reflects_the_board_horizontally():
    """A piece in column 0 must land in column COLS-1, at the same height."""
    sample = Sample(
        state=encode(play([(0, R)]), PLAYER_R),
        policy=np.full(COLS, 1.0 / COLS, dtype=np.float32),
        value=0.0,
    )
    reflected = mirror(sample).state

    assert reflected[0, ROWS - 1, COLS - 1] == 1.0
    assert reflected[0, ROWS - 1, 0] == 0.0
    assert reflected.sum() == 1.0


def test_mirror_leaves_the_value_unchanged():
    """Reflection is a symmetry of Connect-4, so the position is worth the same."""
    sample = a_sample()
    assert mirror(sample).value == sample.value


def test_mirror_is_an_involution():
    sample = a_sample()
    twice = mirror(mirror(sample))

    assert np.array_equal(twice.state, sample.state)
    assert np.array_equal(twice.policy, sample.policy)


def test_mirror_output_is_contiguous():
    """Negative-stride views break torch.from_numpy, and the failure surfaces inside the
    training loop rather than here."""
    reflected = mirror(a_sample())
    assert reflected.state.flags["C_CONTIGUOUS"]
    assert reflected.policy.flags["C_CONTIGUOUS"]
    torch.from_numpy(reflected.state)  # must not raise


def test_mirror_does_not_mutate_the_original():
    sample = a_sample()
    state_before, policy_before = sample.state.copy(), sample.policy.copy()

    mirror(sample)

    assert np.array_equal(sample.state, state_before)
    assert np.array_equal(sample.policy, policy_before)


def test_augment_doubles_the_sample_count():
    samples = [a_sample(), a_sample()]
    assert len(augment(samples)) == 4


# --------------------------------------------------------------------------
# value targets
# --------------------------------------------------------------------------

def test_winner_positions_get_plus_one_and_loser_positions_minus_one():
    """Per-position perspective, not a fixed colour."""
    states = [np.zeros((2, ROWS, COLS), dtype=np.float32)] * 4
    policies = [np.full(COLS, 1.0 / COLS, dtype=np.float32)] * 4
    movers = [PLAYER_R, PLAYER_Y, PLAYER_R, PLAYER_Y]

    samples = assign_values(states, policies, movers, winner=PLAYER_R)

    assert [s.value for s in samples] == [1.0, -1.0, 1.0, -1.0]


def test_a_draw_gives_every_position_zero():
    states = [np.zeros((2, ROWS, COLS), dtype=np.float32)] * 3
    policies = [np.full(COLS, 1.0 / COLS, dtype=np.float32)] * 3
    movers = [PLAYER_R, PLAYER_Y, PLAYER_R]

    samples = assign_values(states, policies, movers, winner=None)

    assert [s.value for s in samples] == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# generate_games
# --------------------------------------------------------------------------

def test_generated_samples_have_the_right_shapes(net):
    samples = generate_games(
        net, games=2, simulations=8, parallel=2, rng=np.random.default_rng(0)
    )
    assert samples

    for sample in samples:
        assert sample.state.shape == (2, ROWS, COLS)
        assert sample.policy.shape == (COLS,)
        assert sample.policy.sum() == pytest.approx(1.0)
        assert sample.value in (-1.0, 0.0, 1.0)


def test_generated_states_are_binary_and_disjoint(net):
    """Every state must still look like a real encoded position after augmentation."""
    samples = generate_games(
        net, games=2, simulations=8, parallel=2, rng=np.random.default_rng(0)
    )
    for sample in samples:
        assert np.isin(sample.state, [0.0, 1.0]).all()
        assert not (sample.state[0] * sample.state[1]).any()


def test_augmentation_doubles_the_output(net):
    kwargs = dict(games=2, simulations=8, parallel=2)
    plain = generate_games(
        net, **kwargs, augment_samples=False, rng=np.random.default_rng(0)
    )
    doubled = generate_games(
        net, **kwargs, augment_samples=True, rng=np.random.default_rng(0)
    )
    assert len(doubled) == 2 * len(plain)


def test_games_produce_a_plausible_number_of_positions(net):
    """One sample per ply, so a game yields between 7 and 42 positions."""
    samples = generate_games(
        net, games=1, simulations=8, parallel=1,
        augment_samples=False, rng=np.random.default_rng(0),
    )
    assert 7 <= len(samples) <= ROWS * COLS


def test_generation_is_reproducible_given_a_seed(net):
    """Same seed, same data."""
    kwargs = dict(games=2, simulations=8, parallel=2, augment_samples=False)
    first = generate_games(net, **kwargs, rng=np.random.default_rng(7))
    second = generate_games(net, **kwargs, rng=np.random.default_rng(7))

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a.state, b.state)
        assert np.array_equal(a.policy, b.policy)
        assert a.value == b.value


def test_noise_changes_the_games_generated(net):
    """Root noise exists to stop self-play replaying one game forever."""
    kwargs = dict(games=4, simulations=16, parallel=4, augment_samples=False)
    with_noise = generate_games(net, **kwargs, add_noise=True, rng=np.random.default_rng(0))
    without = generate_games(net, **kwargs, add_noise=False, rng=np.random.default_rng(0))

    assert not (
        len(with_noise) == len(without)
        and all(np.array_equal(a.policy, b.policy) for a, b in zip(with_noise, without))
    )


def test_parallelism_does_not_change_correctness(net):
    """Batch size is a performance knob."""
    for parallel in (1, 2, 4):
        samples = generate_games(
            net, games=4, simulations=8, parallel=parallel,
            rng=np.random.default_rng(0),
        )
        assert samples
        for sample in samples:
            assert sample.policy.sum() == pytest.approx(1.0)
            assert sample.value in (-1.0, 0.0, 1.0)


def test_zero_games_returns_nothing(net):
    assert generate_games(net, games=0, simulations=8, parallel=2) == []
