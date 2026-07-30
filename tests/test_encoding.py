"""Tests for the board encoding.

The important one is that a position and its colour-swapped twin encode
identically when each is seen from its own player's side. Nothing else in the
network works properly if that's wrong.
"""

import numpy as np
import pytest

from connect4.board import Board, PLAYER_R, PLAYER_Y, ROWS, COLS
from connect4.encoding import (
    ACTION_SIZE,
    INPUT_SHAPE,
    PLANES,
    decode_to_string,
    encode,
    legal_move_mask,
    mask_and_normalise,
)
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y
from tests.test_evaluate import mirror_colours


def sample_position() -> Board:
    """A mid-game position: pieces at several heights, both colours present."""
    return play([(3, R), (3, Y), (4, R), (2, Y), (4, R), (1, Y)])


def full_board() -> Board:
    """A legally-reachable full board with no winner."""
    return play_alternating(DRAW_SEQUENCE)


def filled_column(col: int) -> Board:
    """A board with `col` filled to the top by alternating play."""
    return play_alternating([col] * ROWS)


# --------------------------------------------------------------------------
# shape and dtype
# --------------------------------------------------------------------------

def test_encoding_has_the_declared_shape_and_dtype():
    """(2, 6, 7) float32."""
    encoded = encode(sample_position(), PLAYER_R)
    assert encoded.shape == INPUT_SHAPE == (PLANES, ROWS, COLS)
    assert encoded.dtype == np.float32


def test_empty_board_encodes_to_all_zeros():
    assert not encode(Board(), PLAYER_R).any()


def test_encoding_is_binary():
    """Every entry is exactly 0.0 or 1.0 - no counts, no accumulation."""
    encoded = encode(sample_position(), PLAYER_Y)
    assert np.isin(encoded, [0.0, 1.0]).all()


def test_planes_are_disjoint():
    """No cell is set in both planes; a square holds at most one piece."""
    encoded = encode(sample_position(), PLAYER_R)
    assert not (encoded[0] * encoded[1]).any()


def test_set_cells_equal_move_count():
    """Total ones across both planes equals the number of pieces played."""
    board = sample_position()
    assert encode(board, PLAYER_R).sum() == board.move_count == 6


def test_full_board_encodes_every_cell_exactly_once():
    encoded = encode(full_board(), PLAYER_R)
    assert encoded.sum() == ROWS * COLS
    assert (encoded[0] + encoded[1] == 1.0).all()


# --------------------------------------------------------------------------
# canonical form - the load-bearing property
# --------------------------------------------------------------------------

def test_plane_zero_is_the_mover_and_plane_one_is_the_opponent():
    """Encode the same board as R and as Y; the two planes must swap."""
    board = sample_position()
    as_r = encode(board, PLAYER_R)
    as_y = encode(board, PLAYER_Y)

    assert np.array_equal(as_r[0], as_y[1])
    assert np.array_equal(as_r[1], as_y[0])
    assert not np.array_equal(as_r[0], as_r[1])


def test_colour_swap_encodes_identically_from_each_movers_view():
    """THE canonical-form test."""
    for board in [Board(), sample_position(), full_board()]:
        assert np.array_equal(
            encode(board, PLAYER_R),
            encode(mirror_colours(board), PLAYER_Y),
        )


def test_encoding_matches_the_grid_cell_by_cell():
    """Explicit coordinates, so a transpose or a row flip can't hide behind the aggregate
    tests above - every one of those passes on a flipped encoding."""
    encoded = encode(play([(0, R)]), PLAYER_R)

    assert encoded[0, ROWS - 1, 0] == 1.0
    assert encoded.sum() == 1.0
    assert not encoded[1].any()


def test_encoding_places_a_stack_at_the_right_heights():
    """Two pieces in one column: the mover's at the floor, the opponent's above."""
    encoded = encode(play([(2, R), (2, Y)]), PLAYER_R)

    assert encoded[0, ROWS - 1, 2] == 1.0
    assert encoded[1, ROWS - 2, 2] == 1.0
    assert encoded.sum() == 2.0


def test_encoding_does_not_mutate_the_board():
    board = sample_position()
    grid_before = [row[:] for row in board.grid]
    count_before = board.move_count

    encode(board, PLAYER_R)

    assert board.grid == grid_before
    assert board.move_count == count_before


# --------------------------------------------------------------------------
# legal move mask
# --------------------------------------------------------------------------

def test_mask_is_all_ones_on_an_empty_board():
    assert (legal_move_mask(Board()) == 1.0).all()


def test_mask_has_the_action_shape_and_dtype():
    """(7,) float32."""
    mask = legal_move_mask(sample_position())
    assert mask.shape == (ACTION_SIZE,) == (COLS,)
    assert mask.dtype == np.float32


def test_mask_zeroes_a_full_column():
    mask = legal_move_mask(filled_column(3))
    assert mask[3] == 0.0
    assert mask.sum() == COLS - 1


def test_mask_is_all_zeros_on_a_full_board():
    assert not legal_move_mask(full_board()).any()


def test_mask_agrees_with_available_moves():
    """The mask and available_moves() are two views of the same fact and must never
    disagree."""
    board = Board()
    for i, col in enumerate(DRAW_SEQUENCE):
        board.make_move(col, R if i % 2 == 0 else Y)
        mask = legal_move_mask(board)
        assert [c for c in range(COLS) if mask[c] == 1.0] == board.available_moves()


# --------------------------------------------------------------------------
# masking and renormalising priors
# --------------------------------------------------------------------------

def test_masked_priors_sum_to_one():
    priors = np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32)
    out = mask_and_normalise(priors, legal_move_mask(sample_position()))
    assert out.sum() == pytest.approx(1.0)


def test_masked_priors_are_zero_on_illegal_moves():
    """Probability must not leak onto moves that cannot be played."""
    board = filled_column(0)
    out = mask_and_normalise(np.ones(ACTION_SIZE, dtype=np.float32), legal_move_mask(board))

    assert out[0] == 0.0
    assert out.sum() == pytest.approx(1.0)


def test_masking_preserves_relative_weights_of_legal_moves():
    """Two legal columns at 2:1 must stay 2:1 - masking may only remove mass, never
    redistribute it unevenly."""
    priors = np.array([0, 0, 0, 2, 1, 0, 0], dtype=np.float32)
    out = mask_and_normalise(priors, legal_move_mask(Board()))

    assert out[3] / out[4] == pytest.approx(2.0)
    assert out.sum() == pytest.approx(1.0)


def test_masking_removes_mass_from_an_illegal_favourite():
    """The best column being illegal must not shift the ratio among the rest."""
    priors = np.array([0, 0, 0, 10, 2, 1, 0], dtype=np.float32)
    out = mask_and_normalise(priors, legal_move_mask(filled_column(3)))

    assert out[3] == 0.0
    assert out[4] / out[5] == pytest.approx(2.0)
    assert out.sum() == pytest.approx(1.0)


def test_all_zero_priors_fall_back_to_uniform_over_legal_moves():
    """An untrained or saturated network must not be able to crash the search."""
    out = mask_and_normalise(np.zeros(ACTION_SIZE, dtype=np.float32), legal_move_mask(filled_column(6)))

    assert out[6] == 0.0
    assert out.sum() == pytest.approx(1.0)
    assert out[:6] == pytest.approx(np.full(6, 1.0 / 6))


def test_no_legal_moves_raises():
    """A full board has nothing to normalise over."""
    with pytest.raises(ValueError, match="no legal moves"):
        mask_and_normalise(np.ones(ACTION_SIZE, dtype=np.float32), legal_move_mask(full_board()))


def test_negative_priors_raise():
    """Priors must be probabilities, not logits."""
    logits = np.array([-1.0, -2.0, 0.5, -3.0, 1.0, -1.0, -0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        mask_and_normalise(logits, legal_move_mask(Board()))


def test_masking_does_not_mutate_its_inputs():
    """Returns a new array; the caller's priors and mask are unchanged."""
    priors = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    mask = legal_move_mask(filled_column(0))
    priors_before, mask_before = priors.copy(), mask.copy()

    mask_and_normalise(priors, mask)

    assert np.array_equal(priors, priors_before)
    assert np.array_equal(mask, mask_before)


# --------------------------------------------------------------------------
# debug rendering
# --------------------------------------------------------------------------

def test_decode_round_trips_the_piece_positions():
    """decode_to_string(encode(b, p)) places X and O where the pieces are."""
    rendered = decode_to_string(encode(play([(2, R), (2, Y)]), PLAYER_R)).splitlines()

    assert len(rendered) == ROWS
    assert all(len(row) == COLS for row in rendered)
    assert rendered[ROWS - 1][2] == "X"   # mover's piece at the floor
    assert rendered[ROWS - 2][2] == "O"   # opponent stacked above


def test_decode_agrees_with_the_grid_everywhere():
    """Every cell, both perspectives - catches an off-by-one in the row joining."""
    board = sample_position()
    for player in (PLAYER_R, PLAYER_Y):
        rendered = decode_to_string(encode(board, player)).splitlines()
        for row in range(ROWS):
            for col in range(COLS):
                cell = board.grid[row][col]
                expected = "." if cell not in (PLAYER_R, PLAYER_Y) else (
                    "X" if cell == player else "O"
                )
                assert rendered[row][col] == expected


def test_decode_uses_perspective_symbols_not_colours():
    """X/O, not R/Y - the encoding has thrown away which colour is which, and the renderer
    must not pretend otherwise."""
    rendered = decode_to_string(encode(sample_position(), PLAYER_Y))

    assert PLAYER_R not in rendered
    assert PLAYER_Y not in rendered
    assert "X" in rendered and "O" in rendered


def test_decode_of_an_empty_board_has_no_pieces():
    rendered = decode_to_string(encode(Board(), PLAYER_R))
    assert "X" not in rendered
    assert "O" not in rendered
