"""Tests for the network.

An untrained network has no opinions worth checking, so these are about shapes,
ranges, determinism, devices and saving/loading. All on CPU so they're fast and
the same everywhere.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from connect4.board import Board, PLAYER_R, PLAYER_Y, ROWS, COLS
from connect4.encoding import ACTION_SIZE, INPUT_SHAPE, PLANES, encode
from connect4.network import (
    BLOCKS,
    CHANNELS,
    Connect4Net,
    ResidualBlock,
    architecture,
    count_parameters,
    load,
    predict,
    predict_batch,
    save,
)
from tests.test_board import play, play_alternating, DRAW_SEQUENCE, R, Y
from tests.test_evaluate import mirror_colours

CPU = torch.device("cpu")


@pytest.fixture
def net() -> Connect4Net:
    """A small untrained network on CPU, in eval mode."""
    torch.manual_seed(0)
    return Connect4Net().to(CPU).eval()


def sample_position() -> Board:
    return play([(3, R), (3, Y), (4, R), (2, Y), (4, R), (1, Y)])


def other_positions() -> list[Board]:
    """Assorted positions for batch tests."""
    return [
        Board(),
        play([(0, R)]),
        play([(6, Y), (6, R), (5, Y)]),
        play_alternating(DRAW_SEQUENCE[:20]),
    ]


def batch_of(n: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, *INPUT_SHAPE)


# --------------------------------------------------------------------------
# residual block
# --------------------------------------------------------------------------

def test_residual_block_preserves_shape():
    """A block must be shape-preserving or it can't be stacked."""
    block = ResidualBlock(CHANNELS).eval()
    x = torch.randn(4, CHANNELS, ROWS, COLS)
    assert block(x).shape == x.shape


def test_residual_block_is_not_the_identity():
    """Shape-preserving but not a no-op - catches a forward() that returns its input
    untouched, which would pass every shape test in this file."""
    torch.manual_seed(0)
    block = ResidualBlock(CHANNELS).eval()
    x = torch.randn(4, CHANNELS, ROWS, COLS)
    assert not torch.allclose(block(x), x)


def test_residual_block_passes_gradients_to_its_input():
    """The skip connection's whole purpose."""
    torch.manual_seed(0)
    block = ResidualBlock(CHANNELS).eval()
    x = torch.randn(2, CHANNELS, ROWS, COLS, requires_grad=True)

    block(x).sum().backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_residual_block_does_not_mutate_its_input():
    """`out += x` is in-place on the block's own intermediate, not on x."""
    torch.manual_seed(0)
    block = ResidualBlock(CHANNELS).eval()
    x = torch.randn(2, CHANNELS, ROWS, COLS)
    before = x.clone()

    block(x)

    assert torch.equal(x, before)


# --------------------------------------------------------------------------
# forward pass shapes
# --------------------------------------------------------------------------

def test_forward_returns_the_declared_shapes(net):
    """(B, 2, 6, 7) -> policy (B, 7), value (B,)."""
    logits, value = net(batch_of(8))
    assert logits.shape == (8, ACTION_SIZE)
    assert value.shape == (8,)


def test_forward_handles_a_batch_of_one(net):
    """The shape the search uses on every simulation."""
    logits, value = net(batch_of(1))
    assert logits.shape == (1, ACTION_SIZE)
    assert value.shape == (1,)


def test_forward_handles_a_large_batch(net):
    """The shape training uses."""
    logits, value = net(batch_of(256))
    assert logits.shape == (256, ACTION_SIZE)
    assert value.shape == (256,)


def test_value_is_within_tanh_range(net):
    """Strictly inside [-1, 1] - a missing tanh shows up here immediately."""
    _, value = net(batch_of(64))
    assert value.abs().lt(1.0).all()


def test_policy_output_is_logits_not_probabilities(net):
    """Raw logits must NOT already sum to 1, or a softmax has leaked into forward() and
    training would apply it twice."""
    logits, _ = net(batch_of(16))
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(16), atol=1e-3)
    assert (logits < 0).any()


def test_forward_is_deterministic_in_eval_mode(net):
    """Two identical calls give identical results."""
    x = batch_of(8)
    first = net(x)
    second = net(x)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_rows_of_the_batch_are_independent(net):
    """Position i's output must not depend on what else is in the batch."""
    target = encode(sample_position(), PLAYER_R)
    alone = torch.from_numpy(target[None])
    padded = torch.cat([alone, torch.from_numpy(np.stack([
        encode(b, PLAYER_Y) for b in other_positions()
    ]))])

    logits_alone, value_alone = net(alone)
    logits_batch, value_batch = net(padded)

    assert torch.allclose(logits_alone[0], logits_batch[0], atol=1e-6)
    assert torch.allclose(value_alone[0], value_batch[0], atol=1e-6)


# --------------------------------------------------------------------------
# predict()
# --------------------------------------------------------------------------

def test_predict_returns_probabilities_and_a_scalar(net):
    """Priors: shape (7,), summing to 1."""
    priors, value = predict(net, sample_position(), PLAYER_R)

    assert isinstance(priors, np.ndarray)
    assert priors.shape == (ACTION_SIZE,)
    assert priors.sum() == pytest.approx(1.0)
    assert isinstance(value, float)
    assert -1.0 < value < 1.0


def test_predict_priors_are_non_negative(net):
    """mask_and_normalise() rejects negative priors, so predict() must have applied softmax
    before handing them over."""
    priors, _ = predict(net, sample_position(), PLAYER_R)
    assert (priors >= 0).all()


def test_predict_does_not_mask_illegal_moves(net):
    """Masking is the search's job - it's the only caller that knows the position in
    context."""
    board = play_alternating([3] * ROWS)
    assert 3 not in board.available_moves()

    priors, _ = predict(net, board, PLAYER_R)
    assert priors[3] > 0


def test_predict_agrees_with_forward(net):
    """predict() is a thin wrapper: same value as forward(), and priors equal to
    softmax(logits)."""
    board = sample_position()
    x = torch.from_numpy(encode(board, PLAYER_R)[None])
    # no_grad here for the same reason predict() needs it: .numpy() refuses on a
    # tensor that requires grad.
    with torch.no_grad():
        logits, value = net(x)

    priors, predicted_value = predict(net, board, PLAYER_R)

    assert np.allclose(priors, F.softmax(logits, dim=-1)[0].numpy(), atol=1e-6)
    assert predicted_value == pytest.approx(float(value[0]), abs=1e-6)


def test_predict_leaves_the_network_in_eval_mode(net):
    """A predict() that flips to train() and forgets to flip back would poison every later
    call."""
    net.train()
    predict(net, sample_position(), PLAYER_R)
    assert net.training is False


def test_predict_does_not_mutate_the_board(net):
    board = sample_position()
    grid_before = [row[:] for row in board.grid]

    predict(net, board, PLAYER_R)

    assert board.grid == grid_before
    assert board.move_count == 6


def test_predict_does_not_build_a_graph(net):
    """Asserted by forcing grad to be enabled at the call site."""
    with torch.enable_grad():
        priors, value = predict(net, sample_position(), PLAYER_R)

    assert isinstance(priors, np.ndarray)
    assert isinstance(value, float)


# --------------------------------------------------------------------------
# canonical form flows through the network
# --------------------------------------------------------------------------

def test_colour_mirrored_positions_predict_identically(net):
    """The encoding guarantees identical input, so the network must give identical output."""
    board = sample_position()

    priors_r, value_r = predict(net, board, PLAYER_R)
    priors_y, value_y = predict(net, mirror_colours(board), PLAYER_Y)

    assert np.array_equal(priors_r, priors_y)
    assert value_r == value_y


def test_predict_batch_matches_predict_one_by_one(net):
    """Batched and single evaluation must agree."""
    boards = [sample_position(), *other_positions()]
    players = [PLAYER_R, PLAYER_Y, PLAYER_R, PLAYER_Y, PLAYER_R]

    batch_priors, batch_values = predict_batch(net, boards, players)

    for i, (board, player) in enumerate(zip(boards, players)):
        priors, value = predict(net, board, player)
        assert np.allclose(batch_priors[i], priors, atol=1e-6)
        assert batch_values[i] == pytest.approx(value, abs=1e-6)


def test_predict_batch_shapes(net):
    """(N, 7) priors and (N,) values."""
    boards = other_positions()
    priors, values = predict_batch(net, boards, [PLAYER_R] * len(boards))

    assert priors.shape == (len(boards), ACTION_SIZE)
    assert values.shape == (len(boards),)
    assert np.allclose(priors.sum(axis=-1), 1.0)


def test_predict_batch_on_a_single_position(net):
    """N=1 must not collapse a dimension."""
    priors, values = predict_batch(net, [sample_position()], [PLAYER_R])
    assert priors.shape == (1, ACTION_SIZE)
    assert values.shape == (1,)


def test_predict_batch_rejects_mismatched_lengths(net):
    """One player per board."""
    with pytest.raises(ValueError):
        predict_batch(net, [Board(), Board()], [PLAYER_R])


def test_predict_batch_rejects_an_empty_batch(net):
    with pytest.raises(ValueError):
        predict_batch(net, [], [])


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path, net):
    """A loaded network gives bit-identical predictions to the saved one."""
    path = tmp_path / "ckpt.pt"
    board = sample_position()
    priors_before, value_before = predict(net, board, PLAYER_R)

    save(net, path)
    priors_after, value_after = predict(load(path, CPU), board, PLAYER_R)

    assert np.array_equal(priors_before, priors_after)
    assert value_before == value_after


def test_loaded_network_is_in_eval_mode(tmp_path, net):
    """Loading a checkpoint and forgetting eval() gives a network whose predictions depend
    on batch composition. load() must not leave that trap."""
    path = tmp_path / "ckpt.pt"
    save(net, path)
    assert load(path, CPU).training is False


def test_save_writes_a_state_dict_not_a_pickled_module(tmp_path, net):
    """The checkpoint must not embed the class."""
    path = tmp_path / "ckpt.pt"
    save(net, path)

    raw = torch.load(path, weights_only=True)

    assert set(raw) == {"architecture", "state_dict"}
    assert not isinstance(raw["state_dict"], torch.nn.Module)
    assert all(isinstance(v, torch.Tensor) for v in raw["state_dict"].values())
    assert raw["architecture"] == architecture(net)


def test_load_rejects_an_incompatible_encoding(tmp_path, net):
    """planes and action_size are the encoding contract and cannot vary - the weights would
    describe a different game representation."""
    path = tmp_path / "ckpt.pt"
    save(net, path)

    raw = torch.load(path, weights_only=True)
    raw["architecture"]["planes"] = PLANES + 1
    torch.save(raw, path)

    with pytest.raises(ValueError, match="incompatible with this encoding"):
        load(path, CPU)


def test_load_rejects_weights_that_contradict_the_recorded_architecture(tmp_path, net):
    """A checkpoint claiming a width its own tensors don't have."""
    path = tmp_path / "ckpt.pt"
    save(net, path)

    raw = torch.load(path, weights_only=True)
    raw["architecture"]["channels"] = CHANNELS // 2
    torch.save(raw, path)

    with pytest.raises(ValueError, match="do not match its recorded architecture"):
        load(path, CPU)


def test_load_accepts_a_legitimately_different_size(tmp_path):
    """Width and depth come from the checkpoint: training a smaller or larger net later is
    legitimate, and a stricter check would reject a valid file."""
    torch.manual_seed(0)
    small = Connect4Net(channels=CHANNELS // 2, blocks=BLOCKS // 2).eval()
    path = tmp_path / "small.pt"

    save(small, path)
    loaded = load(path, CPU)

    assert loaded.channels == CHANNELS // 2
    assert loaded.blocks_count == BLOCKS // 2
    assert count_parameters(loaded) == count_parameters(small)


# --------------------------------------------------------------------------
# size and device
# --------------------------------------------------------------------------

def test_parameter_count_is_in_a_sane_range(net):
    """Bounded on both sides."""
    total = count_parameters(net)
    assert 50_000 < total < 2_000_000


def test_heads_are_a_small_fraction_of_the_model(net):
    """The 1x1 reductions exist so the heads don't dominate."""
    head_params = sum(
        p.numel()
        for name, p in net.named_parameters()
        if name.startswith(("policy_", "value_"))
    )
    assert head_params < 0.05 * count_parameters(net)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_works_on_cuda():
    """Inputs must be moved to the network's device, not assumed to be on CPU."""
    torch.manual_seed(0)
    cuda_net = Connect4Net().to(torch.device("cuda")).eval()

    priors, value = predict(cuda_net, sample_position(), PLAYER_R)

    assert isinstance(priors, np.ndarray)
    assert isinstance(value, float)
    assert priors.sum() == pytest.approx(1.0)
