"""The policy + value network: one trunk, two heads.

This is what replaces the hand-written heuristic from Phase 2 and the random
rollout from Phase 3. Same slot in the same algorithm, learned instead of
designed.

    input   (batch, 2, 6, 7)   canonical encoding, see encoding.py
    trunk   a few residual conv blocks
    policy  (batch, 7)         logits over columns
    value   (batch,)           scalar in [-1, 1] via tanh

Both outputs are from the perspective of the player to move, because the input
is. A value of +1 means "the player about to move is winning", regardless of
colour — which is why the sign has to flip on every level of the PUCT backup in
step 3.

The policy head emits **logits**, not probabilities. Softmax is applied at the
call boundary in predict(), so training can use cross-entropy on raw logits
(numerically stabler) while the search receives a distribution.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.board import COLS, ROWS, Board
from connect4.encoding import ACTION_SIZE, INPUT_SHAPE, PLANES, encode

# Small by modern standards, and deliberately so: Connect-4 has 4.5e12 states but
# only 42 cells and 7 actions. A few hundred thousand parameters is plenty, and it
# keeps a self-play iteration to minutes rather than hours on one GPU.
CHANNELS = 64
BLOCKS = 4

# Head widths. Each head first reduces the trunk's CHANNELS down to a couple of
# channels with a 1x1 conv, then flattens. Flattening the full trunk instead
# would make the heads dominate the parameter count and leave no budget for the
# value head's hidden layer.
POLICY_CHANNELS = 2
VALUE_CHANNELS = 1
VALUE_HIDDEN = 64

# Written into every checkpoint and checked on load. Mismatched weights that load
# "successfully" give a network that runs and plays badly — much harder to
# diagnose than a failed load.
ARCHITECTURE_KEYS = ("channels", "blocks", "planes", "action_size")


def default_device() -> torch.device:
    """CUDA if available, else CPU. Kept in one place so tests can override it."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN, plus the skip connection, then ReLU.

    The skip is added *before* the final activation, not after — that ordering is
    what lets gradients bypass the block untouched.
    """

    def __init__(self, channels: int = CHANNELS) -> None:
        super().__init__()
        # Conv2d defaults to padding=0, which would shrink the feature map. We want
        # to preserve the spatial dimensions, so pad=1 for 3x3 kernels.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block's output for input `x`."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # In-place is safe here: BatchNorm's backward needs its input, not its
        # output, and addition's backward needs neither. Same as torchvision's
        # ResNet. The skip is added before the final ReLU, so gradients can pass
        # through the block untouched.
        out += x
        return F.relu(out)


class Connect4Net(nn.Module):
    """Residual trunk with a policy head and a value head.

    Shapes to hold onto, since almost every bug here is a shape bug:
        forward(x: (B, 2, 6, 7)) -> (policy_logits: (B, 7), value: (B,))

    The value head must squeeze its trailing dimension. Returning (B, 1) where
    (B,) is expected doesn't raise — it broadcasts against the (B,) targets and
    silently computes a (B, B) loss matrix.
    """

    def __init__(self, channels: int = CHANNELS, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks

        self.stem_conv = nn.Conv2d(PLANES, channels, kernel_size=3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        # Both heads reduce channels with a 1x1 conv *before* flattening. Going
        # straight from channels*ROWS*COLS to a Linear works, but it makes the
        # heads dominate the parameter count and leaves no room for a hidden
        # layer. Reducing first buys the value head real capacity for a fraction
        # of the weights.
        self.policy_conv = nn.Conv2d(channels, POLICY_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(POLICY_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_CHANNELS * ROWS * COLS, ACTION_SIZE)

        self.value_conv = nn.Conv2d(channels, VALUE_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CHANNELS * ROWS * COLS, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits: (B, 7), value: (B,)) for input `x`."""
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            out = block(out)

        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy_logits = self.policy_fc(policy.flatten(1))

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = F.relu(self.value_fc1(value.flatten(1)))
        # squeeze(-1) turns (B, 1) into (B,). Leaving it as (B, 1) does not raise:
        # it broadcasts against (B,) targets into a (B, B) loss matrix.
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy_logits, value


@torch.no_grad()
def predict(net: Connect4Net, board: Board, player: str) -> tuple[np.ndarray, float]:
    """Evaluate a single position. Returns (priors over 7 columns, value).

    Priors are softmaxed probabilities, NOT logits — mask_and_normalise() rejects
    negative input precisely to catch a mix-up here. Illegal moves are *not*
    masked at this level; the search does that, since only it knows the position's
    legal moves in context.

    Must set eval() mode. With BatchNorm in the trunk, a single-position forward
    pass in train() mode normalises using that one sample's own statistics, which
    produces garbage — and garbage that looks plausible, since the shapes are all
    correct.
    """
    priors, values = predict_batch(net, [board], [player])
    return priors[0], float(values[0])


@torch.no_grad()
def predict_batch(
    net: Connect4Net, boards: list[Board], players: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate many positions at once. Returns (priors: (N, 7), values: (N,)).

    Worth having from the start: a single-position forward pass is dominated by
    kernel-launch overhead, so batching is the difference between a self-play
    iteration taking minutes and taking an hour.
    """
    if len(boards) != len(players):
        raise ValueError(
            f"got {len(boards)} boards but {len(players)} players; "
            "each position needs the player about to move"
        )
    if not boards:
        raise ValueError("nothing to evaluate: boards is empty")

    net.eval()
    device = next(net.parameters()).device

    batch = np.stack([encode(b, p) for b, p in zip(boards, players)])
    x = torch.from_numpy(batch).to(device)

    logits, values = net(x)
    priors = F.softmax(logits, dim=-1)

    # Back to numpy at the boundary so no caller downstream has to be
    # device-aware or remember to detach.
    return priors.cpu().numpy(), values.cpu().numpy()


def architecture(net: Connect4Net) -> dict[str, int]:
    """The constants a checkpoint must agree with to be loadable."""
    return {
        "channels": net.channels,
        "blocks": net.blocks_count,
        "planes": PLANES,
        "action_size": ACTION_SIZE,
    }


def save(net: Connect4Net, path: str | Path) -> None:
    """Write the network's weights to `path`.

    Saves the state_dict, not the module object: pickling the module ties the file
    to this exact class definition, so any later refactor makes old checkpoints
    unloadable. The architecture constants ride along so load() can validate them.
    """
    torch.save(
        {"architecture": architecture(net), "state_dict": net.state_dict()},
        Path(path),
    )


def load(path: str | Path, device: torch.device | None = None) -> Connect4Net:
    """Rebuild a network from `path` and return it on `device` in eval() mode.

    Raises if the saved architecture doesn't match this module's constants.
    """
    device = device if device is not None else default_device()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    saved = checkpoint["architecture"]

    # Width and depth are read *from* the checkpoint, since training a wider or
    # deeper net later is legitimate. What cannot vary is the encoding contract:
    # different planes or action_size means the weights describe a different game
    # representation and are meaningless here.
    for key, current in (("planes", PLANES), ("action_size", ACTION_SIZE)):
        if saved.get(key) != current:
            raise ValueError(
                f"checkpoint is incompatible with this encoding: {key} is "
                f"{saved.get(key)!r} in the file, {current!r} in the code"
            )

    net = Connect4Net(channels=saved["channels"], blocks=saved["blocks"])

    # A checkpoint whose recorded channels/blocks disagree with its own weights
    # fails here. torch's message is a wall of per-tensor shape mismatches, so
    # restate it in terms of the thing that's actually wrong.
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint weights do not match its recorded architecture "
            f"{ {k: saved.get(k) for k in ARCHITECTURE_KEYS} }: {exc}"
        ) from exc
    # eval() before returning: a loaded net left in train() mode has predictions
    # that depend on batch composition, which is a nasty trap to hand a caller.
    return net.to(device).eval()


def count_parameters(net: Connect4Net) -> int:
    """Total trainable parameters. Useful for a sanity check on model size."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
