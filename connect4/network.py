"""The network: a shared conv trunk with a policy head and a value head.

    input   (batch, 2, 6, 7)
    policy  (batch, 7)   logits, one per column
    value   (batch,)     tanh, so between -1 and 1

Both are from the point of view of the player to move, same as the input. +1
means the player about to move is winning, whichever colour that is.

The policy head gives logits rather than probabilities. Softmax happens in
predict(), so training can use log_softmax directly.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.board import COLS, ROWS, Board
from connect4.encoding import ACTION_SIZE, INPUT_SHAPE, PLANES, encode

CHANNELS = 64
BLOCKS = 4

# Each head squashes the trunk down with a 1x1 conv before flattening. Going
# straight from channels*42 into a Linear works but the heads end up holding most
# of the parameters.
POLICY_CHANNELS = 2
VALUE_CHANNELS = 1
VALUE_HIDDEN = 64

ARCHITECTURE_KEYS = ("channels", "blocks", "planes", "action_size")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = CHANNELS) -> None:
        super().__init__()
        # padding=1 keeps the 6x7 shape; without it the map shrinks each block.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Skip goes in before the last ReLU. In-place += is fine here, same as
        # torchvision's ResNet.
        out += x
        return F.relu(out)


class Connect4Net(nn.Module):
    """forward(x: (B, 2, 6, 7)) -> policy (B, 7), value (B,)"""

    def __init__(self, channels: int = CHANNELS, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks

        self.stem_conv = nn.Conv2d(PLANES, channels, kernel_size=3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, POLICY_CHANNELS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(POLICY_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_CHANNELS * ROWS * COLS, ACTION_SIZE)

        self.value_conv = nn.Conv2d(channels, VALUE_CHANNELS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CHANNELS * ROWS * COLS, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            out = block(out)

        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy_logits = self.policy_fc(policy.flatten(1))

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = F.relu(self.value_fc1(value.flatten(1)))
        # squeeze so it's (B,) not (B, 1) - the second one silently broadcasts
        # against the targets and makes a (B, B) loss matrix.
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy_logits, value


@torch.no_grad()
def predict(net: Connect4Net, board: Board, player: str) -> tuple[np.ndarray, float]:
    """One position -> (probabilities over 7 columns, value).

    Illegal moves aren't masked here; the search does that, since it's the only
    caller that knows the position.
    """
    priors, values = predict_batch(net, [board], [player])
    return priors[0], float(values[0])


@torch.no_grad()
def predict_batch(
    net: Connect4Net, boards: list[Board], players: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Many positions at once -> (priors (N, 7), values (N,)).

    A batch of one is nearly all launch overhead, so batching is what makes
    self-play take minutes instead of an hour.
    """
    if len(boards) != len(players):
        raise ValueError(
            f"got {len(boards)} boards but {len(players)} players; "
            "each position needs the player about to move"
        )
    if not boards:
        raise ValueError("nothing to evaluate: boards is empty")

    # eval() matters: BatchNorm in train mode normalises a single position using
    # its own statistics, which gives nonsense.
    net.eval()
    device = next(net.parameters()).device

    batch = np.stack([encode(b, p) for b, p in zip(boards, players)])
    x = torch.from_numpy(batch).to(device)

    logits, values = net(x)
    priors = F.softmax(logits, dim=-1)

    # numpy on the way out, so nothing downstream has to care about devices.
    return priors.cpu().numpy(), values.cpu().numpy()


def architecture(net: Connect4Net) -> dict[str, int]:
    return {
        "channels": net.channels,
        "blocks": net.blocks_count,
        "planes": PLANES,
        "action_size": ACTION_SIZE,
    }


def save(net: Connect4Net, path: str | Path) -> None:
    """Save weights plus the architecture.

    state_dict rather than the module itself, so renaming the class later doesn't
    make old checkpoints unloadable.
    """
    torch.save(
        {"architecture": architecture(net), "state_dict": net.state_dict()},
        Path(path),
    )


def load(path: str | Path, device: torch.device | None = None) -> Connect4Net:
    """Load a checkpoint and return it in eval mode."""
    device = device if device is not None else default_device()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    saved = checkpoint["architecture"]

    # Width and depth come from the file, since training a bigger net later is
    # fine. planes/action_size can't change - that would be a different encoding.
    for key, current in (("planes", PLANES), ("action_size", ACTION_SIZE)):
        if saved.get(key) != current:
            raise ValueError(
                f"checkpoint is incompatible with this encoding: {key} is "
                f"{saved.get(key)!r} in the file, {current!r} in the code"
            )

    net = Connect4Net(channels=saved["channels"], blocks=saved["blocks"])

    # torch's error here is a wall of per-tensor shapes, so say what's wrong.
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint weights do not match its recorded architecture "
            f"{ {k: saved.get(k) for k in ARCHITECTURE_KEYS} }: {exc}"
        ) from exc

    return net.to(device).eval()


def count_parameters(net: Connect4Net) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
