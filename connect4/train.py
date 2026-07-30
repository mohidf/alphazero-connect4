"""Fitting the network to the self-play data.

Loss has three parts: MSE on the value against the game result, cross-entropy on
the policy against the search's visit counts, and weight decay through the
optimiser.

The policy target is a distribution rather than a label, so the cross-entropy is
written out as -(pi * log_softmax(logits)).sum() instead of using
F.cross_entropy, which is built for integer labels.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from connect4.network import Connect4Net
from connect4.selfplay import Sample

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256

# Old positions came from a worse network, so the buffer is capped rather than
# keeping everything forever.
BUFFER_SIZE = 60_000


@dataclass
class LossBreakdown:
    total: float
    policy: float
    value: float

    def __str__(self) -> str:
        return (
            f"total {self.total:.4f}  policy {self.policy:.4f}  value {self.value:.4f}"
        )


class ReplayBuffer:
    """Fixed-size pool of recent training samples."""

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._samples: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def extend(self, samples: list[Sample]) -> None:
        self._samples.extend(samples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Sample]:
        """Random batch, without replacement if there is enough data."""
        if not self._samples:
            raise ValueError("replay buffer is empty")
        size = min(batch_size, len(self._samples))
        indices = rng.choice(len(self._samples), size=size, replace=False)
        return [self._samples[int(i)] for i in indices]


def to_tensors(
    samples: list[Sample], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples -> tensors."""
    states = torch.from_numpy(np.stack([s.state for s in samples])).to(device)
    policies = torch.from_numpy(np.stack([s.policy for s in samples])).to(device)
    values = torch.tensor(
        [s.value for s in samples], dtype=torch.float32, device=device
    )
    return states, policies, values


def losses(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy and value loss for one batch.

    log_softmax rather than log(softmax(...)), which starts to matter once the
    logits get large.
    """
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, value_target)
    return policy_loss, value_loss


def train_step(
    net: Connect4Net,
    optimizer: torch.optim.Optimizer,
    samples: list[Sample],
    device: torch.device,
) -> LossBreakdown:
    """One gradient step."""
    net.train()
    states, policy_target, value_target = to_tensors(samples, device)

    policy_logits, value = net(states)
    policy_loss, value_loss = losses(policy_logits, value, policy_target, value_target)
    total = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()

    return LossBreakdown(
        total=float(total.detach()),
        policy=float(policy_loss.detach()),
        value=float(value_loss.detach()),
    )


def make_optimizer(
    net: Connect4Net,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """Adam. weight_decay is the L2 term of the loss."""
    return torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def train_epoch(
    net: Connect4Net,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    steps: int,
    batch_size: int = BATCH_SIZE,
    rng: np.random.Generator | None = None,
) -> LossBreakdown:
    """Run `steps` gradient steps, return the average loss.

    Leaves the network in eval mode, since everything else (search, arena,
    saving) expects that.
    """
    rng = rng if rng is not None else np.random.default_rng()
    totals = np.zeros(3)

    for _ in range(steps):
        batch = buffer.sample(batch_size, rng)
        result = train_step(net, optimizer, batch, device)
        totals += (result.total, result.policy, result.value)

    net.eval()
    mean = totals / max(steps, 1)
    return LossBreakdown(total=mean[0], policy=mean[1], value=mean[2])
