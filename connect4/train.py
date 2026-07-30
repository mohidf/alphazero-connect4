"""Training: fit the network to self-play targets.

Loss is the sum of three terms:

    value    MSE between the value head and the game outcome z
    policy   cross-entropy between the policy head and the search's visit
             distribution pi
    L2       weight decay, applied through the optimiser

The policy term is a cross-entropy against a *distribution*, not a class label, so
it is computed as -(pi * log_softmax(logits)).sum(). torch's F.cross_entropy
expects integer labels and would quietly do the wrong thing here — it accepts soft
targets in recent versions, but spelling it out keeps the intent visible.
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

# Positions kept for training. Old data comes from a weaker network, so an
# unbounded buffer would keep dragging the policy toward obsolete targets.
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
    """A bounded FIFO of training samples."""

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._samples: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def extend(self, samples: list[Sample]) -> None:
        self._samples.extend(samples)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Sample]:
        """Draw `batch_size` samples without replacement where possible."""
        if not self._samples:
            raise ValueError("replay buffer is empty")
        size = min(batch_size, len(self._samples))
        indices = rng.choice(len(self._samples), size=size, replace=False)
        return [self._samples[int(i)] for i in indices]


def to_tensors(
    samples: list[Sample], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack samples into (states, policies, values) on `device`."""
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
    """Return (policy_loss, value_loss) for one batch.

    Cross-entropy against a soft target: -(pi * log q).sum over actions, averaged
    over the batch. log_softmax rather than log(softmax(...)) for numerical
    stability — the difference matters once logits grow during training.
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
    """One gradient step. Leaves the network in train() mode."""
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
    """Adam with weight decay — the L2 term of the AlphaZero loss."""
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
    """Run `steps` gradient steps and return the mean loss over them.

    Returns the network in eval() mode, because every other consumer — search,
    arena, checkpointing — needs it that way, and a network left in train() mode
    gives predictions that depend on batch composition.
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
