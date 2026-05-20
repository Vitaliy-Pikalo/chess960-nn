"""Policy/value ResNet for Chess960.

AlphaZero-style architecture:
- 8x8 board input with NUM_BOARD_PLANES feature planes (see encoding.py)
- Convolutional stem -> N residual blocks -> two heads
- Policy head: per-square 73 move-type logits, flattened to 4672 actions
- Value head: scalar in [-1, 1] (win probability from STM perspective)

Defaults are tuned for ~8GB VRAM (RTX 3060 Ti class). Tweak ``num_blocks`` and
``num_filters`` to scale up/down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess960_nn.encoding import (
    ACTION_SPACE_SIZE,
    NUM_BOARD_PLANES,
    NUM_MOVE_TYPES,
)

# ============================================================
# Config
# ============================================================


@dataclass
class ModelConfig:
    """Hyperparameters for the policy/value network."""

    num_blocks: int = 10
    num_filters: int = 192
    value_hidden: int = 256
    in_planes: int = NUM_BOARD_PLANES


# ============================================================
# Building blocks
# ============================================================


class _ConvBlock(nn.Module):
    """Conv2d + BN, optionally followed by ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        relu: bool = True,
    ):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class _ResBlock(nn.Module):
    """Residual block: (Conv-BN-ReLU) + (Conv-BN) + skip + ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _ConvBlock(channels, channels, kernel=3, relu=True)
        self.conv2 = _ConvBlock(channels, channels, kernel=3, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


# ============================================================
# Policy/value heads
# ============================================================


class _PolicyHead(nn.Module):
    """Conv(filters -> NUM_MOVE_TYPES, 1x1) + BN -> flatten to ACTION_SPACE_SIZE logits."""

    def __init__(self, in_filters: int):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, NUM_MOVE_TYPES, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(NUM_MOVE_TYPES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))  # (B, 73, 8, 8)
        # Reorder to (B, 64, 73) then flatten so action index =
        # from_square * 73 + action_type (matches encode_move).
        b = x.size(0)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, 73)
        return x.view(b, ACTION_SPACE_SIZE)


class _ValueHead(nn.Module):
    """Conv(filters -> 1, 1x1) + BN + ReLU -> MLP -> tanh scalar."""

    def __init__(self, in_filters: int, hidden: int):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))  # (B, 1, 8, 8)
        x = x.flatten(1)  # (B, 64)
        x = self.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)).squeeze(-1)  # (B,)


# ============================================================
# Full model
# ============================================================


class Chess960Net(nn.Module):
    """Policy/value network."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.stem = _ConvBlock(self.cfg.in_planes, self.cfg.num_filters, kernel=3, relu=True)
        self.blocks = nn.ModuleList(
            _ResBlock(self.cfg.num_filters) for _ in range(self.cfg.num_blocks)
        )
        self.policy_head = _PolicyHead(self.cfg.num_filters)
        self.value_head = _ValueHead(self.cfg.num_filters, self.cfg.value_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, NUM_BOARD_PLANES, 8, 8)`` float tensor.

        Returns:
            policy_logits: ``(B, ACTION_SPACE_SIZE)`` raw logits (no softmax).
            value: ``(B,)`` scalar in ``[-1, 1]``.
        """
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
