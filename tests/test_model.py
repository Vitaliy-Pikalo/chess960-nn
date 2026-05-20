"""Tests for the policy/value ResNet."""

from __future__ import annotations

import pytest
import torch

from chess960_nn.encoding import ACTION_SPACE_SIZE, NUM_BOARD_PLANES
from chess960_nn.model import Chess960Net, ModelConfig

# ============================================================
# CPU forward pass
# ============================================================


def test_forward_shapes_cpu():
    """A small model with a CPU forward pass yields the expected shapes."""
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg)
    net.eval()
    x = torch.randn(4, NUM_BOARD_PLANES, 8, 8)
    with torch.no_grad():
        policy, value = net(x)
    assert policy.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4,)


def test_value_in_tanh_range():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg)
    net.eval()
    x = torch.randn(8, NUM_BOARD_PLANES, 8, 8) * 5  # large inputs
    with torch.no_grad():
        _, value = net(x)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_logits_finite():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg)
    net.eval()
    x = torch.randn(2, NUM_BOARD_PLANES, 8, 8)
    with torch.no_grad():
        policy, _ = net(x)
    assert torch.isfinite(policy).all()


def test_default_param_count_within_target():
    """Default model has ~5-9M params (we target ~7M for 8GB VRAM)."""
    net = Chess960Net()  # default cfg: 10 blocks, 192 filters
    n = net.num_parameters()
    assert 4_000_000 <= n <= 10_000_000, f"unexpected param count: {n}"


# ============================================================
# Backward pass / training mode
# ============================================================


def test_backward_pass_runs():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg)
    net.train()
    x = torch.randn(4, NUM_BOARD_PLANES, 8, 8, requires_grad=False)
    policy, value = net(x)
    target_policy = torch.randint(0, ACTION_SPACE_SIZE, (4,))
    target_value = torch.tensor([1.0, -1.0, 0.0, 1.0])
    loss = (
        torch.nn.functional.cross_entropy(policy, target_policy)
        + torch.nn.functional.mse_loss(value, target_value)
    )
    loss.backward()
    # Verify some gradient flowed
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any((g.abs().sum() > 0) for g in grads)


# ============================================================
# GPU + mixed precision (skipped if no CUDA)
# ============================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_forward_on_gpu():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg).cuda()
    net.eval()
    x = torch.randn(4, NUM_BOARD_PLANES, 8, 8, device="cuda")
    with torch.no_grad():
        policy, value = net(x)
    assert policy.device.type == "cuda"
    assert value.device.type == "cuda"
    assert policy.shape == (4, ACTION_SPACE_SIZE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_forward_with_autocast():
    cfg = ModelConfig(num_blocks=2, num_filters=32)
    net = Chess960Net(cfg).cuda()
    net.eval()
    x = torch.randn(4, NUM_BOARD_PLANES, 8, 8, device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        policy, value = net(x)
    # Outputs stay finite under fp16 autocast
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_default_model_fits_on_gpu_with_batch():
    """Full default model + batch of 64 doesn't OOM."""
    net = Chess960Net().cuda()
    net.eval()
    x = torch.randn(64, NUM_BOARD_PLANES, 8, 8, device="cuda")
    with torch.no_grad():
        policy, value = net(x)
    assert policy.shape == (64, ACTION_SPACE_SIZE)
    assert value.shape == (64,)
    # free the model so it doesn't linger
    del net
    torch.cuda.empty_cache()
