"""Verify PyTorch installation and GPU detection.

Run with: uv run python scripts/check_cuda.py
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is NOT available.")
        return 1

    print(f"CUDA built with: {torch.version.cuda}")
    print(f"cuDNN:           {torch.backends.cudnn.version()}")
    print(f"Device count:    {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(
            f"  Device {i}: {props.name} | "
            f"VRAM: {vram_gb:.2f} GB | "
            f"Compute capability: {props.major}.{props.minor}"
        )

    print("\nGPU smoke test...")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"  matmul OK: device={y.device}, shape={tuple(y.shape)}, dtype={y.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
