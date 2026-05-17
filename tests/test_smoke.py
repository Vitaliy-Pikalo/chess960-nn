"""Smoke tests: verify core deps installed correctly + GPU is alive."""

import chess
import torch


def test_python_chess_960():
    board = chess.Board(chess960=True)
    assert board.chess960 is True
    board.set_chess960_pos(518)
    assert board.is_valid()
    board2 = chess.Board(chess960=True)
    board2.set_chess960_pos(42)
    assert board2.is_valid()
    assert board2.chess960 is True


def test_torch_imports():
    assert torch.__version__


def test_cuda_available():
    assert torch.cuda.is_available(), (
        "CUDA must be available. Check NVIDIA driver + torch CUDA wheel."
    )


def test_gpu_matmul():
    x = torch.randn(512, 512, device="cuda")
    y = x @ x.T
    assert y.device.type == "cuda"
    assert y.shape == (512, 512)


def test_package_imports():
    import chess960_nn

    assert chess960_nn.__version__ == "0.1.0"
