# chess960-nn — Hugging Face Spaces Docker deployment.
#
# Runs the FastAPI dashboard (overview + play + watch + training metrics) on
# CPU at reduced MCTS sims for a usable demo on free tier (2 vCPU / 16 GB).
# The model checkpoint is fetched at build time from a GitHub Release so the
# git repo stays lean.

FROM python:3.11-slim

# System dependencies:
#   - curl + ca-certificates: download the model checkpoint
#   - stockfish: backend engine for the Watch view (NN vs Stockfish)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates stockfish && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# CPU-only torch first. This pin keeps the install ~200 MB instead of the
# ~2 GB CUDA wheel that pyproject.toml would otherwise resolve.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

# Remaining runtime deps. Kept lean — no torchvision, no aiosqlite, no dev tools.
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Source code, scripts, and frontend.
COPY src ./src
COPY scripts ./scripts
COPY web ./web

# Make `import chess960_nn` work without `pip install -e .` (avoids re-resolving
# the CUDA-torch source declared in pyproject.toml).
ENV PYTHONPATH=/app/src

# Trained checkpoint, downloaded at build time. Override on rebuild with
# `--build-arg MODEL_URL=... --build-arg MODEL_SHA256=...`.
ARG MODEL_URL=https://github.com/Vitaliy-Pikalo/chess960-nn/releases/download/v0.3-demo/final_best.pt
ARG MODEL_SHA256=2c189aaf5a555b999694a592e4310f04a308f0541f5ad097430c08f4bc0fc8b5
RUN mkdir -p /app/runs/rl-loop-001 && \
    curl -fL --retry 3 -o /app/runs/rl-loop-001/final_best.pt "$MODEL_URL" && \
    echo "$MODEL_SHA256  /app/runs/rl-loop-001/final_best.pt" | sha256sum -c - && \
    ls -lh /app/runs/rl-loop-001/final_best.pt

# Hugging Face Spaces serves Docker apps on port 7860 by default.
EXPOSE 7860

# Lower sims for usable CPU speed (~2-3 s/move at 30 sims on 2 vCPUs).
# Local-GPU users can override via DEMO launcher.ps1 with --n-simulations 100.
CMD ["python", "scripts/dashboard.py", \
     "--runs-dir", "runs", \
     "--web-dir", "web", \
     "--checkpoint", "runs/rl-loop-001/final_best.pt", \
     "--stockfish", "/usr/games/stockfish", \
     "--n-simulations", "30", \
     "--device", "cpu", \
     "--host", "0.0.0.0", \
     "--port", "7860"]
