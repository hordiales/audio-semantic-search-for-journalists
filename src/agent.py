"""ADK entrypoint expected by agents-cli and Agent Runtime."""

# Initialize PyTorch before FAISS and limit OpenBLAS/OpenMP threads to avoid
# segfaults on Apple Silicon when laion_clap loads alongside FAISS.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from src.agent_service.agent import app, root_agent

__all__ = ["app", "root_agent"]
