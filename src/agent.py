"""ADK entrypoint expected by agents-cli and Agent Runtime."""

# Initialize PyTorch before FAISS and limit OpenBLAS/OpenMP threads to avoid
# segfaults on Apple Silicon when laion_clap loads alongside FAISS.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Only local runs execute the models in this process. The deployed agent image is
# built with `--no-group ml` and delegates retrieval to the Cloud Run search
# service, so PyTorch is legitimately absent there.
try:
    import torch
except ModuleNotFoundError:
    pass
else:
    torch.set_num_threads(1)

from src.agent_service.agent import app, root_agent

__all__ = ["app", "root_agent"]
