"""ADK entrypoint expected by agents-cli and Agent Runtime."""

from src.agent_service.agent import app, root_agent

__all__ = ["app", "root_agent"]
