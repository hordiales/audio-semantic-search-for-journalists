from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys


def _find_project_root() -> Path:
    """Locate the repository root by looking for the src directory."""
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError("Unable to locate project root (missing 'src' directory)")


PROJECT_ROOT: Path = _find_project_root()
TESTS_ROOT: Path = PROJECT_ROOT / "tests"
SRC_ROOT: Path = PROJECT_ROOT / "src"


def ensure_sys_path(entries: Iterable[Path]) -> None:
    """Add the provided directories to sys.path if missing."""
    for entry in entries:
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)


def artifacts_dir(name: str) -> Path:
    """Return the directory that should store artifacts for a test and ensure it exists."""
    target = TESTS_ROOT / "artifacts" / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def resources_dir() -> Path:
    """Return the directory containing shared test resources."""
    return TESTS_ROOT / "resources"


# Ensure src and tests roots are importable by default.
ensure_sys_path([SRC_ROOT, TESTS_ROOT])

