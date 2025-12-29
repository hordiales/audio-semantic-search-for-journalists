#!/usr/bin/env python3
"""Lightweight orchestrator to execute test scripts by category."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from tests.common.path_utils import PROJECT_ROOT

CATALOG_PATH = Path(__file__).with_name("test_catalog.json")


@dataclass
class TestEntry:
    module: str
    description: str


def load_catalog() -> dict[str, list[TestEntry]]:
    raw_catalog = json.loads(CATALOG_PATH.read_text())
    catalog: dict[str, list[TestEntry]] = {}
    for category, entries in raw_catalog.items():
        catalog[category] = [TestEntry(**entry) for entry in entries]
    return catalog


def iter_selected_categories(catalog: dict[str, list[TestEntry]], selections: Iterable[str]) -> list[TestEntry]:
    ordered: list[TestEntry] = []
    for category in selections:
        ordered.extend(catalog.get(category, []))
    return ordered


def run_module(entry: TestEntry) -> int:
    header = f"\n{'=' * 80}\n🧪 Running {entry.module}\n{entry.description}\n{'=' * 80}"
    print(header)
    proc = subprocess.run([sys.executable, "-m", entry.module], cwd=PROJECT_ROOT)
    if proc.returncode == 0:
        print(f"✅ {entry.module} completed successfully")
    else:
        print(f"❌ {entry.module} failed (exit code {proc.returncode})")
    return proc.returncode


def list_catalog(catalog: dict[str, list[TestEntry]]) -> None:
    for category, entries in catalog.items():
        print(f"\n[{category}] ({len(entries)} tests)")
        for entry in entries:
            print(f" - {entry.module}: {entry.description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project tests by category")
    parser.add_argument(
        "categories",
        nargs="*",
        help="Categories to run (default: unit integration functional). Use 'all' to run every entry."
    )
    parser.add_argument("--list", action="store_true", help="List available tests and exit")
    return parser.parse_args()


def main() -> int:
    catalog = load_catalog()
    args = parse_args()

    if args.list:
        list_catalog(catalog)
        return 0

    if not args.categories:
        selections = ["unit", "integration", "functional"]
    else:
        if "all" in args.categories:
            selections = list(catalog.keys())
        else:
            selections = args.categories

    missing = [cat for cat in selections if cat not in catalog]
    if missing:
        print(f"⚠️  Unknown categories requested: {', '.join(missing)}")
        print(f"   Available: {', '.join(sorted(catalog.keys()))}")
        return 1

    tests_to_run = iter_selected_categories(catalog, selections)
    if not tests_to_run:
        print("No tests selected.")
        return 0

    failures = 0
    for entry in tests_to_run:
        result = run_module(entry)
        if result != 0:
            failures += 1

    print("\n" + "=" * 80)
    print(f"✅ Completed with {len(tests_to_run) - failures} successes and {failures} failures")
    print("=" * 80)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
