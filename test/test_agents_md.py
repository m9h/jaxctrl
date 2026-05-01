"""Tests that AGENTS.md stays in sync with the codebase."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS = REPO_ROOT / "AGENTS.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
PKG_DIR = REPO_ROOT / "jaxctrl"


def _module_basenames() -> list[str]:
    return sorted(p.stem for p in PKG_DIR.glob("*.py"))


def test_module_map_lists_all_modules() -> None:
    text = AGENTS.read_text()
    missing = [m for m in _module_basenames() if m not in text]
    assert not missing, f"Modules missing from AGENTS.md Module Map: {missing}"


def test_dependencies_lists_optional_extras() -> None:
    with PYPROJECT.open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    text = AGENTS.read_text()
    missing = [name for name in extras if name not in text]
    assert not missing, f"Optional extras missing from AGENTS.md: {missing}"


def test_quick_start_no_absolute_user_path() -> None:
    text = AGENTS.read_text()
    bad = re.findall(r"/(?:home|Users)/[A-Za-z0-9_-]+", text)
    assert not bad, f"AGENTS.md contains absolute home paths: {bad}"
