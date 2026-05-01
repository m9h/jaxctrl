"""Tests that user-facing install extras in pyproject.toml are documented in README.md."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"

USER_FACING_EXTRAS = {"solvers", "hypergraph", "diffrax"}


def _load_optional_dependencies() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def test_install_section_present() -> None:
    assert "## Installation" in README.read_text()


def test_user_facing_extras_documented() -> None:
    optional_deps = _load_optional_dependencies()
    for extra in USER_FACING_EXTRAS:
        assert extra in optional_deps, f"{extra} missing from pyproject extras"
    readme_text = README.read_text()
    missing = [f"[{e}]" for e in USER_FACING_EXTRAS if f"[{e}]" not in readme_text]
    assert not missing, f"Undocumented extras in README: {missing}"
