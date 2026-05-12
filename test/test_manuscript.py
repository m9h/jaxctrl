"""Tests that manuscript section files exist with valid frontmatter."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTIONS_DIR = REPO_ROOT / "manuscript" / "sections"
README = REPO_ROOT / "README.md"

REQUIRED_KEYS = {"category", "section", "weight", "title", "status"}
SECTION_FILES = ["methods.md", "results.md", "discussion.md"]
EXPECTED_ORDER = ["introduction", "methods", "results", "discussion"]

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _read_frontmatter(path: Path) -> dict:
    text = path.read_text()
    match = FRONTMATTER_RE.match(text)
    assert match, f"{path} has no YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_section_files_exist() -> None:
    for fname in SECTION_FILES:
        assert (SECTIONS_DIR / fname).exists(), f"missing {fname}"


@pytest.mark.parametrize("fname", SECTION_FILES)
def test_frontmatter_required_fields(fname: str) -> None:
    fm = _read_frontmatter(SECTIONS_DIR / fname)
    missing = REQUIRED_KEYS - set(fm)
    assert not missing, f"{fname} missing frontmatter keys: {missing}"


@pytest.mark.parametrize("fname", SECTION_FILES)
def test_section_value_matches_filename(fname: str) -> None:
    fm = _read_frontmatter(SECTIONS_DIR / fname)
    expected = fname.removesuffix(".md")
    assert fm["section"] == expected, (
        f"{fname}: section={fm['section']}, expected {expected}"
    )


def test_weights_are_unique_and_ordered() -> None:
    paths = [README] + [SECTIONS_DIR / f for f in SECTION_FILES]
    weights = [_read_frontmatter(p)["weight"] for p in paths]
    sections = [_read_frontmatter(p)["section"] for p in paths]
    assert sections == EXPECTED_ORDER, f"section order mismatch: {sections}"
    assert weights == sorted(weights), f"weights not sorted: {weights}"
    assert len(set(weights)) == len(weights), f"weights not unique: {weights}"
