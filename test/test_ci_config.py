"""Tests that CI workflow and pre-commit config are present and well-formed."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"


def test_ci_workflow_exists() -> None:
    assert WORKFLOW.exists(), f"missing CI workflow at {WORKFLOW}"


def test_ci_workflow_uses_uv_and_pytest() -> None:
    text = WORKFLOW.read_text()
    assert "uv" in text
    assert "pytest" in text
    assert "actions/checkout" in text


def test_ci_workflow_pins_python_313() -> None:
    text = WORKFLOW.read_text()
    assert "3.13" in text, "CI workflow does not reference Python 3.13"


def test_ci_skips_broken_hypergraph() -> None:
    text = WORKFLOW.read_text()
    assert "test_hypergraph_control" in text and "--ignore" in text, (
        "CI must pass --ignore=test/test_hypergraph_control.py while PyPI hgx is broken"
    )


def test_precommit_config_exists() -> None:
    assert PRECOMMIT.exists()


def test_precommit_uses_ruff() -> None:
    text = PRECOMMIT.read_text()
    assert "astral-sh/ruff-pre-commit" in text
