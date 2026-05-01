"""Tests for the autoresearch experiment harness.

The experiment script must emit exactly one machine-parseable RESULT line
to stdout matching the contract in autoresearch/program.md.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = REPO_ROOT / "autoresearch" / "experiment.py"


def _run_experiment() -> str:
    proc = subprocess.run(
        ["uv", "run", "python", str(EXPERIMENT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    result_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT|")]
    assert len(result_lines) == 1, (
        f"Expected exactly one RESULT line, got {len(result_lines)}.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return result_lines[0]


def test_experiment_runs_and_emits_result_line() -> None:
    line = _run_experiment()
    fields = line.split("|")
    assert len(fields) == 5, f"RESULT line must have 5 fields, got {len(fields)}: {line}"


def test_status_is_not_placeholder() -> None:
    fields = _run_experiment().split("|")
    assert fields[3] != "placeholder", "experiment.py is still in placeholder state"


def test_metric_is_finite() -> None:
    fields = _run_experiment().split("|")
    metric = float(fields[1])
    assert math.isfinite(metric), f"metric is not finite: {metric}"


def test_params_is_valid_json_with_walltimes() -> None:
    fields = _run_experiment().split("|")
    params = json.loads(fields[2])
    assert "wall_time_jaxctrl" in params, params
    assert "wall_time_scipy" in params, params
    assert params["wall_time_jaxctrl"] > 0, params
    assert params["wall_time_scipy"] > 0, params
