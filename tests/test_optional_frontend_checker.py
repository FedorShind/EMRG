"""Unit tests for the optional frontend checker helper logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_checker():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_optional_frontends.py"
    spec = importlib.util.spec_from_file_location("check_optional_frontends", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_extras_preserves_harsh_check_order() -> None:
    checker = _load_checker()

    args = SimpleNamespace(all=True, extra=None)

    assert checker._selected_extras(args) == [
        "braket",
        "pennylane",
        "qibo",
        "pyquil",
        "frontends",
    ]


def test_selected_extras_accepts_single_extra() -> None:
    checker = _load_checker()

    args = SimpleNamespace(all=False, extra="qibo")

    assert checker._selected_extras(args) == ["qibo"]


def test_install_failure_rows_cover_combined_extra_frontends() -> None:
    checker = _load_checker()

    rows = checker._install_failure_rows("frontends", "resolver failed")

    assert [row["frontend"] for row in rows] == [
        "braket",
        "pennylane",
        "qibo",
        "pyquil",
    ]
    assert all(row["install"] == "failed" for row in rows)
    assert all(row["taxonomy"] == "dependency_resolution_failure" for row in rows)
    assert all("resolver failed" in row["message"] for row in rows)


def test_parse_smoke_results_uses_marker_line() -> None:
    checker = _load_checker()
    output = (
        "package log line\n"
        'EMRG_OPTIONAL_FRONTEND_RESULT=[{"frontend": "braket", "status": "passed"}]\n'
    )

    assert checker._parse_smoke_results(output) == [
        {"frontend": "braket", "status": "passed"}
    ]


def test_subprocess_env_uses_utf8_and_disables_pip_progress() -> None:
    checker = _load_checker()

    env = checker._subprocess_env()

    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"
    assert env["PIP_PROGRESS_BAR"] == "off"
