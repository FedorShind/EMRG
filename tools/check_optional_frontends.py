"""Isolated optional frontend install and smoke checker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_MARKER = "EMRG_OPTIONAL_FRONTEND_RESULT="
EXTRA_ORDER = ("braket", "pennylane", "qibo", "pyquil", "frontends")
SMOKE_TARGETS_BY_EXTRA = {
    "braket": ("braket",),
    "pennylane": ("pennylane",),
    "qibo": ("qibo",),
    "pyquil": ("pyquil",),
    "frontends": ("braket", "pennylane", "qibo", "pyquil"),
}

STAGES = (
    "install",
    "import",
    "build",
    "detect",
    "analyze",
    "generate",
    "codegen",
    "preview",
)

SMOKE_SCRIPT = r"""
from __future__ import annotations

import json
import sys
import traceback
import warnings

from emrg import Frontend, detect_frontend, generate_recipe
from emrg.analyzer import analyze_circuit
from emrg.frontends import FrontendConversionError
from emrg.preview import run_preview

RESULT_MARKER = "EMRG_OPTIONAL_FRONTEND_RESULT="
STAGES = (
    "import",
    "build",
    "detect",
    "analyze",
    "generate",
    "codegen",
    "preview",
)


def _new_row(frontend: str) -> dict[str, object]:
    row = {stage: "not_run" for stage in STAGES}
    row.update(
        {
            "frontend": frontend,
            "status": "failed",
            "taxonomy": None,
            "message": "",
        }
    )
    return row


def _message(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _fail(
    row: dict[str, object],
    stage: str,
    taxonomy: str,
    exc: BaseException | str,
) -> dict[str, object]:
    row[stage] = "failed"
    row["status"] = "failed"
    row["taxonomy"] = taxonomy
    row["message"] = str(exc) if isinstance(exc, str) else _message(exc)
    return row


def _build_braket(row: dict[str, object]):
    try:
        import braket  # noqa: F401
        from braket.circuits import Circuit
    except Exception as exc:
        return None, _fail(row, "import", "import_failure", exc)
    row["import"] = "passed"
    try:
        circuit = Circuit().h(0).cnot(0, 1)
    except Exception as exc:
        return None, _fail(row, "build", "test_assumption_failure", exc)
    row["build"] = "passed"
    return circuit, row


def _build_pennylane(row: dict[str, object]):
    try:
        import pennylane as qml
    except Exception as exc:
        return None, _fail(row, "import", "import_failure", exc)
    row["import"] = "passed"
    try:
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
    except Exception as exc:
        return None, _fail(row, "build", "test_assumption_failure", exc)
    row["build"] = "passed"
    return tape, row


def _build_pyquil(row: dict[str, object]):
    try:
        from pyquil import Program
        from pyquil.gates import CNOT, H
    except Exception as exc:
        return None, _fail(row, "import", "import_failure", exc)
    row["import"] = "passed"
    try:
        program = Program(H(0), CNOT(0, 1))
    except Exception as exc:
        return None, _fail(row, "build", "test_assumption_failure", exc)
    row["build"] = "passed"
    return program, row


def _build_qibo(row: dict[str, object]):
    try:
        from qibo import gates
        from qibo.models import Circuit
    except Exception as exc:
        return None, _fail(row, "import", "import_failure", exc)
    row["import"] = "passed"
    try:
        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.CNOT(0, 1))
    except Exception as exc:
        return None, _fail(row, "build", "test_assumption_failure", exc)
    row["build"] = "passed"
    return circuit, row


BUILDERS = {
    "braket": _build_braket,
    "pennylane": _build_pennylane,
    "pyquil": _build_pyquil,
    "qibo": _build_qibo,
}


def _run_frontend(frontend: str) -> dict[str, object]:
    row = _new_row(frontend)
    circuit, row = BUILDERS[frontend](row)
    if circuit is None:
        return row

    expected = getattr(Frontend, frontend.upper())
    try:
        detected = detect_frontend(circuit)
        if detected is not expected:
            raise AssertionError(f"detected {detected!r}, expected {expected!r}")
    except Exception as exc:
        return _fail(row, "detect", "frontend_detection_failure", exc)
    row["detect"] = "passed"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            features = analyze_circuit(circuit)
        if features.frontend != frontend:
            raise AssertionError(f"features.frontend={features.frontend!r}")
        if features.analysis_basis != "cirq-normalized":
            raise AssertionError(
                f"features.analysis_basis={features.analysis_basis!r}"
            )
        if features.num_qubits < 2 or features.total_gate_count <= 0:
            raise AssertionError(f"unexpected features: {features!r}")
    except FrontendConversionError as exc:
        return _fail(row, "analyze", "mitiq_conversion_failure", exc)
    except Exception as exc:
        return _fail(row, "analyze", "analyzer_failure", exc)
    row["analyze"] = "passed"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_recipe(circuit)
    except FrontendConversionError as exc:
        return _fail(row, "generate", "mitiq_conversion_failure", exc)
    except Exception as exc:
        return _fail(row, "generate", "recipe_generation_failure", exc)
    row["generate"] = "passed"

    try:
        compile(result.code, f"<emrg-{frontend}-generated>", "exec")
    except Exception as exc:
        return _fail(row, "codegen", "codegen_failure", exc)
    row["codegen"] = "passed"

    try:
        preview = run_preview(circuit, result.recipe)
        expected_warning = f"Preview for {frontend} inputs is not supported yet"
        if preview.ideal_value is not None:
            raise AssertionError("preview unexpectedly produced simulation values")
        if preview.warning is None or expected_warning not in preview.warning:
            raise AssertionError(f"unexpected preview warning: {preview.warning!r}")
    except Exception as exc:
        return _fail(row, "preview", "preview_behavior_failure", exc)
    row["preview"] = "passed"
    row["status"] = "passed"
    row["message"] = ""
    return row


def main(argv: list[str]) -> int:
    results = [_run_frontend(frontend) for frontend in argv]
    print(RESULT_MARKER + json.dumps(results, sort_keys=True))
    return 1 if any(row["status"] != "passed" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check EMRG optional frontend extras in isolated venvs."
    )
    parser.add_argument("--extra", choices=EXTRA_ORDER)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--keep-venvs", action="store_true")
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args(argv)

    if args.all == bool(args.extra):
        parser.error("Provide exactly one of --all or --extra.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    return args


def _selected_extras(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(EXTRA_ORDER)
    return [args.extra]


def _new_row(extra: str, frontend: str) -> dict[str, Any]:
    row: dict[str, Any] = {stage: "not_run" for stage in STAGES}
    row.update(
        {
            "extra": extra,
            "frontend": frontend,
            "status": "failed",
            "taxonomy": None,
            "message": "",
        }
    )
    return row


def _failure_rows(
    extra: str,
    taxonomy: str,
    message: str,
) -> list[dict[str, Any]]:
    rows = []
    for frontend in SMOKE_TARGETS_BY_EXTRA[extra]:
        row = _new_row(extra, frontend)
        row["install"] = "failed"
        row["taxonomy"] = taxonomy
        row["message"] = message
        rows.append(row)
    return rows


def _install_failure_rows(extra: str, message: str) -> list[dict[str, Any]]:
    return _failure_rows(extra, "dependency_resolution_failure", message)


def _python_executable(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _short_output(result: subprocess.CompletedProcess[str] | SimpleNamespace) -> str:
    output = "\n".join(
        part
        for part in (getattr(result, "stdout", ""), getattr(result, "stderr", ""))
        if part
    ).strip()
    return output[-4000:] if output else f"command exited {result.returncode}"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_PROGRESS_BAR"] = "off"
    return env


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str] | SimpleNamespace:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=_subprocess_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return SimpleNamespace(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=(stderr + f"\nTimed out after {timeout} seconds.").strip(),
        )


def _parse_smoke_results(output: str) -> list[dict[str, Any]]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line.removeprefix(RESULT_MARKER))
    raise ValueError("Smoke script did not emit a result marker.")


def _write_smoke_script(venv_path: Path) -> Path:
    smoke_path = venv_path / "smoke_optional_frontends.py"
    smoke_path.write_text(
        textwrap.dedent(SMOKE_SCRIPT).strip() + "\n", encoding="utf-8"
    )
    return smoke_path


def _remove_tree(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to remove path outside tmp root: {path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _run_extra(
    extra: str,
    *,
    tmp_root: Path,
    timeout: int,
    keep_venvs: bool,
) -> list[dict[str, Any]]:
    venv_path = tmp_root / f"emrg-optional-{extra}"
    if venv_path.exists():
        _remove_tree(venv_path, tmp_root)

    create = _run_command(
        [sys.executable, "-m", "venv", str(venv_path)],
        cwd=ROOT,
        timeout=timeout,
    )
    if create.returncode != 0:
        return _failure_rows(extra, "platform_limitation", _short_output(create))

    python = _python_executable(venv_path)
    upgrade = _run_command(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=ROOT,
        timeout=timeout,
    )
    if upgrade.returncode != 0:
        rows = _install_failure_rows(extra, _short_output(upgrade))
    else:
        install = _run_command(
            [str(python), "-m", "pip", "install", "-e", f".[{extra}]"],
            cwd=ROOT,
            timeout=timeout,
        )
        if install.returncode != 0:
            rows = _install_failure_rows(extra, _short_output(install))
        else:
            smoke_path = _write_smoke_script(venv_path)
            targets = list(SMOKE_TARGETS_BY_EXTRA[extra])
            smoke = _run_command(
                [str(python), str(smoke_path), *targets],
                cwd=ROOT,
                timeout=timeout,
            )
            smoke_output = "\n".join(
                part for part in (smoke.stdout, smoke.stderr) if part
            )
            try:
                rows = _parse_smoke_results(smoke_output)
            except (json.JSONDecodeError, ValueError) as exc:
                rows = _failure_rows(
                    extra,
                    "test_assumption_failure",
                    f"{exc}\n{_short_output(smoke)}",
                )
            for row in rows:
                row["extra"] = extra
                row["install"] = "passed"

    for row in rows:
        row["venv"] = str(venv_path) if keep_venvs else ""

    if not keep_venvs:
        _remove_tree(venv_path, tmp_root)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "extra",
        "frontend",
        "install",
        "import",
        "build",
        "detect",
        "analyze",
        "generate",
        "codegen",
        "preview",
        "status",
        "taxonomy",
    ]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            " | ".join(
                str(row.get(column, "")).ljust(widths[column]) for column in columns
            )
        )


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    selected = _selected_extras(args)

    def run_with_tmp(tmp_root: Path) -> None:
        tmp_root.mkdir(parents=True, exist_ok=True)
        for extra in selected:
            print(f"Checking emrg[{extra}]...")
            extra_rows = _run_extra(
                extra,
                tmp_root=tmp_root,
                timeout=args.timeout,
                keep_venvs=args.keep_venvs,
            )
            rows.extend(extra_rows)
            if args.fail_fast and any(row["status"] != "passed" for row in extra_rows):
                break

    if args.tmp_dir is None:
        with tempfile.TemporaryDirectory(prefix="emrg-optional-frontends-") as tmp:
            run_with_tmp(Path(tmp))
    else:
        run_with_tmp(args.tmp_dir)

    if rows:
        _print_table(rows)
    if args.json_output is not None:
        _write_json(args.json_output, rows)

    failed = any(row["status"] != "passed" for row in rows)
    return 1 if args.require_all and failed else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
