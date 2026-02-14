"""Tests for emrg.cli -- Click command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from emrg import __version__
from emrg.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"

BELL_QASM = EXAMPLES_DIR / "bell_state.qasm"
GHZ_QASM = EXAMPLES_DIR / "ghz_state.qasm"
VQE_QASM = EXAMPLES_DIR / "simple_vqe.qasm"

# Inline QASM for stdin tests
BELL_QASM_STR = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Tests: top-level
# ---------------------------------------------------------------------------


class TestTopLevel:
    """Verify top-level group behaviour."""

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert f"emrg, version {__version__}" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "analyze" in result.output


# ---------------------------------------------------------------------------
# Tests: generate command
# ---------------------------------------------------------------------------


class TestGenerate:
    """Verify the generate command produces correct code."""

    def test_generate_bell(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(BELL_QASM)])
        assert result.exit_code == 0
        assert "LinearFactory" in result.output
        assert "execute_with_zne" in result.output
        assert "fold_global" in result.output

    def test_generate_ghz(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(GHZ_QASM)])
        assert result.exit_code == 0
        assert "execute_with_zne" in result.output
        # GHZ is shallow -> LinearFactory
        assert "LinearFactory" in result.output

    def test_generate_vqe(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(VQE_QASM)])
        assert result.exit_code == 0
        assert "execute_with_zne" in result.output
        # VQE has moderate depth, should get some factory
        assert "Factory" in result.output

    def test_generate_explain(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(BELL_QASM), "--explain"])
        assert result.exit_code == 0
        assert "Rationale:" in result.output
        assert "selected by EMRG heuristics" in result.output

    def test_generate_no_explain(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(BELL_QASM)])
        assert result.exit_code == 0
        assert "Rationale:" not in result.output

    def test_generate_output_file(self, runner: CliRunner, tmp_path: Path) -> None:
        out_file = tmp_path / "output.py"
        result = runner.invoke(main, ["generate", str(BELL_QASM), "-o", str(out_file)])
        assert result.exit_code == 0
        assert f"Written to {out_file}" in result.output
        content = out_file.read_text(encoding="utf-8")
        assert "LinearFactory" in content
        assert "execute_with_zne" in content

    def test_generate_circuit_name(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main, ["generate", str(BELL_QASM), "--circuit-name", "qc"]
        )
        assert result.exit_code == 0
        assert "    qc," in result.output

    def test_generate_default_circuit_name(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", str(BELL_QASM)])
        assert result.exit_code == 0
        assert "    circuit," in result.output

    def test_generate_stdin(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", "-"], input=BELL_QASM_STR)
        assert result.exit_code == 0
        assert "LinearFactory" in result.output
        assert "execute_with_zne" in result.output

    def test_generate_file_not_found(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", "nonexistent.qasm"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_generate_invalid_qasm(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", "-"], input="this is not valid qasm")
        assert result.exit_code != 0
        assert "Failed to parse QASM" in result.output


# ---------------------------------------------------------------------------
# Tests: analyze command
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Verify the analyze command displays circuit features."""

    def test_analyze_bell(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", str(BELL_QASM)])
        assert result.exit_code == 0
        assert "Circuit Analysis:" in result.output
        assert "Qubits:" in result.output
        assert "Depth:" in result.output
        assert "Noise category:" in result.output

    def test_analyze_bell_values(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", str(BELL_QASM)])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 qubits
        assert "low" in result.output  # noise category

    def test_analyze_json(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", str(BELL_QASM), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["num_qubits"] == 2
        assert data["noise_category"] == "low"
        assert data["has_measurements"] is True

    def test_analyze_json_has_all_keys(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", str(BELL_QASM), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        expected_keys = {
            "num_qubits",
            "depth",
            "gate_counts",
            "total_gate_count",
            "multi_qubit_gate_count",
            "single_qubit_gate_count",
            "num_parameters",
            "has_measurements",
            "estimated_noise_factor",
            "noise_category",
        }
        assert set(data.keys()) == expected_keys

    def test_analyze_json_types(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", str(BELL_QASM), "--json"])
        data = json.loads(result.output)
        assert isinstance(data["num_qubits"], int)
        assert isinstance(data["depth"], int)
        assert isinstance(data["gate_counts"], dict)
        assert isinstance(data["estimated_noise_factor"], float)
        assert isinstance(data["noise_category"], str)
        assert isinstance(data["has_measurements"], bool)

    def test_analyze_file_not_found(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", "nonexistent.qasm"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_analyze_stdin(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["analyze", "-"], input=BELL_QASM_STR)
        assert result.exit_code == 0
        assert "Qubits:" in result.output


# ---------------------------------------------------------------------------
# Tests: QASM 3 auto-detect
# ---------------------------------------------------------------------------


class TestQASMAutoDetect:
    """Verify QASM version auto-detection."""

    def test_qasm3_detected(self, runner: CliRunner) -> None:
        """QASM 3.0 header should be detected; since qiskit_qasm3_import
        is not installed, we expect a clear error message."""
        qasm3_str = (
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\nh q[0];\n'
        )
        result = runner.invoke(main, ["analyze", "-"], input=qasm3_str)
        # Should fail gracefully (no traceback), with a clear message
        assert result.exit_code != 0
        assert "QASM 3.0" in result.output or "qasm3" in result.output


# ---------------------------------------------------------------------------
# Tests: stdin edge cases
# ---------------------------------------------------------------------------


class TestStdinEdgeCases:
    """Edge cases for stdin input."""

    def test_empty_stdin(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["generate", "-"], input="")
        assert result.exit_code != 0
        assert "No QASM input" in result.output


# ---------------------------------------------------------------------------
# Tests: error handling edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Cover defensive error paths."""

    def test_generate_no_gates_circuit(self, runner: CliRunner) -> None:
        """Circuit with only measurements -> ValueError from analyzer."""
        qasm_only_measure = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(main, ["generate", "-"], input=qasm_only_measure)
        assert result.exit_code != 0
        assert "no gate operations" in result.output

    def test_analyze_no_gates_circuit(self, runner: CliRunner) -> None:
        """Same for analyze command."""
        qasm_only_measure = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(main, ["analyze", "-"], input=qasm_only_measure)
        assert result.exit_code != 0
        assert "no gate operations" in result.output

    def test_generate_output_to_invalid_path(self, runner: CliRunner) -> None:
        """Writing to an invalid path -> clean error."""
        result = runner.invoke(
            main,
            ["generate", str(BELL_QASM), "-o", "/nonexistent/dir/out.py"],
        )
        assert result.exit_code != 0
        assert "Cannot write to file" in result.output
