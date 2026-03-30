"""Tests for the preview simulation engine.

Covers:
- run_preview() with ZNE and PEC paths
- Circuit too large (>10 qubits) warning
- Observable parsing and edge cases
- format_preview() output structure
- generate_recipe(preview=True) API integration
- CLI --preview flag integration
- Edge cases: single qubit, no multi-qubit gates, very shallow
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from qiskit import QuantumCircuit

from emrg import generate_recipe
from emrg.analyzer import analyze_circuit
from emrg.cli import main
from emrg.heuristics import recommend
from emrg.preview import (
    PreviewResult,
    _parse_observable,
    format_preview,
    run_preview,
)

try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

pytestmark = pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture()
def vqe_4q_circuit() -> QuantumCircuit:
    """4-qubit VQE-style circuit (shallow enough for PEC)."""
    qc = QuantumCircuit(4)
    for q in range(4):
        qc.ry(0.5, q)
    for q in range(3):
        qc.cx(q, q + 1)
    qc.measure_all()
    return qc


@pytest.fixture()
def large_circuit() -> QuantumCircuit:
    """15-qubit circuit -- exceeds preview limit."""
    qc = QuantumCircuit(15)
    for i in range(14):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


@pytest.fixture()
def single_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# A. Unit tests for run_preview()
# ---------------------------------------------------------------------------


class TestRunPreviewZNE:
    """ZNE preview execution path."""

    def test_bell_state_returns_populated_result(self, bell_circuit):
        recipe = recommend(analyze_circuit(bell_circuit))
        result = run_preview(bell_circuit, recipe)

        assert isinstance(result, PreviewResult)
        assert result.ideal_value is not None
        assert result.noisy_value is not None
        assert result.mitigated_value is not None
        assert result.noisy_error is not None
        assert result.mitigated_error is not None
        assert result.error_reduction is not None
        assert result.error_reduction >= 0
        assert result.technique == "ZNE"
        assert result.noise_level == 0.01
        assert result.observable == "Z0"
        assert result.num_qubits == 2

    def test_bell_state_mitigated_closer_to_ideal(self, bell_circuit):
        recipe = recommend(analyze_circuit(bell_circuit))
        result = run_preview(bell_circuit, recipe)

        # Mitigated error should be <= noisy error (or very close).
        assert result.mitigated_error <= result.noisy_error + 0.05

    def test_custom_noise_level(self, bell_circuit):
        recipe = recommend(analyze_circuit(bell_circuit))
        result = run_preview(bell_circuit, recipe, noise_level=0.05)

        assert result.noise_level == 0.05
        assert result.ideal_value is not None

    def test_custom_observable_z1(self, bell_circuit):
        recipe = recommend(analyze_circuit(bell_circuit))
        result = run_preview(bell_circuit, recipe, observable="Z1")

        assert result.observable == "Z1"
        assert result.ideal_value is not None

    def test_zz_observable(self, bell_circuit):
        recipe = recommend(analyze_circuit(bell_circuit))
        result = run_preview(bell_circuit, recipe, observable="ZZ")

        assert result.observable == "ZZ"
        assert result.ideal_value is not None


class TestRunPreviewPEC:
    """PEC preview execution path."""

    def test_pec_path_works(self, vqe_4q_circuit):
        features = analyze_circuit(
            vqe_4q_circuit, noise_model_available=True
        )
        recipe = recommend(features, technique="pec")
        result = run_preview(vqe_4q_circuit, recipe)

        assert result.technique == "PEC"
        assert result.ideal_value is not None
        assert result.mitigated_value is not None
        assert result.warning is not None  # PEC variance note
        assert "approximate" in result.warning.lower()

    def test_pec_with_custom_noise(self, vqe_4q_circuit):
        features = analyze_circuit(
            vqe_4q_circuit, noise_model_available=True
        )
        recipe = recommend(features, technique="pec")
        result = run_preview(vqe_4q_circuit, recipe, noise_level=0.02)

        assert result.noise_level == 0.02
        assert result.ideal_value is not None


class TestRunPreviewLargeCircuit:
    """Circuits exceeding the 10-qubit preview limit."""

    def test_large_circuit_returns_warning(self, large_circuit):
        recipe = recommend(analyze_circuit(large_circuit))
        result = run_preview(large_circuit, recipe)

        assert result.warning is not None
        assert "skip" in result.warning.lower()
        assert result.ideal_value is None
        assert result.noisy_value is None
        assert result.mitigated_value is None
        assert result.num_qubits == 15

    def test_exactly_10_qubits_works(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        for i in range(9):
            qc.cx(i, i + 1)
        qc.measure_all()

        recipe = recommend(analyze_circuit(qc))
        result = run_preview(qc, recipe)

        assert result.ideal_value is not None
        assert result.warning is None or "skip" not in result.warning.lower()


class TestRunPreviewEdgeCases:
    """Edge cases: single qubit, shallow, no multi-qubit gates."""

    def test_single_qubit(self, single_qubit_circuit):
        recipe = recommend(analyze_circuit(single_qubit_circuit))
        result = run_preview(single_qubit_circuit, recipe)

        assert result.ideal_value is not None
        assert result.num_qubits == 1

    def test_no_multi_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.z(2)
        qc.measure_all()

        recipe = recommend(analyze_circuit(qc))
        result = run_preview(qc, recipe)

        assert result.ideal_value is not None

    def test_very_shallow_circuit(self):
        """Depth-1 circuit -- Cirq may drop idle qubits."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()

        recipe = recommend(analyze_circuit(qc))
        result = run_preview(qc, recipe)

        assert result.ideal_value is not None


# ---------------------------------------------------------------------------
# B. Observable parsing tests
# ---------------------------------------------------------------------------


class TestParseObservable:
    def test_z0(self):
        obs = _parse_observable("Z0", 3)
        assert obs.shape == (8, 8)

    def test_z2(self):
        obs = _parse_observable("Z2", 3)
        assert obs.shape == (8, 8)

    def test_zz(self):
        obs = _parse_observable("ZZ", 2)
        assert obs.shape == (4, 4)

    def test_invalid_observable_raises(self):
        with pytest.raises(ValueError, match="Invalid observable"):
            _parse_observable("XY", 2)

    def test_qubit_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _parse_observable("Z5", 3)

    def test_zz_on_single_qubit_raises(self):
        with pytest.raises(ValueError, match="at least 2 qubits"):
            _parse_observable("ZZ", 1)


# ---------------------------------------------------------------------------
# C. Format tests
# ---------------------------------------------------------------------------


class TestFormatPreview:
    def test_contains_box_characters(self, bell_circuit):
        features = analyze_circuit(bell_circuit)
        recipe = recommend(features)
        result = run_preview(bell_circuit, recipe)

        output = format_preview(result, features)
        assert "\u250c" in output  # top-left corner
        assert "\u2514" in output  # bottom-left corner
        assert "\u2502" in output  # vertical bar

    def test_contains_all_fields(self, bell_circuit):
        features = analyze_circuit(bell_circuit)
        recipe = recommend(features)
        result = run_preview(bell_circuit, recipe)

        output = format_preview(result, features)
        assert "EMRG Preview" in output
        assert "Ideal" in output
        assert "Noisy" in output
        assert "Mitigated" in output
        assert "Error reduction" in output
        assert "depolarizing" in output

    def test_warning_displayed(self, large_circuit):
        features = analyze_circuit(large_circuit)
        recipe = recommend(features)
        result = run_preview(large_circuit, recipe)

        output = format_preview(result, features)
        assert "Note:" in output
        assert "skip" in output.lower()

    def test_skipped_simulation_shows_message(self, large_circuit):
        features = analyze_circuit(large_circuit)
        recipe = recommend(features)
        result = run_preview(large_circuit, recipe)

        output = format_preview(result, features)
        assert "Simulation skipped" in output


# ---------------------------------------------------------------------------
# D. API integration tests
# ---------------------------------------------------------------------------


class TestGenerateRecipePreview:
    def test_preview_false_gives_none(self, bell_circuit):
        result = generate_recipe(bell_circuit)
        assert result.preview is None

    def test_preview_true_populates_result(self, bell_circuit):
        result = generate_recipe(bell_circuit, preview=True)
        assert result.preview is not None
        assert isinstance(result.preview, PreviewResult)
        assert result.preview.ideal_value is not None

    def test_preview_with_custom_noise(self, bell_circuit):
        result = generate_recipe(
            bell_circuit, preview=True, noise_level=0.05
        )
        assert result.preview.noise_level == 0.05

    def test_preview_with_pec(self, bell_circuit):
        result = generate_recipe(
            bell_circuit,
            preview=True,
            noise_model_available=True,
        )
        assert result.preview is not None

    def test_str_includes_preview(self, bell_circuit):
        result = generate_recipe(bell_circuit, preview=True)
        output = str(result)
        assert "Preview" in output or "preview" in output
        assert "Ideal" in output or "ideal" in output

    def test_str_without_preview(self, bell_circuit):
        result = generate_recipe(bell_circuit)
        output = str(result)
        # Without preview, str() should just be the code.
        assert output == result.code

    def test_large_circuit_preview_has_warning(self, large_circuit):
        result = generate_recipe(large_circuit, preview=True)
        assert result.preview is not None
        assert result.preview.warning is not None
        assert result.preview.ideal_value is None


# ---------------------------------------------------------------------------
# E. CLI integration tests
# ---------------------------------------------------------------------------


class TestCLIPreview:
    def test_preview_flag_produces_output(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "docs/examples/bell_state.qasm", "--preview"],
        )
        assert result.exit_code == 0
        assert "EMRG Preview" in result.output
        assert "Ideal" in result.output

    def test_no_preview_flag_no_box(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "docs/examples/bell_state.qasm"],
        )
        assert result.exit_code == 0
        assert "EMRG Preview" not in result.output

    def test_preview_with_noise_level(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "docs/examples/bell_state.qasm",
                "--preview",
                "--noise-level",
                "0.05",
            ],
        )
        assert result.exit_code == 0
        assert "p=0.05" in result.output

    def test_preview_with_explain(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "docs/examples/bell_state.qasm",
                "--preview",
                "--explain",
            ],
        )
        assert result.exit_code == 0
        assert "EMRG Preview" in result.output

    def test_noise_level_without_preview_ignored(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "docs/examples/bell_state.qasm",
                "--noise-level",
                "0.05",
            ],
        )
        assert result.exit_code == 0
        assert "EMRG Preview" not in result.output

    def test_nonexistent_file_with_preview(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "nonexistent.qasm", "--preview"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output
