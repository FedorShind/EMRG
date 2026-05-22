"""Preview behavior for Cirq inputs."""

from __future__ import annotations

import pytest

from emrg import generate_recipe
from emrg.analyzer import analyze_circuit
from emrg.heuristics import recommend
from emrg.preview import PreviewResult, run_preview

cirq = pytest.importorskip("cirq")


def _cirq_bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m"),
    )


def test_run_preview_accepts_tiny_cirq_circuit() -> None:
    circuit = _cirq_bell()
    recipe = recommend(analyze_circuit(circuit))

    result = run_preview(circuit, recipe)

    assert isinstance(result, PreviewResult)
    assert result.warning is None or "failed" not in result.warning.lower()
    assert result.ideal_value is not None
    assert result.noisy_value is not None
    assert result.mitigated_value is not None
    assert result.technique == "ZNE"
    assert result.num_qubits == 2


def test_generate_recipe_preview_accepts_tiny_cirq_circuit() -> None:
    result = generate_recipe(_cirq_bell(), preview=True)

    assert result.preview is not None
    assert result.preview.warning is None or "failed" not in result.preview.warning
    assert result.preview.ideal_value is not None
    assert "Preview" in str(result)


def test_cirq_preview_respects_gate_budget() -> None:
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit(cirq.X(qubits[0]) for _ in range(61))
    circuit.append(cirq.measure(*qubits))
    recipe = generate_recipe(circuit).recipe

    result = run_preview(circuit, recipe)

    assert result.warning is not None
    assert "exceeds simulation budget" in result.warning
    assert result.ideal_value is None


def test_cirq_preview_returns_clear_warning_on_invalid_observable() -> None:
    circuit = _cirq_bell()
    recipe = generate_recipe(circuit).recipe

    result = run_preview(circuit, recipe, observable="XY")

    assert result.warning is not None
    assert "Simulation failed" in result.warning
    assert "Invalid observable" in result.warning
