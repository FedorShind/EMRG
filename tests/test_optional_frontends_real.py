"""Real optional SDK smoke tests for converted frontend support."""

from __future__ import annotations

import pytest

from emrg import Frontend, detect_frontend, generate_recipe
from emrg.analyzer import analyze_circuit


def _assert_preview_skips(circuit: object, expected_frontend: str) -> None:
    with pytest.warns(UserWarning, match="no measurements"):
        result = generate_recipe(circuit, preview=True)

    assert result.preview is not None
    assert result.preview.ideal_value is None
    assert result.preview.warning is not None
    assert f"Preview for {expected_frontend} inputs is not supported yet" in (
        result.preview.warning
    )
    assert "connect the executor to your backend" in result.preview.warning


def _assert_real_frontend_recipe(
    circuit: object,
    expected_frontend: Frontend,
) -> None:
    assert detect_frontend(circuit) is expected_frontend

    with pytest.warns(UserWarning, match="no measurements"):
        features = analyze_circuit(circuit)

    assert features.frontend == expected_frontend.value
    assert features.analysis_basis == "cirq-normalized"
    assert features.num_qubits >= 2
    assert features.total_gate_count > 0
    assert features.multi_qubit_gate_count >= 1
    assert features.estimated_noise_factor >= 0
    assert features.pec_overhead_estimate >= 1.0

    with pytest.warns(UserWarning, match="no measurements"):
        result = generate_recipe(circuit)

    assert result.features.frontend == expected_frontend.value
    assert result.recipe.technique in {"zne", "pec", "cdr", "composite"}
    compile(result.code, f"<emrg-{expected_frontend.value}-generated>", "exec")

    _assert_preview_skips(circuit, expected_frontend.value)


def test_real_braket_circuit_smoke() -> None:
    pytest.importorskip("braket")
    braket_circuits = pytest.importorskip("braket.circuits")

    circuit = braket_circuits.Circuit().h(0).cnot(0, 1)

    _assert_real_frontend_recipe(circuit, Frontend.BRAKET)


def test_real_pennylane_quantum_tape_smoke() -> None:
    pennylane = pytest.importorskip("pennylane")

    with pennylane.tape.QuantumTape() as tape:
        pennylane.Hadamard(wires=0)
        pennylane.CNOT(wires=[0, 1])

    _assert_real_frontend_recipe(tape, Frontend.PENNYLANE)


def test_real_pyquil_program_smoke() -> None:
    pyquil = pytest.importorskip("pyquil")
    pyquil_gates = pytest.importorskip("pyquil.gates")

    program = pyquil.Program(pyquil_gates.H(0), pyquil_gates.CNOT(0, 1))

    _assert_real_frontend_recipe(program, Frontend.PYQUIL)


def test_real_qibo_circuit_smoke() -> None:
    qibo_models = pytest.importorskip("qibo.models")
    qibo_gates = pytest.importorskip("qibo.gates")

    circuit = qibo_models.Circuit(2)
    circuit.add(qibo_gates.H(0))
    circuit.add(qibo_gates.CNOT(0, 1))

    _assert_real_frontend_recipe(circuit, Frontend.QIBO)
