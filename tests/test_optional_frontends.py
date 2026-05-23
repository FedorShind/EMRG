"""Optional frontend detection, normalization, and routing behavior."""

from __future__ import annotations

import pytest

import emrg.analyzer as analyzer_module
import emrg.frontends as frontends
from emrg import GeneratedRecipe, generate_recipe
from emrg.analyzer import analyze_circuit
from emrg.frontends import Frontend
from emrg.heuristics import recommend
from emrg.preview import PreviewResult, run_preview

cirq = pytest.importorskip("cirq")


class FakeBraketCircuit:
    """Test double for an installed Braket circuit type."""


def _cirq_bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m"),
    )


def _install_fake_braket_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_frontend_type(frontend: Frontend) -> type:
        if frontend is Frontend.BRAKET:
            return FakeBraketCircuit
        raise frontends.FrontendDependencyError(f"missing {frontend.value}")

    monkeypatch.setattr(frontends, "_load_frontend_type", fake_load_frontend_type)


def test_normalize_to_cirq_returns_cirq_input_unchanged() -> None:
    circuit = _cirq_bell()

    assert frontends.normalize_to_cirq(circuit, Frontend.CIRQ) is circuit


def test_normalize_to_cirq_uses_mitiq_for_optional_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)
    expected = _cirq_bell()

    def fake_convert_to_mitiq(circuit: object):
        assert isinstance(circuit, FakeBraketCircuit)
        return expected, "braket"

    import mitiq.interface

    monkeypatch.setattr(mitiq.interface, "convert_to_mitiq", fake_convert_to_mitiq)

    assert frontends.normalize_to_cirq(FakeBraketCircuit(), Frontend.BRAKET) is expected


def test_normalize_to_cirq_wraps_conversion_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)

    def failing_convert_to_mitiq(_circuit: object):
        raise RuntimeError("custom gate is unsupported")

    import mitiq.interface

    monkeypatch.setattr(mitiq.interface, "convert_to_mitiq", failing_convert_to_mitiq)

    with pytest.raises(ValueError, match="braket.*custom gate is unsupported"):
        frontends.normalize_to_cirq(FakeBraketCircuit(), Frontend.BRAKET)


def test_analyze_circuit_routes_optional_frontend_through_cirq_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)

    def fake_normalize_to_cirq(circuit: object, frontend: Frontend):
        assert isinstance(circuit, FakeBraketCircuit)
        assert frontend is Frontend.BRAKET
        return _cirq_bell()

    monkeypatch.setattr(
        analyzer_module,
        "normalize_to_cirq",
        fake_normalize_to_cirq,
        raising=False,
    )

    features = analyze_circuit(FakeBraketCircuit())

    assert features.num_qubits == 2
    assert features.total_gate_count == 2
    assert features.multi_qubit_gate_count == 1


def test_generate_recipe_accepts_converted_optional_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)

    def fake_normalize_to_cirq(_circuit: object, _frontend: Frontend):
        return _cirq_bell()

    monkeypatch.setattr(analyzer_module, "normalize_to_cirq", fake_normalize_to_cirq)

    result = generate_recipe(FakeBraketCircuit())

    assert isinstance(result, GeneratedRecipe)
    assert result.features.num_qubits == 2
    assert result.recipe.technique in {"zne", "pec", "cdr", "composite"}
    compile(result.code, "<emrg-fake-braket-generated>", "exec")


def test_run_preview_skips_converted_optional_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)
    recipe = recommend(analyze_circuit(_cirq_bell()))

    result = run_preview(FakeBraketCircuit(), recipe)

    assert isinstance(result, PreviewResult)
    assert result.warning is not None
    assert "Preview skipped" in result.warning
    assert "braket" in result.warning
    assert result.ideal_value is None


def test_generate_recipe_preview_skips_converted_optional_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_braket_type(monkeypatch)

    def fake_normalize_to_cirq(_circuit: object, _frontend: Frontend):
        return _cirq_bell()

    monkeypatch.setattr(analyzer_module, "normalize_to_cirq", fake_normalize_to_cirq)

    result = generate_recipe(FakeBraketCircuit(), preview=True)

    assert result.preview is not None
    assert result.preview.warning is not None
    assert "Preview skipped" in result.preview.warning
    assert "braket" in result.preview.warning


def test_installed_braket_frontend_detects_and_analyzes() -> None:
    braket_circuits = pytest.importorskip("braket.circuits")
    circuit = braket_circuits.Circuit().h(0).cnot(0, 1)

    assert frontends.detect_frontend(circuit) is Frontend.BRAKET
    with pytest.warns(UserWarning, match="no measurements"):
        features = analyze_circuit(circuit)

    assert features.num_qubits >= 2
    assert features.total_gate_count > 0


def test_installed_pennylane_frontend_detects_and_analyzes() -> None:
    pennylane = pytest.importorskip("pennylane")

    with pennylane.tape.QuantumTape() as tape:
        pennylane.Hadamard(wires=0)
        pennylane.CNOT(wires=[0, 1])

    assert frontends.detect_frontend(tape) is Frontend.PENNYLANE
    with pytest.warns(UserWarning, match="no measurements"):
        features = analyze_circuit(tape)

    assert features.num_qubits >= 2
    assert features.total_gate_count > 0


def test_installed_pyquil_frontend_detects_and_analyzes() -> None:
    pyquil = pytest.importorskip("pyquil")
    pyquil_gates = pytest.importorskip("pyquil.gates")
    program = pyquil.Program(pyquil_gates.H(0), pyquil_gates.CNOT(0, 1))

    assert frontends.detect_frontend(program) is Frontend.PYQUIL
    with pytest.warns(UserWarning, match="no measurements"):
        features = analyze_circuit(program)

    assert features.num_qubits >= 2
    assert features.total_gate_count > 0


def test_installed_qibo_frontend_detects_and_analyzes() -> None:
    qibo_models = pytest.importorskip("qibo.models")
    qibo_gates = pytest.importorskip("qibo.gates")
    circuit = qibo_models.Circuit(2)
    circuit.add(qibo_gates.H(0))
    circuit.add(qibo_gates.CNOT(0, 1))

    assert frontends.detect_frontend(circuit) is Frontend.QIBO
    with pytest.warns(UserWarning, match="no measurements"):
        features = analyze_circuit(circuit)

    assert features.num_qubits >= 2
    assert features.total_gate_count > 0
