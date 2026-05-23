"""Tests for frontend detection and explicit frontend validation."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

import emrg.frontends as frontends
from emrg.frontends import Frontend, detect_frontend

cirq = pytest.importorskip("cirq")


def _qiskit_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _cirq_bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m"),
    )


def test_auto_detects_qiskit_circuit() -> None:
    assert detect_frontend(_qiskit_bell()) is Frontend.QISKIT


def test_auto_detects_cirq_circuit() -> None:
    assert detect_frontend(_cirq_bell()) is Frontend.CIRQ


def test_frontend_enum_includes_optional_phase3_values() -> None:
    assert Frontend.BRAKET.value == "braket"
    assert Frontend.PENNYLANE.value == "pennylane"
    assert Frontend.PYQUIL.value == "pyquil"
    assert Frontend.QIBO.value == "qibo"


@pytest.mark.parametrize("frontend", ["qiskit", Frontend.QISKIT])
def test_explicit_qiskit_frontend_validates_matching_object(
    frontend: str | Frontend,
) -> None:
    assert detect_frontend(_qiskit_bell(), frontend=frontend) is Frontend.QISKIT


@pytest.mark.parametrize("frontend", ["cirq", Frontend.CIRQ])
def test_explicit_cirq_frontend_validates_matching_object(
    frontend: str | Frontend,
) -> None:
    assert detect_frontend(_cirq_bell(), frontend=frontend) is Frontend.CIRQ


def test_explicit_wrong_frontend_for_qiskit_raises_clear_error() -> None:
    with pytest.raises(TypeError, match="frontend='cirq'.*got QuantumCircuit"):
        detect_frontend(_qiskit_bell(), frontend="cirq")


def test_explicit_wrong_frontend_for_cirq_raises_clear_error() -> None:
    with pytest.raises(TypeError, match="frontend='qiskit'.*got Circuit"):
        detect_frontend(_cirq_bell(), frontend="qiskit")


def test_explicit_optional_frontend_for_qiskit_raises_mismatch() -> None:
    with pytest.raises(TypeError, match="frontend='braket'.*got QuantumCircuit"):
        detect_frontend(_qiskit_bell(), frontend="braket")


def test_explicit_optional_frontend_missing_dependency_has_install_hint() -> None:
    class UnknownCircuit:
        pass

    with pytest.raises(
        ImportError,
        match=(
            r"frontend='braket' requires optional Braket support[\s\S]*"
            r"emrg\[braket\]"
        ),
    ):
        detect_frontend(UnknownCircuit(), frontend="braket")


def test_auto_detects_optional_frontend_when_type_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBraketCircuit:
        pass

    def fake_load_frontend_type(frontend: Frontend) -> type:
        if frontend is Frontend.BRAKET:
            return FakeBraketCircuit
        raise ImportError(f"missing {frontend.value}")

    monkeypatch.setattr(frontends, "_load_frontend_type", fake_load_frontend_type)

    assert detect_frontend(FakeBraketCircuit()) is Frontend.BRAKET


def test_rejects_raw_string_circuit() -> None:
    with pytest.raises(TypeError, match="Raw string circuit input is not supported"):
        detect_frontend("OPENQASM 2.0;")


def test_rejects_unsupported_object() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "Unsupported input type object.*Qiskit QuantumCircuit.*"
            "Cirq Circuit.*optional converted Python frontend objects"
        ),
    ):
        detect_frontend(object())


def test_rejects_invalid_frontend_name() -> None:
    with pytest.raises(
        ValueError,
        match="frontend='openqasm' is not supported.*Allowed values:",
    ):
        detect_frontend(_qiskit_bell(), frontend="openqasm")


def test_rejects_invalid_frontend_type() -> None:
    with pytest.raises(TypeError, match="frontend must be"):
        detect_frontend(_qiskit_bell(), frontend=object())  # type: ignore[arg-type]
