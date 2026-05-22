"""Tests for frontend detection and explicit frontend validation."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

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
    with pytest.raises(TypeError, match="Expected a cirq.Circuit"):
        detect_frontend(_qiskit_bell(), frontend="cirq")


def test_explicit_wrong_frontend_for_cirq_raises_clear_error() -> None:
    with pytest.raises(TypeError, match="Expected a qiskit.QuantumCircuit"):
        detect_frontend(_cirq_bell(), frontend="qiskit")


def test_rejects_raw_string_circuit() -> None:
    with pytest.raises(TypeError, match="Raw string circuit input is not supported"):
        detect_frontend("OPENQASM 2.0;")


def test_rejects_unsupported_object() -> None:
    with pytest.raises(TypeError, match="Unsupported circuit type"):
        detect_frontend(object())


def test_rejects_invalid_frontend_name() -> None:
    with pytest.raises(ValueError, match="Unsupported frontend"):
        detect_frontend(_qiskit_bell(), frontend="braket")


def test_rejects_invalid_frontend_type() -> None:
    with pytest.raises(TypeError, match="frontend must be"):
        detect_frontend(_qiskit_bell(), frontend=object())  # type: ignore[arg-type]
