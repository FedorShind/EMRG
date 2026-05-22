"""Frontend detection for supported circuit objects."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Frontend", "detect_frontend"]


class Frontend(StrEnum):
    """Circuit frontend values supported by EMRG."""

    QISKIT = "qiskit"
    CIRQ = "cirq"


def _coerce_frontend(frontend: str | Frontend) -> Frontend:
    """Return a Frontend enum value from a supported explicit frontend."""
    if isinstance(frontend, Frontend):
        return frontend
    if isinstance(frontend, str):
        value = frontend.strip().lower()
        try:
            return Frontend(value)
        except ValueError as exc:
            supported = ", ".join(item.value for item in Frontend)
            raise ValueError(
                f"Unsupported frontend {frontend!r}. Supported frontends: {supported}."
            ) from exc
    raise TypeError(
        "frontend must be 'qiskit', 'cirq', Frontend.QISKIT, Frontend.CIRQ, "
        f"or None, got {type(frontend).__name__}."
    )


def _is_qiskit_circuit(obj: object) -> bool:
    """Return True when *obj* is a Qiskit QuantumCircuit."""
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return False
    return isinstance(obj, QuantumCircuit)


def _is_cirq_circuit(obj: object) -> bool:
    """Return True when *obj* is a Cirq Circuit."""
    try:
        import cirq
    except ImportError:
        return False
    return isinstance(obj, cirq.Circuit)


def _validate_frontend_object(circuit: object, frontend: Frontend) -> None:
    """Validate that *circuit* is compatible with *frontend*."""
    if frontend is Frontend.QISKIT and not _is_qiskit_circuit(circuit):
        raise TypeError(
            f"Expected a qiskit.QuantumCircuit for frontend 'qiskit', "
            f"got {type(circuit).__name__}."
        )
    if frontend is Frontend.CIRQ and not _is_cirq_circuit(circuit):
        raise TypeError(
            f"Expected a cirq.Circuit for frontend 'cirq', "
            f"got {type(circuit).__name__}."
        )


def detect_frontend(
    circuit: object,
    frontend: str | Frontend | None = None,
) -> Frontend:
    """Detect or validate the circuit frontend.

    Auto-detection intentionally checks Qiskit before Cirq and does not use
    Mitiq's frontend type aliases because optional frontend fallbacks can make
    those aliases too broad for EMRG's Phase 2 boundary.
    """
    if isinstance(circuit, str):
        raise TypeError(
            "Raw string circuit input is not supported in this phase. "
            "Use the QASM CLI for QASM files."
        )

    if frontend is not None:
        explicit_frontend = _coerce_frontend(frontend)
        _validate_frontend_object(circuit, explicit_frontend)
        return explicit_frontend

    if _is_qiskit_circuit(circuit):
        return Frontend.QISKIT
    if _is_cirq_circuit(circuit):
        return Frontend.CIRQ

    raise TypeError(
        "Unsupported circuit type. Expected a qiskit.QuantumCircuit or "
        f"cirq.Circuit, got {type(circuit).__name__}."
    )
