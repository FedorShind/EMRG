"""Frontend detection for supported circuit objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "Frontend",
    "FrontendDependencyError",
    "detect_frontend",
]


class Frontend(StrEnum):
    """Circuit frontend values supported by EMRG."""

    QISKIT = "qiskit"
    CIRQ = "cirq"
    BRAKET = "braket"
    PENNYLANE = "pennylane"
    PYQUIL = "pyquil"
    QIBO = "qibo"


class FrontendDependencyError(ImportError):
    """Raised when an explicit optional frontend cannot be validated."""


_OPTIONAL_FRONTEND_TYPES: dict[Frontend, tuple[str, str, str]] = {
    Frontend.BRAKET: (
        "braket.circuits",
        "Circuit",
        'pip install "emrg[braket]"',
    ),
    Frontend.PENNYLANE: (
        "pennylane.tape",
        "QuantumTape",
        'pip install "emrg[pennylane]"',
    ),
    Frontend.PYQUIL: (
        "pyquil",
        "Program",
        'pip install "emrg[pyquil]"',
    ),
    Frontend.QIBO: (
        "qibo.models.circuit",
        "Circuit",
        'pip install "emrg[qibo]"',
    ),
}


_OPTIONAL_FRONTEND_DISPLAY: dict[Frontend, str] = {
    Frontend.BRAKET: "braket.circuits.Circuit",
    Frontend.PENNYLANE: "pennylane.tape.QuantumTape",
    Frontend.PYQUIL: "pyquil.Program",
    Frontend.QIBO: "qibo.models.circuit.Circuit",
}


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
        "frontend must be a supported frontend string, Frontend value, "
        f"or None, got {type(frontend).__name__}."
    )


def _load_frontend_type(frontend: Frontend) -> type[Any]:
    """Return the SDK circuit type for an optional frontend."""
    import importlib

    if frontend not in _OPTIONAL_FRONTEND_TYPES:
        raise ValueError(f"Frontend {frontend.value!r} is not optional.")

    module_name, attr_name, install_hint = _OPTIONAL_FRONTEND_TYPES[frontend]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise FrontendDependencyError(
            f"Cannot validate frontend '{frontend.value}' because its optional "
            f"dependency is not installed. Install it with: {install_hint}."
        ) from exc

    try:
        frontend_type = getattr(module, attr_name)
    except AttributeError as exc:
        raise FrontendDependencyError(
            f"Cannot validate frontend '{frontend.value}' because "
            f"{module_name}.{attr_name} is unavailable. Install it with: "
            f"{install_hint}."
        ) from exc

    if not isinstance(frontend_type, type):
        raise FrontendDependencyError(
            f"Cannot validate frontend '{frontend.value}' because "
            f"{module_name}.{attr_name} is not a type. Reinstall with: "
            f"{install_hint}."
        )
    return frontend_type


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


def _is_optional_frontend_object(obj: object, frontend: Frontend) -> bool:
    """Return True when *obj* matches an installed optional frontend type."""
    try:
        frontend_type = _load_frontend_type(frontend)
    except FrontendDependencyError:
        return False
    return isinstance(obj, frontend_type)


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
    if frontend in _OPTIONAL_FRONTEND_TYPES:
        if _is_qiskit_circuit(circuit) or _is_cirq_circuit(circuit):
            raise TypeError(
                f"Expected a {_OPTIONAL_FRONTEND_DISPLAY[frontend]} for frontend "
                f"'{frontend.value}', got {type(circuit).__name__}."
            )
        frontend_type = _load_frontend_type(frontend)
        if not isinstance(circuit, frontend_type):
            raise TypeError(
                f"Expected a {_OPTIONAL_FRONTEND_DISPLAY[frontend]} for frontend "
                f"'{frontend.value}', got {type(circuit).__name__}."
            )


def detect_frontend(
    circuit: object,
    frontend: str | Frontend | None = None,
) -> Frontend:
    """Detect or validate the circuit frontend.

    Auto-detection intentionally checks Qiskit before Cirq and does not use
    Mitiq's frontend type aliases because optional frontend fallbacks can make
    those aliases too broad for EMRG's frontend boundary.
    """
    if isinstance(circuit, str):
        raise TypeError(
            "Raw string circuit input is not supported in this phase. "
            "Use the QASM CLI for QASM files. Expected a qiskit.QuantumCircuit "
            "or supported circuit object."
        )

    if frontend is not None:
        explicit_frontend = _coerce_frontend(frontend)
        _validate_frontend_object(circuit, explicit_frontend)
        return explicit_frontend

    if _is_qiskit_circuit(circuit):
        return Frontend.QISKIT
    if _is_cirq_circuit(circuit):
        return Frontend.CIRQ
    for candidate in (
        Frontend.BRAKET,
        Frontend.PENNYLANE,
        Frontend.PYQUIL,
        Frontend.QIBO,
    ):
        if _is_optional_frontend_object(circuit, candidate):
            return candidate

    raise TypeError(
        "Unsupported circuit type. Expected a qiskit.QuantumCircuit, cirq.Circuit, "
        "or installed optional frontend circuit object, "
        f"got {type(circuit).__name__}."
    )
