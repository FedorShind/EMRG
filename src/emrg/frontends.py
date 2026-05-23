"""Frontend detection for supported circuit objects."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "Frontend",
    "FrontendDependencyError",
    "FrontendConversionError",
    "detect_frontend",
    "normalize_to_cirq",
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


class FrontendConversionError(ValueError):
    """Raised when an input circuit cannot be normalized to Cirq."""


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

_OPTIONAL_FRONTEND_NAMES: dict[Frontend, str] = {
    Frontend.BRAKET: "Braket",
    Frontend.PENNYLANE: "PennyLane",
    Frontend.PYQUIL: "PyQuil",
    Frontend.QIBO: "Qibo",
}

_SUPPORTED_INPUTS = (
    "Qiskit QuantumCircuit, Cirq Circuit, and optional converted Python "
    "frontend objects"
)


def _supported_frontend_values() -> str:
    return ", ".join(item.value for item in Frontend)


def _coerce_frontend(frontend: str | Frontend) -> Frontend:
    """Return a Frontend enum value from a supported explicit frontend."""
    if isinstance(frontend, Frontend):
        return frontend
    if isinstance(frontend, str):
        value = frontend.strip().lower()
        try:
            return Frontend(value)
        except ValueError as exc:
            raise ValueError(
                f"frontend={frontend!r} is not supported. "
                f"Allowed values: {_supported_frontend_values()}."
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
            f"frontend='{frontend.value}' requires optional "
            f"{_OPTIONAL_FRONTEND_NAMES[frontend]} support.\n"
            f"Install with: {install_hint}"
        ) from exc

    try:
        frontend_type = getattr(module, attr_name)
    except AttributeError as exc:
        raise FrontendDependencyError(
            f"frontend='{frontend.value}' requires optional "
            f"{_OPTIONAL_FRONTEND_NAMES[frontend]} support, but "
            f"{module_name}.{attr_name} is unavailable.\n"
            f"Install with: {install_hint}"
        ) from exc

    if not isinstance(frontend_type, type):
        raise FrontendDependencyError(
            f"frontend='{frontend.value}' requires optional "
            f"{_OPTIONAL_FRONTEND_NAMES[frontend]} support, but "
            f"{module_name}.{attr_name} is not a type.\n"
            f"Install with: {install_hint}"
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
            "Requested frontend='qiskit' expects qiskit.QuantumCircuit; "
            f"got {type(circuit).__name__}."
        )
    if frontend is Frontend.CIRQ and not _is_cirq_circuit(circuit):
        raise TypeError(
            "Requested frontend='cirq' expects cirq.Circuit; "
            f"got {type(circuit).__name__}."
        )
    if frontend in _OPTIONAL_FRONTEND_TYPES:
        if _is_qiskit_circuit(circuit) or _is_cirq_circuit(circuit):
            raise TypeError(
                f"Requested frontend='{frontend.value}' expects "
                f"{_OPTIONAL_FRONTEND_DISPLAY[frontend]}; "
                f"got {type(circuit).__name__}."
            )
        frontend_type = _load_frontend_type(frontend)
        if not isinstance(circuit, frontend_type):
            raise TypeError(
                f"Requested frontend='{frontend.value}' expects "
                f"{_OPTIONAL_FRONTEND_DISPLAY[frontend]}; "
                f"got {type(circuit).__name__}."
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
            "Use the QASM CLI for QASM files. EMRG currently supports "
            f"{_SUPPORTED_INPUTS}."
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
        f"Unsupported input type {type(circuit).__name__}. "
        f"EMRG currently supports {_SUPPORTED_INPUTS}."
    )


def normalize_to_cirq(circuit: object, frontend: str | Frontend):
    """Normalize a supported circuit object to Cirq through Mitiq."""
    active_frontend = _coerce_frontend(frontend)
    _validate_frontend_object(circuit, active_frontend)

    if active_frontend is Frontend.CIRQ:
        return circuit

    try:
        from mitiq.interface import convert_to_mitiq

        cirq_circuit, _input_type = convert_to_mitiq(circuit)
    except FrontendConversionError:
        raise
    except Exception as exc:
        raise FrontendConversionError(
            f"Could not normalize {active_frontend.value} circuit through Mitiq: {exc}"
        ) from exc

    if not _is_cirq_circuit(cirq_circuit):
        raise FrontendConversionError(
            f"Could not normalize {active_frontend.value} circuit "
            f"through Mitiq: converter returned "
            f"{type(cirq_circuit).__name__}."
        )
    return cirq_circuit
