"""Circuit Analyzer -- extract features from a Qiskit QuantumCircuit.

This module is the foundation of the EMRG pipeline. It inspects a
QuantumCircuit and returns a :class:`CircuitFeatures` dataclass that
downstream modules (heuristics, codegen) consume.

Typical usage::

    from qiskit import QuantumCircuit
    from emrg.analyzer import analyze_circuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    features = analyze_circuit(qc)
    print(features)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qiskit import QuantumCircuit

__all__ = [
    "CircuitFeatures",
    "analyze_circuit",
    "MULTI_QUBIT_GATE_NAMES",
    "DEFAULT_MULTI_QUBIT_ERROR_RATE",
    "DEFAULT_SINGLE_QUBIT_ERROR_RATE",
    "NOISE_THRESHOLD_LOW",
    "NOISE_THRESHOLD_MODERATE",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Gate names that Qiskit uses for two-qubit and multi-qubit operations.
#: Includes 3-qubit gates (ccx, ccz, cswap) that also benefit from
#: multi-qubit error rate estimates.
MULTI_QUBIT_GATE_NAMES: frozenset[str] = frozenset(
    {
        "cx",
        "cz",
        "cy",
        "ch",
        "crx",
        "cry",
        "crz",
        "cp",
        "swap",
        "iswap",
        "ecr",
        "rzx",
        "rzz",
        "rxx",
        "ryy",
        "csx",
        "cu",
        "cu1",
        "cu3",
        "dcx",
        "ccx",  # 3-qubit
        "ccz",  # 3-qubit
        "cswap",  # 3-qubit
    }
)

#: Default proxy for average multi-qubit gate error rate on NISQ hardware.
#: IBM Eagle r3 ~ 0.005-0.01; we use a conservative midpoint.
DEFAULT_MULTI_QUBIT_ERROR_RATE: float = 0.01

#: Default proxy for average single-qubit gate error rate.
DEFAULT_SINGLE_QUBIT_ERROR_RATE: float = 0.001

#: Noise factor below this is classified as "low".
NOISE_THRESHOLD_LOW: float = 0.05

#: Noise factor below this (but >= NOISE_THRESHOLD_LOW) is "moderate";
#: at or above this threshold is "high".
NOISE_THRESHOLD_MODERATE: float = 0.20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _freeze_dict(value: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Wrap a dict in a read-only ``MappingProxyType``."""
    return MappingProxyType(value)


@dataclass(frozen=True)
class CircuitFeatures:
    """Immutable snapshot of circuit properties relevant to error mitigation.

    All fields are derived from Qiskit introspection -- no simulation needed.

    Attributes:
        num_qubits: Number of quantum bits in the circuit.
        depth: Circuit depth (longest path of dependent operations).
        gate_counts: Read-only mapping of gate name to count
            (e.g. ``{'cx': 4, 'h': 3}``).
        total_gate_count: Total number of gate operations.
        multi_qubit_gate_count: Number of multi-qubit gates (2-qubit and
            3-qubit, e.g. CX, CCX, CSWAP).
        single_qubit_gate_count: Number of single-qubit gates.
        num_parameters: Number of unbound parameters (0 for non-variational).
        has_measurements: Whether the circuit contains measurement operations.
        estimated_noise_factor: Simple heuristic noise proxy
            (``multi_qubit_gate_count * multi_qubit_error_rate
              + single_qubit_gate_count * single_qubit_error_rate``).
        noise_category: Human-readable category: ``"low"``, ``"moderate"``,
            or ``"high"``.
    """

    num_qubits: int
    depth: int
    gate_counts: Mapping[str, int] = field(default_factory=dict)
    total_gate_count: int = 0
    multi_qubit_gate_count: int = 0
    single_qubit_gate_count: int = 0
    num_parameters: int = 0
    has_measurements: bool = False
    estimated_noise_factor: float = 0.0
    noise_category: str = "low"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_circuit(qc: QuantumCircuit) -> list[str]:
    """Validate the circuit and return a list of warning messages (may be empty).

    Checks:
    - Circuit must be a ``QuantumCircuit`` instance.
    - Circuit must contain at least one gate operation.
    - Circuit should have measurements for meaningful mitigation.

    Raises:
        TypeError: If *qc* is not a ``QuantumCircuit``.
        ValueError: If the circuit has zero gate operations.
    """
    from qiskit import QuantumCircuit as _QuantumCircuit  # lazy runtime import

    if not isinstance(qc, _QuantumCircuit):
        raise TypeError(f"Expected a qiskit.QuantumCircuit, got {type(qc).__name__}.")

    ops = qc.count_ops()
    # Filter out measurement/barrier/delay -- only count real gates
    gate_ops = {
        name: count
        for name, count in ops.items()
        if name not in ("measure", "barrier", "delay", "reset")
    }

    if not gate_ops:
        raise ValueError(
            "Circuit has no gate operations. "
            "Provide a circuit with at least one quantum gate."
        )

    warns: list[str] = []
    if "measure" not in ops:
        warns.append(
            "Circuit has no measurements. Error mitigation requires "
            "measurement results -- consider adding qc.measure_all()."
        )
    return warns


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------


def _classify_noise(noise_factor: float) -> str:
    """Map a numeric noise factor to a human-readable category.

    Thresholds are deliberately conservative and tunable.
    """
    if noise_factor < NOISE_THRESHOLD_LOW:
        return "low"
    elif noise_factor < NOISE_THRESHOLD_MODERATE:
        return "moderate"
    else:
        return "high"


def _estimate_noise(
    multi_qubit_count: int,
    single_qubit_count: int,
    *,
    multi_qubit_error_rate: float = DEFAULT_MULTI_QUBIT_ERROR_RATE,
    single_qubit_error_rate: float = DEFAULT_SINGLE_QUBIT_ERROR_RATE,
) -> float:
    """Return a simple additive noise proxy.

    This is *not* a rigorous noise model -- it is a fast heuristic used to
    guide mitigation strategy selection.
    """
    return (
        multi_qubit_count * multi_qubit_error_rate
        + single_qubit_count * single_qubit_error_rate
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_circuit(
    qc: QuantumCircuit,
    *,
    multi_qubit_error_rate: float = DEFAULT_MULTI_QUBIT_ERROR_RATE,
    single_qubit_error_rate: float = DEFAULT_SINGLE_QUBIT_ERROR_RATE,
) -> CircuitFeatures:
    """Analyze a Qiskit ``QuantumCircuit`` and extract mitigation-relevant features.

    Parameters
    ----------
    qc:
        The circuit to analyze.
    multi_qubit_error_rate:
        Proxy error rate for multi-qubit gates (default 0.01).
    single_qubit_error_rate:
        Proxy error rate for single-qubit gates (default 0.001).

    Returns
    -------
    CircuitFeatures
        Immutable dataclass with extracted metrics.

    Raises
    ------
    TypeError
        If *qc* is not a ``QuantumCircuit``.
    ValueError
        If the circuit has zero gate operations.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2, 2); _ = qc.h(0); _ = qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    >>> features = analyze_circuit(qc)
    >>> features.num_qubits
    2
    >>> features.multi_qubit_gate_count
    1
    """
    # --- validate -----------------------------------------------------------
    warn_msgs = _validate_circuit(qc)
    for msg in warn_msgs:
        warnings.warn(msg, UserWarning, stacklevel=2)

    # --- extract raw ops ----------------------------------------------------
    ops = dict(qc.count_ops())  # OrderedDict -> dict

    # Separate gate ops from non-gate ops (measure, barrier, etc.)
    non_gate_names = {"measure", "barrier", "delay", "reset"}
    gate_counts: dict[str, int] = {
        name: count for name, count in ops.items() if name not in non_gate_names
    }

    total_gate_count = sum(gate_counts.values())
    multi_qubit_gate_count = sum(
        count for name, count in gate_counts.items() if name in MULTI_QUBIT_GATE_NAMES
    )
    single_qubit_gate_count = total_gate_count - multi_qubit_gate_count

    # --- noise estimate -----------------------------------------------------
    noise_factor = _estimate_noise(
        multi_qubit_gate_count,
        single_qubit_gate_count,
        multi_qubit_error_rate=multi_qubit_error_rate,
        single_qubit_error_rate=single_qubit_error_rate,
    )

    return CircuitFeatures(
        num_qubits=qc.num_qubits,
        depth=qc.depth(),
        gate_counts=_freeze_dict(gate_counts),
        total_gate_count=total_gate_count,
        multi_qubit_gate_count=multi_qubit_gate_count,
        single_qubit_gate_count=single_qubit_gate_count,
        num_parameters=qc.num_parameters,
        has_measurements="measure" in ops,
        estimated_noise_factor=round(noise_factor, 6),
        noise_category=_classify_noise(noise_factor),
    )
