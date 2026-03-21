"""Shared test utilities for EMRG test suite."""

from __future__ import annotations

from emrg.analyzer import CircuitFeatures


def make_features(
    *,
    num_qubits: int = 4,
    depth: int = 10,
    gate_counts: dict[str, int] | None = None,
    total_gate_count: int = 5,
    multi_qubit_gate_count: int = 2,
    single_qubit_gate_count: int = 3,
    num_parameters: int = 0,
    has_measurements: bool = True,
    estimated_noise_factor: float = 0.023,
    noise_category: str = "low",
    noise_model_available: bool = False,
    pec_overhead_estimate: float = 1.0,
) -> CircuitFeatures:
    """Create a CircuitFeatures with sensible defaults, overridable.

    This is a shared helper used across multiple test modules to avoid
    constructing the full frozen dataclass by hand each time.

    Note: ``noise_model_available`` defaults to ``False`` here (unlike the
    real ``analyze_circuit`` pipeline) so that existing ZNE-focused tests
    are not affected by PEC selection logic.
    """
    return CircuitFeatures(
        num_qubits=num_qubits,
        depth=depth,
        gate_counts=gate_counts or {"h": 3, "cx": 2},
        total_gate_count=total_gate_count,
        multi_qubit_gate_count=multi_qubit_gate_count,
        single_qubit_gate_count=single_qubit_gate_count,
        num_parameters=num_parameters,
        has_measurements=has_measurements,
        estimated_noise_factor=estimated_noise_factor,
        noise_category=noise_category,
        noise_model_available=noise_model_available,
        pec_overhead_estimate=pec_overhead_estimate,
    )
