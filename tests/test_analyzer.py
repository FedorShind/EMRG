"""Tests for emrg.analyzer -- circuit feature extraction."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from emrg.analyzer import (
    analyze_circuit,
)

# ---------------------------------------------------------------------------
# Fixtures -- reusable test circuits
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """2-qubit Bell state: H + CX + measure."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def ghz3_circuit() -> QuantumCircuit:
    """3-qubit GHZ: H + 2 CX + measure."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


@pytest.fixture
def vqe_circuit() -> QuantumCircuit:
    """4-qubit, 2-layer VQE-like ansatz (parametric)."""
    qc = QuantumCircuit(4, 4)
    for layer in range(2):
        for q in range(4):
            qc.ry(Parameter(f"t_{layer}_{q}_a"), q)
        for q in range(3):
            qc.cx(q, q + 1)
        for q in range(4):
            qc.ry(Parameter(f"t_{layer}_{q}_b"), q)
    qc.measure(range(4), range(4))
    return qc


@pytest.fixture
def no_measure_circuit() -> QuantumCircuit:
    """Circuit without measurements -- should trigger a warning."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation tests."""

    def test_rejects_non_circuit(self) -> None:
        with pytest.raises(TypeError, match="Expected a qiskit.QuantumCircuit"):
            analyze_circuit("not a circuit")  # type: ignore[arg-type]

    def test_rejects_empty_circuit(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])  # measure only, no gates
        with pytest.raises(ValueError, match="no gate operations"):
            analyze_circuit(qc)

    def test_warns_no_measurements(self, no_measure_circuit: QuantumCircuit) -> None:
        with pytest.warns(UserWarning, match="no measurements"):
            features = analyze_circuit(no_measure_circuit)
        assert features.has_measurements is False


# ---------------------------------------------------------------------------
# Tests: feature extraction
# ---------------------------------------------------------------------------


class TestBellCircuit:
    """Verify features for a simple Bell state."""

    def test_num_qubits(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.num_qubits == 2

    def test_depth(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        # H -> CX -> measure: depth depends on Qiskit scheduling,
        # but should be small (2-4).
        assert 1 <= f.depth <= 10

    def test_gate_counts(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.gate_counts.get("h") == 1
        assert f.gate_counts.get("cx") == 1
        # measure should NOT be in gate_counts
        assert "measure" not in f.gate_counts

    def test_multi_qubit_count(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.multi_qubit_gate_count == 1

    def test_single_qubit_count(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.single_qubit_gate_count == 1  # just H

    def test_has_measurements(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.has_measurements is True

    def test_not_parametric(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.num_parameters == 0

    def test_noise_category_low(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        assert f.noise_category == "low"


class TestGHZCircuit:
    """Verify features for 3-qubit GHZ."""

    def test_num_qubits(self, ghz3_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(ghz3_circuit)
        assert f.num_qubits == 3

    def test_multi_qubit_count(self, ghz3_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(ghz3_circuit)
        assert f.multi_qubit_gate_count == 2

    def test_total_gates(self, ghz3_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(ghz3_circuit)
        # 1 H + 2 CX = 3
        assert f.total_gate_count == 3


class TestVQECircuit:
    """Verify features for a parametric VQE-like ansatz."""

    def test_num_parameters(self, vqe_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(vqe_circuit)
        # 2 layers * (4 + 4) RY params = 16
        assert f.num_parameters == 16

    def test_multi_qubit_count(self, vqe_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(vqe_circuit)
        # 2 layers * 3 CX per layer = 6
        assert f.multi_qubit_gate_count == 6

    def test_single_qubit_count(self, vqe_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(vqe_circuit)
        # 2 layers * 8 RY per layer = 16
        assert f.single_qubit_gate_count == 16

    def test_noise_category(self, vqe_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(vqe_circuit)
        # 6*0.01 + 16*0.001 = 0.076 -> moderate
        assert f.noise_category == "moderate"

    def test_depth_reasonable(self, vqe_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(vqe_circuit)
        assert f.depth >= 4  # at least a few layers deep


# ---------------------------------------------------------------------------
# Tests: noise estimation
# ---------------------------------------------------------------------------


class TestNoiseEstimation:
    """Verify noise factor calculation and category mapping."""

    def test_noise_factor_bell(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        # 1*0.01 + 1*0.001 = 0.011
        assert f.estimated_noise_factor == pytest.approx(0.011, abs=1e-6)

    def test_noise_factor_custom_rates(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(
            bell_circuit,
            multi_qubit_error_rate=0.05,
            single_qubit_error_rate=0.005,
        )
        # 1*0.05 + 1*0.005 = 0.055
        assert f.estimated_noise_factor == pytest.approx(0.055, abs=1e-6)
        assert f.noise_category == "moderate"

    def test_high_noise_circuit(self) -> None:
        """Construct a circuit that crosses the 'high' threshold."""
        qc = QuantumCircuit(4, 4)
        # 30 CX gates -> 30*0.01 = 0.30 (high)
        for _ in range(30):
            qc.cx(0, 1)
        qc.measure(range(4), range(4))
        f = analyze_circuit(qc)
        assert f.noise_category == "high"
        assert f.multi_qubit_gate_count == 30


# ---------------------------------------------------------------------------
# Tests: dataclass properties
# ---------------------------------------------------------------------------


class TestCircuitFeaturesDataclass:
    """Verify CircuitFeatures is frozen and well-formed."""

    def test_immutable(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        with pytest.raises(AttributeError):
            f.depth = 999  # type: ignore[misc]

    def test_repr_contains_key_fields(self, bell_circuit: QuantumCircuit) -> None:
        f = analyze_circuit(bell_circuit)
        r = repr(f)
        assert "num_qubits=2" in r
        assert "noise_category=" in r


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_barrier_not_counted(self) -> None:
        """Barriers should not appear in gate_counts."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        f = analyze_circuit(qc)
        assert "barrier" not in f.gate_counts
        assert f.total_gate_count == 2  # H + CX only

    def test_single_gate_circuit(self) -> None:
        """Minimal circuit: one gate + measure."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        f = analyze_circuit(qc)
        assert f.total_gate_count == 1
        assert f.multi_qubit_gate_count == 0
        assert f.noise_category == "low"

    def test_multiple_multi_qubit_gate_types(self) -> None:
        """Mix of CX, CZ, SWAP gates."""
        qc = QuantumCircuit(3, 3)
        qc.cx(0, 1)
        qc.cz(1, 2)
        qc.swap(0, 2)
        qc.measure(range(3), range(3))
        f = analyze_circuit(qc)
        assert f.multi_qubit_gate_count == 3
        assert f.single_qubit_gate_count == 0
