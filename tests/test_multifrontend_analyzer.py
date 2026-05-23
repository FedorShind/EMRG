"""Analyzer coverage for Qiskit parity and Cirq input support."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from emrg.analyzer import CircuitFeatures, analyze_circuit

cirq = pytest.importorskip("cirq")
sympy = pytest.importorskip("sympy")


def _assert_features(
    features: CircuitFeatures,
    *,
    num_qubits: int,
    depth: int,
    gate_counts: Mapping[str, int],
    total_gate_count: int,
    multi_qubit_gate_count: int,
    single_qubit_gate_count: int,
    num_parameters: int,
    has_measurements: bool,
    estimated_noise_factor: float,
    noise_category: str,
    pec_overhead_estimate: float,
    layer_heterogeneity: float,
    non_clifford_count: int,
    non_clifford_fraction: float,
) -> None:
    assert features.num_qubits == num_qubits
    assert features.depth == depth
    assert dict(features.gate_counts) == dict(gate_counts)
    assert features.total_gate_count == total_gate_count
    assert features.multi_qubit_gate_count == multi_qubit_gate_count
    assert features.single_qubit_gate_count == single_qubit_gate_count
    assert features.num_parameters == num_parameters
    assert features.has_measurements is has_measurements
    assert features.estimated_noise_factor == pytest.approx(estimated_noise_factor)
    assert features.noise_category == noise_category
    assert features.pec_overhead_estimate == pytest.approx(pec_overhead_estimate)
    assert features.layer_heterogeneity == pytest.approx(layer_heterogeneity)
    assert features.non_clifford_count == non_clifford_count
    assert features.non_clifford_fraction == pytest.approx(non_clifford_fraction)


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


class TestQiskitParity:
    """Pin representative Qiskit feature values before adding Cirq support."""

    def test_bell_feature_values_are_unchanged(self) -> None:
        features = analyze_circuit(_qiskit_bell())

        assert features.frontend == "qiskit"
        assert features.analysis_basis == "qiskit"
        _assert_features(
            features,
            num_qubits=2,
            depth=3,
            gate_counts={"h": 1, "cx": 1},
            total_gate_count=2,
            multi_qubit_gate_count=1,
            single_qubit_gate_count=1,
            num_parameters=0,
            has_measurements=True,
            estimated_noise_factor=0.011,
            noise_category="low",
            pec_overhead_estimate=1.022244,
            layer_heterogeneity=0.0,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )

    def test_parameterized_rotation_feature_values_are_unchanged(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(2, 2)
        qc.rx(theta, 0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        with pytest.warns(UserWarning, match="unbound parameter"):
            features = analyze_circuit(qc)

        _assert_features(
            features,
            num_qubits=2,
            depth=3,
            gate_counts={"rx": 1, "cx": 1},
            total_gate_count=2,
            multi_qubit_gate_count=1,
            single_qubit_gate_count=1,
            num_parameters=1,
            has_measurements=True,
            estimated_noise_factor=0.011,
            noise_category="low",
            pec_overhead_estimate=1.022244,
            layer_heterogeneity=0.0,
            non_clifford_count=1,
            non_clifford_fraction=0.5,
        )

    def test_no_measurement_warning_path_is_unchanged(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        with pytest.warns(UserWarning, match="no measurements"):
            features = analyze_circuit(qc)

        _assert_features(
            features,
            num_qubits=2,
            depth=2,
            gate_counts={"h": 1, "cx": 1},
            total_gate_count=2,
            multi_qubit_gate_count=1,
            single_qubit_gate_count=1,
            num_parameters=0,
            has_measurements=False,
            estimated_noise_factor=0.011,
            noise_category="low",
            pec_overhead_estimate=1.022244,
            layer_heterogeneity=0.0,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )

    def test_non_clifford_rotation_feature_values_are_unchanged(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.ry(0.3, 0)
        qc.rz(0.5, 1)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        features = analyze_circuit(qc)

        _assert_features(
            features,
            num_qubits=2,
            depth=3,
            gate_counts={"ry": 1, "rz": 1, "cx": 1},
            total_gate_count=3,
            multi_qubit_gate_count=1,
            single_qubit_gate_count=2,
            num_parameters=0,
            has_measurements=True,
            estimated_noise_factor=0.012,
            noise_category="low",
            pec_overhead_estimate=1.02429,
            layer_heterogeneity=0.0,
            non_clifford_count=2,
            non_clifford_fraction=0.666667,
        )

    def test_multi_layer_cx_cz_feature_values_are_unchanged(self) -> None:
        qc = QuantumCircuit(3, 3)
        for control, target in ((0, 1), (1, 2), (0, 2), (0, 1), (1, 2), (0, 2)):
            qc.h(control)
            qc.s(target)
            if (control + target) % 2 == 0:
                qc.cz(control, target)
            else:
                qc.cx(control, target)
        qc.measure(range(3), range(3))

        features = analyze_circuit(qc)

        _assert_features(
            features,
            num_qubits=3,
            depth=13,
            gate_counts={"h": 6, "s": 6, "cx": 4, "cz": 2},
            total_gate_count=18,
            multi_qubit_gate_count=6,
            single_qubit_gate_count=12,
            num_parameters=0,
            has_measurements=True,
            estimated_noise_factor=0.072,
            noise_category="moderate",
            pec_overhead_estimate=2.372632,
            layer_heterogeneity=0.5,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )


class TestCirqAnalysis:
    """Cirq circuit analysis behavior for the Phase 2 foundation."""

    def test_measured_bell_circuit(self) -> None:
        features = analyze_circuit(_cirq_bell())

        assert features.frontend == "cirq"
        assert features.analysis_basis == "cirq"
        _assert_features(
            features,
            num_qubits=2,
            depth=3,
            gate_counts={"h": 1, "cx": 1},
            total_gate_count=2,
            multi_qubit_gate_count=1,
            single_qubit_gate_count=1,
            num_parameters=0,
            has_measurements=True,
            estimated_noise_factor=0.011,
            noise_category="low",
            pec_overhead_estimate=1.022244,
            layer_heterogeneity=0.0,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )

    def test_no_measurement_circuit_warns(self) -> None:
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        with pytest.warns(UserWarning, match="no measurements"):
            features = analyze_circuit(circuit)

        assert features.has_measurements is False

    @pytest.mark.parametrize(
        "circuit",
        [
            cirq.Circuit(),
            cirq.Circuit(cirq.measure(*cirq.LineQubit.range(2), key="m")),
        ],
    )
    def test_empty_or_measurement_only_circuit_raises_value_error(
        self, circuit
    ) -> None:
        with pytest.raises(ValueError, match="no gate operations"):
            analyze_circuit(circuit)

    def test_parameterized_rotation_warns_and_counts_parameter(self) -> None:
        q0 = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")
        circuit = cirq.Circuit(cirq.rx(theta).on(q0), cirq.measure(q0, key="m"))

        with pytest.warns(UserWarning, match="unbound parameter"):
            features = analyze_circuit(circuit)

        assert features.num_parameters == 1
        assert features.non_clifford_count == 1
        assert features.gate_counts["rx"] == 1

    def test_t_like_zpowgate_counts_as_non_clifford(self) -> None:
        q0 = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ZPowGate(exponent=0.25).on(q0), cirq.measure(q0))

        features = analyze_circuit(circuit)

        assert features.non_clifford_count == 1
        assert features.non_clifford_fraction == 1.0

    def test_clifford_only_circuit_has_zero_non_clifford_count(self) -> None:
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.S(q1),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1),
        )

        features = analyze_circuit(circuit)

        assert features.non_clifford_count == 0

    def test_multi_moment_circuit_has_nonnegative_layer_heterogeneity(self) -> None:
        q0, q1, q2, q3 = cirq.LineQubit.range(4)
        circuit = cirq.Circuit(
            cirq.Moment(cirq.CNOT(q0, q1), cirq.CNOT(q2, q3)),
            cirq.Moment(cirq.CZ(q1, q2)),
            cirq.measure(q0, q1, q2, q3),
        )

        features = analyze_circuit(circuit)

        assert features.layer_heterogeneity >= 0.0
        assert features.multi_qubit_gate_count == 3

    def test_circuit_operation_counts_as_non_clifford_without_crashing(self) -> None:
        q0 = cirq.LineQubit(0)
        subcircuit = cirq.FrozenCircuit(cirq.H(q0))
        circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit), cirq.measure(q0))

        features = analyze_circuit(circuit)

        assert features.total_gate_count == 1
        assert features.non_clifford_count == 1

    def test_cirq_gate_names_cover_common_pow_gate_variants(self) -> None:
        class CustomGate(cirq.Gate):
            def _num_qubits_(self) -> int:
                return 1

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.SWAP(q0, q1),
            cirq.X(q0),
            cirq.Y(q0),
            cirq.Z(q0),
            cirq.ZPowGate(exponent=-0.5).on(q0),
            cirq.ZPowGate(exponent=-0.25).on(q0),
            cirq.I(q0),
            CustomGate().on(q0),
            cirq.measure(q0, q1),
        )

        features = analyze_circuit(circuit)

        assert dict(features.gate_counts) == {
            "swap": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "sdg": 1,
            "tdg": 1,
            "i": 1,
            "customgate": 1,
        }
        assert features.non_clifford_count == 2

    def test_raw_strings_are_rejected(self) -> None:
        with pytest.raises(
            TypeError, match="Raw string circuit input is not supported"
        ):
            analyze_circuit("OPENQASM 2.0;")

    def test_explicit_wrong_frontend_raises_clear_error(self) -> None:
        with pytest.raises(TypeError, match="frontend='cirq'.*got QuantumCircuit"):
            analyze_circuit(_qiskit_bell(), frontend="cirq")
