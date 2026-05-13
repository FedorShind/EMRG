"""Representative circuit validation for the full EMRG pipeline."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import ParameterVector

from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.cli import main
from emrg.codegen import generate_code
from emrg.heuristics import MitigationRecipe, recommend
from emrg.preview import run_preview

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"


@dataclass(frozen=True)
class CircuitCase:
    name: str
    build: Callable[[], QuantumCircuit]
    noise_model_available: bool = False
    expect_unbound_warning: bool = False


def _ghz() -> QuantumCircuit:
    qc = QuantumCircuit(5, 5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure(range(5), range(5))
    return qc


def _bernstein_vazirani() -> QuantumCircuit:
    secret = (1, 0, 1, 1)
    qc = QuantumCircuit(5, 4)
    qc.x(4)
    qc.h(range(5))
    for i, bit in enumerate(secret):
        if bit:
            qc.cx(i, 4)
    qc.h(range(4))
    qc.measure(range(4), range(4))
    return qc


def _qft_style() -> QuantumCircuit:
    qc = QuantumCircuit(4, 4)
    for target in range(4):
        qc.h(target)
        for control in range(target + 1, 4):
            angle = math.pi / (2 ** (control - target))
            qc.cp(angle, control, target)
    qc.swap(0, 3)
    qc.swap(1, 2)
    qc.measure(range(4), range(4))
    return qc


def _variational_unbound() -> QuantumCircuit:
    params = ParameterVector("theta", 8)
    qc = QuantumCircuit(4, 4)
    for q in range(4):
        qc.ry(params[q], q)
    for q in range(3):
        qc.cx(q, q + 1)
    for q in range(4):
        qc.rz(params[q + 4], q)
    qc.measure(range(4), range(4))
    return qc


def _variational_bound() -> QuantumCircuit:
    qc = _variational_unbound()
    bindings = {param: math.pi / 4 for param in qc.parameters}
    return qc.assign_parameters(bindings)


def _random_cliffordish() -> QuantumCircuit:
    rng = random.Random(20260513)
    qc = QuantumCircuit(5, 5)
    single_qubit_gates = ("h", "s", "x", "z")

    for _ in range(5):
        for q in range(5):
            getattr(qc, rng.choice(single_qubit_gates))(q)
        left = rng.randrange(4)
        if rng.random() < 0.5:
            qc.cx(left, left + 1)
        else:
            qc.cz(left, left + 1)

    qc.measure(range(5), range(5))
    return qc


def _non_clifford_rotations() -> QuantumCircuit:
    qc = QuantumCircuit(4, 4)
    for _ in range(3):
        for q in range(4):
            qc.ry(math.pi / 7, q)
            qc.rz(math.pi / 5, q)
        for q in range(3):
            qc.cx(q, q + 1)
    qc.measure(range(4), range(4))
    return qc


def _layered_cx_cz() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    pairs = ((0, 1), (1, 2), (0, 2), (0, 1), (1, 2), (0, 2))
    for control, target in pairs:
        qc.h(control)
        qc.s(target)
        if (control + target) % 2 == 0:
            qc.cz(control, target)
        else:
            qc.cx(control, target)
    qc.measure(range(3), range(3))
    return qc


def _qasm_loaded_ghz() -> QuantumCircuit:
    return qasm2.load(str(EXAMPLES_DIR / "ghz_state.qasm"))


REAL_CIRCUITS = (
    CircuitCase("ghz", _ghz),
    CircuitCase("bernstein_vazirani", _bernstein_vazirani),
    CircuitCase("qft_style", _qft_style),
    CircuitCase(
        "variational_unbound",
        _variational_unbound,
        expect_unbound_warning=True,
    ),
    CircuitCase("variational_bound", _variational_bound),
    CircuitCase("random_cliffordish", _random_cliffordish),
    CircuitCase("non_clifford_rotations", _non_clifford_rotations),
    CircuitCase("layered_cx_cz", _layered_cx_cz, noise_model_available=True),
    CircuitCase("qasm_loaded_ghz", _qasm_loaded_ghz),
)


def _analyze(case: CircuitCase, qc: QuantumCircuit) -> CircuitFeatures:
    if case.expect_unbound_warning:
        with pytest.warns(UserWarning, match="unbound parameter"):
            return analyze_circuit(qc, noise_model_available=case.noise_model_available)
    return analyze_circuit(qc, noise_model_available=case.noise_model_available)


def _assert_sane_features(features: CircuitFeatures) -> None:
    assert features.num_qubits > 0
    assert features.depth >= 0
    assert features.total_gate_count > 0
    assert features.multi_qubit_gate_count >= 0
    assert features.single_qubit_gate_count >= 0
    assert features.num_parameters >= 0
    assert features.estimated_noise_factor >= 0
    assert features.pec_overhead_estimate >= 1.0
    assert features.layer_heterogeneity >= 0
    assert features.non_clifford_count >= 0
    assert 0 <= features.non_clifford_fraction <= 1
    assert features.has_measurements is True
    assert (
        features.single_qubit_gate_count + features.multi_qubit_gate_count
        == features.total_gate_count
    )


def _assert_valid_recipe(recipe: MitigationRecipe) -> None:
    assert recipe.technique in {"zne", "pec", "cdr", "composite"}
    assert recipe.estimated_overhead >= 1.0
    assert recipe.rationale

    if recipe.technique == "zne":
        assert recipe.factory_name in {
            "LinearFactory",
            "RichardsonFactory",
            "PolyFactory",
        }
        assert recipe.scale_factors
        assert recipe.scaling_method in {"fold_global", "fold_gates_at_random"}
    elif recipe.technique == "composite":
        assert len(recipe.components) == 2
        assert [component.technique for component in recipe.components] == [
            "zne",
            "pec",
        ]


@pytest.mark.parametrize("case", REAL_CIRCUITS, ids=lambda case: case.name)
def test_real_circuit_pipeline(case: CircuitCase) -> None:
    qc = case.build()

    features = _analyze(case, qc)
    _assert_sane_features(features)

    recipe = recommend(features)
    _assert_valid_recipe(recipe)

    code = generate_code(recipe, features)
    compile(code, f"<emrg-real-circuit-{case.name}>", "exec")


def test_cli_analyze_and_generate_qasm_example() -> None:
    runner = CliRunner()
    qasm_path = EXAMPLES_DIR / "bell_state.qasm"

    analyze_result = runner.invoke(main, ["analyze", str(qasm_path), "--json"])
    assert analyze_result.exit_code == 0
    assert '"num_qubits": 2' in analyze_result.output
    assert '"gate_counts"' in analyze_result.output

    generate_result = runner.invoke(main, ["generate", str(qasm_path)])
    assert generate_result.exit_code == 0
    assert "Error Mitigation Recipe" in generate_result.output
    assert "execute_with_" in generate_result.output
    compile(generate_result.output, "<emrg-cli-generated-example>", "exec")


def test_preview_runs_on_small_real_circuit_subset() -> None:
    pytest.importorskip("cirq")
    qc = _ghz()
    features = analyze_circuit(qc)
    recipe = recommend(features)

    result = run_preview(qc, recipe, noise_level=0.01)

    assert result.warning is None or "failed" not in result.warning.lower()
    assert result.ideal_value is not None
    assert result.noisy_value is not None
    assert result.mitigated_value is not None
