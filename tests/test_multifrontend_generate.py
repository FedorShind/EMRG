"""Public API coverage for Cirq inputs."""

from __future__ import annotations

import pytest

from emrg import (
    DEFAULT_POLICY,
    Frontend,
    GeneratedRecipe,
    detect_frontend,
    generate_recipe,
)

cirq = pytest.importorskip("cirq")


def _cirq_bell():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="m"),
    )


def test_frontend_exports_are_available_from_public_api() -> None:
    assert Frontend.CIRQ.value == "cirq"
    assert detect_frontend(_cirq_bell()) is Frontend.CIRQ


def test_generate_recipe_accepts_cirq_bell_circuit() -> None:
    result = generate_recipe(_cirq_bell())

    assert isinstance(result, GeneratedRecipe)
    assert result.features.num_qubits == 2
    assert result.recipe.technique in {"zne", "pec", "cdr", "composite"}
    assert result.rationale


def test_generate_recipe_accepts_explicit_cirq_frontend() -> None:
    result = generate_recipe(_cirq_bell(), frontend="cirq")

    assert result.features.gate_counts["cx"] == 1
    assert result.recipe.technique == "zne"


def test_generate_recipe_rejects_wrong_explicit_frontend() -> None:
    with pytest.raises(TypeError, match="Expected a qiskit.QuantumCircuit"):
        generate_recipe(_cirq_bell(), frontend="qiskit")


def test_generated_code_from_cirq_input_compiles() -> None:
    result = generate_recipe(_cirq_bell())

    compile(result.code, "<emrg-cirq-generated>", "exec")


def test_str_for_cirq_result_returns_code_without_preview() -> None:
    result = generate_recipe(_cirq_bell())

    assert str(result) == result.code


def test_generate_recipe_cirq_with_default_policy_works() -> None:
    result = generate_recipe(_cirq_bell(), policy=DEFAULT_POLICY)

    assert result.recipe.technique == "zne"
    assert "LinearFactory" in result.code


def test_forced_pec_warning_surfaces_for_cirq_input() -> None:
    result = generate_recipe(_cirq_bell(), technique="pec")

    assert result.recipe.technique == "pec"
    assert result.recipe.warnings
    assert "# Warnings:" in result.code
