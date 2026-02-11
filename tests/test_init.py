"""Tests for emrg public API -- __init__.py, GeneratedRecipe, generate_recipe."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

import emrg
from emrg import (
    CircuitFeatures,
    GeneratedRecipe,
    MitigationRecipe,
    analyze_circuit,
    generate_code,
    generate_recipe,
    recommend,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# Tests: generate_recipe()
# ---------------------------------------------------------------------------


class TestGenerateRecipe:
    """Verify the one-liner convenience function."""

    def test_returns_generated_recipe(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert isinstance(result, GeneratedRecipe)

    def test_code_is_string(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert isinstance(result.code, str)
        assert "execute_with_zne" in result.code

    def test_rationale_is_tuple_of_strings(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert isinstance(result.rationale, tuple)
        assert len(result.rationale) > 0
        assert all(isinstance(line, str) for line in result.rationale)

    def test_features_attached(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert isinstance(result.features, CircuitFeatures)
        assert result.features.num_qubits == 2
        assert result.features.multi_qubit_gate_count == 1

    def test_recipe_attached(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert isinstance(result.recipe, MitigationRecipe)
        assert result.recipe.technique == "zne"

    def test_explain_mode(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit, explain=True)
        assert "Rationale:" in result.code

    def test_no_explain_by_default(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert "Rationale:" not in result.code

    def test_circuit_name(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit, circuit_name="qc")
        assert "    qc," in result.code

    def test_code_is_valid_python(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        compile(result.code, "<emrg-test>", "exec")


# ---------------------------------------------------------------------------
# Tests: GeneratedRecipe dataclass
# ---------------------------------------------------------------------------


class TestGeneratedRecipe:
    """Verify GeneratedRecipe properties."""

    def test_immutable(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        with pytest.raises(AttributeError):
            result.code = "modified"  # type: ignore[misc]

    def test_str_returns_code(self, bell_circuit: QuantumCircuit) -> None:
        result = generate_recipe(bell_circuit)
        assert str(result) == result.code

    def test_print_works(
        self, bell_circuit: QuantumCircuit, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = generate_recipe(bell_circuit)
        print(result)
        captured = capsys.readouterr()
        assert "execute_with_zne" in captured.out


# ---------------------------------------------------------------------------
# Tests: public imports
# ---------------------------------------------------------------------------


class TestPublicImports:
    """Verify all __all__ names are importable."""

    def test_version(self) -> None:
        assert emrg.__version__ == "0.1.0"

    def test_all_names_importable(self) -> None:
        for name in emrg.__all__:
            assert hasattr(emrg, name), f"emrg.{name} not found"

    def test_reexports_are_correct_types(self) -> None:
        assert callable(analyze_circuit)
        assert callable(recommend)
        assert callable(generate_code)
        assert callable(generate_recipe)
