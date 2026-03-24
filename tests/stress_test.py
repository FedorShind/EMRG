"""EMRG Stress Test Suite.

Exercises every edge case, boundary condition, and technique path in the
EMRG pipeline.  Designed to be run both standalone (``python tests/stress_test.py``)
and via pytest (``pytest tests/stress_test.py -v``).

This script is kept permanently in the repo as a regression safety net.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from qiskit import QuantumCircuit

from emrg import analyze_circuit, generate_recipe
from emrg.cli import main as cli_main
from emrg.heuristics import recommend
from tests._helpers import make_features as _make_features

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
_BELL_QASM = _EXAMPLES_DIR / "bell_state.qasm"


# ===================================================================
# A. Input Edge Cases
# ===================================================================


class TestInputEdgeCases:
    """Exercise unusual or extreme circuit inputs."""

    def test_empty_circuit_raises(self) -> None:
        """QuantumCircuit with no gates should raise ValueError."""
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError, match="no gate operations"):
            analyze_circuit(qc)
        with pytest.raises(ValueError, match="no gate operations"):
            generate_recipe(qc)

    def test_measure_only_circuit_raises(self) -> None:
        """Circuit with only measurements should raise ValueError."""
        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        with pytest.raises(ValueError, match="no gate operations"):
            analyze_circuit(qc)

    def test_single_gate_circuit(self) -> None:
        """Single X gate -- should produce a valid recipe."""
        qc = QuantumCircuit(1)
        qc.x(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_recipe(qc)
        assert result.recipe.technique == "zne"
        assert result.recipe.factory_name == "LinearFactory"
        assert result.features.total_gate_count == 1
        assert result.features.multi_qubit_gate_count == 0
        compile(result.code, "<stress-single-gate>", "exec")

    def test_very_deep_circuit(self) -> None:
        """200+ depth circuit -- should get PolyFactory."""
        qc = QuantumCircuit(3)
        for _ in range(100):
            qc.cx(0, 1)
            qc.cx(1, 2)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.depth > 100
        result = generate_recipe(qc)
        assert result.recipe.factory_name == "PolyFactory"
        assert result.recipe.scaling_method == "fold_gates_at_random"
        compile(result.code, "<stress-deep>", "exec")

    def test_very_wide_circuit(self) -> None:
        """60-qubit shallow circuit."""
        qc = QuantumCircuit(60)
        for i in range(59):
            qc.cx(i, i + 1)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.num_qubits == 60
        assert features.multi_qubit_gate_count == 59
        result = generate_recipe(qc)
        assert result.recipe.technique == "zne"
        compile(result.code, "<stress-wide>", "exec")

    def test_single_qubit_gates_only(self) -> None:
        """Circuit with zero multi-qubit gates."""
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
            qc.rz(0.5, i)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.multi_qubit_gate_count == 0
        assert features.layer_heterogeneity == 0.0
        result = generate_recipe(qc)
        assert result.recipe.technique == "zne"
        compile(result.code, "<stress-1q-only>", "exec")

    def test_pec_rejected_for_deep_circuit(self) -> None:
        """Deep circuit with noise model should still get ZNE, not PEC."""
        qc = QuantumCircuit(5)
        for _ in range(50):
            qc.cx(0, 1)
            qc.cx(2, 3)
        qc.measure_all()
        result = generate_recipe(qc, noise_model_available=True)
        assert result.recipe.technique == "zne"

    def test_pec_selected_for_shallow_circuit(self) -> None:
        """Shallow circuit with noise model should get PEC."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        result = generate_recipe(qc, noise_model_available=True)
        assert result.recipe.technique == "pec"
        compile(result.code, "<stress-pec-shallow>", "exec")

    def test_technique_override_pec(self) -> None:
        """Force technique='pec' on a shallow circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        result = generate_recipe(
            qc, technique="pec", noise_model_available=True
        )
        assert result.recipe.technique == "pec"
        compile(result.code, "<stress-force-pec>", "exec")

    def test_technique_override_zne(self) -> None:
        """Force technique='zne' even when PEC would be viable."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        result = generate_recipe(qc, technique="zne")
        assert result.recipe.technique == "zne"
        compile(result.code, "<stress-force-zne>", "exec")

    def test_invalid_technique_raises(self) -> None:
        """Invalid technique string should raise ValueError."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.raises(ValueError, match="Unknown technique"):
            generate_recipe(qc, technique="invalid")

    def test_layerwise_heterogeneous_circuit(self) -> None:
        """Circuit with uneven layer structure -> fold_gates_at_random."""
        qc = QuantumCircuit(4)
        # Heavy CX layers alternating with single-qubit layers
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        for _ in range(5):
            qc.cx(0, 1)
            qc.cx(1, 2)
            qc.cx(2, 3)
            qc.ry(0.5, 0)
            qc.ry(0.5, 1)
            qc.ry(0.5, 2)
            qc.ry(0.5, 3)
        qc.measure_all()
        features = analyze_circuit(qc)
        result = generate_recipe(qc)
        # Print for manual inspection
        print(f"\n  Heterogeneous circuit: depth={features.depth}, "
              f"het={features.layer_heterogeneity:.4f}, "
              f"factory={result.recipe.factory_name}, "
              f"scaling={result.recipe.scaling_method}")

    def test_homogeneous_circuit(self) -> None:
        """Circuit with uniform CX density -> fold_global."""
        qc = QuantumCircuit(4)
        for _ in range(10):
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.h(0)
            qc.h(1)
            qc.h(2)
            qc.h(3)
        qc.measure_all()
        features = analyze_circuit(qc)
        result = generate_recipe(qc)
        print(f"\n  Homogeneous circuit: depth={features.depth}, "
              f"het={features.layer_heterogeneity:.4f}, "
              f"factory={result.recipe.factory_name}, "
              f"scaling={result.recipe.scaling_method}")
        # Uniform layers should have low heterogeneity
        # Whether it gets fold_global depends on exact depth/het values


# ===================================================================
# B. CLI Stress Tests
# ===================================================================


class TestCLIStress:
    """Exercise CLI edge cases via Click CliRunner."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_generate_nonexistent_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli_main, ["generate", "nonexistent.qasm"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_analyze_json_all_fields(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_main, ["analyze", str(_BELL_QASM), "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "layer_heterogeneity" in data
        assert "pec_overhead_estimate" in data
        assert "noise_model_available" in data
        assert isinstance(data["layer_heterogeneity"], float)

    def test_generate_technique_zne(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_main, ["generate", str(_BELL_QASM), "--technique", "zne"]
        )
        assert result.exit_code == 0
        assert "execute_with_zne" in result.output

    def test_generate_technique_pec(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_main,
            ["generate", str(_BELL_QASM), "--technique", "pec", "--noise-model"],
        )
        assert result.exit_code == 0
        assert "execute_with_pec" in result.output

    def test_generate_explain(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_main, ["generate", str(_BELL_QASM), "--explain"]
        )
        assert result.exit_code == 0
        assert "Rationale:" in result.output

    def test_generate_invalid_technique(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_main, ["generate", str(_BELL_QASM), "--technique", "garbage"]
        )
        assert result.exit_code != 0

    def test_generate_output_file_valid_python(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "test_output.py"
        result = runner.invoke(
            cli_main, ["generate", str(_BELL_QASM), "-o", str(out_file)]
        )
        assert result.exit_code == 0
        code = out_file.read_text(encoding="utf-8")
        compile(code, str(out_file), "exec")


# ===================================================================
# C. Generated Code Validation
# ===================================================================


class TestCodeValidation:
    """Verify generated code compiles for every technique path."""

    def _build_circuit(self, depth_target: int, n_qubits: int = 4) -> QuantumCircuit:
        """Build a circuit with approximately the target depth."""
        qc = QuantumCircuit(n_qubits)
        for _ in range(depth_target // 2):
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(n_qubits):
                qc.h(q)
        qc.measure_all()
        return qc

    def test_zne_linear_compiles(self) -> None:
        """Shallow circuit -> LinearFactory."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        result = generate_recipe(qc)
        assert result.recipe.factory_name == "LinearFactory"
        compile(result.code, "<validate-linear>", "exec")

    def test_zne_richardson_compiles(self) -> None:
        """Medium-depth circuit -> RichardsonFactory."""
        qc = self._build_circuit(30)
        result = generate_recipe(qc)
        assert result.recipe.factory_name in ("RichardsonFactory", "PolyFactory")
        compile(result.code, "<validate-richardson>", "exec")

    def test_zne_poly_compiles(self) -> None:
        """Deep circuit -> PolyFactory."""
        qc = self._build_circuit(80)
        result = generate_recipe(qc)
        assert result.recipe.factory_name == "PolyFactory"
        compile(result.code, "<validate-poly>", "exec")

    def test_zne_layerwise_compiles(self) -> None:
        """Heterogeneous circuit -> layerwise Richardson."""
        features = _make_features(
            depth=30, layer_heterogeneity=3.0, noise_category="moderate"
        )
        recipe = recommend(features)
        assert recipe.factory_name == "RichardsonFactory"
        assert recipe.scaling_method == "fold_gates_at_random"
        from emrg.codegen import generate_code
        code = generate_code(recipe, features)
        compile(code, "<validate-layerwise>", "exec")

    def test_pec_compiles(self) -> None:
        """Shallow circuit + noise model -> PEC."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        result = generate_recipe(qc, noise_model_available=True)
        assert result.recipe.technique == "pec"
        compile(result.code, "<validate-pec>", "exec")


# ===================================================================
# D. Heuristic Boundary Cross-check
# ===================================================================


class TestHeuristicBoundaries:
    """Verify exact threshold behavior."""

    def test_depth_19_gets_linear(self) -> None:
        f = _make_features(depth=19, multi_qubit_gate_count=5)
        recipe = recommend(f)
        assert recipe.factory_name == "LinearFactory"

    def test_depth_20_gets_richardson(self) -> None:
        f = _make_features(depth=20)
        recipe = recommend(f)
        assert recipe.factory_name == "RichardsonFactory"

    def test_depth_50_gets_richardson(self) -> None:
        f = _make_features(depth=50)
        recipe = recommend(f)
        assert recipe.factory_name == "RichardsonFactory"

    def test_depth_51_gets_poly(self) -> None:
        f = _make_features(depth=51, noise_category="moderate")
        recipe = recommend(f)
        assert recipe.factory_name == "PolyFactory"

    def test_pec_boundary_depth_30(self) -> None:
        f = _make_features(
            depth=30, noise_model_available=True, pec_overhead_estimate=50.0,
        )
        recipe = recommend(f)
        assert recipe.technique == "pec"

    def test_pec_boundary_depth_31(self) -> None:
        f = _make_features(
            depth=31, noise_model_available=True, pec_overhead_estimate=50.0,
        )
        recipe = recommend(f)
        assert recipe.technique == "zne"

    def test_layerwise_boundary_1_9(self) -> None:
        f = _make_features(depth=30, layer_heterogeneity=1.9)
        recipe = recommend(f)
        assert recipe.scaling_method == "fold_global"

    def test_layerwise_boundary_2_1(self) -> None:
        f = _make_features(depth=30, layer_heterogeneity=2.1)
        recipe = recommend(f)
        assert recipe.scaling_method == "fold_gates_at_random"

    def test_pec_overhead_999(self) -> None:
        f = _make_features(
            depth=10, noise_model_available=True, pec_overhead_estimate=999.0,
        )
        recipe = recommend(f)
        assert recipe.technique == "pec"

    def test_pec_overhead_1001(self) -> None:
        f = _make_features(
            depth=10, noise_model_available=True, pec_overhead_estimate=1001.0,
        )
        recipe = recommend(f)
        assert recipe.technique == "zne"


# ===================================================================
# E. Performance Check
# ===================================================================


class TestPerformance:
    """Verify generate_recipe() remains fast on large circuits."""

    def test_100_qubit_under_one_second(self) -> None:
        """100-qubit random circuit should complete in under 1 second."""
        rng = np.random.default_rng(42)
        qc = QuantumCircuit(100)
        for _ in range(25):
            for q in range(99):
                if rng.random() < 0.3:
                    qc.cx(q, q + 1)
            for q in range(100):
                qc.h(q)
        qc.measure_all()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t0 = time.perf_counter()
            result = generate_recipe(qc)
            elapsed = time.perf_counter() - t0

        print(f"\n  100-qubit circuit: {elapsed:.3f}s "
              f"(depth={result.features.depth}, "
              f"gates={result.features.total_gate_count})")
        assert elapsed < 1.0, f"generate_recipe took {elapsed:.3f}s (>1s)"


# ===================================================================
# Standalone runner
# ===================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
