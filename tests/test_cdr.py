"""Tests for CDR (Clifford Data Regression) support across all modules."""

from __future__ import annotations

import json
import math

import pytest
from click.testing import CliRunner

from emrg.analyzer import (
    CDR_MAX_DEPTH,
    CDR_MIN_DEPTH,
    CDR_NON_CLIFFORD_FRACTION_THRESHOLD,
    CircuitFeatures,
    analyze_circuit,
)
from emrg.cli import main
from emrg.codegen import generate_code
from emrg.heuristics import (
    CDR_GATE_THRESHOLD_LARGE,
    CDR_GATE_THRESHOLD_MEDIUM,
    CDR_TRAINING_CIRCUITS_LARGE,
    CDR_TRAINING_CIRCUITS_MEDIUM,
    CDR_TRAINING_CIRCUITS_SMALL,
    MitigationRecipe,
    recommend,
)
from tests.conftest import make_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner():
    return CliRunner()


def _make_clifford_circuit():
    """All-Clifford circuit: H, CX, S, X, Z."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.s(1)
    qc.x(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.z(0)
    qc.h(1)
    qc.cx(0, 2)
    qc.measure_all()
    return qc


def _make_non_clifford_circuit(depth_layers: int = 3):
    """Circuit with Rz(0.3) and Ry(0.7) non-Clifford rotations."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4, 4)
    for _ in range(depth_layers):
        for i in range(4):
            qc.rz(0.3, i)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        for i in range(4):
            qc.ry(0.7, i)
        qc.cx(0, 1)
        qc.cx(2, 3)
    qc.measure_all()
    return qc


def _make_t_gate_circuit():
    """Circuit with T and Tdg gates."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.t(0)
    qc.t(1)
    qc.tdg(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _make_clifford_rotation_circuit():
    """Circuit with rotation gates at Clifford angles (multiples of pi/2)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.rz(math.pi / 2, 0)      # S gate equivalent -> Clifford
    qc.rz(math.pi, 1)           # Z gate equivalent -> Clifford
    qc.rx(math.pi, 0)           # X gate equivalent -> Clifford
    qc.ry(math.pi / 2, 1)       # Clifford
    qc.cx(0, 1)
    qc.rz(3 * math.pi / 2, 0)  # Sdg equivalent -> Clifford
    qc.rz(2 * math.pi, 1)       # Identity -> Clifford
    qc.measure_all()
    return qc


def _make_cdr_recipe(
    num_training_circuits: int = 8,
    fit_method: str = "linear",
    noise_category: str = "low",
    estimated_overhead: float = 8.0,
) -> MitigationRecipe:
    from types import MappingProxyType

    return MitigationRecipe(
        technique="cdr",
        factory_name="",
        scale_factors=(),
        factory_kwargs=MappingProxyType({
            "num_training_circuits": num_training_circuits,
            "fit_method": fit_method,
        }),
        scaling_method="",
        rationale=("CDR selected.",),
        noise_category=noise_category,
        estimated_overhead=estimated_overhead,
    )


# ===================================================================
# A. Analyzer tests
# ===================================================================


class TestNonCliffordCounting:
    """Test non_clifford_count and non_clifford_fraction computation."""

    def test_all_clifford_circuit(self):
        """All-Clifford circuit should have fraction 0.0."""
        qc = _make_clifford_circuit()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 0
        assert features.non_clifford_fraction == 0.0

    def test_t_gate_counted(self):
        """T and Tdg gates should be counted as non-Clifford."""
        qc = _make_t_gate_circuit()
        features = analyze_circuit(qc)
        # T, T, Tdg = 3 non-Clifford gates
        assert features.non_clifford_count == 3

    def test_rz_arbitrary_angle(self):
        """Rz with non-Clifford angle should be counted."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.rz(0.123, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 1
        assert features.non_clifford_fraction == 1.0

    def test_rz_pi_over_2_is_clifford(self):
        """Rz(pi/2) is equivalent to S gate -- Clifford."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.rz(math.pi / 2, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 0

    def test_rz_pi_is_clifford(self):
        """Rz(pi) is equivalent to Z gate -- Clifford."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.rz(math.pi, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 0

    def test_rz_pi_over_4_is_non_clifford(self):
        """Rz(pi/4) is T gate equivalent -- non-Clifford."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.rz(math.pi / 4, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 1

    def test_clifford_rotation_circuit(self):
        """All rotations at multiples of pi/2 should be Clifford."""
        qc = _make_clifford_rotation_circuit()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 0
        assert features.non_clifford_fraction == 0.0

    def test_mixed_circuit_fraction(self):
        """Mixed circuit should have correct fraction calculation."""
        qc = _make_non_clifford_circuit(depth_layers=1)
        features = analyze_circuit(qc)
        assert features.non_clifford_count > 0
        assert 0.0 < features.non_clifford_fraction <= 1.0
        assert features.non_clifford_fraction == pytest.approx(
            features.non_clifford_count / features.total_gate_count, abs=1e-6
        )

    def test_zero_gate_fraction(self):
        """Edge case: fraction defaults to 0.0 via dataclass default."""
        f = CircuitFeatures(
            num_qubits=1,
            depth=0,
            total_gate_count=0,
        )
        assert f.non_clifford_fraction == 0.0

    def test_ry_non_clifford_angle(self):
        """Ry with arbitrary angle is non-Clifford."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.ry(0.7, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 1

    def test_rx_pi_is_clifford(self):
        """Rx(pi) is equivalent to X gate -- Clifford."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.rx(math.pi, 0)
        qc.measure_all()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 0


# ===================================================================
# B. Heuristic tests
# ===================================================================


class TestCDRSelection:
    """Test CDR technique selection in the decision engine."""

    def test_high_non_clifford_moderate_depth_selects_cdr(self):
        """High non-Clifford fraction + moderate depth -> CDR."""
        features = make_features(
            depth=25,
            total_gate_count=30,
            non_clifford_count=15,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"

    def test_all_clifford_does_not_select_cdr(self):
        """All-Clifford circuit -> ZNE, not CDR."""
        features = make_features(
            depth=25,
            total_gate_count=30,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )
        recipe = recommend(features)
        assert recipe.technique != "cdr"

    def test_non_clifford_too_deep_selects_zne(self):
        """Non-Clifford but depth > CDR_MAX_DEPTH -> ZNE."""
        features = make_features(
            depth=CDR_MAX_DEPTH + 5,
            total_gate_count=60,
            non_clifford_count=30,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique == "zne"

    def test_non_clifford_too_shallow_selects_zne(self):
        """Non-Clifford but depth < CDR_MIN_DEPTH -> ZNE."""
        features = make_features(
            depth=CDR_MIN_DEPTH - 1,
            total_gate_count=10,
            multi_qubit_gate_count=2,
            non_clifford_count=5,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique != "cdr"

    def test_fraction_below_threshold_selects_zne(self):
        """Non-Clifford fraction below threshold -> ZNE."""
        features = make_features(
            depth=25,
            total_gate_count=30,
            non_clifford_count=3,
            non_clifford_fraction=0.1,
        )
        recipe = recommend(features)
        assert recipe.technique != "cdr"

    def test_pec_wins_over_cdr(self):
        """PEC conditions met AND CDR conditions met -> PEC wins."""
        features = make_features(
            depth=25,
            total_gate_count=30,
            non_clifford_count=15,
            non_clifford_fraction=0.5,
            noise_model_available=True,
            pec_overhead_estimate=10.0,
        )
        recipe = recommend(features)
        assert recipe.technique == "pec"

    def test_technique_cdr_override(self):
        """--technique cdr forces CDR regardless of conditions."""
        features = make_features(
            depth=5,
            total_gate_count=5,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )
        recipe = recommend(features, technique="cdr")
        assert recipe.technique == "cdr"

    def test_cdr_override_on_all_clifford(self):
        """CDR override works even on all-Clifford circuit."""
        features = make_features(
            depth=25,
            non_clifford_count=0,
            non_clifford_fraction=0.0,
        )
        recipe = recommend(features, technique="cdr")
        assert recipe.technique == "cdr"

    def test_cdr_boundary_fraction(self):
        """Fraction exactly at threshold should NOT select CDR (> not >=)."""
        features = make_features(
            depth=25,
            total_gate_count=30,
            non_clifford_count=6,
            non_clifford_fraction=CDR_NON_CLIFFORD_FRACTION_THRESHOLD,
        )
        recipe = recommend(features)
        assert recipe.technique != "cdr"

    def test_cdr_boundary_depth_min(self):
        """Depth exactly at CDR_MIN_DEPTH should select CDR."""
        features = make_features(
            depth=CDR_MIN_DEPTH,
            total_gate_count=30,
            non_clifford_count=15,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"

    def test_cdr_boundary_depth_max(self):
        """Depth exactly at CDR_MAX_DEPTH should select CDR."""
        features = make_features(
            depth=CDR_MAX_DEPTH,
            total_gate_count=30,
            non_clifford_count=15,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"

    def test_invalid_technique_raises(self):
        """Unknown technique string should raise ValueError."""
        features = make_features()
        with pytest.raises(ValueError, match="Unknown technique"):
            recommend(features, technique="invalid")


class TestCDRTrainingCircuits:
    """Test num_training_circuits scaling."""

    def test_small_circuit(self):
        """< 20 gates -> CDR_TRAINING_CIRCUITS_SMALL."""
        features = make_features(
            depth=15,
            total_gate_count=15,
            non_clifford_count=8,
            non_clifford_fraction=0.53,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"
        num_tc = recipe.factory_kwargs["num_training_circuits"]
        assert num_tc == CDR_TRAINING_CIRCUITS_SMALL

    def test_medium_circuit(self):
        """20-50 gates -> CDR_TRAINING_CIRCUITS_MEDIUM."""
        features = make_features(
            depth=25,
            total_gate_count=35,
            non_clifford_count=18,
            non_clifford_fraction=0.51,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"
        num_tc = recipe.factory_kwargs["num_training_circuits"]
        assert num_tc == CDR_TRAINING_CIRCUITS_MEDIUM

    def test_large_circuit(self):
        """> 50 gates -> CDR_TRAINING_CIRCUITS_LARGE."""
        features = make_features(
            depth=35,
            total_gate_count=55,
            non_clifford_count=30,
            non_clifford_fraction=0.55,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"
        num_tc = recipe.factory_kwargs["num_training_circuits"]
        assert num_tc == CDR_TRAINING_CIRCUITS_LARGE

    def test_boundary_medium(self):
        """Exactly CDR_GATE_THRESHOLD_MEDIUM gates -> medium."""
        features = make_features(
            depth=20,
            total_gate_count=CDR_GATE_THRESHOLD_MEDIUM,
            non_clifford_count=10,
            non_clifford_fraction=0.5,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"
        num_tc = recipe.factory_kwargs["num_training_circuits"]
        assert num_tc == CDR_TRAINING_CIRCUITS_MEDIUM

    def test_boundary_large(self):
        """Exactly CDR_GATE_THRESHOLD_LARGE gates -> medium."""
        features = make_features(
            depth=30,
            total_gate_count=CDR_GATE_THRESHOLD_LARGE,
            non_clifford_count=26,
            non_clifford_fraction=0.52,
        )
        recipe = recommend(features)
        assert recipe.technique == "cdr"
        num_tc = recipe.factory_kwargs["num_training_circuits"]
        assert num_tc == CDR_TRAINING_CIRCUITS_MEDIUM


# ===================================================================
# C. Codegen tests
# ===================================================================


class TestCDRCodegen:
    """Test CDR code generation template."""

    def test_cdr_generates_valid_python(self):
        """CDR template should produce code that compiles."""
        recipe = _make_cdr_recipe()
        features = make_features(
            non_clifford_count=5,
            non_clifford_fraction=0.5,
        )
        code = generate_code(recipe, features)
        compile(code, "<cdr_test>", "exec")

    def test_cdr_imports(self):
        """CDR template should import execute_with_cdr."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features)
        assert "from mitiq.cdr import execute_with_cdr" in code

    def test_cdr_includes_simulator(self):
        """CDR template should include a simulator function."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features)
        assert "def simulator(circuit):" in code
        assert "DensityMatrixSimulator" in code

    def test_cdr_includes_num_training(self):
        """CDR template should include num_training_circuits from recipe."""
        recipe = _make_cdr_recipe(num_training_circuits=12)
        features = make_features()
        code = generate_code(recipe, features)
        assert "num_training_circuits = 12" in code

    def test_cdr_includes_execute_call(self):
        """CDR template should include execute_with_cdr call."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features)
        assert "execute_with_cdr(" in code
        assert "simulator=simulator" in code

    def test_cdr_header_recommendation(self):
        """CDR template header should say CDR."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features)
        assert "CDR (Clifford Data Regression)" in code

    def test_cdr_explain_mode(self):
        """CDR template with explain=True should include rationale."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features, explain=True)
        assert "Rationale:" in code
        assert "training circuits" in code.lower()

    def test_cdr_parameter_warning(self):
        """CDR template should show parameter warning when applicable."""
        recipe = _make_cdr_recipe()
        features = make_features(num_parameters=3)
        code = generate_code(recipe, features)
        assert "WARNING" in code
        assert "unbound parameter" in code

    def test_cdr_custom_circuit_name(self):
        """CDR template should use custom circuit name."""
        recipe = _make_cdr_recipe()
        features = make_features()
        code = generate_code(recipe, features, circuit_name="my_circuit")
        assert "my_circuit," in code


# ===================================================================
# D. CLI tests
# ===================================================================


class TestCDRCli:
    """Test CDR-related CLI behavior."""

    def test_analyze_shows_non_clifford(self, runner, tmp_path):
        """emrg analyze should show non_clifford fields."""
        qasm = tmp_path / "test.qasm"
        qasm.write_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(main, ["analyze", str(qasm)])
        assert result.exit_code == 0
        assert "Non-Clifford gates:" in result.output
        assert "Non-Clifford frac:" in result.output

    def test_analyze_json_includes_non_clifford(self, runner, tmp_path):
        """emrg analyze --json should include new fields."""
        qasm = tmp_path / "test.qasm"
        qasm.write_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(main, ["analyze", "--json", str(qasm)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "non_clifford_count" in data
        assert "non_clifford_fraction" in data

    def test_generate_technique_cdr(self, runner, tmp_path):
        """emrg generate --technique cdr should work."""
        qasm = tmp_path / "test.qasm"
        qasm.write_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(main, ["generate", "--technique", "cdr", str(qasm)])
        assert result.exit_code == 0
        assert "execute_with_cdr" in result.output

    def test_generate_technique_invalid(self, runner, tmp_path):
        """emrg generate --technique invalid should error."""
        qasm = tmp_path / "test.qasm"
        qasm.write_text(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\n"
            "h q[0];\ncx q[0],q[1];\n"
            "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        result = runner.invoke(
            main, ["generate", "--technique", "invalid", str(qasm)]
        )
        assert result.exit_code != 0


# ===================================================================
# E. Preview tests (cirq-dependent)
# ===================================================================


try:
    import cirq  # noqa: F401
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


@pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")
class TestCDRPreview:
    """Test CDR preview simulation path."""

    def test_cdr_preview_produces_result(self):
        """CDR preview should produce a populated PreviewResult."""
        from emrg.preview import run_preview

        qc = _make_non_clifford_circuit(depth_layers=1)
        recipe = _make_cdr_recipe()
        result = run_preview(qc, recipe, noise_level=0.01)
        # CDR may or may not succeed depending on Mitiq internals,
        # but it should not raise and should return a PreviewResult.
        assert result is not None
        assert result.technique == "CDR"

    def test_cdr_preview_custom_noise_level(self):
        """CDR preview with custom noise level should work."""
        from emrg.preview import run_preview

        qc = _make_non_clifford_circuit(depth_layers=1)
        recipe = _make_cdr_recipe()
        result = run_preview(qc, recipe, noise_level=0.05)
        assert result is not None
        assert result.noise_level == 0.05


# ===================================================================
# F. Integration tests
# ===================================================================


class TestCDRIntegration:
    """Full pipeline integration tests for CDR."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_non_clifford_circuit_gets_cdr(self):
        """Circuit with non-Clifford gates should get CDR recommendation."""
        from emrg import generate_recipe

        qc = _make_non_clifford_circuit(depth_layers=2)
        result = generate_recipe(qc)
        features = result.features
        assert features.non_clifford_fraction > CDR_NON_CLIFFORD_FRACTION_THRESHOLD
        assert result.recipe.technique == "cdr"
        # Verify generated code is valid Python
        compile(result.code, "<integration_test>", "exec")
        assert "execute_with_cdr" in result.code

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_clifford_circuit_does_not_get_cdr(self):
        """All-Clifford circuit should NOT get CDR."""
        from emrg import generate_recipe

        qc = _make_clifford_circuit()
        result = generate_recipe(qc)
        assert result.recipe.technique != "cdr"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_cdr_override_produces_valid_code(self):
        """Forcing CDR on any circuit should produce valid Python."""
        from emrg import generate_recipe

        qc = _make_clifford_circuit()
        result = generate_recipe(qc, technique="cdr")
        assert result.recipe.technique == "cdr"
        compile(result.code, "<cdr_override_test>", "exec")

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_t_gate_circuit_analysis(self):
        """T gate circuit should be analyzed correctly."""
        qc = _make_t_gate_circuit()
        features = analyze_circuit(qc)
        assert features.non_clifford_count == 3
        assert features.non_clifford_fraction > 0.0
