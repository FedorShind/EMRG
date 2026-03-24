"""Tests for emrg.codegen -- Python code generation from mitigation recipes."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from emrg import __version__
from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.codegen import generate_code
from emrg.heuristics import MitigationRecipe, recommend
from tests._helpers import make_features as _make_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_recipe() -> MitigationRecipe:
    return MitigationRecipe(
        technique="zne",
        factory_name="LinearFactory",
        scale_factors=(1.0, 1.5, 2.0),
        factory_kwargs={},
        scaling_method="fold_global",
        rationale=(
            "Low depth, linear sufficient.",
            "Li & Benjamin, PRX 7, 021050, 2017.",
        ),
        noise_category="low",
        estimated_overhead=3.0,
    )


def _make_richardson_recipe() -> MitigationRecipe:
    return MitigationRecipe(
        technique="zne",
        factory_name="RichardsonFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5),
        factory_kwargs={},
        scaling_method="fold_global",
        rationale=(
            "Moderate depth, Richardson handles non-linearity.",
            "Temme et al., PRL 119, 180509, 2017.",
        ),
        noise_category="moderate",
        estimated_overhead=4.0,
    )


def _make_poly_recipe() -> MitigationRecipe:
    return MitigationRecipe(
        technique="zne",
        factory_name="PolyFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5, 3.0),
        factory_kwargs={"order": 2},
        scaling_method="fold_gates_at_random",
        rationale=(
            "Deep/noisy circuit, polynomial fit needed.",
            "arXiv:2307.05203.",
        ),
        noise_category="high",
        estimated_overhead=5.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shallow_features() -> CircuitFeatures:
    return _make_features(depth=4, multi_qubit_gate_count=1)


@pytest.fixture
def moderate_features() -> CircuitFeatures:
    return _make_features(
        depth=35,
        multi_qubit_gate_count=6,
        single_qubit_gate_count=16,
        total_gate_count=22,
        estimated_noise_factor=0.076,
        noise_category="moderate",
    )


@pytest.fixture
def deep_features() -> CircuitFeatures:
    return _make_features(
        depth=80,
        multi_qubit_gate_count=40,
        single_qubit_gate_count=60,
        total_gate_count=100,
        estimated_noise_factor=0.46,
        noise_category="high",
    )


@pytest.fixture
def parametric_features() -> CircuitFeatures:
    return _make_features(
        depth=12,
        num_parameters=16,
        multi_qubit_gate_count=6,
    )


# ---------------------------------------------------------------------------
# Tests: import correctness
# ---------------------------------------------------------------------------


class TestImports:
    """Verify generated code imports the right Mitiq classes."""

    def test_imports_linear(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "from mitiq.zne.inference import LinearFactory" in code

    def test_imports_richardson(self, moderate_features: CircuitFeatures) -> None:
        code = generate_code(_make_richardson_recipe(), moderate_features)
        assert "from mitiq.zne.inference import RichardsonFactory" in code

    def test_imports_poly(self, deep_features: CircuitFeatures) -> None:
        code = generate_code(_make_poly_recipe(), deep_features)
        assert "from mitiq.zne.inference import PolyFactory" in code

    def test_imports_execute_with_zne(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "from mitiq.zne import execute_with_zne" in code


# ---------------------------------------------------------------------------
# Tests: factory construction
# ---------------------------------------------------------------------------


class TestFactory:
    """Verify factory construction lines are correct."""

    def test_linear_factory(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "factory = LinearFactory(scale_factors=[1.0, 1.5, 2.0])" in code

    def test_richardson_factory(self, moderate_features: CircuitFeatures) -> None:
        code = generate_code(_make_richardson_recipe(), moderate_features)
        assert "factory = RichardsonFactory(scale_factors=[1.0, 1.5, 2.0, 2.5])" in code

    def test_poly_factory_with_order(self, deep_features: CircuitFeatures) -> None:
        code = generate_code(_make_poly_recipe(), deep_features)
        assert (
            "factory = PolyFactory(scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0], "
            "order=2)" in code
        )


# ---------------------------------------------------------------------------
# Tests: scaling method
# ---------------------------------------------------------------------------


class TestScalingMethod:
    """Verify the correct scaling function appears in the output."""

    def test_fold_global(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "from mitiq.zne.scaling import fold_global" in code
        assert "scale_noise=fold_global," in code

    def test_fold_gates_at_random(self, deep_features: CircuitFeatures) -> None:
        code = generate_code(_make_poly_recipe(), deep_features)
        assert "from mitiq.zne.scaling import fold_gates_at_random" in code
        assert "scale_noise=fold_gates_at_random," in code


# ---------------------------------------------------------------------------
# Tests: header content
# ---------------------------------------------------------------------------


class TestHeader:
    """Verify the comment header contains circuit stats and version."""

    def test_contains_version(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert f"EMRG v{__version__}" in code

    def test_contains_circuit_stats(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "4 qubits" in code
        assert "depth 4" in code
        assert "1 multi-qubit gates" in code

    def test_contains_noise_estimate(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "0.023" in code
        assert "(low)" in code

    def test_contains_recommendation(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "Recommendation: LinearFactory + fold_global" in code

    def test_parametric_shows_parameters(
        self, parametric_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), parametric_features)
        assert "16 parameters" in code

    def test_non_parametric_omits_parameters(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "parameters" not in code


# ---------------------------------------------------------------------------
# Tests: parameter warning
# ---------------------------------------------------------------------------


class TestParameterWarning:
    """Verify generated code warns about unbound parameters."""

    def test_parametric_has_warning_block(
        self, parametric_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), parametric_features)
        assert "WARNING" in code
        assert "unbound parameter" in code
        assert "assign_parameters" in code

    def test_non_parametric_no_warning(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "unbound parameter" not in code

    def test_parametric_code_still_compiles(
        self, parametric_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), parametric_features)
        compile(code, "<emrg-param-warning>", "exec")

    def test_parametric_explain_mode(
        self, parametric_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), parametric_features, explain=True)
        assert "WARNING" in code
        assert "assign_parameters" in code
        assert "Rationale:" in code  # explain content still present


# ---------------------------------------------------------------------------
# Tests: explain mode
# ---------------------------------------------------------------------------


class TestExplainMode:
    """Verify explain mode adds rationale and inline comments."""

    def test_explain_has_rationale_lines(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features, explain=True)
        assert "Rationale:" in code
        assert "Li & Benjamin" in code

    def test_normal_mode_no_rationale_section(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features, explain=False)
        assert "Rationale:" not in code

    def test_explain_has_factory_comment(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features, explain=True)
        assert "selected by EMRG heuristics" in code

    def test_explain_has_overhead_comment(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features, explain=True)
        assert "Estimated overhead" in code

    def test_explain_poly_shows_extra_kwargs(
        self, deep_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_poly_recipe(), deep_features, explain=True)
        assert "Extra parameters: order=2" in code


# ---------------------------------------------------------------------------
# Tests: executor placeholder
# ---------------------------------------------------------------------------


class TestExecutor:
    """Verify the executor stub is present and informative."""

    def test_executor_function_defined(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "def execute(circuit):" in code

    def test_executor_raises_not_implemented(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "NotImplementedError" in code

    def test_executor_has_aer_example(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "AerSimulator" in code


# ---------------------------------------------------------------------------
# Tests: syntax validity via exec()
# ---------------------------------------------------------------------------


class TestSyntaxValidity:
    """Verify generated code is valid Python that can be exec'd.

    We exec() in an isolated namespace. The executor raises
    NotImplementedError so we don't actually call execute_with_zne,
    but the import statements and factory construction must all resolve.
    """

    def _exec_imports_and_factory(self, code: str) -> None:
        """Extract and exec everything up to and including factory setup."""
        # Split at the executor definition -- only run imports + factory
        parts = code.split("def execute(circuit):")
        assert len(parts) == 2, "Expected exactly one 'def execute' in code"
        preamble = parts[0]
        namespace: dict[str, object] = {}
        exec(compile(preamble, "<emrg-test>", "exec"), namespace)
        # Verify factory was created
        assert "factory" in namespace

    def test_exec_linear(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        self._exec_imports_and_factory(code)

    def test_exec_richardson(self, moderate_features: CircuitFeatures) -> None:
        code = generate_code(_make_richardson_recipe(), moderate_features)
        self._exec_imports_and_factory(code)

    def test_exec_poly(self, deep_features: CircuitFeatures) -> None:
        code = generate_code(_make_poly_recipe(), deep_features)
        self._exec_imports_and_factory(code)

    def test_full_compile(self, shallow_features: CircuitFeatures) -> None:
        """The entire generated file must be compilable Python."""
        code = generate_code(_make_linear_recipe(), shallow_features)
        compile(code, "<emrg-test-full>", "exec")


# ---------------------------------------------------------------------------
# Tests: circuit_name parameter
# ---------------------------------------------------------------------------


class TestCircuitName:
    """Verify custom circuit_name appears in generated code."""

    def test_custom_name(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features, circuit_name="qc")
        assert "    qc," in code

    def test_default_name(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_linear_recipe(), shallow_features)
        assert "    circuit," in code

    def test_rejects_invalid_identifier_numeric(
        self, shallow_features: CircuitFeatures
    ) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            generate_code(
                _make_linear_recipe(), shallow_features, circuit_name="123bad"
            )

    def test_rejects_invalid_identifier_space(
        self, shallow_features: CircuitFeatures
    ) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            generate_code(
                _make_linear_recipe(), shallow_features, circuit_name="my circuit"
            )

    def test_rejects_empty_string(self, shallow_features: CircuitFeatures) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            generate_code(_make_linear_recipe(), shallow_features, circuit_name="")

    def test_accepts_underscore_name(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(
            _make_linear_recipe(), shallow_features, circuit_name="_qc"
        )
        assert "    _qc," in code


# ---------------------------------------------------------------------------
# Tests: fallback recipe
# ---------------------------------------------------------------------------


class TestFallbackRecipe:
    """Verify codegen works with a fallback recipe."""

    def test_fallback_generates_valid_code(
        self, shallow_features: CircuitFeatures
    ) -> None:
        fallback = MitigationRecipe(
            technique="zne",
            factory_name="LinearFactory",
            scale_factors=(1.0, 1.5, 2.0),
            factory_kwargs={},
            scaling_method="fold_global",
            rationale=("No specific rule matched -- conservative default.",),
            noise_category="low",
            estimated_overhead=3.0,
        )
        code = generate_code(fallback, shallow_features)
        compile(code, "<emrg-test-fallback>", "exec")
        assert "LinearFactory" in code


# ---------------------------------------------------------------------------
# Tests: full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: real circuit -> analyze -> recommend -> generate_code."""

    def test_bell_state_pipeline(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        features = analyze_circuit(qc)
        recipe = recommend(features)
        code = generate_code(recipe, features, explain=True)

        # Must be valid Python
        compile(code, "<emrg-pipeline>", "exec")

        # Must contain expected elements
        assert f"EMRG v{__version__}" in code
        assert "2 qubits" in code
        assert recipe.factory_name in code
        assert "execute_with_zne" in code
        assert "Rationale:" in code

    @pytest.mark.filterwarnings("ignore:Circuit has.*unbound parameter")
    def test_vqe_pipeline(self) -> None:
        """Moderate-depth VQE should get RichardsonFactory."""
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(4, 4)
        for layer in range(3):
            for q in range(4):
                qc.ry(Parameter(f"p_{layer}_{q}_a"), q)
            for q in range(3):
                qc.cx(q, q + 1)
            for q in range(4):
                qc.ry(Parameter(f"p_{layer}_{q}_b"), q)
        qc.measure(range(4), range(4))

        features = analyze_circuit(qc)
        recipe = recommend(features)
        code = generate_code(recipe, features)

        compile(code, "<emrg-vqe-pipeline>", "exec")
        assert recipe.factory_name in code
        assert "execute_with_zne" in code
        assert "assign_parameters" in code  # parameter warning present


# ---------------------------------------------------------------------------
# Helpers: PEC recipe
# ---------------------------------------------------------------------------


def _make_pec_recipe() -> MitigationRecipe:
    return MitigationRecipe(
        technique="pec",
        factory_name="",
        scale_factors=(),
        factory_kwargs={"num_samples": 200, "noise_level": 0.01},
        scaling_method="",
        rationale=(
            "PEC selected for shallow circuit.",
            "Temme et al., PRL 119, 180509, 2017.",
        ),
        noise_category="low",
        estimated_overhead=50.0,
    )


# ---------------------------------------------------------------------------
# Tests: PEC import correctness
# ---------------------------------------------------------------------------


class TestPECImports:
    """Verify PEC code imports the right Mitiq classes."""

    def test_imports_execute_with_pec(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "from mitiq.pec import execute_with_pec" in code

    def test_imports_representations(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "represent_operations_in_circuit_with_local_depolarizing_noise" in code

    def test_no_zne_imports(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "execute_with_zne" not in code
        assert "from mitiq.zne" not in code


# ---------------------------------------------------------------------------
# Tests: PEC setup
# ---------------------------------------------------------------------------


class TestPECSetup:
    """Verify PEC noise level and sample count appear in output."""

    def test_noise_level_in_code(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "noise_level = 0.01" in code

    def test_num_samples_in_code(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "num_samples = 200" in code

    def test_representations_call(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "represent_operations_in_circuit_with_local_depolarizing_noise(" in code
        assert "noise_level=noise_level," in code


# ---------------------------------------------------------------------------
# Tests: PEC execution
# ---------------------------------------------------------------------------


class TestPECExecution:
    """Verify execute_with_pec call appears with correct args."""

    def test_execute_with_pec_call(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "mitigated_value = execute_with_pec(" in code

    def test_representations_arg(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "representations=representations," in code

    def test_num_samples_arg(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "num_samples=num_samples," in code

    def test_custom_circuit_name(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(
            _make_pec_recipe(), shallow_features, circuit_name="qc"
        )
        assert "    qc," in code


# ---------------------------------------------------------------------------
# Tests: PEC syntax validity
# ---------------------------------------------------------------------------


class TestPECSyntaxValidity:
    """Verify PEC-generated code compiles as valid Python."""

    def test_full_compile(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        compile(code, "<emrg-pec-test>", "exec")

    def test_full_compile_explain(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features, explain=True)
        compile(code, "<emrg-pec-explain-test>", "exec")

    def test_full_compile_parametric(
        self, parametric_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_pec_recipe(), parametric_features)
        compile(code, "<emrg-pec-param-test>", "exec")


# ---------------------------------------------------------------------------
# Tests: PEC header
# ---------------------------------------------------------------------------


class TestPECHeader:
    """Verify PEC header says PEC, not factory name."""

    def test_pec_recommendation_line(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert "Recommendation: PEC (Probabilistic Error Cancellation)" in code

    def test_no_factory_name_in_recommendation(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        # Should NOT show "Recommendation:  + " (empty factory + scaling)
        assert "Recommendation:  + " not in code

    def test_contains_version(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features)
        assert f"EMRG v{__version__}" in code


# ---------------------------------------------------------------------------
# Tests: PEC explain mode
# ---------------------------------------------------------------------------


class TestPECExplainMode:
    """Verify PEC explain mode includes appropriate comments."""

    def test_explain_has_rationale(self, shallow_features: CircuitFeatures) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features, explain=True)
        assert "Rationale:" in code
        assert "Temme" in code

    def test_explain_has_quasi_probability_comment(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features, explain=True)
        assert "quasi-probability" in code

    def test_explain_has_overhead_comment(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features, explain=True)
        assert "Estimated overhead" in code

    def test_normal_mode_no_noise_level_comment(
        self, shallow_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_pec_recipe(), shallow_features, explain=False)
        assert "matches the expected hardware noise" not in code


# ---------------------------------------------------------------------------
# Helpers: layerwise Richardson recipe
# ---------------------------------------------------------------------------


def _make_layerwise_richardson_recipe() -> MitigationRecipe:
    return MitigationRecipe(
        technique="zne",
        factory_name="RichardsonFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5),
        factory_kwargs={},
        scaling_method="fold_gates_at_random",
        rationale=(
            "Moderate depth with high layer heterogeneity.",
            "arXiv:2005.10921.",
        ),
        noise_category="moderate",
        estimated_overhead=4.0,
    )


# ---------------------------------------------------------------------------
# Tests: layerwise Richardson codegen
# ---------------------------------------------------------------------------


class TestLayerwiseRichardsonCodegen:
    """Verify codegen handles layerwise Richardson correctly."""

    def test_layerwise_has_inline_comment(
        self, moderate_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_layerwise_richardson_recipe(), moderate_features)
        assert "fold_gates_at_random selected" in code
        assert "uneven noise density" in code

    def test_standard_richardson_no_layerwise_comment(
        self, moderate_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_richardson_recipe(), moderate_features)
        assert "fold_gates_at_random selected" not in code

    def test_poly_no_layerwise_comment(
        self, deep_features: CircuitFeatures
    ) -> None:
        """PolyFactory also uses fold_gates_at_random but should NOT get
        the layerwise comment."""
        code = generate_code(_make_poly_recipe(), deep_features)
        assert "fold_gates_at_random selected" not in code

    def test_layerwise_imports_fold_gates_at_random(
        self, moderate_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_layerwise_richardson_recipe(), moderate_features)
        assert "from mitiq.zne.scaling import fold_gates_at_random" in code

    def test_layerwise_execution_uses_fold_gates_at_random(
        self, moderate_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_layerwise_richardson_recipe(), moderate_features)
        assert "scale_noise=fold_gates_at_random," in code

    def test_layerwise_compiles(
        self, moderate_features: CircuitFeatures
    ) -> None:
        code = generate_code(_make_layerwise_richardson_recipe(), moderate_features)
        compile(code, "<emrg-layerwise-test>", "exec")
