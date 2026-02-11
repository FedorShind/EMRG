"""Tests for emrg.heuristics -- rule-based mitigation recipe selection."""

from __future__ import annotations

import pytest

from emrg.analyzer import CircuitFeatures
from emrg.heuristics import (
    DEFAULT_RULES,
    MitigationRecipe,
    _is_deep_or_high_noise,
    _is_moderate_depth,
    _is_shallow,
    recommend,
)
from tests._helpers import make_features as _make_features

# ---------------------------------------------------------------------------
# Fixtures -- representative circuits for each tier
# ---------------------------------------------------------------------------


@pytest.fixture
def shallow_features() -> CircuitFeatures:
    """Shallow circuit: depth=4, 1 CX, low noise."""
    return _make_features(
        depth=4,
        multi_qubit_gate_count=1,
        single_qubit_gate_count=1,
        total_gate_count=2,
        estimated_noise_factor=0.011,
        noise_category="low",
    )


@pytest.fixture
def moderate_features() -> CircuitFeatures:
    """Moderate circuit: depth=35, 6 CX, moderate noise."""
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
    """Deep circuit: depth=80, many gates, high noise."""
    return _make_features(
        depth=80,
        multi_qubit_gate_count=40,
        single_qubit_gate_count=60,
        total_gate_count=100,
        estimated_noise_factor=0.46,
        noise_category="high",
    )


@pytest.fixture
def high_noise_shallow_features() -> CircuitFeatures:
    """Shallow depth but high noise (many 2q gates in parallel)."""
    return _make_features(
        depth=15,
        multi_qubit_gate_count=30,
        single_qubit_gate_count=10,
        total_gate_count=40,
        estimated_noise_factor=0.31,
        noise_category="high",
    )


# ---------------------------------------------------------------------------
# Tests: predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    """Verify each predicate function matches the correct features."""

    def test_is_deep_or_high_noise_matches_deep(self) -> None:
        f = _make_features(depth=60, noise_category="moderate")
        assert _is_deep_or_high_noise(f) is True

    def test_is_deep_or_high_noise_matches_high_noise(self) -> None:
        f = _make_features(depth=10, noise_category="high")
        assert _is_deep_or_high_noise(f) is True

    def test_is_deep_or_high_noise_rejects_shallow_low(self) -> None:
        f = _make_features(depth=10, noise_category="low")
        assert _is_deep_or_high_noise(f) is False

    def test_is_moderate_depth_matches(self) -> None:
        f = _make_features(depth=35)
        assert _is_moderate_depth(f) is True

    def test_is_moderate_depth_boundary_low(self) -> None:
        f = _make_features(depth=20)
        assert _is_moderate_depth(f) is True

    def test_is_moderate_depth_boundary_high(self) -> None:
        f = _make_features(depth=50)
        assert _is_moderate_depth(f) is True

    def test_is_moderate_depth_rejects_shallow(self) -> None:
        f = _make_features(depth=19)
        assert _is_moderate_depth(f) is False

    def test_is_moderate_depth_rejects_deep(self) -> None:
        f = _make_features(depth=51)
        assert _is_moderate_depth(f) is False

    def test_is_shallow_matches(self) -> None:
        f = _make_features(depth=10, multi_qubit_gate_count=5)
        assert _is_shallow(f) is True

    def test_is_shallow_rejects_deep(self) -> None:
        f = _make_features(depth=25, multi_qubit_gate_count=5)
        assert _is_shallow(f) is False

    def test_is_shallow_rejects_many_2q_gates(self) -> None:
        f = _make_features(depth=10, multi_qubit_gate_count=50)
        assert _is_shallow(f) is False

    def test_is_shallow_boundary_depth_19(self) -> None:
        f = _make_features(depth=19, multi_qubit_gate_count=49)
        assert _is_shallow(f) is True

    def test_is_shallow_boundary_depth_20(self) -> None:
        f = _make_features(depth=20, multi_qubit_gate_count=10)
        assert _is_shallow(f) is False


# ---------------------------------------------------------------------------
# Tests: recommend() -- rule selection
# ---------------------------------------------------------------------------


class TestRecommend:
    """Verify recommend() picks the right rule for each circuit tier."""

    def test_shallow_gets_linear(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        assert recipe.factory_name == "LinearFactory"
        assert recipe.scaling_method == "fold_global"

    def test_moderate_gets_richardson(self, moderate_features: CircuitFeatures) -> None:
        recipe = recommend(moderate_features)
        assert recipe.factory_name == "RichardsonFactory"
        assert recipe.scaling_method == "fold_global"

    def test_deep_gets_poly(self, deep_features: CircuitFeatures) -> None:
        recipe = recommend(deep_features)
        assert recipe.factory_name == "PolyFactory"
        assert recipe.factory_kwargs == {"order": 2}
        assert recipe.scaling_method == "fold_gates_at_random"

    def test_high_noise_shallow_gets_poly(
        self, high_noise_shallow_features: CircuitFeatures
    ) -> None:
        """High noise should trigger poly even if depth is shallow."""
        recipe = recommend(high_noise_shallow_features)
        assert recipe.factory_name == "PolyFactory"

    def test_technique_always_zne(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        assert recipe.technique == "zne"

    def test_noise_category_propagated(
        self, moderate_features: CircuitFeatures
    ) -> None:
        recipe = recommend(moderate_features)
        assert recipe.noise_category == "moderate"


# ---------------------------------------------------------------------------
# Tests: rule priority
# ---------------------------------------------------------------------------


class TestRulePriority:
    """Verify that higher-priority rules take precedence."""

    def test_deep_high_noise_beats_moderate(self) -> None:
        """depth=60 + high noise -> poly, NOT richardson (even though 60>20)."""
        f = _make_features(depth=60, noise_category="high")
        recipe = recommend(f)
        assert recipe.factory_name == "PolyFactory"

    def test_depth_51_gets_poly_not_richardson(self) -> None:
        """depth=51 triggers deep rule (>50) before moderate (20-50)."""
        f = _make_features(depth=51, noise_category="moderate")
        recipe = recommend(f)
        assert recipe.factory_name == "PolyFactory"

    def test_depth_50_gets_richardson_not_poly(self) -> None:
        """depth=50 stays in moderate range (<=50)."""
        f = _make_features(depth=50, noise_category="moderate")
        recipe = recommend(f)
        assert recipe.factory_name == "RichardsonFactory"


# ---------------------------------------------------------------------------
# Tests: fallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Verify the fallback recipe fires when no rule matches."""

    def test_fallback_on_gap(self) -> None:
        """depth=19, 2q_gates=50 -> shallow says no (>=50 gates), moderate
        says no (depth<20) -> fallback should fire."""
        f = _make_features(
            depth=19,
            multi_qubit_gate_count=50,
            noise_category="low",
        )
        recipe = recommend(f)
        assert recipe.factory_name == "LinearFactory"
        assert "No specific heuristic rule matched" in recipe.rationale[0]

    def test_fallback_with_empty_rules(self, shallow_features: CircuitFeatures) -> None:
        """Passing an empty rule list should trigger fallback."""
        recipe = recommend(shallow_features, rules=[])
        assert "No specific heuristic rule matched" in recipe.rationale[0]


# ---------------------------------------------------------------------------
# Tests: custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    """Verify that users can inject custom rules."""

    def test_custom_rule_overrides_default(
        self, shallow_features: CircuitFeatures
    ) -> None:
        """A custom rule that always matches should override defaults."""

        def always_true(f: CircuitFeatures) -> bool:
            return True

        def custom_builder(f: CircuitFeatures) -> MitigationRecipe:
            return MitigationRecipe(
                technique="zne",
                factory_name="ExpFactory",
                scale_factors=(1.0, 2.0, 3.0),
                rationale=("Custom rule fired.",),
                noise_category=f.noise_category,
                estimated_overhead=3.0,
            )

        recipe = recommend(shallow_features, rules=[(always_true, custom_builder)])
        assert recipe.factory_name == "ExpFactory"
        assert recipe.rationale[0] == "Custom rule fired."


# ---------------------------------------------------------------------------
# Tests: recipe content quality
# ---------------------------------------------------------------------------


class TestRecipeContent:
    """Verify recipe fields are well-formed and informative."""

    def test_scale_factors_are_sorted(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        assert recipe.scale_factors == tuple(sorted(recipe.scale_factors))

    def test_scale_factors_start_at_one(
        self, shallow_features: CircuitFeatures
    ) -> None:
        recipe = recommend(shallow_features)
        assert recipe.scale_factors[0] == 1.0

    def test_overhead_matches_scale_count(
        self, shallow_features: CircuitFeatures
    ) -> None:
        recipe = recommend(shallow_features)
        assert recipe.estimated_overhead == float(len(recipe.scale_factors))

    def test_rationale_not_empty(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        assert len(recipe.rationale) >= 2

    def test_rationale_contains_reference_shallow(
        self, shallow_features: CircuitFeatures
    ) -> None:
        recipe = recommend(shallow_features)
        all_text = " ".join(recipe.rationale)
        # Li & Benjamin is the seminal ZNE paper
        assert "Li & Benjamin" in all_text or "PRX" in all_text

    def test_rationale_contains_reference_deep(
        self, deep_features: CircuitFeatures
    ) -> None:
        recipe = recommend(deep_features)
        all_text = " ".join(recipe.rationale)
        assert "arXiv" in all_text

    def test_rationale_contains_reference_moderate(
        self, moderate_features: CircuitFeatures
    ) -> None:
        recipe = recommend(moderate_features)
        all_text = " ".join(recipe.rationale)
        assert "Temme" in all_text or "PRL" in all_text

    def test_poly_recipe_has_order(self, deep_features: CircuitFeatures) -> None:
        recipe = recommend(deep_features)
        assert "order" in recipe.factory_kwargs
        assert recipe.factory_kwargs["order"] == 2

    def test_linear_recipe_no_extra_kwargs(
        self, shallow_features: CircuitFeatures
    ) -> None:
        recipe = recommend(shallow_features)
        assert recipe.factory_kwargs == {}


# ---------------------------------------------------------------------------
# Tests: MitigationRecipe dataclass
# ---------------------------------------------------------------------------


class TestMitigationRecipeDataclass:
    """Verify MitigationRecipe is well-behaved."""

    def test_immutable(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        with pytest.raises(AttributeError):
            recipe.technique = "pec"  # type: ignore[misc]

    def test_repr_contains_key_fields(self, shallow_features: CircuitFeatures) -> None:
        recipe = recommend(shallow_features)
        r = repr(recipe)
        assert "technique=" in r
        assert "factory_name=" in r

    def test_default_factory_kwargs_empty(self) -> None:
        """Verify default factory_kwargs is an empty dict."""
        recipe = MitigationRecipe(
            technique="zne",
            factory_name="LinearFactory",
            scale_factors=(1.0, 2.0),
        )
        assert recipe.factory_kwargs == {}

    def test_default_scaling_method(self) -> None:
        recipe = MitigationRecipe(
            technique="zne",
            factory_name="LinearFactory",
            scale_factors=(1.0, 2.0),
        )
        assert recipe.scaling_method == "fold_global"


# ---------------------------------------------------------------------------
# Tests: DEFAULT_RULES structure
# ---------------------------------------------------------------------------


class TestDefaultRules:
    """Verify the rule list itself is well-formed."""

    def test_rules_is_tuple(self) -> None:
        assert isinstance(DEFAULT_RULES, tuple)

    def test_rules_has_three_entries(self) -> None:
        assert len(DEFAULT_RULES) == 3

    def test_each_rule_is_callable_pair(self) -> None:
        for predicate, builder in DEFAULT_RULES:
            assert callable(predicate)
            assert callable(builder)
