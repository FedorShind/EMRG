"""Integration tests for policy-aware recommendation and generation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from qiskit import QuantumCircuit

from emrg import DEFAULT_POLICY, RecipePolicy, generate_recipe, recommend
from emrg.cli import main
from emrg.codegen import generate_code
from emrg.policy import policy_to_dict, save_policy
from tests.conftest import make_features


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _default_data() -> dict[str, Any]:
    return deepcopy(policy_to_dict(DEFAULT_POLICY))


def _policy_with(mutator: Any) -> RecipePolicy:
    data = _default_data()
    mutator(data)
    return RecipePolicy.from_dict(data)


def _recipe_json(recipe: Any) -> str:
    return json.dumps(recipe.to_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("features", "technique"),
    [
        (make_features(depth=4, multi_qubit_gate_count=1, noise_category="low"), None),
        (make_features(depth=35, noise_category="moderate"), None),
        (
            make_features(
                depth=25,
                noise_category="moderate",
                layer_heterogeneity=3.0,
            ),
            None,
        ),
        (make_features(depth=80, noise_category="high"), None),
        (
            make_features(
                depth=10,
                noise_model_available=True,
                pec_overhead_estimate=50.0,
            ),
            None,
        ),
        (
            make_features(
                depth=12,
                total_gate_count=30,
                non_clifford_count=12,
                non_clifford_fraction=0.4,
            ),
            None,
        ),
        (
            make_features(
                depth=25,
                total_gate_count=30,
                noise_category="moderate",
                noise_model_available=True,
                pec_overhead_estimate=20.0,
            ),
            None,
        ),
        (
            make_features(
                depth=19,
                multi_qubit_gate_count=50,
                noise_category="low",
            ),
            None,
        ),
        (
            make_features(
                depth=80,
                noise_model_available=False,
                pec_overhead_estimate=5000.0,
                noise_category="high",
            ),
            "pec",
        ),
        (make_features(depth=2, non_clifford_fraction=0.0), "cdr"),
        (
            make_features(
                depth=25,
                noise_category="moderate",
                noise_model_available=False,
                pec_overhead_estimate=500.0,
            ),
            "composite",
        ),
    ],
)
def test_default_policy_matches_existing_recommendation(
    features: Any,
    technique: str | None,
) -> None:
    without_policy = recommend(features, technique=technique)
    with_policy = recommend(features, technique=technique, policy=DEFAULT_POLICY)

    assert _recipe_json(with_policy) == _recipe_json(without_policy)


def test_generate_recipe_default_policy_matches_existing_output() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    without_policy = generate_recipe(qc)
    with_policy = generate_recipe(qc, policy=DEFAULT_POLICY)

    assert with_policy.code == without_policy.code
    assert _recipe_json(with_policy.recipe) == _recipe_json(without_policy.recipe)


def test_generate_recipe_accepts_policy_path(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    save_policy(DEFAULT_POLICY, path)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    result = generate_recipe(qc, policy=path)

    assert result.recipe.technique == "zne"
    assert "LinearFactory" in result.code


def test_recommend_rejects_rules_and_policy_together() -> None:
    features = make_features(depth=4)

    with pytest.raises(ValueError, match="rules and policy"):
        recommend(features, rules=[], policy=DEFAULT_POLICY)


def test_custom_zne_scale_factors_reach_recipe_and_codegen() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["zne"]["shallow"].update(
            {"scale_factors": [1.0, 2.0]}
        )
    )
    features = make_features(depth=4, multi_qubit_gate_count=1)

    recipe = recommend(features, policy=policy)
    code = generate_code(recipe, features)

    assert recipe.scale_factors == (1.0, 2.0)
    assert "scale_factors=[1.0, 2.0]" in code


def test_policy_rationale_reflects_configured_zne_profile() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["zne"]["moderate"].update(
            {
                "factory": "LinearFactory",
                "scale_factors": [1.0, 1.5, 2.0, 2.5],
                "scaling_method": "fold_gates_at_random",
            }
        )
    )
    features = make_features(
        depth=30,
        total_gate_count=20,
        non_clifford_fraction=0.0,
        noise_category="moderate",
    )

    recipe = recommend(features, policy=policy)
    rationale = " ".join(recipe.rationale)

    assert recipe.factory_name == "LinearFactory"
    assert recipe.scaling_method == "fold_gates_at_random"
    assert "LinearFactory" in rationale
    assert "fold_gates_at_random" in rationale
    assert "RichardsonFactory uses" not in rationale


def test_disabled_pec_prevents_auto_pec_selection() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["pec"].update({"enabled": False})
    )
    features = make_features(
        depth=10,
        noise_model_available=True,
        pec_overhead_estimate=50.0,
    )

    recipe = recommend(features, policy=policy)

    assert recipe.technique == "zne"


def test_disabled_cdr_prevents_auto_cdr_selection() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["cdr"].update({"enabled": False})
    )
    features = make_features(
        depth=12,
        total_gate_count=30,
        non_clifford_count=12,
        non_clifford_fraction=0.4,
    )

    recipe = recommend(features, policy=policy)

    assert recipe.technique == "zne"


def test_disabled_composite_falls_back_to_pec() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["composite"].update({"enabled": False})
    )
    features = make_features(
        depth=25,
        total_gate_count=30,
        noise_category="moderate",
        noise_model_available=True,
        pec_overhead_estimate=20.0,
    )

    recipe = recommend(features, policy=policy)

    assert recipe.technique == "pec"


def test_custom_pec_settings_reach_recipe_and_codegen() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["pec"].update(
            {"noise_level": 0.02, "min_samples": 321, "max_samples": 321}
        )
    )
    features = make_features(
        depth=10,
        noise_model_available=True,
        pec_overhead_estimate=10.0,
    )

    recipe = recommend(features, policy=policy)
    code = generate_code(recipe, features)

    assert recipe.factory_kwargs["noise_level"] == 0.02
    assert recipe.factory_kwargs["num_samples"] == 321
    assert "noise_level = 0.02" in code
    assert "num_samples = 321" in code


def test_custom_cdr_threshold_changes_selection() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["cdr"].update(
            {"min_non_clifford_fraction": 0.5}
        )
    )
    features = make_features(
        depth=12,
        total_gate_count=30,
        non_clifford_count=12,
        non_clifford_fraction=0.4,
    )

    recipe = recommend(features, policy=policy)

    assert recipe.technique == "zne"


def test_budget_prevents_composite_when_combined_overhead_exceeds_cap() -> None:
    policy = _policy_with(lambda data: data["budget"].update({"max_overhead": 10}))
    features = make_features(
        depth=25,
        total_gate_count=30,
        noise_category="moderate",
        noise_model_available=True,
        pec_overhead_estimate=20.0,
    )

    recipe = recommend(features, policy=policy)

    assert recipe.technique == "pec"


def test_forced_disabled_technique_returns_recipe_with_warning() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["pec"].update({"enabled": False})
    )
    features = make_features(depth=10, noise_model_available=True)

    recipe = recommend(features, technique="pec", policy=policy)
    code = generate_code(recipe, features)

    assert recipe.technique == "pec"
    assert any("disabled" in warning.lower() for warning in recipe.warnings)
    assert "# Warnings:" in code


def test_forced_disabled_cdr_warns() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["cdr"].update({"enabled": False})
    )
    features = make_features(
        depth=12,
        total_gate_count=30,
        non_clifford_count=12,
        non_clifford_fraction=0.4,
    )

    recipe = recommend(features, technique="cdr", policy=policy)

    assert recipe.technique == "cdr"
    assert any("disabled" in warning.lower() for warning in recipe.warnings)


def test_forced_disabled_composite_warns_for_disabled_components() -> None:
    policy = _policy_with(
        lambda data: (
            data["budget"].update({"allow_composite": False}),
            data["techniques"]["zne"].update({"enabled": False}),
            data["techniques"]["pec"].update({"enabled": False}),
            data["techniques"]["composite"].update({"enabled": False}),
        )
    )
    features = make_features(
        depth=25,
        noise_category="moderate",
        noise_model_available=True,
        pec_overhead_estimate=20.0,
    )

    recipe = recommend(features, technique="composite", policy=policy)
    warning_text = " ".join(recipe.warnings).lower()

    assert recipe.technique == "composite"
    assert "budget" in warning_text
    assert "zne component is disabled" in warning_text
    assert "pec component is disabled" in warning_text


def test_forced_disabled_zne_warns() -> None:
    policy = _policy_with(
        lambda data: data["techniques"]["zne"].update({"enabled": False})
    )
    features = make_features(depth=4)

    recipe = recommend(features, technique="zne", policy=policy)

    assert recipe.technique == "zne"
    assert recipe.warnings == ("Forced ZNE: ZNE is disabled by policy.",)


def test_auto_mode_errors_when_policy_disables_every_available_path() -> None:
    policy = _policy_with(
        lambda data: (
            data["techniques"]["zne"].update({"enabled": False}),
            data["techniques"]["pec"].update({"enabled": False}),
            data["techniques"]["cdr"].update({"enabled": False}),
            data["techniques"]["composite"].update({"enabled": False}),
        )
    )
    features = make_features(depth=4)

    with pytest.raises(ValueError, match="Policy disables ZNE"):
        recommend(features, policy=policy)


def test_recipe_to_dict_is_json_compatible_and_recursive() -> None:
    features = make_features(
        depth=25,
        total_gate_count=30,
        noise_category="moderate",
        noise_model_available=True,
        pec_overhead_estimate=20.0,
    )

    recipe = recommend(features, policy=DEFAULT_POLICY)
    data = recipe.to_dict()

    assert data["technique"] == "composite"
    assert [component["technique"] for component in data["components"]] == [
        "zne",
        "pec",
    ]
    assert isinstance(data["warnings"], list)
    json.dumps(data)


def test_cli_generate_with_policy(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    save_policy(DEFAULT_POLICY, path)

    result = runner.invoke(
        main,
        ["generate", "docs/examples/bell_state.qasm", "--policy", str(path)],
    )

    assert result.exit_code == 0
    assert "LinearFactory" in result.output


def test_cli_policy_init_and_validate(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "policy.json"

    init_result = runner.invoke(main, ["policy", "init", str(path)])
    validate_result = runner.invoke(main, ["policy", "validate", str(path)])

    assert init_result.exit_code == 0
    assert "Wrote default policy" in init_result.output
    assert validate_result.exit_code == 0
    assert "Policy is valid" in validate_result.output


def test_cli_invalid_policy_path_errors(runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        ["generate", "docs/examples/bell_state.qasm", "--policy", "missing.json"],
    )

    assert result.exit_code != 0
    assert "Policy file not found" in result.output


def test_cli_invalid_policy_content_errors(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"version": 1}', encoding="utf-8")

    result = runner.invoke(
        main,
        ["generate", "docs/examples/bell_state.qasm", "--policy", str(path)],
    )

    assert result.exit_code != 0
    assert "Invalid policy" in result.output


def test_analyze_json_remains_clean_with_policy_feature(runner: CliRunner) -> None:
    result = runner.invoke(main, ["analyze", "docs/examples/bell_state.qasm", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "policy" not in data
    assert "warnings" not in data
