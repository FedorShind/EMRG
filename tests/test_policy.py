"""Tests for EMRG policy loading, validation, and serialization."""

from __future__ import annotations

import builtins
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from emrg.policy import (
    DEFAULT_POLICY,
    RecipePolicy,
    load_policy,
    policy_to_dict,
    save_policy,
)


def _default_data() -> dict[str, Any]:
    return deepcopy(policy_to_dict(DEFAULT_POLICY))


def test_default_policy_is_version_one() -> None:
    assert DEFAULT_POLICY.version == 1
    assert DEFAULT_POLICY.name == "conservative-default"


def test_policy_dataclass_is_immutable() -> None:
    with pytest.raises(AttributeError):
        DEFAULT_POLICY.name = "changed"  # type: ignore[misc]


def test_valid_json_policy_loads(tmp_path: Path) -> None:
    path = tmp_path / "emrg-policy.json"
    path.write_text(json.dumps(_default_data()), encoding="utf-8")

    policy = load_policy(path)

    assert isinstance(policy, RecipePolicy)
    assert policy == DEFAULT_POLICY


def test_valid_yaml_policy_loads_when_pyyaml_is_installed(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "emrg-policy.yaml"
    save_policy(DEFAULT_POLICY, path)

    policy = load_policy(path)

    assert policy == DEFAULT_POLICY


def test_yaml_missing_dependency_has_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "emrg-policy.yaml"
    path.write_text(json.dumps(_default_data()), encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match=r"emrg\[config\]"):
        load_policy(path)


def test_invalid_extension_errors(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported policy file extension"):
        load_policy(path)


def test_malformed_json_errors_clearly(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse JSON policy"):
        load_policy(path)


def test_missing_required_field_errors() -> None:
    data = _default_data()
    del data["techniques"]

    with pytest.raises(ValueError, match="Missing required field: techniques"):
        RecipePolicy.from_dict(data)


def test_unknown_top_level_field_errors() -> None:
    data = _default_data()
    data["extra"] = True

    with pytest.raises(ValueError, match="Unknown field: extra"):
        RecipePolicy.from_dict(data)


def test_unknown_technique_field_errors() -> None:
    data = _default_data()
    data["techniques"]["magic"] = {"enabled": True}

    with pytest.raises(ValueError, match="Unknown field: techniques.magic"):
        RecipePolicy.from_dict(data)


def test_unknown_zne_profile_field_errors() -> None:
    data = _default_data()
    data["techniques"]["zne"]["shallow"]["callable"] = "os.system"

    with pytest.raises(
        ValueError,
        match="Unknown field: techniques.zne.shallow.callable",
    ):
        RecipePolicy.from_dict(data)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update({"version": 2}), "Unsupported policy version"),
        (lambda data: data.update({"name": ""}), "name must be a nonempty string"),
        (
            lambda data: data["budget"].update({"max_overhead": "high"}),
            "budget.max_overhead must be a number",
        ),
        (
            lambda data: data["budget"].update({"max_overhead": float("inf")}),
            "budget.max_overhead must be finite",
        ),
        (
            lambda data: data["budget"].update({"max_overhead": 0}),
            "budget.max_overhead must be positive",
        ),
        (
            lambda data: data["techniques"]["pec"].update({"noise_level": -0.1}),
            "techniques.pec.noise_level must be nonnegative",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"scale_factors": "1,2,3"}
            ),
            "scale_factors must be a list",
        ),
        (
            lambda data: data["techniques"]["cdr"].update(
                {"min_depth": 41, "max_depth": 40}
            ),
            "techniques.cdr.min_depth cannot exceed max_depth",
        ),
        (
            lambda data: data["techniques"]["composite"].update(
                {"min_depth": 31, "max_depth": 30}
            ),
            "techniques.composite.min_depth cannot exceed max_depth",
        ),
    ],
)
def test_policy_validation_rejects_more_invalid_scalars(
    mutate: Any,
    match: str,
) -> None:
    data = _default_data()
    mutate(data)

    with pytest.raises(ValueError, match=match):
        RecipePolicy.from_dict(data)


def test_policy_document_must_be_mapping() -> None:
    with pytest.raises(ValueError, match="policy must be an object"):
        RecipePolicy.from_dict([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda data: data["techniques"]["zne"]["moderate"].update(
                {"min_depth": -1}
            ),
            "nonnegative integer",
        ),
        (
            lambda data: data["techniques"]["zne"]["moderate"].update(
                {"min_depth": 51, "max_depth": 20}
            ),
            "min_depth cannot exceed max_depth",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"scale_factors": []}
            ),
            "scale_factors must not be empty",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"scale_factors": [1.5, 2.0]}
            ),
            "scale_factors must include 1.0",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"scale_factors": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
            ),
            "exceeds budget.max_scale_factors",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"factory": "ExpFactory"}
            ),
            "Unsupported factory",
        ),
        (
            lambda data: data["techniques"]["zne"]["shallow"].update(
                {"scaling_method": "eval"}
            ),
            "Unsupported scaling method",
        ),
        (
            lambda data: data["techniques"]["zne"]["deep"].update(
                {"factory_kwargs": {"degree": 2}}
            ),
            "Unsupported factory kwarg",
        ),
        (
            lambda data: data["techniques"]["pec"].update(
                {"min_samples": 500, "max_samples": 100}
            ),
            "min_samples cannot exceed max_samples",
        ),
        (
            lambda data: data["techniques"]["cdr"]["training_circuits"].update(
                {"small": 0}
            ),
            "positive integer",
        ),
        (
            lambda data: data["techniques"]["cdr"]["training_circuits"].update(
                {"medium_gate_threshold": 60, "large_gate_threshold": 50}
            ),
            "medium_gate_threshold cannot exceed large_gate_threshold",
        ),
        (
            lambda data: data["techniques"]["pec"].update({"enabled": "yes"}),
            "must be a boolean",
        ),
    ],
)
def test_policy_validation_rejects_invalid_values(
    mutate: Any,
    match: str,
) -> None:
    data = _default_data()
    mutate(data)

    with pytest.raises(ValueError, match=match):
        RecipePolicy.from_dict(data)


def test_save_policy_json_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"

    save_policy(DEFAULT_POLICY, path)

    loaded = load_policy(path)
    assert loaded == DEFAULT_POLICY
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_malformed_yaml_errors_clearly(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "policy.yaml"
    path.write_text("version: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse YAML policy"):
        load_policy(path)


def test_save_policy_yaml_missing_dependency_has_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "policy.yaml"
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match=r"emrg\[config\]"):
        save_policy(DEFAULT_POLICY, path)


def test_save_policy_invalid_extension_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported policy file extension"):
        save_policy(DEFAULT_POLICY, tmp_path / "policy.txt")
