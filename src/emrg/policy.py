"""Policy model and file loading for configurable EMRG heuristics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "BudgetPolicy",
    "CdrPolicy",
    "CdrTrainingPolicy",
    "CompositePolicy",
    "DEFAULT_POLICY",
    "PecPolicy",
    "RecipePolicy",
    "TechniquePolicySet",
    "ZnePolicy",
    "ZneProfilePolicy",
    "load_policy",
    "policy_to_dict",
    "save_policy",
]

SUPPORTED_POLICY_VERSION = 1
ALLOWED_FACTORIES = frozenset({"LinearFactory", "RichardsonFactory", "PolyFactory"})
ALLOWED_SCALING_METHODS = frozenset({"fold_global", "fold_gates_at_random"})
ALLOWED_FACTORY_KWARGS = {"PolyFactory": frozenset({"order"})}


@dataclass(frozen=True)
class BudgetPolicy:
    """Global recommendation budget limits."""

    max_overhead: float
    max_scale_factors: int
    allow_composite: bool


@dataclass(frozen=True)
class ZneProfilePolicy:
    """Configuration for one ZNE profile."""

    factory: str
    scale_factors: tuple[float, ...]
    scaling_method: str
    factory_kwargs: Mapping[str, Any] = field(default_factory=dict)
    min_depth: int | None = None
    max_depth: int | None = None
    max_multi_qubit_gates: int | None = None
    min_layer_heterogeneity: float | None = None


@dataclass(frozen=True)
class ZnePolicy:
    """ZNE technique policy."""

    enabled: bool
    shallow: ZneProfilePolicy
    moderate: ZneProfilePolicy
    heterogeneous: ZneProfilePolicy
    deep: ZneProfilePolicy


@dataclass(frozen=True)
class PecPolicy:
    """PEC technique policy."""

    enabled: bool
    requires_noise_model: bool
    max_depth: int
    max_overhead: float
    noise_level: float
    min_samples: int
    max_samples: int


@dataclass(frozen=True)
class CdrTrainingPolicy:
    """CDR training circuit count policy."""

    small: int
    medium: int
    large: int
    medium_gate_threshold: int
    large_gate_threshold: int


@dataclass(frozen=True)
class CdrPolicy:
    """CDR technique policy."""

    enabled: bool
    min_depth: int
    max_depth: int
    min_non_clifford_fraction: float
    training_circuits: CdrTrainingPolicy


@dataclass(frozen=True)
class CompositePolicy:
    """Composite ZNE-over-PEC policy."""

    enabled: bool
    min_depth: int
    max_depth: int
    max_combined_overhead: float


@dataclass(frozen=True)
class TechniquePolicySet:
    """All technique policies."""

    zne: ZnePolicy
    pec: PecPolicy
    cdr: CdrPolicy
    composite: CompositePolicy


@dataclass(frozen=True)
class RecipePolicy:
    """Strict versioned policy for EMRG recommendation heuristics."""

    version: int
    name: str
    budget: BudgetPolicy
    techniques: TechniquePolicySet

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipePolicy:
        """Build and validate a policy from a plain mapping."""
        root = _expect_mapping(data, "policy")
        _check_fields(root, {"version", "name", "budget", "techniques"}, set(), "")

        version = _expect_int(root["version"], "version", positive=True)
        if version != SUPPORTED_POLICY_VERSION:
            raise ValueError(
                f"Unsupported policy version: {version}. "
                f"Only version {SUPPORTED_POLICY_VERSION} is supported."
            )

        name = _expect_nonempty_str(root["name"], "name")
        budget = _parse_budget(root["budget"])
        techniques = _parse_techniques(root["techniques"], budget)
        return cls(
            version=version,
            name=name,
            budget=budget,
            techniques=techniques,
        )


def _freeze_mapping(value: Mapping[str, Any]) -> MappingProxyType[str, Any]:
    return MappingProxyType(dict(value))


def _field_path(path: str, key: str) -> str:
    return key if not path else f"{path}.{key}"


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    return value


def _check_fields(
    data: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    allowed = required | optional
    for key in data:
        if key not in allowed:
            raise ValueError(f"Unknown field: {_field_path(path, key)}")
    for key in sorted(required):
        if key not in data:
            raise ValueError(f"Missing required field: {_field_path(path, key)}")


def _expect_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean.")
    return value


def _expect_int(value: Any, path: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{path} must be a {qualifier} integer.")
    if positive and value <= 0:
        raise ValueError(f"{path} must be a positive integer.")
    if not positive and value < 0:
        raise ValueError(f"{path} must be a nonnegative integer.")
    return value


def _expect_number(value: Any, path: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{path} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{path} must be positive.")
    if not positive and result < 0:
        raise ValueError(f"{path} must be nonnegative.")
    return result


def _expect_nonempty_str(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a nonempty string.")
    return value


def _parse_budget(value: Any) -> BudgetPolicy:
    data = _expect_mapping(value, "budget")
    _check_fields(
        data,
        {"max_overhead", "max_scale_factors", "allow_composite"},
        set(),
        "budget",
    )
    return BudgetPolicy(
        max_overhead=_expect_number(
            data["max_overhead"], "budget.max_overhead", positive=True
        ),
        max_scale_factors=_expect_int(
            data["max_scale_factors"], "budget.max_scale_factors", positive=True
        ),
        allow_composite=_expect_bool(data["allow_composite"], "budget.allow_composite"),
    )


def _parse_techniques(value: Any, budget: BudgetPolicy) -> TechniquePolicySet:
    data = _expect_mapping(value, "techniques")
    _check_fields(data, {"zne", "pec", "cdr", "composite"}, set(), "techniques")
    return TechniquePolicySet(
        zne=_parse_zne(data["zne"], budget),
        pec=_parse_pec(data["pec"]),
        cdr=_parse_cdr(data["cdr"]),
        composite=_parse_composite(data["composite"]),
    )


def _parse_zne(value: Any, budget: BudgetPolicy) -> ZnePolicy:
    data = _expect_mapping(value, "techniques.zne")
    _check_fields(
        data,
        {"enabled", "shallow", "moderate", "heterogeneous", "deep"},
        set(),
        "techniques.zne",
    )
    return ZnePolicy(
        enabled=_expect_bool(data["enabled"], "techniques.zne.enabled"),
        shallow=_parse_zne_profile(
            data["shallow"],
            budget,
            "techniques.zne.shallow",
            {"max_depth", "max_multi_qubit_gates"},
        ),
        moderate=_parse_zne_profile(
            data["moderate"],
            budget,
            "techniques.zne.moderate",
            {"min_depth", "max_depth"},
        ),
        heterogeneous=_parse_zne_profile(
            data["heterogeneous"],
            budget,
            "techniques.zne.heterogeneous",
            {"min_depth", "max_depth", "min_layer_heterogeneity"},
        ),
        deep=_parse_zne_profile(
            data["deep"],
            budget,
            "techniques.zne.deep",
            {"min_depth"},
        ),
    )


def _parse_zne_profile(
    value: Any,
    budget: BudgetPolicy,
    path: str,
    threshold_fields: set[str],
) -> ZneProfilePolicy:
    data = _expect_mapping(value, path)
    _check_fields(
        data,
        {"factory", "scale_factors", "scaling_method"} | threshold_fields,
        {"factory_kwargs"},
        path,
    )
    min_depth = _optional_int(data, "min_depth", path)
    max_depth = _optional_int(data, "max_depth", path)
    if min_depth is not None and max_depth is not None and min_depth > max_depth:
        raise ValueError(f"{path}.min_depth cannot exceed max_depth.")

    factory = _expect_nonempty_str(data["factory"], f"{path}.factory")
    if factory not in ALLOWED_FACTORIES:
        raise ValueError(f"Unsupported factory: {factory!r}.")

    scaling_method = _expect_nonempty_str(
        data["scaling_method"], f"{path}.scaling_method"
    )
    if scaling_method not in ALLOWED_SCALING_METHODS:
        raise ValueError(f"Unsupported scaling method: {scaling_method!r}.")

    scale_factors = _parse_scale_factors(data["scale_factors"], budget, path)
    factory_kwargs = _parse_factory_kwargs(
        data.get("factory_kwargs", {}), factory, f"{path}.factory_kwargs"
    )
    max_multi = _optional_int(data, "max_multi_qubit_gates", path)
    min_layer_heterogeneity = _optional_number(data, "min_layer_heterogeneity", path)

    return ZneProfilePolicy(
        factory=factory,
        scale_factors=scale_factors,
        scaling_method=scaling_method,
        factory_kwargs=factory_kwargs,
        min_depth=min_depth,
        max_depth=max_depth,
        max_multi_qubit_gates=max_multi,
        min_layer_heterogeneity=min_layer_heterogeneity,
    )


def _optional_int(data: Mapping[str, Any], key: str, path: str) -> int | None:
    if key not in data:
        return None
    return _expect_int(data[key], f"{path}.{key}")


def _optional_number(data: Mapping[str, Any], key: str, path: str) -> float | None:
    if key not in data:
        return None
    return _expect_number(data[key], f"{path}.{key}")


def _parse_scale_factors(
    value: Any,
    budget: BudgetPolicy,
    path: str,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path}.scale_factors must be a list.")
    if not value:
        raise ValueError(f"{path}.scale_factors must not be empty.")
    if len(value) > budget.max_scale_factors:
        raise ValueError(
            f"{path}.scale_factors exceeds budget.max_scale_factors "
            f"({budget.max_scale_factors})."
        )

    scale_factors = tuple(
        _expect_number(item, f"{path}.scale_factors[{index}]", positive=True)
        for index, item in enumerate(value)
    )
    if 1.0 not in scale_factors:
        raise ValueError(f"{path}.scale_factors must include 1.0.")
    return scale_factors


def _parse_factory_kwargs(
    value: Any,
    factory: str,
    path: str,
) -> MappingProxyType[str, Any]:
    data = _expect_mapping(value, path)
    allowed = ALLOWED_FACTORY_KWARGS.get(factory, frozenset())
    for key in data:
        if key not in allowed:
            raise ValueError(f"Unsupported factory kwarg: {path}.{key}.")

    if factory == "PolyFactory" and "order" in data:
        return _freeze_mapping(
            {"order": _expect_int(data["order"], f"{path}.order", positive=True)}
        )
    return _freeze_mapping({})


def _parse_pec(value: Any) -> PecPolicy:
    data = _expect_mapping(value, "techniques.pec")
    _check_fields(
        data,
        {
            "enabled",
            "requires_noise_model",
            "max_depth",
            "max_overhead",
            "noise_level",
            "min_samples",
            "max_samples",
        },
        set(),
        "techniques.pec",
    )
    min_samples = _expect_int(
        data["min_samples"], "techniques.pec.min_samples", positive=True
    )
    max_samples = _expect_int(
        data["max_samples"], "techniques.pec.max_samples", positive=True
    )
    if min_samples > max_samples:
        raise ValueError("techniques.pec.min_samples cannot exceed max_samples.")
    return PecPolicy(
        enabled=_expect_bool(data["enabled"], "techniques.pec.enabled"),
        requires_noise_model=_expect_bool(
            data["requires_noise_model"], "techniques.pec.requires_noise_model"
        ),
        max_depth=_expect_int(data["max_depth"], "techniques.pec.max_depth"),
        max_overhead=_expect_number(
            data["max_overhead"], "techniques.pec.max_overhead", positive=True
        ),
        noise_level=_expect_number(data["noise_level"], "techniques.pec.noise_level"),
        min_samples=min_samples,
        max_samples=max_samples,
    )


def _parse_cdr(value: Any) -> CdrPolicy:
    data = _expect_mapping(value, "techniques.cdr")
    _check_fields(
        data,
        {
            "enabled",
            "min_depth",
            "max_depth",
            "min_non_clifford_fraction",
            "training_circuits",
        },
        set(),
        "techniques.cdr",
    )
    min_depth = _expect_int(data["min_depth"], "techniques.cdr.min_depth")
    max_depth = _expect_int(data["max_depth"], "techniques.cdr.max_depth")
    if min_depth > max_depth:
        raise ValueError("techniques.cdr.min_depth cannot exceed max_depth.")
    return CdrPolicy(
        enabled=_expect_bool(data["enabled"], "techniques.cdr.enabled"),
        min_depth=min_depth,
        max_depth=max_depth,
        min_non_clifford_fraction=_expect_number(
            data["min_non_clifford_fraction"],
            "techniques.cdr.min_non_clifford_fraction",
        ),
        training_circuits=_parse_cdr_training(data["training_circuits"]),
    )


def _parse_cdr_training(value: Any) -> CdrTrainingPolicy:
    data = _expect_mapping(value, "techniques.cdr.training_circuits")
    _check_fields(
        data,
        {
            "small",
            "medium",
            "large",
            "medium_gate_threshold",
            "large_gate_threshold",
        },
        set(),
        "techniques.cdr.training_circuits",
    )
    medium_threshold = _expect_int(
        data["medium_gate_threshold"],
        "techniques.cdr.training_circuits.medium_gate_threshold",
    )
    large_threshold = _expect_int(
        data["large_gate_threshold"],
        "techniques.cdr.training_circuits.large_gate_threshold",
    )
    if medium_threshold > large_threshold:
        raise ValueError(
            "techniques.cdr.training_circuits.medium_gate_threshold cannot "
            "exceed large_gate_threshold."
        )
    return CdrTrainingPolicy(
        small=_expect_int(
            data["small"], "techniques.cdr.training_circuits.small", positive=True
        ),
        medium=_expect_int(
            data["medium"], "techniques.cdr.training_circuits.medium", positive=True
        ),
        large=_expect_int(
            data["large"], "techniques.cdr.training_circuits.large", positive=True
        ),
        medium_gate_threshold=medium_threshold,
        large_gate_threshold=large_threshold,
    )


def _parse_composite(value: Any) -> CompositePolicy:
    data = _expect_mapping(value, "techniques.composite")
    _check_fields(
        data,
        {"enabled", "min_depth", "max_depth", "max_combined_overhead"},
        set(),
        "techniques.composite",
    )
    min_depth = _expect_int(data["min_depth"], "techniques.composite.min_depth")
    max_depth = _expect_int(data["max_depth"], "techniques.composite.max_depth")
    if min_depth > max_depth:
        raise ValueError("techniques.composite.min_depth cannot exceed max_depth.")
    return CompositePolicy(
        enabled=_expect_bool(data["enabled"], "techniques.composite.enabled"),
        min_depth=min_depth,
        max_depth=max_depth,
        max_combined_overhead=_expect_number(
            data["max_combined_overhead"],
            "techniques.composite.max_combined_overhead",
            positive=True,
        ),
    )


def policy_to_dict(policy: RecipePolicy) -> dict[str, Any]:
    """Return a JSON-compatible representation of *policy*."""
    return {
        "version": policy.version,
        "name": policy.name,
        "budget": {
            "max_overhead": policy.budget.max_overhead,
            "max_scale_factors": policy.budget.max_scale_factors,
            "allow_composite": policy.budget.allow_composite,
        },
        "techniques": {
            "zne": {
                "enabled": policy.techniques.zne.enabled,
                "shallow": _zne_profile_to_dict(policy.techniques.zne.shallow),
                "moderate": _zne_profile_to_dict(policy.techniques.zne.moderate),
                "heterogeneous": _zne_profile_to_dict(
                    policy.techniques.zne.heterogeneous
                ),
                "deep": _zne_profile_to_dict(policy.techniques.zne.deep),
            },
            "pec": {
                "enabled": policy.techniques.pec.enabled,
                "requires_noise_model": policy.techniques.pec.requires_noise_model,
                "max_depth": policy.techniques.pec.max_depth,
                "max_overhead": policy.techniques.pec.max_overhead,
                "noise_level": policy.techniques.pec.noise_level,
                "min_samples": policy.techniques.pec.min_samples,
                "max_samples": policy.techniques.pec.max_samples,
            },
            "cdr": {
                "enabled": policy.techniques.cdr.enabled,
                "min_depth": policy.techniques.cdr.min_depth,
                "max_depth": policy.techniques.cdr.max_depth,
                "min_non_clifford_fraction": (
                    policy.techniques.cdr.min_non_clifford_fraction
                ),
                "training_circuits": {
                    "small": policy.techniques.cdr.training_circuits.small,
                    "medium": policy.techniques.cdr.training_circuits.medium,
                    "large": policy.techniques.cdr.training_circuits.large,
                    "medium_gate_threshold": (
                        policy.techniques.cdr.training_circuits.medium_gate_threshold
                    ),
                    "large_gate_threshold": (
                        policy.techniques.cdr.training_circuits.large_gate_threshold
                    ),
                },
            },
            "composite": {
                "enabled": policy.techniques.composite.enabled,
                "min_depth": policy.techniques.composite.min_depth,
                "max_depth": policy.techniques.composite.max_depth,
                "max_combined_overhead": (
                    policy.techniques.composite.max_combined_overhead
                ),
            },
        },
    }


def _zne_profile_to_dict(profile: ZneProfilePolicy) -> dict[str, Any]:
    data: dict[str, Any] = {
        "factory": profile.factory,
        "scale_factors": list(profile.scale_factors),
        "scaling_method": profile.scaling_method,
    }
    if profile.min_depth is not None:
        data["min_depth"] = profile.min_depth
    if profile.max_depth is not None:
        data["max_depth"] = profile.max_depth
    if profile.max_multi_qubit_gates is not None:
        data["max_multi_qubit_gates"] = profile.max_multi_qubit_gates
    if profile.min_layer_heterogeneity is not None:
        data["min_layer_heterogeneity"] = profile.min_layer_heterogeneity
    if profile.factory_kwargs:
        data["factory_kwargs"] = dict(profile.factory_kwargs)
    return data


def load_policy(path: str | Path) -> RecipePolicy:
    """Load and validate a JSON or YAML policy file."""
    policy_path = Path(path)
    if not policy_path.exists():
        raise ValueError(f"Policy file not found: {policy_path}")

    suffix = policy_path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON policy: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml_policy(policy_path)
    else:
        raise ValueError(
            "Unsupported policy file extension. Use .json, .yaml, or .yml."
        )

    try:
        return RecipePolicy.from_dict(data)
    except ValueError as exc:
        raise ValueError(f"Invalid policy: {exc}") from exc


def _load_yaml_policy(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            "YAML policy support requires PyYAML. Install with: pip install "
            "emrg[config]"
        ) from exc

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML policy: {exc}") from exc


def save_policy(policy: RecipePolicy, path: str | Path) -> None:
    """Write *policy* as JSON or YAML based on the file extension."""
    policy_path = Path(path)
    suffix = policy_path.suffix.lower()
    data = policy_to_dict(policy)
    if suffix == ".json":
        policy_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "YAML policy output requires PyYAML. Install with: pip install "
                "emrg[config]"
            ) from exc
        policy_path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
        return
    raise ValueError("Unsupported policy file extension. Use .json, .yaml, or .yml.")


_DEFAULT_POLICY_DATA: dict[str, Any] = {
    "version": 1,
    "name": "conservative-default",
    "budget": {
        "max_overhead": 1000.0,
        "max_scale_factors": 5,
        "allow_composite": True,
    },
    "techniques": {
        "zne": {
            "enabled": True,
            "shallow": {
                "max_depth": 20,
                "max_multi_qubit_gates": 50,
                "factory": "LinearFactory",
                "scale_factors": [1.0, 1.5, 2.0],
                "scaling_method": "fold_global",
            },
            "moderate": {
                "min_depth": 20,
                "max_depth": 50,
                "factory": "RichardsonFactory",
                "scale_factors": [1.0, 1.5, 2.0, 2.5],
                "scaling_method": "fold_global",
            },
            "heterogeneous": {
                "min_depth": 15,
                "max_depth": 50,
                "min_layer_heterogeneity": 2.0,
                "factory": "RichardsonFactory",
                "scale_factors": [1.0, 1.5, 2.0, 2.5],
                "scaling_method": "fold_gates_at_random",
            },
            "deep": {
                "min_depth": 51,
                "factory": "PolyFactory",
                "factory_kwargs": {"order": 2},
                "scale_factors": [1.0, 1.5, 2.0, 2.5, 3.0],
                "scaling_method": "fold_gates_at_random",
            },
        },
        "pec": {
            "enabled": True,
            "requires_noise_model": True,
            "max_depth": 30,
            "max_overhead": 1000.0,
            "noise_level": 0.01,
            "min_samples": 100,
            "max_samples": 500,
        },
        "cdr": {
            "enabled": True,
            "min_depth": 10,
            "max_depth": 40,
            "min_non_clifford_fraction": 0.2,
            "training_circuits": {
                "small": 8,
                "medium": 12,
                "large": 16,
                "medium_gate_threshold": 20,
                "large_gate_threshold": 50,
            },
        },
        "composite": {
            "enabled": True,
            "min_depth": 15,
            "max_depth": 30,
            "max_combined_overhead": 1000.0,
        },
    },
}

DEFAULT_POLICY = RecipePolicy.from_dict(_DEFAULT_POLICY_DATA)
