"""Heuristic Engine -- select error mitigation recipes from circuit features.

This module is the decision-making core of EMRG. It takes a
:class:`~emrg.analyzer.CircuitFeatures` snapshot and returns a
:class:`MitigationRecipe` describing the recommended Mitiq configuration.

The engine uses a **list-of-rules** pattern: each rule is a
``(predicate, builder)`` pair evaluated in priority order.  The first
matching predicate wins.  A conservative fallback always matches last.

Design rationale
~~~~~~~~~~~~~~~~
* **Deterministic** -- no ML/AI, purely rule-based for reproducibility.
* **Extensible** -- add a rule by appending to :data:`DEFAULT_RULES`.
  Structured policies can tune the built-in thresholds and recipe parameters.
* **Explainable** -- every recipe carries human-readable rationale lines
  with literature references.

Typical usage::

    from emrg.analyzer import analyze_circuit
    from emrg.heuristics import recommend

    features = analyze_circuit(qc)
    recipe = recommend(features)
    print(recipe.factory_name, recipe.scale_factors)
    for line in recipe.rationale:
        print(f"  - {line}")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from emrg.analyzer import CircuitFeatures
from emrg.policy import (
    DEFAULT_POLICY,
    CdrPolicy,
    PecPolicy,
    RecipePolicy,
    ZneProfilePolicy,
)

__all__ = [
    "MitigationRecipe",
    "Rule",
    "DEFAULT_RULES",
    "recommend",
    "DEPTH_DEEP_THRESHOLD",
    "DEPTH_MODERATE_THRESHOLD",
    "MULTI_QUBIT_GATE_SHALLOW_MAX",
    "PEC_MAX_DEPTH",
    "PEC_MAX_OVERHEAD",
    "PEC_DEFAULT_NOISE_LEVEL",
    "PEC_MIN_SAMPLES",
    "PEC_MAX_SAMPLES",
    "COMPOSITE_MIN_DEPTH",
    "COMPOSITE_MAX_DEPTH",
    "COMPOSITE_MAX_OVERHEAD",
    "LAYERWISE_MIN_DEPTH",
    "LAYERWISE_MAX_DEPTH",
    "LAYERWISE_HETEROGENEITY_THRESHOLD",
    "CDR_TRAINING_CIRCUITS_SMALL",
    "CDR_TRAINING_CIRCUITS_MEDIUM",
    "CDR_TRAINING_CIRCUITS_LARGE",
    "CDR_GATE_THRESHOLD_MEDIUM",
    "CDR_GATE_THRESHOLD_LARGE",
]

# ---------------------------------------------------------------------------
# Threshold constants (tunable; referenced by predicates)
# ---------------------------------------------------------------------------

#: Circuits with depth above this are considered "deep".
DEPTH_DEEP_THRESHOLD: int = 50

#: Circuits with depth at or above this are "moderate" (up to DEPTH_DEEP_THRESHOLD).
DEPTH_MODERATE_THRESHOLD: int = 20

#: Shallow circuits must have fewer multi-qubit gates than this.
MULTI_QUBIT_GATE_SHALLOW_MAX: int = 50

#: Maximum circuit depth for PEC to be considered.
PEC_MAX_DEPTH: int = 30

#: PEC is only practical when the estimated overhead is below this.
PEC_MAX_OVERHEAD: float = 1000.0

#: Default depolarizing noise level assumed for PEC representations.
PEC_DEFAULT_NOISE_LEVEL: float = 0.01

#: Minimum number of PEC samples to request.
PEC_MIN_SAMPLES: int = 100

#: Maximum number of PEC samples to request.
PEC_MAX_SAMPLES: int = 500

# --- Composite (ZNE over PEC) constants ---------------------------------------

#: Minimum circuit depth for composite mitigation consideration.
COMPOSITE_MIN_DEPTH: int = 15

#: Maximum circuit depth for composite mitigation consideration.
COMPOSITE_MAX_DEPTH: int = PEC_MAX_DEPTH

#: Maximum combined shot multiplier for automatic composite selection.
COMPOSITE_MAX_OVERHEAD: float = 1000.0

#: Minimum circuit depth for layerwise Richardson consideration.
LAYERWISE_MIN_DEPTH: int = 15

#: Maximum circuit depth for layerwise Richardson consideration.
LAYERWISE_MAX_DEPTH: int = 50

#: Minimum layer heterogeneity (exclusive) to trigger layerwise Richardson.
LAYERWISE_HETEROGENEITY_THRESHOLD: float = 2.0

# --- CDR (Clifford Data Regression) constants --------------------------------

#: Number of CDR training circuits for small circuits (< 20 gates).
CDR_TRAINING_CIRCUITS_SMALL: int = 8

#: Number of CDR training circuits for medium circuits (20-50 gates).
CDR_TRAINING_CIRCUITS_MEDIUM: int = 12

#: Number of CDR training circuits for large circuits (> 50 gates).
CDR_TRAINING_CIRCUITS_LARGE: int = 16

#: Gate count threshold between small and medium CDR training.
CDR_GATE_THRESHOLD_MEDIUM: int = 20

#: Gate count threshold between medium and large CDR training.
CDR_GATE_THRESHOLD_LARGE: int = 50

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _freeze_kwargs(value: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Wrap a dict in a read-only ``MappingProxyType``."""
    return MappingProxyType(value)


@dataclass(frozen=True)
class MitigationRecipe:
    """Immutable recommendation for an error mitigation strategy.

    This dataclass is consumed by the codegen module to produce runnable
    Python code.  It is intentionally technique-generic to support future
    PEC / CDR extensions.

    Attributes:
        technique: Mitigation family (``"zne"``; future: ``"pec"``, ``"cdr"``).
        factory_name: Mitiq factory class name
            (e.g. ``"LinearFactory"``, ``"RichardsonFactory"``).
        scale_factors: Noise-scaling levels passed to the factory.
        factory_kwargs: Read-only mapping of extra keyword arguments for the
            factory constructor (e.g. ``{"order": 2}`` for :class:`PolyFactory`).
        scaling_method: Mitiq scaling function name
            (``"fold_global"`` or ``"fold_gates_at_random"``).
        rationale: Human-readable explanation lines, including literature
            references where applicable.
        noise_category: Copied from :class:`CircuitFeatures` for convenience.
        estimated_overhead: Approximate shot-count multiplier
            (roughly ``len(scale_factors)``).
        components: Child recipes used by composite techniques. Empty for
            single-technique recipes.
        warnings: Validation notes for forced techniques that do not satisfy
            the automatic selection predicates. Empty for normal auto-selected
            recipes.
    """

    technique: str
    factory_name: str
    scale_factors: tuple[float, ...]
    factory_kwargs: Mapping[str, Any] = field(default_factory=dict)
    scaling_method: str = "fold_global"
    rationale: tuple[str, ...] = ()
    noise_category: str = "low"
    estimated_overhead: float = 1.0
    components: tuple[MitigationRecipe, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the recipe."""
        return {
            "technique": self.technique,
            "factory_name": self.factory_name,
            "scale_factors": list(self.scale_factors),
            "factory_kwargs": dict(self.factory_kwargs),
            "scaling_method": self.scaling_method,
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "noise_category": self.noise_category,
            "estimated_overhead": self.estimated_overhead,
            "components": [component.to_dict() for component in self.components],
        }


def _with_warnings(
    recipe: MitigationRecipe,
    warnings: Sequence[str],
) -> MitigationRecipe:
    """Return *recipe* with immutable warning notes attached."""
    return replace(recipe, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Type alias for rules
# ---------------------------------------------------------------------------

#: A rule is a (predicate, builder) pair.
#:
#: * **predicate** receives :class:`CircuitFeatures` and returns ``True``
#:   if this rule applies.
#: * **builder** receives :class:`CircuitFeatures` and returns a fully
#:   populated :class:`MitigationRecipe`.
Rule = tuple[
    Callable[[CircuitFeatures], bool],
    Callable[[CircuitFeatures], MitigationRecipe],
]


# ---------------------------------------------------------------------------
# PEC helpers
# ---------------------------------------------------------------------------


def _effective_pec_max_overhead(policy: RecipePolicy) -> float:
    return policy.techniques.pec.max_overhead


def _should_use_pec(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> bool:
    """Return ``True`` when PEC is viable for the given circuit.

    PEC is recommended when **all** of:
    * depth <= ``PEC_MAX_DEPTH``
    * ``noise_model_available`` is ``True``
    * ``pec_overhead_estimate`` < ``PEC_MAX_OVERHEAD``
    """
    active_policy = policy or DEFAULT_POLICY
    pec_policy = active_policy.techniques.pec
    if not pec_policy.enabled:
        return False
    if features.depth > pec_policy.max_depth:
        return False
    if pec_policy.requires_noise_model and not features.noise_model_available:
        return False
    return features.pec_overhead_estimate < _effective_pec_max_overhead(active_policy)


def _compute_pec_samples(
    overhead: float,
    policy: PecPolicy | None = None,
) -> int:
    """Derive the number of PEC samples from the overhead estimate.

    The result is clamped between ``PEC_MIN_SAMPLES`` and ``PEC_MAX_SAMPLES``.
    ``math.inf`` and ``NaN`` overheads (possible when a user forces PEC on
    a circuit deep enough that the exponential cap kicks in) collapse to
    ``PEC_MAX_SAMPLES``.
    """
    import math

    active_policy = policy or DEFAULT_POLICY.techniques.pec
    if math.isinf(overhead) or math.isnan(overhead):
        return active_policy.max_samples
    return max(
        active_policy.min_samples,
        min(active_policy.max_samples, int(overhead * 2)),
    )


def _build_pec_recipe(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> MitigationRecipe:
    """Build a PEC recipe for shallow circuits with an available noise model."""
    active_policy = policy or DEFAULT_POLICY
    pec_policy = active_policy.techniques.pec
    num_samples = _compute_pec_samples(features.pec_overhead_estimate, pec_policy)
    max_overhead = _effective_pec_max_overhead(active_policy)
    return MitigationRecipe(
        technique="pec",
        factory_name="",
        scale_factors=(),
        factory_kwargs=_freeze_kwargs(
            {
                "num_samples": num_samples,
                "noise_level": pec_policy.noise_level,
            }
        ),
        scaling_method="",
        rationale=(
            f"Circuit depth ({features.depth}) <= {pec_policy.max_depth} with "
            f"noise model available and PEC overhead "
            f"({features.pec_overhead_estimate:.1f}) < {max_overhead:.0f}.",
            "PEC provides unbiased error mitigation by probabilistically "
            "cancelling noise using quasi-probability representations "
            "(Temme et al., PRL 119, 180509, 2017).",
            f"Using {num_samples} samples (derived from overhead estimate).",
            "PEC is preferred over ZNE when a noise model is available "
            "and the sampling overhead is manageable.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=features.pec_overhead_estimate,
    )


def _forced_pec_warnings(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> tuple[str, ...]:
    """Return validation notes for a forced PEC recipe."""
    active_policy = policy or DEFAULT_POLICY
    pec_policy = active_policy.techniques.pec
    max_overhead = _effective_pec_max_overhead(active_policy)
    warnings: list[str] = []
    if not pec_policy.enabled:
        warnings.append("Forced PEC: PEC is disabled by policy.")
    if pec_policy.requires_noise_model and not features.noise_model_available:
        warnings.append(
            "Forced PEC: no noise model is marked available, so the generated "
            "depolarizing representations may not match the target backend."
        )
    if features.depth > pec_policy.max_depth:
        warnings.append(
            f"Forced PEC: circuit depth ({features.depth}) exceeds the "
            f"automatic PEC limit ({pec_policy.max_depth})."
        )
    if features.pec_overhead_estimate >= max_overhead:
        warnings.append(
            f"Forced PEC: estimated sampling overhead "
            f"({features.pec_overhead_estimate:.1f}) is at or above the "
            f"automatic PEC limit ({max_overhead:.0f})."
        )
    return tuple(warnings)


# ---------------------------------------------------------------------------
# CDR helpers
# ---------------------------------------------------------------------------


def _should_use_cdr(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> bool:
    """Return ``True`` when CDR is viable for the given circuit.

    CDR is recommended when **all** of:
    * ``non_clifford_fraction`` > ``CDR_NON_CLIFFORD_FRACTION_THRESHOLD``
    * ``CDR_MIN_DEPTH`` <= depth <= ``CDR_MAX_DEPTH``
    """
    active_policy = policy or DEFAULT_POLICY
    cdr_policy = active_policy.techniques.cdr
    return (
        cdr_policy.enabled
        and features.non_clifford_fraction > cdr_policy.min_non_clifford_fraction
        and cdr_policy.min_depth <= features.depth <= cdr_policy.max_depth
    )


def _compute_cdr_training_circuits(
    total_gate_count: int,
    policy: CdrPolicy | None = None,
) -> int:
    """Derive the number of CDR training circuits from the gate count."""
    active_policy = policy or DEFAULT_POLICY.techniques.cdr
    training = active_policy.training_circuits
    if total_gate_count < training.medium_gate_threshold:
        return training.small
    if total_gate_count <= training.large_gate_threshold:
        return training.medium
    return training.large


def _build_cdr_recipe(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> MitigationRecipe:
    """Build a CDR recipe for circuits with significant non-Clifford content."""
    active_policy = policy or DEFAULT_POLICY
    cdr_policy = active_policy.techniques.cdr
    num_training = _compute_cdr_training_circuits(
        features.total_gate_count,
        cdr_policy,
    )
    return MitigationRecipe(
        technique="cdr",
        factory_name="",
        scale_factors=(),
        factory_kwargs=_freeze_kwargs(
            {
                "num_training_circuits": num_training,
                "fit_method": "linear",
            }
        ),
        scaling_method="",
        rationale=(
            f"Circuit has {features.non_clifford_fraction:.0%} non-Clifford "
            f"gates (threshold: "
            f">{cdr_policy.min_non_clifford_fraction:.0%}) at depth "
            f"{features.depth} (range: "
            f"{cdr_policy.min_depth}-{cdr_policy.max_depth}).",
            "CDR replaces non-Clifford gates with near-Clifford substitutes "
            "to create training circuits that can be simulated classically, "
            "then fits a regression model to correct the noisy results "
            "(Czarnik et al., Quantum 5, 592, 2021).",
            f"Using {num_training} training circuits with linear regression.",
            "CDR is more accurate than ZNE on circuits with many "
            "non-Clifford gates, and has lower overhead than PEC.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=float(num_training),
    )


def _forced_cdr_warnings(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> tuple[str, ...]:
    """Return validation notes for a forced CDR recipe."""
    active_policy = policy or DEFAULT_POLICY
    cdr_policy = active_policy.techniques.cdr
    warnings: list[str] = []
    if not cdr_policy.enabled:
        warnings.append("Forced CDR: CDR is disabled by policy.")
    if features.non_clifford_fraction <= cdr_policy.min_non_clifford_fraction:
        warnings.append(
            f"Forced CDR: non-Clifford fraction "
            f"({features.non_clifford_fraction:.0%}) is at or below the "
            f"automatic CDR threshold "
            f"({cdr_policy.min_non_clifford_fraction:.0%})."
        )
    if not cdr_policy.min_depth <= features.depth <= cdr_policy.max_depth:
        warnings.append(
            f"Forced CDR: circuit depth ({features.depth}) is outside the "
            f"automatic CDR range ({cdr_policy.min_depth}-{cdr_policy.max_depth})."
        )
    return tuple(warnings)


# ---------------------------------------------------------------------------
# Composite helpers
# ---------------------------------------------------------------------------


def _build_zne_recipe(
    features: CircuitFeatures,
    rules: Sequence[Rule] | None = None,
    policy: RecipePolicy | None = None,
) -> MitigationRecipe:
    """Return the ZNE recipe selected by the configured ZNE rule chain."""
    if policy is not None:
        return _build_policy_zne_recipe(features, policy)

    if rules is None:
        active_rules = DEFAULT_RULES
    else:
        active_rules = tuple(rules)

    for predicate, builder in active_rules:
        if predicate(features):
            return builder(features)

    return _build_fallback_recipe(features)


def _build_composite_recipe(
    features: CircuitFeatures,
    rules: Sequence[Rule] | None = None,
    policy: RecipePolicy | None = None,
    force_components: bool = False,
) -> MitigationRecipe:
    """Build a composite recipe that runs ZNE over a PEC executor."""
    if policy is not None and force_components:
        zne_recipe = _build_policy_zne_recipe(features, policy, force=True)
    else:
        zne_recipe = _build_zne_recipe(features, rules, policy)
    pec_recipe = _build_pec_recipe(features, policy)
    combined_overhead = zne_recipe.estimated_overhead * pec_recipe.estimated_overhead
    return MitigationRecipe(
        technique="composite",
        factory_name="",
        scale_factors=(),
        factory_kwargs=_freeze_kwargs({}),
        scaling_method="",
        rationale=(
            "Composite recipe selected: PEC corrects each noise-scaled "
            "circuit before ZNE extrapolates the residual bias.",
            f"ZNE component uses {zne_recipe.factory_name} with "
            f"{zne_recipe.scaling_method}; PEC component uses "
            f"{pec_recipe.factory_kwargs['num_samples']} samples.",
            f"Combined estimated overhead is ~{combined_overhead:.1f}x "
            "the base shot count.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=combined_overhead,
        components=(zne_recipe, pec_recipe),
    )


def _effective_composite_max_overhead(policy: RecipePolicy) -> float:
    return min(
        policy.budget.max_overhead,
        policy.techniques.composite.max_combined_overhead,
    )


def _should_use_composite(
    features: CircuitFeatures,
    policy: RecipePolicy | None = None,
) -> bool:
    """Return ``True`` when ZNE-over-PEC is viable and worth the cost."""
    active_policy = policy or DEFAULT_POLICY
    composite_policy = active_policy.techniques.composite
    if not active_policy.budget.allow_composite or not composite_policy.enabled:
        return False
    if not active_policy.techniques.zne.enabled:
        return False
    if not _should_use_pec(features, active_policy):
        return False
    if features.noise_category == "low":
        return False
    if _should_use_cdr(features, active_policy):
        return False
    if not composite_policy.min_depth <= features.depth <= composite_policy.max_depth:
        return False

    zne_recipe = _build_zne_recipe(features, policy=active_policy)
    combined_overhead = zne_recipe.estimated_overhead * features.pec_overhead_estimate
    return combined_overhead <= _effective_composite_max_overhead(active_policy)


def _forced_composite_warnings(
    features: CircuitFeatures,
    rules: Sequence[Rule] | None = None,
    policy: RecipePolicy | None = None,
) -> tuple[str, ...]:
    """Return validation notes for a forced composite recipe."""
    active_policy = policy or DEFAULT_POLICY
    pec_policy = active_policy.techniques.pec
    zne_policy = active_policy.techniques.zne
    cdr_policy = active_policy.techniques.cdr
    composite_policy = active_policy.techniques.composite
    pec_max_overhead = _effective_pec_max_overhead(active_policy)
    composite_max_overhead = _effective_composite_max_overhead(active_policy)
    warnings: list[str] = []

    if not active_policy.budget.allow_composite:
        warnings.append("Forced composite: composite recipes are disabled by budget.")
    if not composite_policy.enabled:
        warnings.append("Forced composite: composite is disabled by policy.")
    if not pec_policy.enabled:
        warnings.append("Forced composite: the required PEC component is disabled.")
    if not zne_policy.enabled:
        warnings.append("Forced composite: the required ZNE component is disabled.")
    if pec_policy.requires_noise_model and not features.noise_model_available:
        warnings.append(
            "Forced composite: no noise model is marked available, so the PEC "
            "component may not match the target backend."
        )
    if features.pec_overhead_estimate >= pec_max_overhead:
        warnings.append(
            f"Forced composite: PEC overhead "
            f"({features.pec_overhead_estimate:.1f}) is at or above the "
            f"automatic PEC limit ({pec_max_overhead:.0f})."
        )
    if features.noise_category == "low":
        warnings.append(
            "Forced composite: automatic selection skips low-noise circuits "
            "because composite overhead is usually not justified."
        )
    if cdr_policy.enabled and _should_use_cdr(features, active_policy):
        warnings.append(
            "Forced composite: automatic selection would prefer CDR for this "
            "non-Clifford circuit profile."
        )
    if not composite_policy.min_depth <= features.depth <= composite_policy.max_depth:
        warnings.append(
            f"Forced composite: circuit depth ({features.depth}) is outside "
            f"the automatic composite range "
            f"({composite_policy.min_depth}-{composite_policy.max_depth})."
        )

    if policy is not None:
        zne_recipe = _build_policy_zne_recipe(features, policy, force=True)
    else:
        zne_recipe = _build_zne_recipe(features, rules)
    combined_overhead = zne_recipe.estimated_overhead * features.pec_overhead_estimate
    if combined_overhead > composite_max_overhead:
        warnings.append(
            f"Forced composite: combined overhead ({combined_overhead:.1f}) "
            f"exceeds the automatic composite limit "
            f"({composite_max_overhead:.0f})."
        )

    return tuple(warnings)


# ---------------------------------------------------------------------------
# ZNE predicates
# ---------------------------------------------------------------------------


def _is_deep_or_high_noise(features: CircuitFeatures) -> bool:
    """Match deep circuits (depth > DEPTH_DEEP_THRESHOLD) or high estimated noise."""
    return features.depth > DEPTH_DEEP_THRESHOLD or features.noise_category == "high"


def _is_moderate_depth(features: CircuitFeatures) -> bool:
    """Match moderate-depth circuits (20 <= depth <= 50)."""
    return DEPTH_MODERATE_THRESHOLD <= features.depth <= DEPTH_DEEP_THRESHOLD


def _is_moderate_heterogeneous(features: CircuitFeatures) -> bool:
    """Match moderate-depth circuits with high layer heterogeneity.

    These circuits benefit from per-gate noise folding rather than
    uniform folding because their layers have uneven multi-qubit gate
    density.
    """
    return (
        LAYERWISE_MIN_DEPTH <= features.depth <= LAYERWISE_MAX_DEPTH
        and features.layer_heterogeneity > LAYERWISE_HETEROGENEITY_THRESHOLD
    )


def _is_shallow(features: CircuitFeatures) -> bool:
    """Match shallow, low-gate-count circuits (depth < 20, multi-qubit < 50)."""
    return (
        features.depth < DEPTH_MODERATE_THRESHOLD
        and features.multi_qubit_gate_count < MULTI_QUBIT_GATE_SHALLOW_MAX
    )


# ---------------------------------------------------------------------------
# Policy-backed ZNE helpers
# ---------------------------------------------------------------------------


def _build_policy_zne_recipe(
    features: CircuitFeatures,
    policy: RecipePolicy,
    *,
    force: bool = False,
) -> MitigationRecipe:
    """Return the ZNE recipe selected by a structured policy."""
    zne_policy = policy.techniques.zne
    if not zne_policy.enabled and not force:
        raise ValueError("Policy disables ZNE and no enabled technique matched.")

    if _matches_deep_profile(features, zne_policy.deep):
        return _build_zne_profile_recipe(features, zne_policy.deep, "deep")
    if _matches_heterogeneous_profile(features, zne_policy.heterogeneous):
        return _build_zne_profile_recipe(
            features,
            zne_policy.heterogeneous,
            "heterogeneous",
        )
    if _matches_moderate_profile(features, zne_policy.moderate):
        return _build_zne_profile_recipe(features, zne_policy.moderate, "moderate")
    if _matches_shallow_profile(features, zne_policy.shallow):
        return _build_zne_profile_recipe(features, zne_policy.shallow, "shallow")
    return _build_policy_fallback_recipe(features, zne_policy.shallow)


def _matches_deep_profile(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
) -> bool:
    min_depth = profile.min_depth or 0
    return features.depth >= min_depth or features.noise_category == "high"


def _matches_heterogeneous_profile(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
) -> bool:
    return (
        profile.min_depth is not None
        and profile.max_depth is not None
        and profile.min_layer_heterogeneity is not None
        and profile.min_depth <= features.depth <= profile.max_depth
        and features.layer_heterogeneity > profile.min_layer_heterogeneity
    )


def _matches_moderate_profile(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
) -> bool:
    return (
        profile.min_depth is not None
        and profile.max_depth is not None
        and profile.min_depth <= features.depth <= profile.max_depth
    )


def _matches_shallow_profile(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
) -> bool:
    return (
        profile.max_depth is not None
        and profile.max_multi_qubit_gates is not None
        and features.depth < profile.max_depth
        and features.multi_qubit_gate_count < profile.max_multi_qubit_gates
    )


def _build_zne_profile_recipe(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
    profile_name: str,
) -> MitigationRecipe:
    rationale = _profile_rationale(features, profile, profile_name)
    return MitigationRecipe(
        technique="zne",
        factory_name=profile.factory,
        scale_factors=profile.scale_factors,
        factory_kwargs=_freeze_kwargs(dict(profile.factory_kwargs)),
        scaling_method=profile.scaling_method,
        rationale=rationale,
        noise_category=features.noise_category,
        estimated_overhead=float(len(profile.scale_factors)),
    )


def _profile_rationale(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
    profile_name: str,
) -> tuple[str, ...]:
    if profile_name == "deep":
        threshold = (profile.min_depth or 51) - 1
        return (
            f"Circuit depth ({features.depth}) > {threshold} or noise category "
            f"'{features.noise_category}' indicates strong non-linear noise.",
            "PolyFactory with order=2 captures quadratic noise scaling "
            "better than linear/Richardson for deep circuits.",
            "fold_gates_at_random reduces coherent error accumulation "
            "compared to uniform folding (arXiv:2005.10921).",
            "Five scale factors provide sufficient data points for a "
            "degree-2 polynomial fit (arXiv:2307.05203).",
        )
    if profile_name == "heterogeneous":
        return (
            f"Circuit depth ({features.depth}) is moderate "
            f"({profile.min_depth}-{profile.max_depth}) with high layer "
            f"heterogeneity ({features.layer_heterogeneity:.2f} "
            f"> {profile.min_layer_heterogeneity}).",
            "Layers have uneven multi-qubit gate density, so "
            "fold_gates_at_random targets the noisiest gates rather than "
            "amplifying noise uniformly (arXiv:2005.10921).",
            "RichardsonFactory provides polynomial interpolation suited "
            "for moderate-depth circuits with non-uniform noise.",
            "Four scale factors balance accuracy vs. shot overhead.",
        )
    if profile_name == "moderate":
        return (
            f"Circuit depth ({features.depth}) is in the moderate range "
            f"({profile.min_depth}-{profile.max_depth}) with noise category "
            f"'{features.noise_category}'.",
            "RichardsonFactory uses polynomial interpolation that handles "
            "moderate non-linear noise better than linear extrapolation "
            "(Temme et al., PRL 119, 180509, 2017).",
            "fold_global provides uniform noise amplification suitable "
            "for structured circuits at moderate depth.",
            "Four scale factors balance accuracy vs. shot overhead.",
        )
    return (
        f"Circuit depth ({features.depth}) < {profile.max_depth} with "
        f"{features.multi_qubit_gate_count} multi-qubit gates -- "
        f"noise category '{features.noise_category}'.",
        "LinearFactory is sufficient when noise scales approximately "
        "linearly with the folding factor (Li & Benjamin, PRX 7, "
        "021050, 2017).",
        "fold_global provides deterministic, reproducible noise "
        "amplification for shallow circuits.",
        "Three conservative scale factors minimize shot overhead "
        "while providing a reliable linear fit.",
    )


def _build_policy_fallback_recipe(
    features: CircuitFeatures,
    profile: ZneProfilePolicy,
) -> MitigationRecipe:
    """Build a policy-backed conservative fallback recipe."""
    return MitigationRecipe(
        technique="zne",
        factory_name=profile.factory,
        scale_factors=profile.scale_factors,
        factory_kwargs=_freeze_kwargs(dict(profile.factory_kwargs)),
        scaling_method=profile.scaling_method,
        rationale=(
            "No specific heuristic rule matched -- applying conservative "
            "LinearFactory as a safe default.",
            f"Circuit: depth={features.depth}, "
            f"multi_qubit_gates={features.multi_qubit_gate_count}, "
            f"noise='{features.noise_category}'.",
            "LinearFactory with fold_global is the most general and "
            "lowest-risk ZNE configuration.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=float(len(profile.scale_factors)),
    )


# ---------------------------------------------------------------------------
# Recipe builders
# ---------------------------------------------------------------------------


def _build_poly_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a PolyFactory(order=2) recipe for deep / high-noise circuits."""
    return MitigationRecipe(
        technique="zne",
        factory_name="PolyFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5, 3.0),
        factory_kwargs=_freeze_kwargs({"order": 2}),
        scaling_method="fold_gates_at_random",
        rationale=(
            f"Circuit depth ({features.depth}) > 50 or noise category "
            f"'{features.noise_category}' indicates strong non-linear noise.",
            "PolyFactory with order=2 captures quadratic noise scaling "
            "better than linear/Richardson for deep circuits.",
            "fold_gates_at_random reduces coherent error accumulation "
            "compared to uniform folding (arXiv:2005.10921).",
            "Five scale factors provide sufficient data points for a "
            "degree-2 polynomial fit (arXiv:2307.05203).",
        ),
        noise_category=features.noise_category,
        estimated_overhead=5.0,
    )


def _build_richardson_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a RichardsonFactory recipe for moderate-depth circuits."""
    return MitigationRecipe(
        technique="zne",
        factory_name="RichardsonFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5),
        factory_kwargs=_freeze_kwargs({}),
        scaling_method="fold_global",
        rationale=(
            f"Circuit depth ({features.depth}) is in the moderate range "
            f"(20-50) with noise category '{features.noise_category}'.",
            "RichardsonFactory uses polynomial interpolation that handles "
            "moderate non-linear noise better than linear extrapolation "
            "(Temme et al., PRL 119, 180509, 2017).",
            "fold_global provides uniform noise amplification suitable "
            "for structured circuits at moderate depth.",
            "Four scale factors balance accuracy vs. shot overhead.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=4.0,
    )


def _build_layerwise_richardson_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a layerwise RichardsonFactory recipe for heterogeneous circuits."""
    return MitigationRecipe(
        technique="zne",
        factory_name="RichardsonFactory",
        scale_factors=(1.0, 1.5, 2.0, 2.5),
        factory_kwargs=_freeze_kwargs({}),
        scaling_method="fold_gates_at_random",
        rationale=(
            f"Circuit depth ({features.depth}) is moderate (15-50) with "
            f"high layer heterogeneity ({features.layer_heterogeneity:.2f} "
            f"> {LAYERWISE_HETEROGENEITY_THRESHOLD}).",
            "Layers have uneven multi-qubit gate density, so "
            "fold_gates_at_random targets the noisiest gates rather than "
            "amplifying noise uniformly (arXiv:2005.10921).",
            "RichardsonFactory provides polynomial interpolation suited "
            "for moderate-depth circuits with non-uniform noise.",
            "Four scale factors balance accuracy vs. shot overhead.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=4.0,
    )


def _build_linear_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a LinearFactory recipe for shallow / low-noise circuits."""
    return MitigationRecipe(
        technique="zne",
        factory_name="LinearFactory",
        scale_factors=(1.0, 1.5, 2.0),
        factory_kwargs=_freeze_kwargs({}),
        scaling_method="fold_global",
        rationale=(
            f"Circuit depth ({features.depth}) < 20 with "
            f"{features.multi_qubit_gate_count} multi-qubit gates -- "
            f"noise category '{features.noise_category}'.",
            "LinearFactory is sufficient when noise scales approximately "
            "linearly with the folding factor (Li & Benjamin, PRX 7, "
            "021050, 2017).",
            "fold_global provides deterministic, reproducible noise "
            "amplification for shallow circuits.",
            "Three conservative scale factors minimize shot overhead "
            "while providing a reliable linear fit.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=3.0,
    )


def _build_fallback_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a conservative fallback recipe when no specific rule matches."""
    return MitigationRecipe(
        technique="zne",
        factory_name="LinearFactory",
        scale_factors=(1.0, 1.5, 2.0),
        factory_kwargs=_freeze_kwargs({}),
        scaling_method="fold_global",
        rationale=(
            "No specific heuristic rule matched -- applying conservative "
            "LinearFactory as a safe default.",
            f"Circuit: depth={features.depth}, "
            f"multi_qubit_gates={features.multi_qubit_gate_count}, "
            f"noise='{features.noise_category}'.",
            "LinearFactory with fold_global is the most general and "
            "lowest-risk ZNE configuration.",
        ),
        noise_category=features.noise_category,
        estimated_overhead=3.0,
    )


# ---------------------------------------------------------------------------
# Default rule set
# ---------------------------------------------------------------------------

#: Ordered list of ``(predicate, builder)`` rules.  First match wins.
#:
#: Priority order:
#: 1. Deep / high-noise           ->  PolyFactory + fold_gates_at_random
#: 2. Moderate + heterogeneous    ->  RichardsonFactory + fold_gates_at_random
#: 3. Moderate depth (uniform)    ->  RichardsonFactory + fold_global
#: 4. Shallow / low-gate          ->  LinearFactory + fold_global
#:
#: Order is load-bearing: ``_is_moderate_heterogeneous`` uses ``15 <= depth
#: <= 50`` which overlaps with ``_is_shallow``'s ``depth < 20`` at depths
#: 15-19, so heterogeneous must be checked before shallow.
#:
#: If none match, :func:`recommend` uses :func:`_build_fallback_recipe`.
DEFAULT_RULES: tuple[Rule, ...] = (
    (_is_deep_or_high_noise, _build_poly_recipe),
    (_is_moderate_heterogeneous, _build_layerwise_richardson_recipe),
    (_is_moderate_depth, _build_richardson_recipe),
    (_is_shallow, _build_linear_recipe),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend(
    features: CircuitFeatures,
    *,
    rules: Sequence[Rule] | None = None,
    technique: str | None = None,
    policy: RecipePolicy | None = None,
) -> MitigationRecipe:
    """Select the optimal error mitigation recipe for a circuit.

    When *technique* is ``None`` (the default), the engine checks
    techniques in priority order: composite (for moderate PEC-eligible
    circuits), PEC (if noise model available, shallow, low overhead), CDR
    (if non-Clifford fraction > threshold, moderate depth), then ZNE rules.
    A conservative LinearFactory fallback is used if no rule matches.

    Parameters
    ----------
    features:
        Circuit analysis output from :func:`~emrg.analyzer.analyze_circuit`.
    rules:
        Optional custom rule list.  Defaults to :data:`DEFAULT_RULES`.
    technique:
        Force a specific technique: ``"pec"``, ``"cdr"``, ``"composite"``,
        or ``"zne"``. When ``None``, the engine auto-selects.
    policy:
        Optional structured policy controlling thresholds, enabled
        techniques, budgets, and generated recipe parameters.

    Returns
    -------
    MitigationRecipe
        Fully populated recommendation ready for code generation.

    Examples
    --------
    >>> from emrg.analyzer import CircuitFeatures
    >>> shallow = CircuitFeatures(
    ...     num_qubits=2, depth=4, gate_counts={"h": 1, "cx": 1},
    ...     total_gate_count=2, multi_qubit_gate_count=1,
    ...     single_qubit_gate_count=1, num_parameters=0,
    ...     has_measurements=True, estimated_noise_factor=0.011,
    ...     noise_category="low",
    ... )
    >>> recipe = recommend(shallow)
    >>> recipe.factory_name
    'LinearFactory'
    """
    # --- validate technique ---------------------------------------------------
    _valid_techniques = {"zne", "pec", "cdr", "composite", None}
    if technique not in _valid_techniques:
        raise ValueError(
            f"Unknown technique {technique!r}. "
            f"Must be 'zne', 'pec', 'cdr', 'composite', or None."
        )
    if rules is not None and policy is not None:
        raise ValueError("recommend() does not accept both rules and policy.")

    # --- technique override --------------------------------------------------
    if technique == "pec":
        return _with_warnings(
            _build_pec_recipe(features, policy),
            _forced_pec_warnings(features, policy),
        )
    if technique == "cdr":
        return _with_warnings(
            _build_cdr_recipe(features, policy),
            _forced_cdr_warnings(features, policy),
        )
    if technique == "composite":
        return _with_warnings(
            _build_composite_recipe(
                features,
                rules,
                policy,
                force_components=True,
            ),
            _forced_composite_warnings(features, rules, policy),
        )
    if technique == "zne":
        if policy is None:
            return _build_zne_recipe(features, rules)
        warnings = ()
        if not policy.techniques.zne.enabled:
            warnings = ("Forced ZNE: ZNE is disabled by policy.",)
        return _with_warnings(
            _build_policy_zne_recipe(features, policy, force=True),
            warnings,
        )

    # --- auto-select: try composite, PEC, then CDR ---------------------------
    if technique is None and _should_use_composite(features, policy):
        return _build_composite_recipe(features, rules, policy)
    if technique is None and _should_use_pec(features, policy):
        return _build_pec_recipe(features, policy)
    if technique is None and _should_use_cdr(features, policy):
        return _build_cdr_recipe(features, policy)

    # --- ZNE rule chain ------------------------------------------------------
    return _build_zne_recipe(features, rules, policy)
