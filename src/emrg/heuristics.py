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
  Phase 3 will load rules from a config file.
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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from emrg.analyzer import CircuitFeatures

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
    """

    technique: str
    factory_name: str
    scale_factors: tuple[float, ...]
    factory_kwargs: Mapping[str, Any] = field(default_factory=dict)
    scaling_method: str = "fold_global"
    rationale: tuple[str, ...] = ()
    noise_category: str = "low"
    estimated_overhead: float = 1.0


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


def _should_use_pec(features: CircuitFeatures) -> bool:
    """Return ``True`` when PEC is viable for the given circuit.

    PEC is recommended when **all** of:
    * depth <= ``PEC_MAX_DEPTH``
    * ``noise_model_available`` is ``True``
    * ``pec_overhead_estimate`` < ``PEC_MAX_OVERHEAD``
    """
    return (
        features.depth <= PEC_MAX_DEPTH
        and features.noise_model_available
        and features.pec_overhead_estimate < PEC_MAX_OVERHEAD
    )


def _compute_pec_samples(overhead: float) -> int:
    """Derive the number of PEC samples from the overhead estimate.

    The result is clamped between ``PEC_MIN_SAMPLES`` and ``PEC_MAX_SAMPLES``.
    """
    return max(PEC_MIN_SAMPLES, min(PEC_MAX_SAMPLES, int(overhead * 2)))


def _build_pec_recipe(features: CircuitFeatures) -> MitigationRecipe:
    """Build a PEC recipe for shallow circuits with an available noise model."""
    num_samples = _compute_pec_samples(features.pec_overhead_estimate)
    return MitigationRecipe(
        technique="pec",
        factory_name="",
        scale_factors=(),
        factory_kwargs=_freeze_kwargs({
            "num_samples": num_samples,
            "noise_level": PEC_DEFAULT_NOISE_LEVEL,
        }),
        scaling_method="",
        rationale=(
            f"Circuit depth ({features.depth}) <= {PEC_MAX_DEPTH} with "
            f"noise model available and PEC overhead "
            f"({features.pec_overhead_estimate:.1f}) < {PEC_MAX_OVERHEAD:.0f}.",
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


# ---------------------------------------------------------------------------
# ZNE predicates
# ---------------------------------------------------------------------------


def _is_deep_or_high_noise(features: CircuitFeatures) -> bool:
    """Match deep circuits (depth > DEPTH_DEEP_THRESHOLD) or high estimated noise."""
    return features.depth > DEPTH_DEEP_THRESHOLD or features.noise_category == "high"


def _is_moderate_depth(features: CircuitFeatures) -> bool:
    """Match moderate-depth circuits (20 <= depth <= 50)."""
    return DEPTH_MODERATE_THRESHOLD <= features.depth <= DEPTH_DEEP_THRESHOLD


def _is_shallow(features: CircuitFeatures) -> bool:
    """Match shallow, low-gate-count circuits (depth < 20, multi-qubit < 50)."""
    return (
        features.depth < DEPTH_MODERATE_THRESHOLD
        and features.multi_qubit_gate_count < MULTI_QUBIT_GATE_SHALLOW_MAX
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
#: 1. Deep / high-noise  ->  PolyFactory + fold_gates_at_random
#: 2. Moderate depth     ->  RichardsonFactory + fold_global
#: 3. Shallow / low-gate ->  LinearFactory + fold_global
#:
#: If none match, :func:`recommend` uses :func:`_build_fallback_recipe`.
DEFAULT_RULES: tuple[Rule, ...] = (
    (_is_deep_or_high_noise, _build_poly_recipe),
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
) -> MitigationRecipe:
    """Select the optimal error mitigation recipe for a circuit.

    When *technique* is ``None`` (the default), the engine first checks
    whether PEC is viable (shallow circuit, noise model available, low
    overhead).  If so, a PEC recipe is returned.  Otherwise, *rules* are
    evaluated in order and the first matching predicate wins.  A
    conservative LinearFactory fallback is used if no rule matches.

    Parameters
    ----------
    features:
        Circuit analysis output from :func:`~emrg.analyzer.analyze_circuit`.
    rules:
        Optional custom rule list.  Defaults to :data:`DEFAULT_RULES`.
    technique:
        Force a specific technique: ``"pec"`` or ``"zne"``.  When
        ``None``, the engine auto-selects.

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
    # --- technique override --------------------------------------------------
    if technique == "pec":
        return _build_pec_recipe(features)

    # --- auto-select: try PEC first ------------------------------------------
    if technique is None and _should_use_pec(features):
        return _build_pec_recipe(features)

    # --- ZNE rule chain ------------------------------------------------------
    if rules is None:
        active_rules = DEFAULT_RULES
    else:
        active_rules = tuple(rules)

    for predicate, builder in active_rules:
        if predicate(features):
            return builder(features)

    return _build_fallback_recipe(features)
