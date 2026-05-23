"""EMRG -- Error Mitigation Recipe Generator.

Automatic quantum error mitigation recipe generator for NISQ circuits.
Analyzes quantum circuits and generates explained Mitiq-powered code
that is ready to connect to a simulator or hardware executor.

Quick start::

    from qiskit import QuantumCircuit
    from emrg import generate_recipe

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    result = generate_recipe(qc)
    print(result.code)        # Ready-to-run Python script
    print(result.rationale)   # Why these parameters were chosen
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from emrg._version import __version__
from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.codegen import generate_code
from emrg.frontends import Frontend, detect_frontend
from emrg.heuristics import MitigationRecipe, recommend
from emrg.policy import (
    DEFAULT_POLICY,
    RecipePolicy,
    load_policy,
    policy_to_dict,
    save_policy,
)
from emrg.preview import PreviewResult, format_preview, run_preview

__all__ = [
    "__version__",
    "GeneratedRecipe",
    "generate_recipe",
    "Frontend",
    "detect_frontend",
    "analyze_circuit",
    "CircuitFeatures",
    "recommend",
    "MitigationRecipe",
    "RecipePolicy",
    "DEFAULT_POLICY",
    "load_policy",
    "policy_to_dict",
    "save_policy",
    "generate_code",
    "run_preview",
    "PreviewResult",
]


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedRecipe:
    """Complete output of the EMRG pipeline.

    Bundles the generated code, human-readable rationale, and all
    intermediate analysis results into a single object.

    Attributes:
        code: Python script with Mitiq imports, mitigation setup,
            backend executor adapter, and execution call.
        rationale: Tuple of explanation strings describing why the
            mitigation strategy was chosen, with literature references.
        features: Circuit analysis from :func:`analyze_circuit`.
        recipe: Mitigation strategy from :func:`recommend`.

    ``str(result)`` returns :attr:`code`, so ``print(generate_recipe(qc))``
    outputs the script directly.  When :attr:`preview` is populated,
    ``str()`` appends the formatted preview comparison.
    """

    code: str
    rationale: tuple[str, ...]
    features: CircuitFeatures
    recipe: MitigationRecipe
    preview: PreviewResult | None = field(default=None)

    def __str__(self) -> str:
        """Return the generated code, with preview output if present."""
        if self.preview is None:
            return self.code
        return self.code + "\n" + format_preview(self.preview, self.features)


def generate_recipe(
    circuit: object,
    *,
    frontend: str | Frontend | None = None,
    explain: bool = False,
    circuit_name: str = "circuit",
    technique: str | None = None,
    noise_model_available: bool = False,
    preview: bool = False,
    noise_level: float = 0.01,
    observable: str = "Z0",
    policy: RecipePolicy | str | Path | None = None,
) -> GeneratedRecipe:
    """Analyze a circuit and generate a complete error mitigation recipe.

    This is the primary one-liner entry point for EMRG. It chains the
    full pipeline: analyze -> recommend -> generate code.

    Parameters
    ----------
    circuit:
        A Qiskit ``QuantumCircuit``, Cirq ``Circuit``, or supported optional
        frontend object to mitigate.
    frontend:
        Optional explicit frontend. When omitted, EMRG auto-detects Qiskit,
        Cirq, then installed optional frontend objects.
    explain:
        If ``True``, include full rationale and inline comments in the
        generated code.
    circuit_name:
        Variable name for the circuit in generated code
        (default ``"circuit"``).
    technique:
        Force a specific mitigation technique: ``"zne"``, ``"pec"``,
        ``"cdr"``, or ``"composite"``.  When ``None`` (default), the
        engine auto-selects based on circuit characteristics.
    noise_model_available:
        Whether a noise model is available for PEC (default ``False``).
        Set to ``True`` to enable PEC consideration.
    preview:
        If ``True``, run a noisy simulation to preview the mitigation
        effect.  Requires ``cirq`` (install with ``pip install
        emrg[preview]``).  Default ``False``.
    noise_level:
        Per-gate depolarizing noise probability for preview simulation
        (default 0.01).  Only used when *preview* is ``True``.
    observable:
        Observable to measure in preview: ``"Z0"``, ``"Z1"``, ..., or
        ``"ZZ"`` (default ``"Z0"``).  Only used when *preview* is
        ``True``.
    policy:
        Optional :class:`RecipePolicy` or path to a JSON/YAML policy file.
        When ``None``, EMRG uses the built-in default policy.

    Returns
    -------
    GeneratedRecipe
        Object with ``.code``, ``.rationale``, ``.features``,
        ``.recipe``, and ``.preview`` attributes.  When *preview* is
        ``False``, ``.preview`` is ``None``.

    Raises
    ------
    TypeError
        If *circuit* is not a supported circuit object, or if an explicit
        frontend does not match the object.
    ValueError
        If the circuit has zero gate operations.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2, 2)
    >>> _ = qc.h(0); _ = qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    >>> result = generate_recipe(qc)
    >>> "LinearFactory" in result.code
    True
    >>> len(result.rationale) > 0
    True
    """
    active_policy = load_policy(policy) if isinstance(policy, (str, Path)) else policy
    features = analyze_circuit(
        circuit,
        frontend=frontend,
        noise_model_available=noise_model_available,
    )
    recipe = recommend(features, technique=technique, policy=active_policy)
    code = generate_code(
        recipe,
        features,
        circuit_name=circuit_name,
        explain=explain,
    )

    preview_result = None
    if preview:
        from emrg.preview import run_preview

        preview_result = run_preview(
            circuit, recipe, noise_level=noise_level, observable=observable
        )

    return GeneratedRecipe(
        code=code,
        rationale=recipe.rationale,
        features=features,
        recipe=recipe,
        preview=preview_result,
    )
