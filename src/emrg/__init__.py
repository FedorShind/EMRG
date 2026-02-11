"""EMRG -- Error Mitigation Recipe Generator.

Automatic quantum error mitigation recipe generator for NISQ circuits.
Analyzes quantum circuits and generates ready-to-run, explained
Mitiq-powered error mitigation code.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emrg._version import __version__
from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.codegen import generate_code
from emrg.heuristics import MitigationRecipe, recommend

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

__all__ = [
    "__version__",
    "GeneratedRecipe",
    "generate_recipe",
    "analyze_circuit",
    "CircuitFeatures",
    "recommend",
    "MitigationRecipe",
    "generate_code",
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
        code: Ready-to-run Python script with Mitiq imports, factory
            setup, executor placeholder, and ``execute_with_zne`` call.
        rationale: Tuple of explanation strings describing why the
            mitigation strategy was chosen, with literature references.
        features: Circuit analysis from :func:`analyze_circuit`.
        recipe: Mitigation strategy from :func:`recommend`.

    ``str(result)`` returns :attr:`code`, so ``print(generate_recipe(qc))``
    outputs the script directly.
    """

    code: str
    rationale: tuple[str, ...]
    features: CircuitFeatures
    recipe: MitigationRecipe

    def __str__(self) -> str:
        """Return the generated code for convenient printing."""
        return self.code


def generate_recipe(
    qc: QuantumCircuit,
    *,
    explain: bool = False,
    circuit_name: str = "circuit",
) -> GeneratedRecipe:
    """Analyze a circuit and generate a complete error mitigation recipe.

    This is the primary one-liner entry point for EMRG. It chains the
    full pipeline: analyze -> recommend -> generate code.

    Parameters
    ----------
    qc:
        A Qiskit ``QuantumCircuit`` to mitigate.
    explain:
        If ``True``, include full rationale and inline comments in the
        generated code.
    circuit_name:
        Variable name for the circuit in generated code
        (default ``"circuit"``).

    Returns
    -------
    GeneratedRecipe
        Object with ``.code``, ``.rationale``, ``.features``, and
        ``.recipe`` attributes.

    Raises
    ------
    TypeError
        If *qc* is not a ``QuantumCircuit``.
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
    features = analyze_circuit(qc)
    recipe = recommend(features)
    code = generate_code(
        recipe,
        features,
        circuit_name=circuit_name,
        explain=explain,
    )
    return GeneratedRecipe(
        code=code,
        rationale=recipe.rationale,
        features=features,
        recipe=recipe,
    )
