"""Code Generator -- produce runnable Mitiq code from a mitigation recipe.

This module is the user-facing output stage of the EMRG pipeline. It takes
a :class:`~emrg.heuristics.MitigationRecipe` and
:class:`~emrg.analyzer.CircuitFeatures`, and renders a complete Python
script that the user can copy-paste, adapt the executor, and run.

Two verbosity levels are supported:

* **Normal** (default): compact header with 1-line recommendation, clean code.
* **Explain** (``explain=True``): full rationale with literature references,
  plus inline comments on every code section.

Typical usage::

    from emrg.analyzer import analyze_circuit
    from emrg.heuristics import recommend
    from emrg.codegen import generate_code

    features = analyze_circuit(qc)
    recipe = recommend(features)
    code = generate_code(recipe, features, explain=True)
    print(code)
"""

from __future__ import annotations

from emrg._version import __version__
from emrg.analyzer import CircuitFeatures
from emrg.heuristics import MitigationRecipe

__all__ = ["generate_code"]

# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_header(
    recipe: MitigationRecipe,
    features: CircuitFeatures,
    explain: bool,
) -> str:
    """Render the top-of-file comment block."""
    lines: list[str] = [
        "# =============================================================",
        f"# EMRG v{__version__} -- Error Mitigation Recipe",
        f"# Circuit: {features.num_qubits} qubits, depth {features.depth}, "
        f"{features.multi_qubit_gate_count} multi-qubit gates"
        + (
            f", {features.num_parameters} parameters"
            if features.num_parameters > 0
            else ""
        ),
        f"# Noise estimate: {features.estimated_noise_factor} "
        f"({features.noise_category})",
        "# =============================================================",
        "#",
        f"# Recommendation: {recipe.factory_name} + {recipe.scaling_method}",
    ]

    if explain and recipe.rationale:
        lines.append("#")
        lines.append("# Rationale:")
        for rationale_line in recipe.rationale:
            lines.append(f"#   - {rationale_line}")

    lines.append("#")
    lines.append("# =============================================================")
    return "\n".join(lines)


def _render_imports(recipe: MitigationRecipe) -> str:
    """Render import statements tailored to the recipe."""
    lines: list[str] = [
        "from mitiq.zne import execute_with_zne",
        f"from mitiq.zne.inference import {recipe.factory_name}",
        f"from mitiq.zne.scaling import {recipe.scaling_method}",
    ]
    return "\n".join(lines)


def _render_factory(recipe: MitigationRecipe, explain: bool) -> str:
    """Render the factory construction statement."""
    lines: list[str] = []

    if explain:
        lines.append(f"# {recipe.factory_name} selected by EMRG heuristics.")
        if recipe.factory_kwargs:
            kwargs_desc = ", ".join(
                f"{k}={v}" for k, v in recipe.factory_kwargs.items()
            )
            lines.append(f"# Extra parameters: {kwargs_desc}")

    # Build the constructor call
    scale_list = list(recipe.scale_factors)
    kwargs_parts: list[str] = [f"scale_factors={scale_list}"]
    for key, value in recipe.factory_kwargs.items():
        kwargs_parts.append(f"{key}={value!r}")

    constructor = f"{recipe.factory_name}({', '.join(kwargs_parts)})"
    lines.append(f"factory = {constructor}")

    return "\n".join(lines)


def _render_parameter_warning(features: CircuitFeatures) -> str | None:
    """Render a warning comment block when the circuit has unbound parameters.

    Returns ``None`` if the circuit has no parameters (caller should skip).
    """
    if features.num_parameters == 0:
        return None

    return "\n".join(
        [
            "# " + "!" * 61,
            f"# WARNING: Circuit has {features.num_parameters} unbound parameter(s).",
            "# You must bind all parameters before execution, e.g.:",
            "#",
            "#     from numpy import pi",
            "#     params = {p: pi / 4 for p in circuit.parameters}",
            "#     bound_circuit = circuit.assign_parameters(params)",
            "#",
            "# Mitiq cannot execute a circuit with unbound parameters.",
            "# " + "!" * 61,
        ]
    )


def _render_executor(explain: bool) -> str:
    """Render the executor placeholder with a commented-out Aer example."""
    lines: list[str] = []

    if explain:
        lines.append("# The executor is the bridge between Mitiq and your backend.")
        lines.append(
            "# It must accept a quantum circuit and return a float (expectation value)."
        )

    lines.extend(
        [
            "def execute(circuit):",
            '    """Execute a circuit and return an expectation value (float).',
            "",
            "    This is a placeholder -- replace with your actual backend.",
            "    The function must accept a quantum circuit and return a float.",
            '    """',
            "    # Example using Qiskit Aer (requires: pip install qiskit-aer):",
            "    # from qiskit_aer import AerSimulator",
            "    # from qiskit import transpile",
            '    # backend = AerSimulator(method="density_matrix")',
            "    # transpiled = transpile(circuit, backend)",
            "    # result = backend.run(transpiled, shots=8192).result()",
            "    # counts = result.get_counts()",
            "    # # Compute expectation value from counts...",
            "    # return expectation_value",
            '    raise NotImplementedError("Replace this with your executor.")',
        ]
    )

    return "\n".join(lines)


def _render_execution(
    recipe: MitigationRecipe,
    circuit_name: str,
    explain: bool,
) -> str:
    """Render the execute_with_zne call and result print."""
    lines: list[str] = []

    if explain:
        lines.append("# Run zero-noise extrapolation with the configured factory.")
        lines.append(
            f"# Estimated overhead: ~{recipe.estimated_overhead:.0f}x "
            "the base shot count."
        )

    lines.extend(
        [
            "mitigated_value = execute_with_zne(",
            f"    {circuit_name},",
            "    execute,",
            "    factory=factory,",
            f"    scale_noise={recipe.scaling_method},",
            ")",
            "",
            'print(f"Mitigated expectation value: {mitigated_value}")',
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_code(
    recipe: MitigationRecipe,
    features: CircuitFeatures,
    *,
    circuit_name: str = "circuit",
    explain: bool = False,
) -> str:
    """Generate runnable Python code from a mitigation recipe.

    Parameters
    ----------
    recipe:
        The mitigation strategy from :func:`~emrg.heuristics.recommend`.
    features:
        Circuit analysis from :func:`~emrg.analyzer.analyze_circuit`.
    circuit_name:
        Variable name for the quantum circuit in generated code
        (default ``"circuit"``).
    explain:
        If ``True``, include full rationale and inline comments.

    Returns
    -------
    str
        Complete Python script as a string.

    Examples
    --------
    >>> from emrg.analyzer import CircuitFeatures
    >>> from emrg.heuristics import MitigationRecipe
    >>> recipe = MitigationRecipe(
    ...     technique="zne", factory_name="LinearFactory",
    ...     scale_factors=(1.0, 1.5, 2.0), scaling_method="fold_global",
    ...     rationale=("Low depth.",), noise_category="low",
    ...     estimated_overhead=3.0,
    ... )
    >>> features = CircuitFeatures(
    ...     num_qubits=2, depth=4, gate_counts={"h": 1, "cx": 1},
    ...     total_gate_count=2, multi_qubit_gate_count=1,
    ...     single_qubit_gate_count=1, has_measurements=True,
    ...     estimated_noise_factor=0.011, noise_category="low",
    ... )
    >>> code = generate_code(recipe, features)
    >>> "LinearFactory" in code
    True

    Raises
    ------
    ValueError
        If *circuit_name* is not a valid Python identifier.
    """
    if not circuit_name.isidentifier():
        raise ValueError(
            f"circuit_name must be a valid Python identifier, got {circuit_name!r}."
        )

    sections = [
        _render_header(recipe, features, explain),
        "",
        _render_imports(recipe),
        "",
        _render_factory(recipe, explain),
    ]

    param_warning = _render_parameter_warning(features)
    if param_warning is not None:
        sections.append("")
        sections.append(param_warning)

    sections.extend(
        [
            "",
            _render_executor(explain),
            "",
            _render_execution(recipe, circuit_name, explain),
            "",  # trailing newline
        ]
    )

    return "\n".join(sections)
