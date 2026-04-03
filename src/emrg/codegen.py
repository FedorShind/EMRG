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
        "# Recommendation: "
        + (
            "PEC (Probabilistic Error Cancellation)"
            if recipe.technique == "pec"
            else (
                "CDR (Clifford Data Regression)"
                if recipe.technique == "cdr"
                else f"{recipe.factory_name} + {recipe.scaling_method}"
            )
        ),
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

    if (
        recipe.scaling_method == "fold_gates_at_random"
        and recipe.factory_name == "RichardsonFactory"
    ):
        lines.append(
            "# fold_gates_at_random selected: circuit layers have "
            "uneven noise density."
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
# PEC renderers
# ---------------------------------------------------------------------------


def _render_pec_imports() -> str:
    """Render PEC-specific import statements."""
    return "\n".join(
        [
            "from mitiq.pec import execute_with_pec",
            "from mitiq.pec.representations.depolarizing import (",
            "    represent_operations_in_circuit_with_local_depolarizing_noise,",
            ")",
        ]
    )


def _render_pec_setup(
    recipe: MitigationRecipe,
    circuit_name: str,
    explain: bool,
) -> str:
    """Render PEC noise level, sample count, and representation setup."""
    noise_level = recipe.factory_kwargs.get("noise_level", 0.01)
    num_samples = recipe.factory_kwargs.get("num_samples", 100)

    lines: list[str] = []
    if explain:
        lines.append(
            "# PEC uses quasi-probability representations of noisy gates"
        )
        lines.append(
            "# to probabilistically cancel errors at the cost of extra samples."
        )
        lines.append(
            f"# noise_level={noise_level} matches the expected hardware noise."
        )
        lines.append("")

    lines.extend(
        [
            f"noise_level = {noise_level}",
            f"num_samples = {num_samples}",
            "",
            "# Build quasi-probability representations for the circuit's operations.",
            "# In practice, these should be derived from your noise model.",
            "representations = "
            "represent_operations_in_circuit_with_local_depolarizing_noise(",
            f"    {circuit_name},",
            "    noise_level=noise_level,",
            ")",
        ]
    )
    return "\n".join(lines)


def _render_pec_execution(
    recipe: MitigationRecipe,
    circuit_name: str,
    explain: bool,
) -> str:
    """Render the execute_with_pec call and result print."""
    lines: list[str] = []

    if explain:
        lines.append("# Run probabilistic error cancellation.")
        lines.append(
            f"# Estimated overhead: ~{recipe.estimated_overhead:.1f}x "
            "the base shot count."
        )

    lines.extend(
        [
            "mitigated_value = execute_with_pec(",
            f"    {circuit_name},",
            "    execute,",
            "    representations=representations,",
            "    num_samples=num_samples,",
            ")",
            "",
            'print(f"Mitigated expectation value: {mitigated_value}")',
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CDR renderers
# ---------------------------------------------------------------------------


def _render_cdr_imports() -> str:
    """Render CDR-specific import statements."""
    return "\n".join(
        [
            "from mitiq.cdr import execute_with_cdr",
        ]
    )


def _render_cdr_simulator(explain: bool) -> str:
    """Render the classical simulator function for CDR training circuits."""
    lines: list[str] = []

    if explain:
        lines.append(
            "# CDR needs a classical simulator to evaluate near-Clifford"
        )
        lines.append(
            "# training circuits exactly. Cirq's DensityMatrixSimulator"
        )
        lines.append(
            "# runs these circuits without noise (noiseless reference)."
        )
        lines.append(
            "# Install cirq if needed: pip install emrg[preview]"
        )

    lines.extend(
        [
            "def simulator(circuit):",
            '    """Simulate a near-Clifford circuit classically (noiseless).',
            "",
            "    CDR uses this to get exact expectation values for training",
            "    circuits where non-Clifford gates have been replaced with",
            "    Clifford gates.",
            '    """',
            "    import cirq",
            "    rho = (",
            "        cirq.DensityMatrixSimulator()",
            "        .simulate(circuit)",
            "        .final_density_matrix",
            "    )",
            "    # Compute expectation value from the density matrix.",
            "    # Adjust the observable to match your measurement.",
            "    import numpy as np",
            "    n = int(np.log2(rho.shape[0]))",
            "    z = np.array([[1, 0], [0, -1]], dtype=complex)",
            "    obs = z",
            "    for _ in range(n - 1):",
            "        obs = np.kron(obs, np.eye(2, dtype=complex))",
            "    return float(np.real(np.trace(rho @ obs)))",
        ]
    )

    return "\n".join(lines)


def _render_cdr_setup(
    recipe: MitigationRecipe,
    explain: bool,
) -> str:
    """Render CDR configuration (num_training_circuits, fit_method)."""
    num_training = recipe.factory_kwargs.get("num_training_circuits", 8)

    lines: list[str] = []
    if explain:
        lines.append(
            "# CDR creates near-Clifford training circuits by replacing"
        )
        lines.append(
            "# non-Clifford gates with Clifford substitutes. These can be"
        )
        lines.append(
            "# simulated classically and used to fit a regression model"
        )
        lines.append(
            "# that corrects the noisy results on the original circuit."
        )
        lines.append("")

    lines.append(f"num_training_circuits = {num_training}")

    return "\n".join(lines)


def _render_cdr_execution(
    recipe: MitigationRecipe,
    circuit_name: str,
    explain: bool,
) -> str:
    """Render the execute_with_cdr call and result print."""
    lines: list[str] = []

    if explain:
        lines.append("# Run Clifford Data Regression.")
        lines.append(
            f"# Estimated overhead: ~{recipe.estimated_overhead:.0f}x "
            "the base shot count (training circuits + original)."
        )

    lines.extend(
        [
            "mitigated_value = execute_with_cdr(",
            f"    {circuit_name},",
            "    execute,",
            "    simulator=simulator,",
            "    num_training_circuits=num_training_circuits,",
            ")",
            "",
            'print(f"Mitigated expectation value: {mitigated_value}")',
        ]
    )

    return "\n".join(lines)


def _generate_cdr_code(
    recipe: MitigationRecipe,
    features: CircuitFeatures,
    circuit_name: str,
    explain: bool,
) -> str:
    """Assemble sections for a CDR recipe."""
    sections = [
        _render_header(recipe, features, explain),
        "",
        _render_cdr_imports(),
        "",
        _render_cdr_setup(recipe, explain),
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
            _render_cdr_simulator(explain),
            "",
            _render_cdr_execution(recipe, circuit_name, explain),
            "",  # trailing newline
        ]
    )

    return "\n".join(sections)


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

    if recipe.technique == "pec":
        return _generate_pec_code(recipe, features, circuit_name, explain)
    if recipe.technique == "cdr":
        return _generate_cdr_code(recipe, features, circuit_name, explain)
    return _generate_zne_code(recipe, features, circuit_name, explain)


def _generate_zne_code(
    recipe: MitigationRecipe,
    features: CircuitFeatures,
    circuit_name: str,
    explain: bool,
) -> str:
    """Assemble sections for a ZNE recipe."""
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


def _generate_pec_code(
    recipe: MitigationRecipe,
    features: CircuitFeatures,
    circuit_name: str,
    explain: bool,
) -> str:
    """Assemble sections for a PEC recipe."""
    sections = [
        _render_header(recipe, features, explain),
        "",
        _render_pec_imports(),
        "",
        _render_pec_setup(recipe, circuit_name, explain),
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
            _render_pec_execution(recipe, circuit_name, explain),
            "",  # trailing newline
        ]
    )

    return "\n".join(sections)
