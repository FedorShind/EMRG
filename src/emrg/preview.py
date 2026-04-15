"""Preview Engine -- noisy simulation of EMRG mitigation recipes.

This module provides opt-in simulation so users can verify that EMRG's
recommended mitigation actually reduces noise *before* running on real
hardware.

All simulation dependencies (cirq, numpy) are lazily imported so that
the core EMRG package works without them installed.

Typical usage::

    from qiskit import QuantumCircuit
    from emrg import generate_recipe
    from emrg.preview import run_preview

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    result = generate_recipe(qc)
    preview = run_preview(qc, result.recipe, noise_level=0.01)
    print(preview.error_reduction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mitiq.cdr import execute_with_cdr
from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
from mitiq.pec import execute_with_pec
from mitiq.pec.representations.depolarizing import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
)
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory, PolyFactory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random, fold_global

if TYPE_CHECKING:
    import numpy as np
    from qiskit import QuantumCircuit

    from emrg.analyzer import CircuitFeatures
    from emrg.heuristics import MitigationRecipe

logger = logging.getLogger("emrg.preview")

__all__ = [
    "PreviewResult",
    "run_preview",
    "format_preview",
]

MAX_PREVIEW_QUBITS = 10
PEC_PREVIEW_SAMPLES = 200
CDR_PREVIEW_TRAINING_CIRCUITS = 8

#: Below this, errors are treated as numerically zero when computing the
#: ratio of noisy-to-mitigated error.
_ERROR_TOLERANCE = 1e-10

# Gate-count limits for preview simulation, indexed by qubit count.
# Density-matrix ops scale as O(4^n) per gate, and ZNE folding
# multiplies gate count by the max scale factor.  These limits keep
# preview under ~30 s on typical hardware.  Circuits with n <= 7
# qubits are intentionally unbounded: 4^7 = 16384 state elements is
# cheap enough that any realistic gate count finishes fast.
_GATE_LIMITS: dict[int, int] = {
    8: 60,
    9: 40,
    10: 30,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreviewResult:
    """Output of a preview simulation run.

    When simulation is skipped (e.g. circuit too large), the value
    fields are ``None`` and :attr:`warning` explains why.
    """

    ideal_value: float | None
    noisy_value: float | None
    mitigated_value: float | None
    noisy_error: float | None
    mitigated_error: float | None
    error_reduction: float | None
    technique: str
    noise_level: float
    observable: str
    num_qubits: int
    warning: str | None = None


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _check_cirq_available() -> None:
    """Raise ImportError with install hint if cirq is not available."""
    try:
        import cirq  # noqa: F401
    except ImportError:
        raise ImportError(
            "Preview requires cirq. Install with: pip install emrg[preview]"
        ) from None


# ---------------------------------------------------------------------------
# Observable helpers
# ---------------------------------------------------------------------------


def _z_on_qubit(qubit_index: int, n_qubits: int) -> np.ndarray:
    """Build Z on a specific qubit: I (x) ... (x) Z (x) ... (x) I."""
    import numpy as np_

    z = np_.array([[1, 0], [0, -1]], dtype=complex)
    eye = np_.eye(2, dtype=complex)

    obs = z if qubit_index == 0 else eye
    for i in range(1, n_qubits):
        obs = np_.kron(obs, z if i == qubit_index else eye)
    return obs


def _zz_observable(n_qubits: int) -> np.ndarray:
    """Build Z_0 Z_1 = Z (x) Z (x) I (x) ..."""
    import numpy as np_

    z = np_.array([[1, 0], [0, -1]], dtype=complex)
    obs = np_.kron(z, z)
    for _ in range(n_qubits - 2):
        obs = np_.kron(obs, np_.eye(2, dtype=complex))
    return obs


def _classify_observable(observable: str) -> tuple[str, int | None]:
    """Split an observable string into ``(kind, qubit_index)``.

    Returns ``("zz", None)`` for ``"ZZ"`` or ``("z", i)`` for ``"Zi"``.
    Raises ``ValueError`` for any other input.  This is the single source
    of truth for the observable grammar used by parsing and labelling.
    """
    obs_upper = observable.upper().strip()

    if obs_upper == "ZZ":
        return "zz", None

    if obs_upper.startswith("Z") and len(obs_upper) >= 2:
        try:
            return "z", int(obs_upper[1:])
        except ValueError:
            pass

    raise ValueError(
        f"Invalid observable: {observable!r}. "
        f"Expected 'Z0', 'Z1', ..., or 'ZZ'."
    )


def _parse_observable(observable: str, n_qubits: int) -> np.ndarray:
    """Parse an observable string into a numpy matrix.

    Accepted formats:
    - ``"Z0"``, ``"Z1"``, ... ``"Z9"`` -- single-qubit Z on qubit N
    - ``"ZZ"`` -- Z_0 Z_1
    """
    kind, qubit_index = _classify_observable(observable)

    if kind == "zz":
        if n_qubits < 2:
            raise ValueError("ZZ observable requires at least 2 qubits.")
        return _zz_observable(n_qubits)

    # kind == "z"; _classify_observable guarantees qubit_index is an int.
    if qubit_index is None or qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError(
            f"Qubit index {qubit_index} out of range for "
            f"{n_qubits}-qubit circuit."
        )
    return _z_on_qubit(qubit_index, n_qubits)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _make_noisy_cirq_circuit(circuit, noise_level: float):
    """Apply per-gate depolarizing noise to a Cirq circuit."""
    import cirq

    if noise_level <= 0:
        return circuit
    noisy = cirq.Circuit()
    for moment in circuit.moments:
        noisy.append(moment)
        for op in moment.operations:
            nq = len(op.qubits)
            noisy.append(
                cirq.depolarize(p=noise_level, n_qubits=nq).on(*op.qubits)
            )
    return noisy


def _compute_expectation(rho, observable) -> float:
    """Tr(rho @ observable)."""
    import numpy as np_

    return float(np_.real(np_.trace(rho @ observable)))


def _make_executor(noise_level: float, observable_matrix, n_qubits: int):
    """Return a Cirq DensityMatrixSimulator executor for Mitiq."""
    import cirq

    # NOTE: no return type annotation here -- Mitiq inspects annotations
    # at runtime, and `from __future__ import annotations` turns `-> float`
    # into the string 'float', which Mitiq cannot resolve.
    def executor(circuit):
        sim_circuit = _make_noisy_cirq_circuit(circuit, noise_level)
        rho = (
            cirq.DensityMatrixSimulator()
            .simulate(sim_circuit)
            .final_density_matrix
        )
        return _compute_expectation(rho, observable_matrix)

    return executor


def _compute_error_reduction(noisy_err: float, mitigated_err: float) -> float:
    """Return the ratio ``noisy_err / mitigated_err``, handling zero edges."""
    if mitigated_err < _ERROR_TOLERANCE:
        return float("inf") if noisy_err > _ERROR_TOLERANCE else 1.0
    if noisy_err < _ERROR_TOLERANCE:
        return 1.0
    return noisy_err / mitigated_err


# ---------------------------------------------------------------------------
# ZNE execution
# ---------------------------------------------------------------------------


def _run_zne(cirq_circuit, noisy_executor, recipe: MitigationRecipe) -> float:
    """Execute ZNE using the actual recipe parameters."""
    factory_cls = {
        "LinearFactory": LinearFactory,
        "RichardsonFactory": RichardsonFactory,
        "PolyFactory": PolyFactory,
    }[recipe.factory_name]

    extra_kwargs = dict(recipe.factory_kwargs) if recipe.factory_kwargs else {}
    factory = factory_cls(
        scale_factors=list(recipe.scale_factors), **extra_kwargs
    )

    scale_fn = {
        "fold_global": fold_global,
        "fold_gates_at_random": fold_gates_at_random,
    }[recipe.scaling_method]

    return execute_with_zne(
        cirq_circuit, noisy_executor, factory=factory, scale_noise=scale_fn,
    )


# ---------------------------------------------------------------------------
# PEC execution
# ---------------------------------------------------------------------------


def _run_pec(
    cirq_circuit, noisy_executor, noise_level: float
) -> float:
    """Execute PEC with depolarizing representations."""
    representations = (
        represent_operations_in_circuit_with_local_depolarizing_noise(
            cirq_circuit, noise_level=noise_level,
        )
    )

    return execute_with_pec(
        cirq_circuit,
        noisy_executor,
        representations=representations,
        num_samples=PEC_PREVIEW_SAMPLES,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# CDR execution
# ---------------------------------------------------------------------------


def _run_cdr(
    cirq_circuit, noisy_executor, observable_matrix, n_qubits: int, recipe,
) -> float:
    """Execute CDR with a noiseless simulator for training circuits."""
    sim_executor = _make_executor(0.0, observable_matrix, n_qubits)

    num_training = recipe.factory_kwargs.get(
        "num_training_circuits", CDR_PREVIEW_TRAINING_CIRCUITS
    )

    return execute_with_cdr(
        cirq_circuit,
        noisy_executor,
        simulator=sim_executor,
        num_training_circuits=num_training,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_preview(
    qc: QuantumCircuit,
    recipe: MitigationRecipe,
    *,
    noise_level: float = 0.01,
    observable: str = "Z0",
) -> PreviewResult:
    """Run a noisy simulation to preview the effect of a mitigation recipe.

    Parameters
    ----------
    qc:
        A Qiskit ``QuantumCircuit``.
    recipe:
        The ``MitigationRecipe`` returned by ``recommend()``.
    noise_level:
        Per-gate depolarizing noise probability (default 0.01).
    observable:
        Observable to measure: ``"Z0"``, ``"Z1"``, ..., or ``"ZZ"``
        (default ``"Z0"``).

    Returns
    -------
    PreviewResult
        Simulation results with ideal, noisy, and mitigated values.
        If simulation is skipped or fails, value fields are ``None``
        and ``warning`` explains why.
    """
    _check_cirq_available()

    n_qubits = qc.num_qubits
    technique = recipe.technique.upper()

    # Guard: density matrix simulation cost is roughly
    # O(4^n * total_gates * max_scale_factor).  Cap by qubit count
    # and by an estimated gate-cost budget for wide circuits.
    if n_qubits > MAX_PREVIEW_QUBITS:
        return PreviewResult(
            ideal_value=None,
            noisy_value=None,
            mitigated_value=None,
            noisy_error=None,
            mitigated_error=None,
            error_reduction=None,
            technique=technique,
            noise_level=noise_level,
            observable=observable,
            num_qubits=n_qubits,
            warning=(
                f"Preview skipped: density matrix simulation is impractical "
                f"above {MAX_PREVIEW_QUBITS} qubits ({n_qubits} qubits in "
                f"circuit). Test on a smaller circuit."
            ),
        )

    gate_limit = _GATE_LIMITS.get(n_qubits)
    if gate_limit is not None:
        total_gates = sum(
            count for name, count in qc.count_ops().items()
            if name not in ("measure", "barrier", "delay", "reset")
        )
        if total_gates > gate_limit:
            return PreviewResult(
                ideal_value=None,
                noisy_value=None,
                mitigated_value=None,
                noisy_error=None,
                mitigated_error=None,
                error_reduction=None,
                technique=technique,
                noise_level=noise_level,
                observable=observable,
                num_qubits=n_qubits,
                warning=(
                    f"Preview skipped: {total_gates} gates on "
                    f"{n_qubits} qubits exceeds simulation budget "
                    f"(limit: {gate_limit} gates). "
                    f"Density matrix simulation with ZNE folding "
                    f"would be impractically slow. "
                    f"Test on a smaller or shallower circuit."
                ),
            )

    try:
        # Strip measurements -- Cirq density matrix sim doesn't need them.
        gate_only = qc.copy()
        gate_only.remove_final_measurements()
        cirq_circuit = from_qiskit(gate_only)

        # Use Cirq circuit's actual qubit count for the observable matrix.
        # Cirq may drop idle qubits during conversion, so this can differ
        # from qc.num_qubits.
        cirq_n_qubits = len(sorted(cirq_circuit.all_qubits()))
        obs_matrix = _parse_observable(observable, cirq_n_qubits)

        # Ideal (noiseless) execution.
        ideal_exec = _make_executor(0.0, obs_matrix, cirq_n_qubits)
        ideal_value = ideal_exec(cirq_circuit)

        # Noisy execution.
        noisy_exec = _make_executor(noise_level, obs_matrix, cirq_n_qubits)
        noisy_value = noisy_exec(cirq_circuit)

        # Mitigated execution using the actual recipe.
        if recipe.technique == "pec":
            mitigated_value = _run_pec(
                cirq_circuit, noisy_exec, noise_level
            )
            warning = (
                f"PEC results are approximate ({PEC_PREVIEW_SAMPLES} samples). "
                f"Variance decreases with more samples."
            )
        elif recipe.technique == "cdr":
            mitigated_value = _run_cdr(
                cirq_circuit, noisy_exec, obs_matrix, cirq_n_qubits, recipe,
            )
            num_tc = recipe.factory_kwargs.get(
                "num_training_circuits", CDR_PREVIEW_TRAINING_CIRCUITS
            )
            warning = (
                f"CDR results use {num_tc} training circuits. "
                f"Accuracy improves with more training circuits."
            )
        else:
            mitigated_value = _run_zne(cirq_circuit, noisy_exec, recipe)
            warning = None

        noisy_err = abs(ideal_value - noisy_value)
        mitigated_err = abs(ideal_value - mitigated_value)
        reduction = _compute_error_reduction(noisy_err, mitigated_err)

        return PreviewResult(
            ideal_value=ideal_value,
            noisy_value=noisy_value,
            mitigated_value=mitigated_value,
            noisy_error=noisy_err,
            mitigated_error=mitigated_err,
            error_reduction=reduction,
            technique=technique,
            noise_level=noise_level,
            observable=observable,
            num_qubits=n_qubits,
            warning=warning,
        )

    except (ValueError, RuntimeError, ImportError, NotImplementedError) as exc:
        logger.debug("Preview simulation failed", exc_info=True)
        return PreviewResult(
            ideal_value=None,
            noisy_value=None,
            mitigated_value=None,
            noisy_error=None,
            mitigated_error=None,
            error_reduction=None,
            technique=technique,
            noise_level=noise_level,
            observable=observable,
            num_qubits=n_qubits,
            warning=f"Simulation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_observable_label(observable: str) -> str:
    """Turn 'Z0' into '<Z> on qubit 0', 'ZZ' into '<ZZ> on qubits 0,1'.

    Falls back to the raw string for unparseable inputs so the preview
    box always renders something.
    """
    try:
        kind, qubit_index = _classify_observable(observable)
    except ValueError:
        return observable
    if kind == "zz":
        return "<ZZ> on qubits 0,1"
    return f"<Z> on qubit {qubit_index}"


def format_preview(
    result: PreviewResult,
    features: CircuitFeatures,
) -> str:
    """Format a PreviewResult as a box-drawing table for terminal output.

    Parameters
    ----------
    result:
        The preview simulation result.
    features:
        Circuit features from ``analyze_circuit()``, used for the
        header line.

    Returns
    -------
    str
        Formatted multi-line string ready for ``click.echo()`` or
        ``print()``.
    """
    width = 49
    hline = "\u2500" * width

    top = f"\u250c{hline}\u2510"
    mid = f"\u251c{hline}\u2524"
    bot = f"\u2514{hline}\u2518"

    def row(text: str) -> str:
        return f"\u2502  {text:<{width - 3}}\u2502"

    obs_label = _format_observable_label(result.observable)

    # Technique description.
    tech_desc = result.technique
    if result.technique == "PEC":
        tech_desc = f"PEC ({PEC_PREVIEW_SAMPLES} samples)"
    elif result.technique == "CDR":
        tech_desc = "CDR (Clifford Data Regression)"

    lines = [
        top,
        row("EMRG Preview -- Simulation Comparison"),
        mid,
        row(f"Circuit:    {features.num_qubits} qubits, depth {features.depth}"),
        row(f"Noise:      depolarizing p={result.noise_level}"),
        row(f"Observable: {obs_label}"),
        row(f"Technique:  {tech_desc}"),
        mid,
    ]

    if result.ideal_value is not None:
        lines.append(row(f"Ideal:      {result.ideal_value:+.4f}"))
        lines.append(
            row(
                f"Noisy:      {result.noisy_value:+.4f}"
                f"  (error: {result.noisy_error:.4f})"
            )
        )
        lines.append(
            row(
                f"Mitigated:  {result.mitigated_value:+.4f}"
                f"  (error: {result.mitigated_error:.4f})"
            )
        )
        lines.append(row(""))

        if result.error_reduction == float("inf"):
            reduction_str = "inf (mitigated error ~ 0)"
        else:
            reduction_str = f"{result.error_reduction:.1f}x"
        lines.append(row(f"Error reduction: {reduction_str}"))
    else:
        lines.append(row("Simulation skipped."))
        lines.append(row(""))

    lines.append(bot)

    if result.warning:
        lines.append("")
        lines.append(f"  Note: {result.warning}")

    return "\n".join(lines)
