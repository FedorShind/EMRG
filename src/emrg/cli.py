"""Command-line interface for EMRG.

Provides two commands:

* ``emrg generate`` -- full pipeline: QASM -> mitigation code.
* ``emrg analyze``  -- inspection: QASM -> circuit features.

Entry point is :func:`main`, wired via ``pyproject.toml``
(``emrg = "emrg.cli:main"``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from emrg._version import __version__
from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.codegen import generate_code
from emrg.heuristics import recommend

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

__all__ = ["main"]

# ---------------------------------------------------------------------------
# QASM loading
# ---------------------------------------------------------------------------


def _load_circuit(qasm_source: str) -> QuantumCircuit:
    """Load a QuantumCircuit from a QASM file path or stdin (``"-"``).

    Auto-detects QASM 2.0 vs 3.0 from the ``OPENQASM`` header.

    Raises:
        click.ClickException: On file-not-found, parse errors, or
            missing QASM 3 importer.
    """
    from qiskit import qasm2

    # --- read source text ---------------------------------------------------
    if qasm_source == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise click.ClickException("No QASM input received on stdin.")
    else:
        path = Path(qasm_source)
        if not path.exists():
            raise click.ClickException(f"File not found: {qasm_source}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(f"Cannot read file: {exc}") from exc

    # --- auto-detect version ------------------------------------------------
    stripped = text.strip()
    if stripped.startswith("OPENQASM 3"):
        try:
            from qiskit import qasm3
        except ImportError as exc:
            raise click.ClickException(
                "QASM 3.0 detected but the 'qiskit_qasm3_import' package "
                "is not installed. Install it with: "
                "pip install qiskit_qasm3_import"
            ) from exc
        try:
            return qasm3.loads(text)
        except Exception as exc:
            raise click.ClickException(f"Failed to parse QASM 3.0: {exc}") from exc

    # Default: QASM 2.0
    try:
        return qasm2.loads(text)
    except Exception as exc:
        raise click.ClickException(f"Failed to parse QASM: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature formatting
# ---------------------------------------------------------------------------


def _format_features_table(features: CircuitFeatures) -> str:
    """Render CircuitFeatures as a human-readable table."""
    gate_str = ", ".join(
        f"{name}: {count}" for name, count in sorted(features.gate_counts.items())
    )
    lines = [
        f"  Qubits:              {features.num_qubits}",
        f"  Depth:               {features.depth}",
        f"  Total gates:         {features.total_gate_count}",
        f"  Multi-qubit gates:   {features.multi_qubit_gate_count}",
        f"  Single-qubit gates:  {features.single_qubit_gate_count}",
        f"  Gate breakdown:      {gate_str}",
        f"  Parameters:          {features.num_parameters}",
        f"  Has measurements:    {features.has_measurements}",
        f"  Noise estimate:      {features.estimated_noise_factor}",
        f"  Noise category:      {features.noise_category}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="emrg")
@click.pass_context
def main(ctx: click.Context) -> None:
    """EMRG -- Error Mitigation Recipe Generator.

    Analyze quantum circuits and generate ready-to-run Mitiq
    error mitigation code.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("qasm_file")
@click.option(
    "--explain",
    is_flag=True,
    help="Include full rationale and inline comments.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    help="Write to file instead of stdout.",
)
@click.option(
    "--circuit-name",
    default="circuit",
    show_default=True,
    help="Variable name for the circuit in generated code.",
)
def generate(
    qasm_file: str,
    explain: bool,
    output_path: str | None,
    circuit_name: str,
) -> None:
    """Generate error mitigation code from a QASM circuit.

    QASM_FILE is a path to a .qasm file, or "-" to read from stdin.
    """
    qc = _load_circuit(qasm_file)

    try:
        features = analyze_circuit(qc)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    recipe = recommend(features)
    code = generate_code(
        recipe,
        features,
        circuit_name=circuit_name,
        explain=explain,
    )

    if output_path is not None:
        try:
            Path(output_path).write_text(code, encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(f"Cannot write to file: {exc}") from exc
        click.echo(f"Written to {output_path}")
    else:
        click.echo(code)


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("qasm_file")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
def analyze(qasm_file: str, json_output: bool) -> None:
    """Analyze a QASM circuit and display its features.

    QASM_FILE is a path to a .qasm file, or "-" to read from stdin.
    """
    qc = _load_circuit(qasm_file)

    try:
        features = analyze_circuit(qc)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        # Manual dict conversion -- dataclasses.asdict deep-copies
        # which fails on MappingProxyType fields.
        data = {
            "num_qubits": features.num_qubits,
            "depth": features.depth,
            "gate_counts": dict(features.gate_counts),
            "total_gate_count": features.total_gate_count,
            "multi_qubit_gate_count": features.multi_qubit_gate_count,
            "single_qubit_gate_count": features.single_qubit_gate_count,
            "num_parameters": features.num_parameters,
            "has_measurements": features.has_measurements,
            "estimated_noise_factor": features.estimated_noise_factor,
            "noise_category": features.noise_category,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo("Circuit Analysis:")
        click.echo(_format_features_table(features))
