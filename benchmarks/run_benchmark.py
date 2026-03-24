"""EMRG Benchmark Suite.

Measures four things:

1. **Tool performance** -- wall-clock time and memory overhead of
   ``analyze_circuit()`` / ``generate_recipe()`` across circuits ranging
   from 2 to 50+ qubits.  No simulation is involved; EMRG relies on pure
   Qiskit introspection, so even large circuits complete in milliseconds.

2. **ZNE fidelity** -- end-to-end validation that EMRG-generated ZNE
   recipes actually reduce noise.  We run ideal, noisy, and ZNE-mitigated
   simulations (Cirq ``DensityMatrixSimulator`` with per-gate depolarizing
   noise) and compare the resulting ``<Z_0>`` expectation values.

3. **PEC fidelity** -- same methodology as ZNE, but using Mitiq's
   ``execute_with_pec`` on shallow circuits where PEC is viable.

4. **Layerwise vs global folding** -- compares ``fold_global`` and
   ``fold_gates_at_random`` on circuits with heterogeneous layer structure
   to validate EMRG's layerwise Richardson recommendation.

All numbers printed to stdout are real measurements.  A machine-readable
JSON copy is written to ``benchmarks/results.json``.

Usage::

    # from the project root, with the venv activated
    python benchmarks/run_benchmark.py
"""

import json
import math
import platform
import sys
import time
import tracemalloc
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from emrg import generate_recipe
from emrg.analyzer import analyze_circuit

# Suppress noisy Qiskit / NumPy deprecation chatter.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def make_bell() -> QuantumCircuit:
    """Two-qubit Bell state (H + CX)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def make_x_circuit(n_qubits: int) -> QuantumCircuit:
    """X on qubit 0 followed by a CX chain.

    Ideal ``<Z_0>`` = -1.0, which gives a clear non-zero baseline for
    fidelity benchmarks.
    """
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_ghz(n_qubits: int) -> QuantumCircuit:
    """GHZ state: H on q0, then a CX chain."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def make_random_circuit(
    n_qubits: int,
    n_layers: int,
    seed: int = 42,
) -> QuantumCircuit:
    """Pseudo-random circuit: alternating single-qubit + CX layers.

    Built manually so the benchmark has zero dependency on Qiskit's
    ``random_circuit`` utility (which has changed API across versions).
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    gates_1q = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz")

    for _ in range(n_layers):
        # Single-qubit layer
        for q in range(n_qubits):
            gate = gates_1q[rng.integers(len(gates_1q))]
            if gate in ("rx", "ry", "rz"):
                getattr(qc, gate)(float(rng.uniform(0, 2 * np.pi)), q)
            else:
                getattr(qc, gate)(q)
        # CX layer -- pair up random qubits
        order = list(range(n_qubits))
        rng.shuffle(order)
        for j in range(0, n_qubits - 1, 2):
            qc.cx(order[j], order[j + 1])

    qc.measure_all()
    return qc


def make_hardware_efficient_ansatz(
    n_qubits: int,
    n_layers: int,
) -> QuantumCircuit:
    """VQE-style hardware-efficient ansatz (RY + CX, parameters bound)."""
    rng = np.random.default_rng(123)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def make_heterogeneous_circuit(
    n_qubits: int,
    n_layers: int,
) -> QuantumCircuit:
    """Circuit with alternating heavy and light CX layers.

    Every fourth layer has parallel CX on all qubit pairs; other layers
    have a single CX. This creates varied layer structure (heterogeneity
    depends on qubit count and layer count).
    """
    rng = np.random.default_rng(77)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        # Heavy CX layer (every 4th) vs light CX layer
        if i % 4 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            qc.cx(0, 1)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Part 1: Tool Performance
# ---------------------------------------------------------------------------

@dataclass
class PerfResult:
    """Timing + metadata for a single performance benchmark run."""

    name: str
    n_qubits: int
    depth: int
    total_gates: int
    multi_qubit_gates: int
    layer_heterogeneity: float
    technique_config: str
    time_ms: float        # median wall-clock for generate_recipe()
    peak_memory_kb: float  # tracemalloc peak (Python-heap only)


def benchmark_performance(
    circuit: QuantumCircuit,
    name: str,
    *,
    noise_model_available: bool = False,
    n_runs: int = 100,
) -> PerfResult:
    """Benchmark ``generate_recipe()`` over *n_runs* and return the median."""
    # Warm-up (JIT, import caches, etc.)
    for _ in range(3):
        generate_recipe(circuit, noise_model_available=noise_model_available)

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate_recipe(circuit, noise_model_available=noise_model_available)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # Peak memory (single run, Python heap only via tracemalloc).
    tracemalloc.start()
    generate_recipe(circuit, noise_model_available=noise_model_available)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    features = analyze_circuit(
        circuit, noise_model_available=noise_model_available
    )
    recipe = generate_recipe(
        circuit, noise_model_available=noise_model_available
    ).recipe
    median_ms = float(np.median(times))

    if recipe.technique == "pec":
        config = "PEC"
    else:
        config = f"{recipe.factory_name} + {recipe.scaling_method}"

    return PerfResult(
        name=name,
        n_qubits=features.num_qubits,
        depth=features.depth,
        total_gates=features.total_gate_count,
        multi_qubit_gates=features.multi_qubit_gate_count,
        layer_heterogeneity=round(features.layer_heterogeneity, 4),
        technique_config=config,
        time_ms=round(median_ms, 3),
        peak_memory_kb=round(peak / 1024, 1),
    )


# ---------------------------------------------------------------------------
# Part 2: ZNE Fidelity Benchmark
# ---------------------------------------------------------------------------

@dataclass
class FidelityResult:
    """Ideal / noisy / mitigated comparison for one circuit + noise level."""

    name: str
    n_qubits: int
    depth: int
    noise_level: str
    technique: str
    config: str
    ideal_value: float
    noisy_value: float
    mitigated_value: float
    noisy_error: float
    mitigated_error: float
    improvement_factor: float


def _make_cirq_executor(noise_level: float = 0.0):
    """Return a Cirq-based ``<Z_0>`` executor for use with Mitiq.

    The executor computes the exact expectation value via density-matrix
    simulation (no shot noise), which eliminates statistical fluctuations
    and makes fidelity comparisons clean and reproducible.
    """
    import cirq

    def executor(circuit) -> float:
        if noise_level > 0:
            noisy = cirq.Circuit()
            for moment in circuit.moments:
                noisy.append(moment)
                for op in moment.operations:
                    nq = len(op.qubits)
                    noisy.append(
                        cirq.depolarize(p=noise_level, n_qubits=nq)
                        .on(*op.qubits)
                    )
            sim_circuit = noisy
        else:
            sim_circuit = circuit

        rho = cirq.DensityMatrixSimulator().simulate(sim_circuit).final_density_matrix

        # <Z_0> = Tr(rho * (Z (x) I (x) ...))
        n = len(sorted(circuit.all_qubits()))
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        observable = z
        for _ in range(n - 1):
            observable = np.kron(observable, np.eye(2, dtype=complex))

        return float(np.real(np.trace(rho @ observable)))

    return executor


def _compute_improvement(
    ideal: float, noisy: float, mitigated: float,
) -> float:
    """Compute improvement factor: noisy_error / mitigated_error."""
    noisy_err = abs(ideal - noisy)
    mitigated_err = abs(ideal - mitigated)
    if mitigated_err < 1e-10:
        return float("inf") if noisy_err > 1e-10 else 1.0
    if noisy_err < 1e-10:
        return 1.0
    return noisy_err / mitigated_err


def benchmark_zne_fidelity(
    circuit: QuantumCircuit,
    name: str,
    noise_level: float,
    noise_label: str,
) -> FidelityResult:
    """Compare ideal / noisy / EMRG-mitigated ``<Z_0>`` using ZNE."""
    from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import LinearFactory, PolyFactory, RichardsonFactory
    from mitiq.zne.scaling import fold_gates_at_random, fold_global

    gate_only = circuit.copy()
    gate_only.remove_final_measurements()
    cirq_circuit = from_qiskit(gate_only)

    ideal_value = _make_cirq_executor(noise_level=0.0)(cirq_circuit)
    noisy_exec = _make_cirq_executor(noise_level=noise_level)
    noisy_value = noisy_exec(cirq_circuit)

    recipe = generate_recipe(circuit).recipe
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

    mitigated_value = execute_with_zne(
        cirq_circuit, noisy_exec, factory=factory, scale_noise=scale_fn,
    )

    noisy_err = abs(ideal_value - noisy_value)
    mitigated_err = abs(ideal_value - mitigated_value)
    improvement = _compute_improvement(ideal_value, noisy_value, mitigated_value)

    return FidelityResult(
        name=name,
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        noise_level=noise_label,
        technique="ZNE",
        config=f"{recipe.factory_name} + {recipe.scaling_method}",
        ideal_value=round(ideal_value, 4),
        noisy_value=round(noisy_value, 4),
        mitigated_value=round(mitigated_value, 4),
        noisy_error=round(noisy_err, 4),
        mitigated_error=round(mitigated_err, 4),
        improvement_factor=round(improvement, 2),
    )


# ---------------------------------------------------------------------------
# Part 3: PEC Fidelity Benchmark
# ---------------------------------------------------------------------------

def benchmark_pec_fidelity(
    circuit: QuantumCircuit,
    name: str,
    noise_level: float,
    noise_label: str,
) -> FidelityResult:
    """Compare ideal / noisy / PEC-mitigated ``<Z_0>``."""
    from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
    from mitiq.pec import execute_with_pec
    from mitiq.pec.representations.depolarizing import (
        represent_operations_in_circuit_with_local_depolarizing_noise,
    )

    gate_only = circuit.copy()
    gate_only.remove_final_measurements()
    cirq_circuit = from_qiskit(gate_only)

    ideal_value = _make_cirq_executor(noise_level=0.0)(cirq_circuit)
    noisy_exec = _make_cirq_executor(noise_level=noise_level)
    noisy_value = noisy_exec(cirq_circuit)

    # Build quasi-probability representations for PEC.
    representations = represent_operations_in_circuit_with_local_depolarizing_noise(
        cirq_circuit, noise_level=noise_level,
    )

    mitigated_value = execute_with_pec(
        cirq_circuit,
        noisy_exec,
        representations=representations,
        num_samples=200,
        random_state=42,
    )

    noisy_err = abs(ideal_value - noisy_value)
    mitigated_err = abs(ideal_value - mitigated_value)
    improvement = _compute_improvement(ideal_value, noisy_value, mitigated_value)

    return FidelityResult(
        name=name,
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        noise_level=noise_label,
        technique="PEC",
        config="PEC (depolarizing)",
        ideal_value=round(ideal_value, 4),
        noisy_value=round(noisy_value, 4),
        mitigated_value=round(mitigated_value, 4),
        noisy_error=round(noisy_err, 4),
        mitigated_error=round(mitigated_err, 4),
        improvement_factor=round(improvement, 2),
    )


# ---------------------------------------------------------------------------
# Part 4: Layerwise vs Global Folding Comparison
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseResult:
    """Comparison of fold_global vs fold_gates_at_random on one circuit."""

    name: str
    n_qubits: int
    depth: int
    layer_heterogeneity: float
    noise_level: str
    ideal_value: float
    noisy_value: float
    global_value: float
    layerwise_value: float
    global_improvement: float
    layerwise_improvement: float
    layerwise_wins: bool


def benchmark_layerwise(
    circuit: QuantumCircuit,
    name: str,
    noise_level: float,
    noise_label: str,
) -> LayerwiseResult:
    """Run ZNE with both fold_global and fold_gates_at_random and compare."""
    from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import RichardsonFactory
    from mitiq.zne.scaling import fold_gates_at_random, fold_global

    gate_only = circuit.copy()
    gate_only.remove_final_measurements()
    cirq_circuit = from_qiskit(gate_only)

    ideal_value = _make_cirq_executor(noise_level=0.0)(cirq_circuit)
    noisy_exec = _make_cirq_executor(noise_level=noise_level)
    noisy_value = noisy_exec(cirq_circuit)

    global_value = execute_with_zne(
        cirq_circuit, noisy_exec,
        factory=RichardsonFactory(scale_factors=[1.0, 1.5, 2.0, 2.5]),
        scale_noise=fold_global,
    )
    layerwise_value = execute_with_zne(
        cirq_circuit, noisy_exec,
        factory=RichardsonFactory(scale_factors=[1.0, 1.5, 2.0, 2.5]),
        scale_noise=fold_gates_at_random,
    )

    features = analyze_circuit(circuit)
    global_imp = _compute_improvement(ideal_value, noisy_value, global_value)
    layerwise_imp = _compute_improvement(ideal_value, noisy_value, layerwise_value)

    return LayerwiseResult(
        name=name,
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        layer_heterogeneity=round(features.layer_heterogeneity, 4),
        noise_level=noise_label,
        ideal_value=round(ideal_value, 4),
        noisy_value=round(noisy_value, 4),
        global_value=round(global_value, 4),
        layerwise_value=round(layerwise_value, 4),
        global_improvement=round(global_imp, 2),
        layerwise_improvement=round(layerwise_imp, 2),
        layerwise_wins=bool(layerwise_imp > global_imp),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import emrg  # noqa: I001 -- deferred imports for version display
    import mitiq
    import qiskit

    print("=" * 78)
    print("EMRG Benchmark Suite")
    print("=" * 78)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  OS:      {platform.system()} {platform.release()}")
    print(f"  Qiskit:  {qiskit.__version__}")
    print(f"  Mitiq:   {mitiq.__version__}")
    print(f"  EMRG:    {emrg.__version__}")
    print()

    # -------------------------------------------------------------------
    # Part 1: Tool Performance
    # -------------------------------------------------------------------
    print("=" * 78)
    print("PART 1: Tool Performance (generate_recipe, median of 100 runs)")
    print("=" * 78)
    print()

    perf_circuits = [
        ("Bell (2q)",              make_bell(), False),
        ("Bell (2q, PEC)",         make_bell(), True),
        ("GHZ-5",                  make_ghz(5), False),
        ("GHZ-10",                 make_ghz(10), False),
        ("Random 10q (3 layers)",  make_random_circuit(10, 3), False),
        ("Random 20q (6 layers)",  make_random_circuit(20, 6), False),
        ("VQE 10q (4 layers)",     make_hardware_efficient_ansatz(10, 4), False),
        ("Hetero 4q (8 layers)",   make_heterogeneous_circuit(4, 8), False),
        ("Random 30q (10 layers)", make_random_circuit(30, 10), False),
        ("Random 50q (15 layers)", make_random_circuit(50, 15), False),
    ]

    perf_results: list[PerfResult] = []
    for name, circ, noise_avail in perf_circuits:
        print(f"  {name} ...", end=" ", flush=True)
        r = benchmark_performance(circ, name, noise_model_available=noise_avail)
        perf_results.append(r)
        print(f"{r.time_ms:.2f} ms")

    print()
    header = (
        f"{'Circuit':<26} {'Qb':>3} {'Dep':>4} {'Gates':>6} "
        f"{'MQ':>4} {'Het':>6} {'Technique / Config':<34} "
        f"{'Time':>10} {'Mem':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in perf_results:
        print(
            f"{r.name:<26} {r.n_qubits:>3} {r.depth:>4} "
            f"{r.total_gates:>6} {r.multi_qubit_gates:>4} "
            f"{r.layer_heterogeneity:>6.2f} {r.technique_config:<34} "
            f"{r.time_ms:>8.3f}ms {r.peak_memory_kb:>8.1f}KB"
        )
    print()

    # -------------------------------------------------------------------
    # Part 2: ZNE Fidelity Benchmark
    # -------------------------------------------------------------------
    print("=" * 78)
    print("PART 2: ZNE Fidelity (ideal vs noisy vs EMRG-mitigated)")
    print("=" * 78)
    print()

    zne_tests = [
        ("X-flip 2q",            make_x_circuit(2), 0.01),
        ("X-flip 3q",            make_x_circuit(3), 0.01),
        ("X-flip 2q (noisy)",    make_x_circuit(2), 0.05),
        ("X-flip 3q (noisy)",    make_x_circuit(3), 0.05),
        ("VQE 4q (2 layers)",    make_hardware_efficient_ansatz(4, 2), 0.01),
        ("VQE 4q (4 layers)",    make_hardware_efficient_ansatz(4, 4), 0.01),
        ("VQE 4q (2L, noisy)",   make_hardware_efficient_ansatz(4, 2), 0.05),
    ]

    zne_results: list[FidelityResult] = []
    for name, circ, noise_p in zne_tests:
        print(f"  {name} ...", end=" ", flush=True)
        r = benchmark_zne_fidelity(circ, name, noise_p, f"depol p={noise_p}")
        zne_results.append(r)
        print(f"{r.improvement_factor:.1f}x improvement")

    print()
    _print_fidelity_table(zne_results)

    # -------------------------------------------------------------------
    # Part 3: PEC Fidelity Benchmark
    # -------------------------------------------------------------------
    print("=" * 78)
    print("PART 3: PEC Fidelity (ideal vs noisy vs PEC-mitigated)")
    print("=" * 78)
    print()

    pec_tests = [
        ("Bell (PEC)",      make_bell(), 0.01),
        ("GHZ-3 (PEC)",     make_ghz(3), 0.01),
        ("VQE 4q 2L (PEC)", make_hardware_efficient_ansatz(4, 2), 0.01),
    ]

    pec_results: list[FidelityResult] = []
    for name, circ, noise_p in pec_tests:
        print(f"  {name} ...", end=" ", flush=True)
        r = benchmark_pec_fidelity(circ, name, noise_p, f"depol p={noise_p}")
        pec_results.append(r)
        print(f"{r.improvement_factor:.1f}x improvement")

    print()
    _print_fidelity_table(pec_results)

    # -------------------------------------------------------------------
    # Part 4: Layerwise vs Global Folding
    # -------------------------------------------------------------------
    print("=" * 78)
    print("PART 4: Layerwise (fold_gates_at_random) vs Global (fold_global)")
    print("=" * 78)
    print()

    layerwise_tests = [
        ("Hetero 4q (6L)",  make_heterogeneous_circuit(4, 6), 0.01),
        ("Hetero 4q (8L)",  make_heterogeneous_circuit(4, 8), 0.01),
        ("Hetero 4q (6L, noisy)", make_heterogeneous_circuit(4, 6), 0.05),
    ]

    layerwise_results: list[LayerwiseResult] = []
    for name, circ, noise_p in layerwise_tests:
        print(f"  {name} ...", end=" ", flush=True)
        r = benchmark_layerwise(circ, name, noise_p, f"depol p={noise_p}")
        layerwise_results.append(r)
        winner = "layerwise" if r.layerwise_wins else "global"
        print(f"global={r.global_improvement:.1f}x, "
              f"layerwise={r.layerwise_improvement:.1f}x ({winner} wins)")

    print()
    lhdr = (
        f"{'Circuit':<22} {'Qb':>3} {'Dep':>4} {'Het':>6} "
        f"{'Noise':<14} {'Ideal':>8} {'Noisy':>8} "
        f"{'Global':>8} {'Layer':>8} {'G_imp':>6} {'L_imp':>6} {'Winner':>8}"
    )
    print(lhdr)
    print("-" * len(lhdr))
    for r in layerwise_results:
        winner = "layer" if r.layerwise_wins else "global"
        print(
            f"{r.name:<22} {r.n_qubits:>3} {r.depth:>4} "
            f"{r.layer_heterogeneity:>6.2f} {r.noise_level:<14} "
            f"{r.ideal_value:>8.4f} {r.noisy_value:>8.4f} "
            f"{r.global_value:>8.4f} {r.layerwise_value:>8.4f} "
            f"{r.global_improvement:>5.1f}x {r.layerwise_improvement:>5.1f}x "
            f"{winner:>8}"
        )
    print()

    # -------------------------------------------------------------------
    # JSON output (machine-readable, gitignored)
    # -------------------------------------------------------------------
    output = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.release()}",
            "qiskit": qiskit.__version__,
            "mitiq": mitiq.__version__,
            "emrg": emrg.__version__,
        },
        "performance": [asdict(r) for r in perf_results],
        "zne_fidelity": [asdict(r) for r in zne_results],
        "pec_fidelity": [asdict(r) for r in pec_results],
        "layerwise_comparison": [asdict(r) for r in layerwise_results],
    }

    _RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results saved to {_RESULTS_PATH.relative_to(Path.cwd())}")


def _print_fidelity_table(results: list[FidelityResult]) -> None:
    """Print a formatted fidelity results table."""
    fhdr = (
        f"{'Circuit':<22} {'Qb':>3} {'Dep':>4} {'Noise':<14} "
        f"{'Technique / Config':<26} {'Ideal':>8} {'Noisy':>8} "
        f"{'Mitigated':>10} {'Err_N':>8} {'Err_M':>8} {'Improv':>8}"
    )
    print(fhdr)
    print("-" * len(fhdr))
    for r in results:
        imp_str = (
            f"{r.improvement_factor:>7.1f}x"
            if not math.isinf(r.improvement_factor)
            else "     inf"
        )
        print(
            f"{r.name:<22} {r.n_qubits:>3} {r.depth:>4} "
            f"{r.noise_level:<14} {r.config:<26} "
            f"{r.ideal_value:>8.4f} {r.noisy_value:>8.4f} "
            f"{r.mitigated_value:>10.4f} {r.noisy_error:>8.4f} "
            f"{r.mitigated_error:>8.4f} {imp_str}"
        )
    print()


if __name__ == "__main__":
    main()
