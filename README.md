# EMRG

[![CI](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml/badge.svg)](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml)

**Error Mitigation Recipe Generator** -- Automatic quantum error mitigation for NISQ circuits.

EMRG analyzes your quantum circuit and generates ready-to-run, explained [Mitiq](https://mitiq.readthedocs.io/)-powered error mitigation code. No manual tuning required.

> **Status:** v0.1.0 -- MVP. Actively developed, [grant-funded roadmap](#roadmap) ahead.

---

## Why EMRG?

Noise limits every computation on today's hardware. Error mitigation techniques like **Zero-Noise Extrapolation (ZNE)** can boost fidelity 2--10x, but configuring them manually is tedious:

* Which extrapolation factory? Linear, Richardson, Polynomial?
* What scale factors for your circuit depth?
* How do you balance overhead vs. accuracy?

**EMRG handles this automatically.** Give it a circuit, get back optimized mitigation code with clear explanations of *why* each choice was made.

## How It Works

```
Quantum Circuit --> [Analyze] --> [Heuristic Engine] --> [Code Generator] --> Mitigated Code
```

1. **Parse & Validate** -- Load a Qiskit `QuantumCircuit` or QASM file
2. **Extract Features** -- Depth, gate counts, multi-qubit gate density, estimated noise factor
3. **Apply Heuristics** -- Rule-based decision tree selects the best mitigation recipe
4. **Generate Code** -- Output runnable Python with Mitiq imports, factory config, and inline rationale

### Heuristic Rules (v0.1)

| Circuit Profile | Factory | Scale Factors | Rationale |
| --- | --- | --- | --- |
| Depth < 20, low multi-qubit gates | `LinearFactory` | `[1.0, 1.5, 2.0]` | Conservative for shallow circuits |
| Depth 20--50 | `RichardsonFactory` | `[1.0, 1.5, 2.0, 2.5]` | Better extrapolation for moderate noise |
| Depth > 50 or high noise | `PolyFactory` (deg 2--3) | `[1.0, 1.5, 2.0, 2.5, 3.0]` | Handles non-linear noise scaling |

## Quick Start

### Installation

```
pip install emrg
```

Or from source:

```
git clone https://github.com/FedorShind/EMRG.git
cd EMRG
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Generate mitigation recipe from a QASM file
emrg generate docs/examples/bell_state.qasm

# With verbose explanation
emrg generate docs/examples/bell_state.qasm --explain

# Save to file
emrg generate circuit.qasm -o mitigated.py

# Analyze circuit features
emrg analyze docs/examples/simple_vqe.qasm

# JSON output (for scripting)
emrg analyze circuit.qasm --json
```

### Python API

```python
from qiskit import QuantumCircuit
from emrg import generate_recipe

# Create a circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Generate mitigation recipe (one-liner)
result = generate_recipe(qc)
print(result)             # Ready-to-run Python script
print(result.rationale)   # Why these parameters were chosen
print(result.features)    # Circuit analysis details

# With verbose explanations
result = generate_recipe(qc, explain=True)
```

### Example Output

```python
# =============================================================
# EMRG v0.1.0 -- Error Mitigation Recipe
# Circuit: 2 qubits, depth 3, 1 multi-qubit gates
# Noise estimate: 0.011 (low)
# =============================================================
#
# Recommendation: LinearFactory + fold_global
#
# =============================================================

from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory
from mitiq.zne.scaling import fold_global

factory = LinearFactory(scale_factors=[1.0, 1.5, 2.0])

def execute(circuit):
    """Execute a circuit and return an expectation value (float)."""
    # Replace with your actual backend
    raise NotImplementedError("Replace this with your executor.")

mitigated_value = execute_with_zne(
    circuit,
    execute,
    factory=factory,
    scale_noise=fold_global,
)

print(f"Mitigated expectation value: {mitigated_value}")
```

## Project Structure

```
EMRG/
├── src/emrg/
│   ├── __init__.py      # Public API and generate_recipe()
│   ├── _version.py      # Single source of truth for version
│   ├── analyzer.py      # Circuit feature extraction
│   ├── heuristics.py    # Rule-based decision engine
│   ├── codegen.py       # Template-based code generation
│   ├── cli.py           # Click CLI interface
│   └── py.typed         # PEP 561 type marker
├── tests/               # 144 pytest tests, 98% coverage
├── docs/examples/       # Example circuits (Python + QASM)
└── pyproject.toml       # Package configuration
```

## Benchmarks

Real measurements from EMRG v0.1.0, collected automatically by [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py).

> **Environment:** Python 3.12, Windows 11 | Qiskit 2.3.0, Mitiq 0.48.1

### Tool Performance

EMRG relies on pure Qiskit introspection (no simulation), so `generate_recipe()` completes in sub-millisecond time even for large circuits. Median of 100 runs:

| Circuit | Qubits | Depth | Gates | Multi-Q | Factory | Time | Memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bell state | 2 | 3 | 2 | 1 | `LinearFactory` | 0.033 ms | 3.8 KB |
| GHZ-5 | 5 | 6 | 5 | 4 | `LinearFactory` | 0.047 ms | 3.8 KB |
| GHZ-10 | 10 | 11 | 10 | 9 | `LinearFactory` | 0.069 ms | 3.8 KB |
| Random 10q, 3 layers | 10 | 7 | 45 | 15 | `LinearFactory` | 0.159 ms | 4.1 KB |
| VQE 10q, 4 layers | 10 | 20 | 76 | 36 | `PolyFactory` | 0.234 ms | 3.9 KB |
| Random 20q, 6 layers | 20 | 13 | 180 | 60 | `PolyFactory` | 0.478 ms | 6.2 KB |
| Random 30q, 10 layers | 30 | 21 | 450 | 150 | `PolyFactory` | 1.10 ms | 7.9 KB |
| Random 50q, 15 layers | 50 | 31 | 1125 | 375 | `PolyFactory` | 2.59 ms | 11.1 KB |

A 50-qubit, 1125-gate circuit is analyzed and produces a full mitigation recipe in under 3 ms with ~11 KB memory overhead.

### ZNE Fidelity

To validate that EMRG selects effective mitigation parameters, we ran ZNE end-to-end on noisy simulations (Cirq `DensityMatrixSimulator` with per-gate depolarizing noise) and compared the `<Z>` expectation value on qubit 0:

| Circuit | Qubits | Depth | Noise | Factory | Ideal | Noisy | Mitigated | Error Reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X-flip, 2q | 2 | 3 | p=0.01 | `LinearFactory` | -1.0000 | -0.9761 | -1.0003 | **77x** |
| X-flip, 3q | 3 | 4 | p=0.01 | `LinearFactory` | -1.0000 | -0.9761 | -1.0003 | **77x** |
| X-flip, 2q | 2 | 3 | p=0.05 | `LinearFactory` | -1.0000 | -0.8836 | -0.9906 | **12x** |
| X-flip, 3q | 3 | 4 | p=0.05 | `LinearFactory` | -1.0000 | -0.8836 | -0.9906 | **12x** |
| VQE 4q, 2 layers | 4 | 8 | p=0.01 | `LinearFactory` | 0.0850 | 0.0775 | 0.0794 | **1.4x** |
| VQE 4q, 4 layers | 4 | 14 | p=0.01 | `LinearFactory` | -0.1915 | -0.1766 | -0.1850 | **2.3x** |
| VQE 4q, 2 layers | 4 | 8 | p=0.05 | `LinearFactory` | 0.0850 | 0.0523 | 0.0586 | **1.2x** |

EMRG-generated ZNE recipes reduce error across all tested circuits, with improvements from 1.2x on high-noise VQE ansatze up to 77x on structured low-noise circuits.

### Reproduce

```bash
pip install -e ".[dev]" qiskit-aer
python benchmarks/run_benchmark.py
```

## Roadmap

### Phase 1 -- MVP (current)

Everything needed to go from circuit to mitigation recipe in one command:

- [x] Project structure and packaging
- [x] Circuit analyzer (feature extraction)
- [x] Heuristic engine (ZNE: Linear + Richardson + Poly)
- [x] Code generator (template-based)
- [x] CLI with `generate` and `analyze` commands
- [x] Public Python API (`generate_recipe()`)
- [x] Example circuits (Python + QASM) and documentation
- [x] 144 tests, 98% coverage, zero lint warnings

### Phase 2 -- More techniques, better validation

Expand beyond ZNE so EMRG can recommend the right technique, not just the right ZNE settings:

- [ ] Probabilistic Error Cancellation (PEC) support
- [ ] Layerwise Richardson integration
- [ ] `--preview` mode (noisy simulation + fidelity plots)
- [ ] Real hardware benchmarks (IBM Quantum devices)
- [ ] Expanded tutorials (VQE, QAOA, random circuits)

### Phase 3 -- Multi-framework and community

Make EMRG useful regardless of which framework you use:

- [ ] Cirq and PennyLane input support
- [ ] Configurable heuristics file
- [ ] Web/Colab interface

## Tech Stack

* **Python 3.10+**
* **Qiskit** >= 1.0 -- Circuit representation and introspection
* **Mitiq** >= 0.48 -- Error mitigation primitives
* **Click** >= 8.0 -- CLI framework

## Contributing

EMRG is open source and contributions are welcome. If you have ideas, find bugs, or want to add support for new mitigation techniques, open an issue or PR.

## License

[MIT](LICENSE) -- Free for academic and commercial use.

## Acknowledgments

Built on [Mitiq](https://mitiq.readthedocs.io/) by [Unitary Fund](https://unitary.fund/).
Inspired by the need to make quantum error mitigation accessible to everyone working with NISQ hardware.
