# EMRG

[![CI](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml/badge.svg)](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml)

## **Error Mitigation Recipe Generator** -- Automatic quantum error mitigation for NISQ circuits.

EMRG analyzes your quantum circuit and generates ready-to-run, explained [Mitiq](https://mitiq.readthedocs.io/)-powered error mitigation code. No manual tuning required.

> **Status:** v0.2.5 -- ZNE + PEC support. Actively developed, [grant-funded roadmap](#roadmap) ahead.

---

## Why EMRG?

Noise limits every computation on today's hardware. Error mitigation techniques like **Zero-Noise Extrapolation (ZNE)** and **Probabilistic Error Cancellation (PEC)** can boost fidelity 2--10x, but configuring them manually is tedious:

* Which technique -- ZNE or PEC?
* Which extrapolation factory? Linear, Richardson, Polynomial?
* What scale factors for your circuit depth?
* How do you balance overhead vs. accuracy?

**EMRG handles this automatically.** Give it a circuit, get back optimized mitigation code with clear explanations of *why* each choice was made. EMRG selects between techniques, not just tunes settings.

## How It Works

```
Quantum Circuit --> [Analyze] --> [Technique Selection] --> [Code Generator] --> Mitigated Code
                                   ZNE or PEC
```

1. **Parse & Validate** -- Load a Qiskit `QuantumCircuit` or QASM file
2. **Extract Features** -- Depth, gate counts, multi-qubit gate density, estimated noise factor, PEC overhead
3. **Select Technique** -- Choose between ZNE and PEC based on circuit characteristics
4. **Generate Code** -- Output runnable Python with Mitiq imports, config, and inline rationale

### Heuristic Rules (v0.2)

| Circuit Profile | Technique | Configuration | Rationale |
| --- | --- | --- | --- |
| Depth ≤ 30 + noise model + overhead < 1000 | **PEC** | Depolarizing representations | Unbiased error cancellation when overhead is manageable |
| Depth < 20, low multi-qubit gates | ZNE `LinearFactory` | `[1.0, 1.5, 2.0]` | Conservative for shallow circuits |
| Depth 20--50 | ZNE `RichardsonFactory` | `[1.0, 1.5, 2.0, 2.5]` | Better extrapolation for moderate noise |
| Depth > 50 or high noise | ZNE `PolyFactory` (deg 2--3) | `[1.0, 1.5, 2.0, 2.5, 3.0]` | Handles non-linear noise scaling |

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

# Force PEC technique (requires noise model)
emrg generate circuit.qasm --technique pec --noise-model

# Force ZNE even when PEC is viable
emrg generate circuit.qasm --technique zne
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

# With PEC (when a noise model is available)
result = generate_recipe(qc, noise_model_available=True)
print(result.recipe.technique)  # "pec" for shallow circuits
```

### Example Output

```python
# =============================================================
# EMRG v0.2.0 -- Error Mitigation Recipe
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
├── tests/               # 215+ pytest tests, 99% coverage
├── docs/examples/       # Example circuits (Python + QASM)
└── pyproject.toml       # Package configuration
```

## Benchmarks

Real measurements from EMRG v0.2.5, collected automatically by [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py).

> **Environment:** Python 3.12, Windows 11 | Qiskit 2.3.0, Mitiq 0.48.1

### Tool Performance

EMRG relies on pure Qiskit introspection (no simulation), so `generate_recipe()` completes in sub-millisecond time even for large circuits. Median of 100 runs:

| Circuit | Qubits | Depth | Gates | Multi-Q | Het | Technique / Config | Time | Memory |
|---|---|---|---|---|---|---|---|---|
| Bell state | 2 | 3 | 2 | 1 | 0.00 | `LinearFactory` + fold_global | 0.069 ms | 9.4 KB |
| Bell state (PEC) | 2 | 3 | 2 | 1 | 0.00 | PEC | 0.068 ms | 9.4 KB |
| GHZ-5 | 5 | 6 | 5 | 4 | 0.50 | `LinearFactory` + fold_global | 0.114 ms | 15.2 KB |
| GHZ-10 | 10 | 11 | 10 | 9 | 0.50 | `LinearFactory` + fold_global | 0.193 ms | 24.9 KB |
| Random 10q, 3 layers | 10 | 7 | 45 | 15 | 0.83 | `LinearFactory` + fold_global | 0.296 ms | 21.2 KB |
| VQE 10q, 4 layers | 10 | 20 | 76 | 36 | 1.50 | `PolyFactory` + fold_gates_at_random | 0.492 ms | 43.8 KB |
| Random 20q, 6 layers | 20 | 13 | 180 | 60 | 0.91 | `PolyFactory` + fold_gates_at_random | 0.836 ms | 40.4 KB |
| Random 30q, 10 layers | 30 | 21 | 450 | 150 | 0.94 | `PolyFactory` + fold_gates_at_random | 1.917 ms | 69.6 KB |
| Random 50q, 15 layers | 50 | 31 | 1125 | 375 | 0.96 | `PolyFactory` + fold_gates_at_random | 4.518 ms | 124.6 KB |

A 50-qubit, 1125-gate circuit is analyzed and produces a full mitigation recipe in under 5 ms with ~125 KB memory overhead.

### ZNE Fidelity

End-to-end ZNE on noisy simulations (Cirq `DensityMatrixSimulator` with per-gate depolarizing noise), comparing the ⟨Z⟩ expectation value on qubit 0:

| Circuit | Qubits | Depth | Noise | Technique / Config | Ideal | Noisy | Mitigated | Error Reduction |
|---|---|---|---|---|---|---|---|---|
| X-flip, 2q | 2 | 3 | p=0.01 | `LinearFactory` + fold_global | -1.0000 | -0.9761 | -1.0003 | **77x** |
| X-flip, 3q | 3 | 4 | p=0.01 | `LinearFactory` + fold_global | -1.0000 | -0.9761 | -1.0003 | **77x** |
| X-flip, 2q | 2 | 3 | p=0.05 | `LinearFactory` + fold_global | -1.0000 | -0.8836 | -0.9906 | **12x** |
| X-flip, 3q | 3 | 4 | p=0.05 | `LinearFactory` + fold_global | -1.0000 | -0.8836 | -0.9906 | **12x** |
| VQE 4q, 2 layers | 4 | 8 | p=0.01 | `LinearFactory` + fold_global | 0.0850 | 0.0775 | 0.0794 | **1.4x** |
| VQE 4q, 4 layers | 4 | 14 | p=0.01 | `LinearFactory` + fold_global | -0.1915 | -0.1766 | -0.1850 | **2.3x** |
| VQE 4q, 2 layers | 4 | 8 | p=0.05 | `LinearFactory` + fold_global | 0.0850 | 0.0523 | 0.0586 | **1.2x** |

### PEC vs ZNE: Head-to-Head

Same circuits, same noise, both techniques. PEC uses 1000 samples for benchmark accuracy. ZNE is deterministic; PEC results have inherent variance due to stochastic sampling.

**Single-qubit observable ⟨Z⟩:**

| Circuit | Noise | ZNE Error | ZNE Reduction | PEC Error | PEC Reduction | Better |
|---|---|---|---|---|---|---|
| VQE 4q, 2 layers | p=0.01 | 0.0055 | 1.4x | 0.0007 | **10.4x** | PEC |
| VQE 4q, 2 layers | p=0.03 | 0.0162 | 1.3x | 0.0138 | **1.5x** | PEC |
| VQE 4q, 2 layers | p=0.05 | 0.0264 | 1.2x | 0.0176 | **1.9x** | PEC |
| X-flip, 3q | p=0.03 | 0.0024 | **28.9x** | 0.0245 | 2.9x | ZNE |

**Multi-qubit observable ⟨ZZ⟩:**

| Circuit | Noise | ZNE Error | ZNE Reduction | PEC Error | PEC Reduction | Better |
|---|---|---|---|---|---|---|
| VQE 4q, 2 layers | p=0.01 | 0.0021 | **5.7x** | 0.0064 | 1.9x | ZNE |
| VQE 4q, 2 layers | p=0.03 | 0.0102 | **3.4x** | 0.0173 | 2.0x | ZNE |
| VQE 4q, 2 layers | p=0.05 | 0.0216 | 2.5x | 0.0147 | **3.6x** | PEC |

The pattern: ZNE excels on structured circuits where the noise-vs-scale relationship is predictable (X-flip: 28.9x). PEC excels on irregular circuits at higher noise levels, where ZNE's extrapolation assumptions start to break down. On multi-qubit observables, PEC overtakes ZNE as noise increases -- at p=0.05, PEC achieves 3.6x vs ZNE's 2.5x on ⟨ZZ⟩. This is the tradeoff EMRG's heuristic engine navigates: it recommends PEC for shallow, noisy circuits where a noise model is available, and ZNE elsewhere.

### Layerwise Folding

EMRG supports layerwise folding (`fold_gates_at_random`) as an alternative to global folding for circuits with heterogeneous layer structure. This feature is in active development -- current benchmarks show mixed results, and the heuristic thresholds are being refined.

| Circuit | Qubits | Depth | Het | Noise | Global | Layerwise | Winner |
|---|---|---|---|---|---|---|---|
| VQE 10q, 3 reps | 10 | 13 | 2.50 | p=0.01 | 0.9x | **12.6x** | layerwise |
| VQE 10q, 3 reps | 10 | 13 | 2.50 | p=0.03 | 1.1x | 1.1x | -- |
| QAOA 10q | 10 | 14 | 2.50 | p=0.01 | **4.2x** | 0.2x | global |
| QAOA 10q | 10 | 14 | 2.50 | p=0.03 | **5.9x** | 0.7x | global |
| Extreme 10q | 10 | 13 | 2.50 | p=0.01 | 0.5x | 0.4x | -- |
| Extreme 10q | 10 | 13 | 2.50 | p=0.03 | 0.5x | 0.1x | global |

At this scale, `fold_global` is generally more reliable because `fold_gates_at_random` introduces stochastic variation into the extrapolation fit. Layerwise folding shows occasional strong results (12.6x on VQE at low noise) but is not yet consistent enough to be the default. EMRG currently defaults to `fold_global` for most circuits and recommends layerwise folding conservatively. Improvements to the layerwise heuristic -- including noise-aware layer selection and deterministic layer folding strategies -- are planned for future releases.

### Reproduce
```
pip install -e ".[dev]" qiskit-aer
python benchmarks/run_benchmark.py
```

## Roadmap

### Phase 1 -- MVP (complete)

Everything needed to go from circuit to mitigation recipe in one command:

- [x] Project structure and packaging
- [x] Circuit analyzer (feature extraction)
- [x] Heuristic engine (ZNE: Linear + Richardson + Poly)
- [x] Code generator (template-based)
- [x] CLI with `generate` and `analyze` commands
- [x] Public Python API (`generate_recipe()`)
- [x] Example circuits (Python + QASM) and documentation
- [x] 144 tests, 98% coverage, zero lint warnings

### Phase 2 -- More techniques, better validation (current)

Expand beyond ZNE so EMRG can recommend the right technique, not just the right ZNE settings:

- [x] Probabilistic Error Cancellation (PEC) support
- [x] Multi-technique selection (ZNE vs PEC)
- [x] PEC code generation template
- [x] `--technique` override and `--noise-model` CLI flags
- [x] 215+ tests, 99% coverage, zero lint warnings
- [ ] Clifford Data Regression (CDR) support
- [x] Layerwise Richardson integration
- [ ] Composite recipes -- combine ZNE + PEC for circuits that benefit from both
- [ ] `--preview` mode (noisy simulation + fidelity plots)
- [ ] Real hardware benchmarks (IBM Quantum devices)
- [x] Expanded tutorials (VQE for H₂, QAOA on MaxCut, random circuits)

### Phase 3 -- Multi-framework and community

Make EMRG useful regardless of which framework you use:

- [ ] Cirq and PennyLane circuit input support
- [ ] Noise model import from Qiskit Aer / real device calibration data
- [ ] Configurable heuristics via YAML/JSON
- [ ] Jupyter widget for interactive recipe exploration
- [ ] Web/Colab interface

### Phase 4 -- Intelligence layer

Replace static rules with data-driven mitigation selection:

- [ ] Train on benchmark data to predict optimal mitigation strategy
- [ ] Circuit similarity search -- match against known-good configurations
- [ ] Auto-tuning -- run `--preview` internally and iterate on parameters before output
- [ ] Cost-aware optimization -- user specifies a shot budget, EMRG optimizes within that constraint

### Phase 5 -- Ecosystem integration

Make EMRG part of the standard quantum development workflow:

- [ ] Qiskit Runtime integration
- [ ] Mitiq Calibration API integration -- use calibration data to refine recommendations
- [ ] VS Code extension -- analyze circuits inline while writing them
- [ ] CI/CD integration -- add EMRG to quantum testing pipelines for automatic mitigation

## Tech Stack

* **Python 3.11+**
* **Qiskit** >= 1.0 -- Circuit representation and introspection
* **Mitiq** >= 0.48 -- Error mitigation primitives
* **Click** >= 8.0 -- CLI framework

## Contributing

EMRG is open source and contributions are welcome. If you have ideas, find bugs, or want to add support for new mitigation techniques, open an issue or PR.

## License

[MIT](LICENSE) -- Free for academic and commercial use.

## Acknowledgments

Built on [Mitiq](https://mitiq.readthedocs.io/) by [Unitary Foundation](https://unitary.foundation/).
Inspired by the need to make quantum error mitigation accessible to everyone working with NISQ hardware.
