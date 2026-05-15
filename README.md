# EMRG

[![CI](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml/badge.svg)](https://github.com/FedorShind/EMRG/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedorShind/EMRG/blob/main/docs/tutorials/vqe_h2_mitigation.ipynb)

**Error Mitigation Recipe Generator** for [Mitiq](https://mitiq.readthedocs.io/).

EMRG reads a Qiskit circuit, chooses right mitigation recipe, and renders the
Mitiq code to run it. The problem of choosing the recipe is what EMRG helps with.
EMRG is the small deterministic layer in between.

## The idea

NISQ error mitigation has too many knobs. ZNE needs scale factors and a folding
method. PEC needs a noise model and a sampling budget. CDR needs enough
non-Clifford structure to train against. Composite recipes can help, but only
when their cost is still sane.

EMRG *is a recipe layer*. It is not a new mitigation primitive.

It inspects a circuit, picks from ZNE, PEC, CDR, or composite ZNE-over-PEC, then
outputs Mitiq-native Python with the reason for the choice. It is intentionally
boring: inspect, choose, render.

## How it works

```text
Qiskit circuit or QASM
        |
        v
analyze features -> choose policy recipe -> render Mitiq code -> optional preview
```

1. Analyze circuit features: depth, gate mix, noise proxy, PEC overhead,
   layer heterogeneity, and non-Clifford fraction.
2. Pick a recipe from the active policy.
3. Generate Mitiq code with imports, parameters, and rationale.
4. Optionally preview the recipe with a local simulator.

## Quick start

```bash
pip install emrg
```

From a source checkout:

```powershell
emrg analyze docs/examples/bell_state.qasm
emrg generate docs/examples/bell_state.qasm
```

Python:

```python
from qiskit import QuantumCircuit
from emrg import generate_recipe

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

result = generate_recipe(qc)
print(result.code)
print(result.rationale)
```


Or try it without installing: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedorShind/EMRG/blob/main/docs/tutorials/vqe_h2_mitigation.ipynb)

The generated script contains a backend executor adapter. You still connect
that adapter to your simulator or hardware backend.

### CLI Usage
```
# Generate mitigation recipe from a QASM file
emrg generate circuit.qasm

# With verbose explanation
emrg generate circuit.qasm --explain

# Save to file
emrg generate circuit.qasm -o mitigated.py

# Create and validate a policy file
emrg policy init emrg-policy.json
emrg policy validate emrg-policy.json

# Generate with a policy file
emrg generate circuit.qasm --policy emrg-policy.json

# Force a specific technique
emrg generate circuit.qasm --technique pec --noise-model
emrg generate circuit.qasm --technique composite --noise-model
emrg generate circuit.qasm --technique cdr

# Forced techniques bypass automatic viability checks. EMRG still returns the
# requested recipe, but generated output includes warnings when the circuit
# falls outside the automatic selection criteria.

# Preview: simulate and compare before/after mitigation
emrg generate circuit.qasm --preview

# Preview with custom noise level and observable
emrg generate circuit.qasm --preview --noise-level 0.03 --observable ZZ

# Analyze circuit features
emrg analyze circuit.qasm

# JSON output (for scripting)
emrg analyze circuit.qasm --json
```

## Policies

EMRG ships with a built-in default policy. Policies tune thresholds, overhead
budgets, and Mitiq factory/scaling choices. Policies are data, not code: they
do not execute Python, import modules, or define arbitrary logic.

JSON policies work in the base install. YAML policies require:

```bash
pip install "emrg[config]"
```

Smallest useful flow:

```powershell
emrg policy init emrg-policy.json
emrg policy validate emrg-policy.json
emrg generate circuit.qasm --policy emrg-policy.json
```

Use the generated policy file as the schema reference. The implementation lives
in `src/emrg/policy.py`.

## Preview mode

Preview mode runs a neat local simulator check:

```bash
pip install "emrg[preview]"
emrg generate docs/examples/bell_state.qasm --preview
```

It uses density-matrix simulation, so size matters. Circuits above the preview
budget are skipped with a plain warning. PEC, CDR, and composite previews can
vary because they include stochastic pieces.

## Benchmarks

v0.5.1 includes a reproducible local benchmark harness. It compares policy
choices under simulator and noise-model setups. This is useful for regression
testing and local policy calibration. It is not a hardware performance claim.

Compact local calibration snapshot:

| Policy | Score | Median error reduction | Failures | Skips |
|---|---:|---:|---:|---:|
| `default-v050.json` | 0.7872 | 2.357x | 0 | 8/18 |
| `default-v051.json` | 1.8455 | 4.779x | 0 | 8/18 |

Benchmark numbers are local simulator results. Run
`benchmarks/run_benchmark.py` to write machine-readable JSON under
`benchmarks/results/`, then score it with `benchmarks/score_results.py`.
Skipped simulations are recorded explicitly, not counted as passes.
Do not commit generated benchmark JSON or external QASM datasets.

Older detailed benchmark tables live in
[`benchmarks/HISTORICAL_RESULTS.md`](benchmarks/HISTORICAL_RESULTS.md).

## Project structure

```
EMRG/
├── src/emrg/
│   ├── __init__.py      # Public API and generate_recipe()
│   ├── _version.py      # Single source of truth for version
│   ├── analyzer.py      # Circuit feature extraction
│   ├── heuristics.py    # Rule-based decision engine
│   ├── policy.py        # JSON/YAML policy model and validation
│   ├── codegen.py       # Template-based code generation
│   ├── preview.py       # Simulation preview engine
│   ├── cli.py           # Click CLI interface
│   └── py.typed         # PEP 561 type marker
├── tests/               # 486 tests, coverage checked in CI/local validation
├── docs/
│   ├── examples/        # Example circuits (Python + QASM)
│   └── tutorials/       # Jupyter notebooks (VQE, QAOA)
├── benchmarks/          # Automated benchmark suite plus historical data
└── pyproject.toml       # Package configuration
```

## Design choices

- Deterministic by default.
- Policy files, not arbitrary code.
- Mitiq-native output instead of a wrapper runtime.
- Qiskit input today.
- Benchmark harness is local and reproducible.
- Conservative about hardware claims.

## Limitations

- Qiskit input is the main path for now.
- Generated code needs a backend executor.
- Preview is simulation, not hardware validation of course.
- PEC, CDR, and composite recipes can have stochastic or backend-specific costs.

## Roadmap

### Phase 1 -- MVP (complete)

- [x] Project structure and packaging
- [x] Circuit analyzer (feature extraction)
- [x] Heuristic engine (ZNE: Linear + Richardson + Poly)
- [x] Code generator (template-based)
- [x] CLI with `generate` and `analyze` commands
- [x] Public Python API (`generate_recipe()`)
- [x] Example circuits (Python + QASM) and documentation

### Phase 2 -- Multi-technique support (complete)

- [x] Probabilistic Error Cancellation (PEC) support
- [x] Multi-technique selection (ZNE vs PEC)
- [x] PEC code generation template
- [x] `--technique` override and `--noise-model` CLI flags
- [x] Layerwise Richardson integration
- [x] `--preview` mode (noisy simulation + before/after comparison)
- [x] Expanded tutorials (VQE, QAOA)
- [x] 486 tests, coverage checked in CI/local validation, zero lint warnings
- [x] Clifford Data Regression (CDR) support
- [x] Composite recipes -- combine ZNE + PEC for circuits that benefit from both
- [x] Configurable heuristics via YAML/JSON

### Phase 3 -- Multi-framework support (in progress)

- [ ] Cirq, PennyLane, and Amazon Braket input support
- [ ] Noise model import from Qiskit Aer / real device calibration data
- [ ] Jupyter widget for interactive recipe exploration
- [ ] Web/Colab interface

### Phase 4 -- Data-driven selection

- [ ] Train on benchmark data to predict optimal mitigation strategy
- [ ] Circuit similarity search against known-good configurations
- [ ] Auto-tuning via internal `--preview` iterations before output
- [ ] Cost-aware optimization within user-specified shot budgets
- [ ] Real hardware benchmarks (IBM Quantum devices)

### Phase 5 -- Ecosystem integration

- [ ] Qiskit Runtime integration
- [ ] Mitiq Calibration API integration
- [ ] VS Code extension for inline circuit analysis
- [ ] CI/CD integration for quantum testing pipelines

## Tech Stack

* **Python 3.11+**
* **Qiskit** >= 1.0 -- circuit representation and introspection
* **Mitiq** >= 0.48 -- error mitigation primitives
* **Click** >= 8.0 -- CLI framework
* **Cirq** >= 1.0 -- simulation backend (optional, for preview and CDR)

## Contributing

Open an issue or PR on [GitHub](https://github.com/FedorShind/EMRG). Always welcome!

## License

[MIT](LICENSE) - go build something quantum!

## Acknowledgments

Built on [Mitiq](https://mitiq.readthedocs.io/) by
[Unitary Foundation](https://unitary.foundation/).
