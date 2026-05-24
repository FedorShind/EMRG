<div align="center">
  <img src="docs/banner-emrg.jpg" alt="EMRG" width="390">
  <p><strong>Error Mitigation Recipe Generator</strong></p>
  <p>
    <a href="https://github.com/FedorShind/EMRG/tree/main/docs">Docs</a> &middot;
    <a href="https://mitiq.readthedocs.io/">Mitiq</a> &middot;
    <a href="https://pypi.org/project/emrg/">PyPI</a>
  </p>
  <p>
    <a href="https://github.com/FedorShind/EMRG/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/FedorShind/EMRG/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://pypi.org/project/emrg/"><img alt="PyPI" src="https://img.shields.io/pypi/v/emrg.svg"></a>
    <a href="https://colab.research.google.com/github/FedorShind/EMRG/blob/main/docs/tutorials/vqe_h2_mitigation.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  </p>
</div>

**Error Mitigation Recipe Generator** for [Mitiq](https://mitiq.readthedocs.io/).

EMRG reads a quantum circuit, chooses the right mitigation recipe, and renders
the Mitiq code to run it. Qiskit and QASM are native, Cirq is supported
directly, and Braket, PennyLane, PyQuil, and Qibo can be used through
Mitiq/Cirq normalization.

## The idea

NISQ error mitigation has too many knobs. ZNE needs scale factors and a folding
method. PEC needs a noise model and a sampling budget. CDR needs enough
non-Clifford structure to train against. Composite recipes can help, but only
when their cost is still sane. It is a problem that I myself faced.

EMRG *is a recipe layer*.

It inspects a circuit, picks from ZNE, PEC, CDR, or composite ZNE-over-PEC, then
outputs Mitiq-native Python with the reason for the choice. It is intentionally made
boring: inspect, choose, render.

## How it works

```text
Qiskit / QASM / Cirq / converted frontend
        |
        v
analyze features -> choose policy recipe -> render Mitiq code -> optional preview
```

1. Analyze circuit features: depth, gate mix, noise proxy, PEC overhead,
   layer heterogeneity, and non-Clifford fraction. Qiskit and QASM use the
   native Qiskit path, Cirq is analyzed directly, and optional frontends are
   converted through Mitiq/Cirq first.
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

Cirq:

```python
import cirq
from emrg import generate_recipe

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))

result = generate_recipe(circuit)
print(result.code)
```

Or try it without installing:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedorShind/EMRG/blob/main/docs/tutorials/vqe_h2_mitigation.ipynb)

The generated script contains a backend executor adapter. You still connect
that adapter to your simulator or hardware backend.

## Inputs

| Input | Status | Notes |
|---|---|---|
| Qiskit `QuantumCircuit` | native | Main Python API path |
| QASM files/stdin | native | CLI path, loaded through Qiskit |
| Cirq `Circuit` | direct | Python API |
| Braket `Circuit` | experimental normalized | converted through Mitiq/Cirq |
| PennyLane `QuantumTape` | experimental normalized | QNodes are not the target yet |
| PyQuil `Program` | experimental normalized | converted through Mitiq/Cirq |
| Qibo `Circuit` | experimental normalized | converted through Mitiq/Cirq |

For normalized frontends, EMRG analyzes the Mitiq/Cirq-normalized circuit.
Depth, gate counts, and non-Clifford counts may differ from native SDK
semantics. The CLI is still QASM-only; other frontends are Python API inputs.

Optional installs:

```bash
pip install "emrg[preview]"
pip install "emrg[config]"
pip install "emrg[braket]"
pip install "emrg[pennylane]"
pip install "emrg[pyquil]"
pip install "emrg[qibo]"
pip install "emrg[frontends]"
```

`emrg[frontends]` installs the optional converted frontend stack in one shot.

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

EMRG has a built-in default policy. Policies tune thresholds, overhead
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

Preview mode runs a local simulator check for Qiskit and Cirq inputs:

```bash
pip install "emrg[preview]"
emrg generate docs/examples/bell_state.qasm --preview
```

It uses density-matrix simulation, so size matters. Circuits above the preview
budget are skipped with a plain warning. PEC, CDR, and composite previews can
vary because they include stochastic pieces. For Braket, PennyLane, PyQuil, and
Qibo inputs, recipe generation works but preview is intentionally skipped for
now.

## Benchmarks

v0.6.0 includes a reproducible local benchmark harness. It compares policy
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
Skipped simulations are recorded, not counted as passes.

Older detailed benchmark tables live in
[`benchmarks/HISTORICAL_RESULTS.md`](benchmarks/HISTORICAL_RESULTS.md).

## Project structure

```
EMRG/
|-- src/emrg/
|   |-- __init__.py      # Public API and generate_recipe()
|   |-- _version.py      # Single source of truth for version
|   |-- analyzer.py      # Circuit feature extraction
|   |-- frontends.py     # Frontend detection and normalization
|   |-- heuristics.py    # Rule-based decision engine
|   |-- policy.py        # JSON/YAML policy model and validation
|   |-- codegen.py       # Template-based code generation
|   |-- preview.py       # Simulation preview engine
|   |-- cli.py           # Click CLI interface
|   `-- py.typed         # PEP 561 type marker
|-- tests/               # Unit, integration, frontend, benchmark, and docs checks
|-- docs/
|   |-- examples/        # Example circuits (Python + QASM)
|   |-- tutorials/       # Jupyter notebooks (VQE, QAOA)
|   `-- banner-emrg.jpg  # Cool README banner
|-- benchmarks/          # Automated benchmark suite plus historical data
|-- tools/               # Maintainer checks for optional frontend extras
`-- pyproject.toml       # Package configuration
```

## Design choices

- Deterministic by default.
- Policy files, not arbitrary code.
- Mitiq-native output instead of a wrapper runtime.
- Qiskit/QASM native, Cirq direct, optional frontends normalized through Mitiq/Cirq.
- Benchmark harness is local and reproducible.
- Conservative about hardware claims.

## Limitations

- Converted frontend analysis uses Cirq-normalized features, so counts may
  differ from native SDK semantics.
- CLI input is QASM-only.
- Generated code needs a backend executor.
- Preview is Qiskit/Cirq simulation, not hardware validation.
- PEC, CDR, and composite recipes can have stochastic or backend-specific costs.
- Native analyzers for optional frontends are future work only if conversion
  proves insufficient.

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
- [x] Unit, integration, preview, policy, and docs checks in CI/local validation
- [x] Clifford Data Regression (CDR) support
- [x] Composite recipes -- combine ZNE + PEC for circuits that benefit from both
- [x] Configurable heuristics via YAML/JSON

### Phase 3 -- Multi-framework support (in progress)

- [x] Cirq Python API input support
- [x] Experimental normalized Braket, PennyLane QuantumTape, PyQuil Program, and Qibo Circuit input support
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
* **Cirq** >= 1.0 -- direct Python input and simulation preview backend
* **Braket, PennyLane, PyQuil, Qibo** -- optional converted frontend extras

## Contributing

Open an issue or PR on [GitHub](https://github.com/FedorShind/EMRG). Always welcome!

## License

[MIT](LICENSE) - go build something quantum!

## Acknowledgments

Built on [Mitiq](https://mitiq.readthedocs.io/) by
[Unitary Foundation](https://unitary.foundation/).
