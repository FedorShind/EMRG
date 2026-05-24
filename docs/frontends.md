# Frontend support

EMRG accepts several circuit frontends, but support levels are different.
Qiskit and QASM use native paths. Cirq is analyzed directly through the Python
API. Braket, PennyLane, PyQuil, and Qibo are experimental Python inputs that
EMRG normalizes to Cirq through [Mitiq](https://mitiq.readthedocs.io/).

## Support matrix

| Input | Status | Install | API | Notes |
|---|---|---|---|---|
| Qiskit `QuantumCircuit` | native | base | Python | Main Python API path |
| QASM file/stdin | native | base, plus `emrg[qasm3]` for QASM 3 | CLI | Loaded through Qiskit |
| Cirq `Circuit` | direct | base/dev via Mitiq/Cirq | Python | Analyzed as Cirq |
| Braket `Circuit` | experimental normalized | `emrg[braket]` | Python | Converted through Mitiq/Cirq |
| PennyLane `QuantumTape` | experimental normalized | `emrg[pennylane]` | Python | QNodes are not the target yet |
| PyQuil `Program` | experimental normalized | `emrg[pyquil]` | Python | Converted through Mitiq/Cirq |
| Qibo `Circuit` | experimental normalized | `emrg[qibo]` | Python | Converted through Mitiq/Cirq |

Use `emrg[frontends]` to install all optional converted frontend extras.

## What normalized means

For Braket, PennyLane, PyQuil, and Qibo, EMRG asks Mitiq to convert the input
to a Cirq circuit, then extracts features from that Cirq circuit. This keeps
EMRG small and aligned with Mitiq, but feature values can differ from native SDK
counts.

The feature record tells you which path was used:

```python
features.frontend       # "braket", "pennylane", "pyquil", or "qibo"
features.analysis_basis # "cirq-normalized"
```

Qiskit reports `analysis_basis == "qiskit"`. Cirq reports
`analysis_basis == "cirq"`.

## Python examples

### Qiskit

```python
from qiskit import QuantumCircuit

from emrg import generate_recipe

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

result = generate_recipe(qc)
print(result.features.frontend)
print(result.features.analysis_basis)
print(result.code)
```

### Cirq

```python
import cirq

from emrg import generate_recipe

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))

result = generate_recipe(circuit)
print(result.features.frontend)
print(result.features.analysis_basis)
print(result.code)
```

### Braket

Install:

```bash
pip install "emrg[braket]"
```

```python
from braket.circuits import Circuit

from emrg import generate_recipe

circuit = Circuit().h(0).cnot(0, 1)
result = generate_recipe(circuit)

print(result.features.frontend)       # "braket"
print(result.features.analysis_basis) # "cirq-normalized"
```

### PennyLane

Install:

```bash
pip install "emrg[pennylane]"
```

```python
import pennylane as qml

from emrg import generate_recipe

with qml.tape.QuantumTape() as tape:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

result = generate_recipe(tape)

print(result.features.frontend)       # "pennylane"
print(result.features.analysis_basis) # "cirq-normalized"
```

### PyQuil

Install:

```bash
pip install "emrg[pyquil]"
```

```python
from pyquil import Program
from pyquil.gates import CNOT, H

from emrg import generate_recipe

program = Program(H(0), CNOT(0, 1))
result = generate_recipe(program)

print(result.features.frontend)       # "pyquil"
print(result.features.analysis_basis) # "cirq-normalized"
```

### Qibo

Install:

```bash
pip install "emrg[qibo]"
```

```python
from qibo import gates
from qibo.models import Circuit

from emrg import generate_recipe

circuit = Circuit(2)
circuit.add(gates.H(0))
circuit.add(gates.CNOT(0, 1))

result = generate_recipe(circuit)

print(result.features.frontend)       # "qibo"
print(result.features.analysis_basis) # "cirq-normalized"
```

## CLI support

The CLI is QASM-only for now. Use the Python API for Cirq, Braket, PennyLane,
PyQuil, and Qibo objects.

```bash
emrg analyze docs/examples/bell_state.qasm
emrg generate docs/examples/bell_state.qasm
```

QASM files and stdin are loaded through Qiskit. QASM 2 works with the base
install; QASM 3 needs the `emrg[qasm3]` extra.

## Preview support

Preview supports Qiskit and Cirq inputs.

For converted optional frontends, EMRG generates recipes but skips preview with
an explicit warning. This is intentional until the normalized preview path has
strong enough frontend-specific coverage.

## Optional frontend checker

Maintainers can use the optional checker as a release preflight. It installs
extras in isolated virtual environments, builds tiny real SDK circuits, checks
frontend detection, analyzes the circuit, generates a recipe, compiles the
generated code, and confirms preview skip behavior for converted frontends.

```bash
python tools/check_optional_frontends.py --all
```

Ordinary users do not need to run this checker.

## Known limits

- Optional frontends use Cirq-normalized features.
- Depth, gate counts, and non-Clifford counts may differ from native SDK counts.
- PennyLane QNodes are not the target yet; pass a `QuantumTape`.
- CLI input remains QASM-only.
- Preview is Qiskit/Cirq only.
- Generated code still needs your backend executor.
