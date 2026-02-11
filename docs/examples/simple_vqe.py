"""Simple VQE-like Ansatz Circuit -- parametric variational circuit.

A small variational circuit typical of VQE/QAOA workloads. Uses RY
rotations and CX entangling layers. This represents a more realistic
use case for error mitigation.

Expected profile:
- Depth: ~15-25 depending on layers
- Multi-qubit gates: moderate (scales with layers * qubits)
- Noise estimate: Moderate
- Expected EMRG recommendation: RichardsonFactory
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def create_simple_vqe(n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """Create a simple hardware-efficient VQE ansatz.

    Structure per layer: RY on all qubits -> CX chain -> RY on all qubits.

    Args:
        n_qubits: Number of qubits (default 4).
        n_layers: Number of variational layers (default 2).

    Returns:
        QuantumCircuit with parametric gates and measurements.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    param_idx = 0

    for layer in range(n_layers):
        # RY rotation layer
        for qubit in range(n_qubits):
            theta = Parameter(f"theta_{param_idx}")
            qc.ry(theta, qubit)
            param_idx += 1

        # CX entangling layer (linear chain)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

        # Second RY rotation layer
        for qubit in range(n_qubits):
            theta = Parameter(f"theta_{param_idx}")
            qc.ry(theta, qubit)
            param_idx += 1

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


if __name__ == "__main__":
    qc = create_simple_vqe(n_qubits=4, n_layers=2)
    print("=== Simple VQE Ansatz (4 qubits, 2 layers) ===")
    print(qc.draw(output="text", fold=120))
    print(f"\nDepth: {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
    print(f"Qubits: {qc.num_qubits}")
    print(f"Parameters: {qc.num_parameters}")
