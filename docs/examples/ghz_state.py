"""GHZ State Circuit -- 3-qubit maximally entangled state.

A slightly deeper circuit than Bell state, with a chain of CX gates.
Expected profile:
- Depth: ~5 (with measurements)
- Multi-qubit gates: 2 CX
- Noise estimate: Low-moderate
- Expected EMRG recommendation: LinearFactory or RichardsonFactory
"""

from qiskit import QuantumCircuit


def create_ghz_state(n_qubits: int = 3) -> QuantumCircuit:
    """Create an n-qubit GHZ state circuit (|00...0> + |11...1>) / sqrt(2).

    Args:
        n_qubits: Number of qubits (default 3).

    Returns:
        QuantumCircuit with GHZ state preparation and measurements.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


if __name__ == "__main__":
    qc = create_ghz_state(3)
    print("=== GHZ State Circuit (3 qubits) ===")
    print(qc.draw(output="text"))
    print(f"\nDepth: {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
    print(f"Qubits: {qc.num_qubits}")

    # Also show a larger GHZ
    qc5 = create_ghz_state(5)
    print("\n=== GHZ State Circuit (5 qubits) ===")
    print(qc5.draw(output="text"))
    print(f"\nDepth: {qc5.depth()}")
    print(f"Gate counts: {qc5.count_ops()}")
    print(f"Qubits: {qc5.num_qubits}")
