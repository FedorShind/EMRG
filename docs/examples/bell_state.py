"""Bell State Circuit -- 2-qubit entangled pair.

This is one of the simplest quantum circuits and serves as a baseline
test case for EMRG's circuit analyzer. Expected profile:
- Depth: ~4 (with barriers/measurements)
- Multi-qubit gates: 1 CX
- Noise estimate: Low
- Expected EMRG recommendation: LinearFactory with conservative scales
"""

from qiskit import QuantumCircuit


def create_bell_state() -> QuantumCircuit:
    """Create a 2-qubit Bell state circuit (|00> + |11>) / sqrt(2)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


if __name__ == "__main__":
    qc = create_bell_state()
    print("=== Bell State Circuit ===")
    print(qc.draw(output="text"))
    print(f"\nDepth: {qc.depth()}")
    print(f"Gate counts: {qc.count_ops()}")
    print(f"Qubits: {qc.num_qubits}")
