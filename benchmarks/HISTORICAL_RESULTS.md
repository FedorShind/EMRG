# Historical Benchmark Results

These tables were originally kept in the main README as an EMRG v0.3.0
reference snapshot. They are preserved here so the README can stay short.

Environment: Python 3.12, Windows 11. Qiskit 2.3.0, Mitiq 0.48.1.

## Tool Performance

`generate_recipe()` used pure Qiskit introspection in this snapshot, so it
completed quickly even for large circuits. Median of 100 runs:

| Circuit | Qubits | Depth | Gates | Multi-Q | Het | Technique / Config | Time | Memory |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Bell state | 2 | 3 | 2 | 1 | 0.00 | `LinearFactory` + fold_global | 0.09 ms | 9.4 KB |
| Bell state (PEC) | 2 | 3 | 2 | 1 | 0.00 | PEC | 0.09 ms | 9.4 KB |
| GHZ-5 | 5 | 6 | 5 | 4 | 0.50 | `LinearFactory` + fold_global | 0.14 ms | 15.2 KB |
| GHZ-10 | 10 | 11 | 10 | 9 | 0.50 | `LinearFactory` + fold_global | 0.24 ms | 24.9 KB |
| Random 10q, 3 layers | 10 | 7 | 45 | 15 | 0.83 | `LinearFactory` + fold_global | 0.39 ms | 21.2 KB |
| VQE 10q, 4 layers | 10 | 20 | 76 | 36 | 1.50 | CDR (16 training) | 0.64 ms | 43.8 KB |
| Hetero 4q, 8 layers | 4 | 17 | 42 | 10 | 1.00 | CDR (12 training) | 0.40 ms | 34.6 KB |
| T-gate 4q | 4 | 7 | 12 | 3 | 0.50 | `LinearFactory` + fold_global | 0.15 ms | 16.7 KB |
| Rz-rot 4q, 4 layers | 4 | 14 | 28 | 12 | 0.50 | CDR (12 training) | 0.29 ms | 29.3 KB |
| Random 20q, 6 layers | 20 | 13 | 180 | 60 | 0.91 | CDR (16 training) | 1.06 ms | 49.0 KB |
| Random 30q, 10 layers | 30 | 21 | 450 | 150 | 0.94 | CDR (16 training) | 2.41 ms | 116.3 KB |
| Random 50q, 15 layers | 50 | 31 | 1125 | 375 | 0.96 | CDR (16 training) | 5.81 ms | 282.0 KB |

In this historical snapshot, a 50-qubit, 1125-gate circuit was analyzed and
produced a full mitigation recipe in under 6 ms. Several non-Clifford rotation
circuits were routed to CDR under the older default policy.

## ZNE Fidelity

End-to-end ZNE on Cirq `DensityMatrixSimulator` with per-gate depolarizing
noise, comparing `<Z>` on qubit 0:

| Circuit | Qubits | Depth | Noise | Technique / Config | Ideal | Noisy | Mitigated | Error Reduction |
|---|---:|---:|---|---|---:|---:|---:|---:|
| X-flip, 2q | 2 | 3 | p=0.01 | `LinearFactory` + fold_global | -1.0000 | -0.9761 | -1.0003 | 77x |
| X-flip, 3q | 3 | 4 | p=0.01 | `LinearFactory` + fold_global | -1.0000 | -0.9761 | -1.0003 | 77x |
| X-flip, 2q | 2 | 3 | p=0.05 | `LinearFactory` + fold_global | -1.0000 | -0.8836 | -0.9906 | 12x |
| X-flip, 3q | 3 | 4 | p=0.05 | `LinearFactory` + fold_global | -1.0000 | -0.8836 | -0.9906 | 12x |
| VQE 4q, 2 layers | 4 | 8 | p=0.01 | `LinearFactory` + fold_global | 0.0850 | 0.0775 | 0.0794 | 1.4x |
| VQE 4q, 4 layers | 4 | 14 | p=0.01 | `LinearFactory` + fold_global | -0.1915 | -0.1766 | -0.1850 | 2.3x |
| VQE 4q, 2 layers | 4 | 8 | p=0.05 | `LinearFactory` + fold_global | 0.0850 | 0.0523 | 0.0586 | 1.2x |

## PEC vs ZNE

Same circuits, same noise, both techniques. PEC used 1000 samples for benchmark
accuracy in this snapshot. Its results have variance from stochastic sampling.
ZNE is deterministic.

Single-qubit observable `<Z>`:

| Circuit | Noise | ZNE Error | ZNE Reduction | PEC Error | PEC Reduction | Better |
|---|---|---:|---:|---:|---:|---|
| VQE 4q, 2 layers | p=0.01 | 0.0055 | 1.4x | 0.0007 | 10.4x | PEC |
| VQE 4q, 2 layers | p=0.03 | 0.0162 | 1.3x | 0.0138 | 1.5x | PEC |
| VQE 4q, 2 layers | p=0.05 | 0.0264 | 1.2x | 0.0176 | 1.9x | PEC |
| X-flip, 3q | p=0.03 | 0.0024 | 28.9x | 0.0245 | 2.9x | ZNE |

Multi-qubit observable `<ZZ>`:

| Circuit | Noise | ZNE Error | ZNE Reduction | PEC Error | PEC Reduction | Better |
|---|---|---:|---:|---:|---:|---|
| VQE 4q, 2 layers | p=0.01 | 0.0021 | 5.7x | 0.0064 | 1.9x | ZNE |
| VQE 4q, 2 layers | p=0.03 | 0.0102 | 3.4x | 0.0173 | 2.0x | ZNE |
| VQE 4q, 2 layers | p=0.05 | 0.0216 | 2.5x | 0.0147 | 3.6x | PEC |

ZNE did well on structured circuits where noise scaled predictably with folding
in this snapshot. PEC did better on some irregular circuits at higher noise.

## Layerwise Folding

`fold_gates_at_random` targets gates instead of folding the whole circuit
uniformly. The v0.3.0 snapshot showed mixed results:

| Circuit | Qubits | Depth | Het | Noise | Global | Layerwise | Winner |
|---|---:|---:|---:|---|---:|---:|---|
| VQE 10q, 3 reps | 10 | 13 | 2.50 | p=0.01 | 0.9x | 12.6x | layerwise |
| VQE 10q, 3 reps | 10 | 13 | 2.50 | p=0.03 | 1.1x | 1.1x | -- |
| QAOA 10q | 10 | 14 | 2.50 | p=0.01 | 4.2x | 0.2x | global |
| QAOA 10q | 10 | 14 | 2.50 | p=0.03 | 5.9x | 0.7x | global |
| Extreme 10q | 10 | 13 | 2.50 | p=0.01 | 0.5x | 0.4x | -- |
| Extreme 10q | 10 | 13 | 2.50 | p=0.03 | 0.5x | 0.1x | global |

## CDR vs ZNE

CDR replaces non-Clifford gates with Clifford substitutes to create classically
simulable training circuits, then fits a regression model to correct the noisy
result. Compared to ZNE on circuits with non-Clifford gates:

| Circuit | Noise | ZNE Error | ZNE Reduction | CDR Error | CDR Reduction | Better |
|---|---|---:|---:|---:|---:|---|
| Rz-rot 4q | p=0.01 | 0.0253 | 2.8x | ~0.0000 | >1000x | CDR |
| Rz-rot 4q | p=0.03 | 0.0866 | 2.3x | ~0.0000 | >1000x | CDR |
| VQE 4q, 2 layers | p=0.01 | 0.0055 | 1.4x | 0.0036 | 2.1x | CDR |
| VQE 4q, 2 layers | p=0.03 | 0.0162 | 1.3x | 0.0108 | 1.9x | CDR |

This historical snapshot favored CDR on the listed rotation-heavy and VQE
circuits. Current policy behavior is defined by `src/emrg/policy.py` and the
policy snapshot files under `benchmarks/policies/`.
