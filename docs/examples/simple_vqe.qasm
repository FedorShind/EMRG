// Simple VQE-like Ansatz -- 4-qubit, 2-layer variational circuit.
// Parameters bound to pi/4 for QASM compatibility.
// See simple_vqe.py for the parametric (unbound) version.
// Expected EMRG recommendation: LinearFactory or RichardsonFactory
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(pi/4) q[0];
ry(pi/4) q[1];
ry(pi/4) q[2];
ry(pi/4) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
ry(pi/4) q[0];
ry(pi/4) q[1];
ry(pi/4) q[2];
ry(pi/4) q[3];
ry(pi/4) q[0];
ry(pi/4) q[1];
ry(pi/4) q[2];
ry(pi/4) q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
ry(pi/4) q[0];
ry(pi/4) q[1];
ry(pi/4) q[2];
ry(pi/4) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
