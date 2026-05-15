"""Deterministic benchmark circuit corpus for EMRG."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class BenchmarkCase:
    """One reproducible benchmark case."""

    case_id: str
    family: str
    build: Callable[[], QuantumCircuit]
    stress_target: str
    observable: str = "Z0"
    noise_level: float = 0.01
    noise_model: str = "depolarizing"
    noise_model_available: bool = False
    run_quality_by_default: bool = True
    speed_only: bool = False
    quick: bool = False
    split: str = "train"


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def ghz(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def bernstein_vazirani() -> QuantumCircuit:
    oracle_bits = (1, 0, 1, 1)
    qc = QuantumCircuit(5)
    qc.x(4)
    qc.h(range(5))
    for index, bit in enumerate(oracle_bits):
        if bit:
            qc.cx(index, 4)
    qc.h(range(4))
    qc.measure_all()
    return qc


def deutsch_jozsa() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.x(3)
    qc.h(range(4))
    qc.cx(0, 3)
    qc.cx(2, 3)
    qc.h(range(3))
    qc.measure_all()
    return qc


def qft_style(n_qubits: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for target in range(n_qubits):
        qc.h(target)
        for control in range(target + 1, n_qubits):
            qc.cp(math.pi / (2 ** (control - target)), control, target)
    for q in range(n_qubits // 2):
        qc.swap(q, n_qubits - q - 1)
    qc.measure_all()
    return qc


def qaoa_style(n_qubits: int = 6, layers: int = 2) -> QuantumCircuit:
    rng = np.random.default_rng(20260515)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(layers):
        gamma = float(rng.uniform(0.1, 1.2))
        beta = float(rng.uniform(0.1, 1.2))
        for q in range(n_qubits - 1):
            qc.rzz(gamma, q, q + 1)
        for q in range(n_qubits):
            qc.rx(beta, q)
    qc.measure_all()
    return qc


def vqe_ansatz(n_qubits: int = 4, layers: int = 2) -> QuantumCircuit:
    rng = np.random.default_rng(12345 + n_qubits + layers)
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(0.1, 2.4)), q)
            qc.rz(float(rng.uniform(0.1, 2.4)), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def random_cliffordish(n_qubits: int = 8, layers: int = 4) -> QuantumCircuit:
    rng = random.Random(20260515 + n_qubits + layers)
    qc = QuantumCircuit(n_qubits)
    single_qubit_gates = ("h", "s", "x", "z")
    for _ in range(layers):
        for q in range(n_qubits):
            getattr(qc, rng.choice(single_qubit_gates))(q)
        order = list(range(n_qubits))
        rng.shuffle(order)
        for index in range(0, n_qubits - 1, 2):
            if rng.random() < 0.5:
                qc.cx(order[index], order[index + 1])
            else:
                qc.cz(order[index], order[index + 1])
    qc.measure_all()
    return qc


def rotation_heavy(n_qubits: int = 4, layers: int = 3) -> QuantumCircuit:
    rng = np.random.default_rng(404 + n_qubits + layers)
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            qc.rx(float(rng.uniform(0.13, 1.11)), q)
            qc.ry(float(rng.uniform(0.17, 1.19)), q)
            qc.rz(float(rng.uniform(0.23, 1.31)), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def repeated_cx_cz_layers(n_qubits: int = 6, layers: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits - 1):
            if (q + layer) % 2:
                qc.cz(q, q + 1)
            else:
                qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def heterogeneous_layers(n_qubits: int = 8, layers: int = 6) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        qc.h(0)
        qc.s(0)
        qc.h(0)
        if layer % 3 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            qc.cx(0, 1)
    qc.measure_all()
    return qc


def shallow_pec_eligible() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(math.pi / 5, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


def moderate_cdr_eligible() -> QuantumCircuit:
    return rotation_heavy(4, 3)


def deeper_zne_oriented() -> QuantumCircuit:
    qc = QuantumCircuit(5)
    for _ in range(18):
        for q in range(5):
            qc.h(q)
        for q in range(4):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def composite_candidate() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    for _ in range(4):
        for q in range(4):
            qc.h(q)
            qc.s(q)
        for q in range(3):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def speed_random(n_qubits: int, layers: int) -> QuantumCircuit:
    rng = np.random.default_rng(9000 + n_qubits + layers)
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            gate = int(rng.integers(4))
            if gate == 0:
                qc.h(q)
            elif gate == 1:
                qc.s(q)
            elif gate == 2:
                qc.rx(float(rng.uniform(0.1, 1.3)), q)
            else:
                qc.rz(float(rng.uniform(0.1, 1.3)), q)
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def build_corpus(
    *,
    seed: int = 1234,
    quick: bool = False,
    split: str = "all",
) -> tuple[BenchmarkCase, ...]:
    """Return the fixed internal benchmark corpus.

    The seed is accepted for API symmetry with the runner. Individual circuit
    builders use stable local seeds so case IDs remain reproducible.
    """
    if split not in {"train", "holdout", "all"}:
        raise ValueError("split must be 'train', 'holdout', or 'all'.")

    _ = seed
    cases = (
        BenchmarkCase(
            "bell_2q_zz_p001",
            "bell",
            bell,
            "zne_linear",
            observable="ZZ",
            quick=True,
        ),
        BenchmarkCase(
            "ghz_5q_zz_p001",
            "ghz",
            lambda: ghz(5),
            "zne_linear",
            observable="ZZ",
            quick=True,
            split="holdout",
        ),
        BenchmarkCase(
            "bernstein_vazirani_5q_z0_p001",
            "bernstein_vazirani",
            bernstein_vazirani,
            "zne_linear",
            split="train",
        ),
        BenchmarkCase(
            "deutsch_jozsa_4q_z0_p001",
            "deutsch_jozsa",
            deutsch_jozsa,
            "zne_linear",
            split="holdout",
        ),
        BenchmarkCase(
            "qft_5q_z0_p001",
            "qft",
            qft_style,
            "cdr",
            run_quality_by_default=False,
            split="train",
        ),
        BenchmarkCase(
            "qaoa_6q_2layers_zz_p001",
            "qaoa",
            qaoa_style,
            "cdr",
            observable="ZZ",
            run_quality_by_default=False,
            split="holdout",
        ),
        BenchmarkCase(
            "vqe_4q_2layers_z0_p001",
            "vqe",
            lambda: vqe_ansatz(4, 2),
            "cdr",
            quick=True,
        ),
        BenchmarkCase(
            "random_cliffordish_8q_z0_p001",
            "random_cliffordish",
            random_cliffordish,
            "zne_linear",
            split="holdout",
        ),
        BenchmarkCase(
            "rotation_heavy_4q_z0_p001",
            "rotation_heavy",
            moderate_cdr_eligible,
            "cdr",
            quick=True,
            split="train",
        ),
        BenchmarkCase(
            "cx_cz_layers_6q_z0_p001",
            "cx_cz_layers",
            repeated_cx_cz_layers,
            "zne_richardson",
            split="train",
        ),
        BenchmarkCase(
            "heterogeneous_12q_z0_p001",
            "heterogeneous",
            lambda: heterogeneous_layers(12, 6),
            "layerwise",
            run_quality_by_default=False,
            split="holdout",
        ),
        BenchmarkCase(
            "shallow_pec_3q_zz_p001",
            "pec",
            shallow_pec_eligible,
            "pec",
            observable="ZZ",
            noise_model_available=True,
            quick=True,
            split="train",
        ),
        BenchmarkCase(
            "moderate_cdr_4q_z0_p003",
            "cdr",
            moderate_cdr_eligible,
            "cdr",
            noise_level=0.03,
            split="holdout",
        ),
        BenchmarkCase(
            "deep_zne_5q_z0_p001",
            "deep_zne",
            deeper_zne_oriented,
            "zne_deep",
            run_quality_by_default=False,
            split="train",
        ),
        BenchmarkCase(
            "composite_4q_z0_p001",
            "composite",
            composite_candidate,
            "composite",
            noise_model_available=True,
            run_quality_by_default=False,
            split="holdout",
        ),
        BenchmarkCase(
            "speed_random_20q_6layers",
            "speed",
            lambda: speed_random(20, 6),
            "speed_only",
            speed_only=True,
            run_quality_by_default=False,
            split="train",
        ),
        BenchmarkCase(
            "speed_random_30q_8layers",
            "speed",
            lambda: speed_random(30, 8),
            "speed_only",
            speed_only=True,
            run_quality_by_default=False,
            split="train",
        ),
        BenchmarkCase(
            "speed_random_50q_10layers",
            "speed",
            lambda: speed_random(50, 10),
            "speed_only",
            speed_only=True,
            run_quality_by_default=False,
            split="holdout",
        ),
    )
    selected = cases
    if split != "all":
        selected = tuple(case for case in selected if case.split == split)
    if quick:
        return tuple(case for case in selected if case.quick or case.speed_only)
    return selected
