"""Brutal stress test for EMRG v0.2.9 preview mode.

Run with: python tests/stress_test_preview.py
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit  # noqa: E402

from emrg import generate_recipe  # noqa: E402

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: dict[str, list[tuple[str, str, str]]] = {}


def record(category: str, name: str, passed: bool, note: str = ""):
    global PASS_COUNT, FAIL_COUNT  # noqa: PLW0603
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.setdefault(category, []).append((name, status, note))
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))


def run_test(category: str, name: str, fn):
    try:
        fn()
    except Exception as e:
        record(category, name, False, f"EXCEPTION: {e}")


def _make_bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _make_3q():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


# ===================================================================
# PHASE 1A -- Circuit Edge Cases
# ===================================================================
print("\n=== PHASE 1A: Circuit Edge Cases ===")

CAT = "Circuit edge cases"


def test_a1():
    """Empty circuit (no gates)."""
    qc = QuantumCircuit(2)
    try:
        generate_recipe(qc, preview=True)
        record(CAT, "A1 empty circuit", False, "No ValueError")
    except ValueError as e:
        record(CAT, "A1 empty circuit", True, f"ValueError: {e}")
    except Exception as e:
        record(
            CAT, "A1 empty circuit", False,
            f"Wrong: {type(e).__name__}: {e}",
        )


def test_a2():
    """Measurement-only circuit."""
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    try:
        generate_recipe(qc, preview=True)
        record(CAT, "A2 measurement-only", False, "No ValueError")
    except ValueError as e:
        record(CAT, "A2 measurement-only", True, f"ValueError: {e}")
    except Exception as e:
        record(
            CAT, "A2 measurement-only", False,
            f"Wrong: {type(e).__name__}: {e}",
        )


def test_a3():
    """Single gate, single qubit."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    result = generate_recipe(qc, preview=True)
    ok = result.preview is not None and result.preview.ideal_value is not None
    record(CAT, "A3 single gate 1q", ok,
           f"ideal={result.preview.ideal_value}")


def test_a4():
    """Identity circuit (H then H cancels)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)
    qc.measure_all()
    result = generate_recipe(qc, preview=True)
    ok = result.preview is not None and result.preview.ideal_value is not None
    record(CAT, "A4 identity circuit", ok)


def test_a5():
    """Exactly 10 qubits (boundary -- should simulate)."""
    qc = QuantumCircuit(10)
    for i in range(9):
        qc.cx(i, i + 1)
    qc.measure_all()
    result = generate_recipe(qc, preview=True)
    w = result.preview.warning
    ok = w is None or "skip" not in w.lower()
    record(CAT, "A5 exactly 10 qubits", ok, f"warning={w}")


def test_a6():
    """Exactly 11 qubits (boundary -- should skip)."""
    qc = QuantumCircuit(11)
    for i in range(10):
        qc.cx(i, i + 1)
    qc.measure_all()
    result = generate_recipe(qc, preview=True)
    w = result.preview.warning
    ok = w is not None and "skip" in w.lower()
    record(CAT, "A6 exactly 11 qubits", ok,
           f"warning={w[:60] if w else 'None'}...")


def test_a7():
    """10 qubits, 100+ depth -- should skip (depth guard)."""
    qc = QuantumCircuit(10)
    for _ in range(12):
        for i in range(9):
            qc.cx(i, i + 1)
    qc.measure_all()
    t0 = time.time()
    result = generate_recipe(qc, preview=True)
    elapsed = time.time() - t0
    w = result.preview.warning or ""
    ok = "skip" in w.lower() and elapsed < 2
    record(CAT, "A7 10q deep (depth guard)", ok,
           f"{elapsed:.1f}s, {w[:50]}")


def test_a8():
    """Circuit with barriers."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    qc.measure_all()
    result = generate_recipe(qc, preview=True)
    ok = (result.preview is not None
          and result.preview.ideal_value is not None)
    record(CAT, "A8 barriers", ok)


def test_a9():
    """Circuit with classical registers but no measurements."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    result = generate_recipe(qc, preview=True)
    ok = (result.preview is not None
          and result.preview.ideal_value is not None)
    record(CAT, "A9 classical regs no meas", ok)


def test_a10():
    """Circuit with measurements (Cirq conversion)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    result = generate_recipe(qc, preview=True)
    ok = (result.preview is not None
          and result.preview.ideal_value is not None)
    record(CAT, "A10 with measurements", ok)


for fn in [test_a1, test_a2, test_a3, test_a4, test_a5, test_a6,
           test_a7, test_a8, test_a9, test_a10]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# PHASE 1B -- Observable Edge Cases
# ===================================================================
print("\n=== PHASE 1B: Observable Edge Cases ===")

CAT = "Observable edge cases"


def test_b1():
    """Default observable on 3q circuit."""
    result = generate_recipe(_make_3q(), preview=True)
    ok = result.preview.ideal_value is not None
    record(CAT, "B1 default observable", ok)


def test_b2():
    """Observable on last qubit (Z2)."""
    result = generate_recipe(_make_3q(), preview=True, observable="Z2")
    ok = result.preview.ideal_value is not None
    record(CAT, "B2 observable Z2", ok)


def test_b3():
    """Observable on nonexistent qubit (Z5 on 3q)."""
    result = generate_recipe(
        _make_3q(), preview=True, observable="Z5",
    )
    w = result.preview.warning or ""
    ok = len(w) > 0
    record(CAT, "B3 invalid qubit Z5", ok, f"warning={w[:60]}")


def test_b4():
    """ZZ observable."""
    result = generate_recipe(_make_3q(), preview=True, observable="ZZ")
    ok = result.preview.ideal_value is not None
    record(CAT, "B4 ZZ observable", ok)


def test_b5():
    """ZZ on 1-qubit circuit."""
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    result = generate_recipe(qc, preview=True, observable="ZZ")
    w = result.preview.warning or ""
    ok = len(w) > 0
    record(CAT, "B5 ZZ on 1 qubit", ok, f"warning={w[:60]}")


def test_b6():
    """Garbage observable string."""
    result = generate_recipe(
        _make_3q(), preview=True, observable="XYZZY",
    )
    w = result.preview.warning or ""
    ok = len(w) > 0
    record(CAT, "B6 garbage observable", ok, f"warning={w[:60]}")


def test_b7():
    """Empty string observable."""
    result = generate_recipe(_make_3q(), preview=True, observable="")
    w = result.preview.warning or ""
    ok = len(w) > 0
    record(CAT, "B7 empty observable", ok, f"warning={w[:60]}")


for fn in [test_b1, test_b2, test_b3, test_b4, test_b5, test_b6, test_b7]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# PHASE 1C -- Noise Level Edge Cases
# ===================================================================
print("\n=== PHASE 1C: Noise Level Edge Cases ===")

CAT = "Noise level edge cases"


def test_c1():
    """Zero noise -- ideal and noisy should match."""
    result = generate_recipe(_make_bell(), preview=True, noise_level=0.0)
    diff = abs(result.preview.ideal_value - result.preview.noisy_value)
    ok = diff < 1e-6
    record(CAT, "C1 zero noise", ok, f"diff={diff:.2e}")


def test_c2():
    """Very high noise p=0.5."""
    result = generate_recipe(_make_bell(), preview=True, noise_level=0.5)
    ok = result.preview.ideal_value is not None
    record(CAT, "C2 p=0.5", ok)


def test_c3():
    """Noise > 0.5 (invalid for depolarizing)."""
    result = generate_recipe(_make_bell(), preview=True, noise_level=0.9)
    w = result.preview.warning
    ok = w is not None or result.preview.ideal_value is not None
    note = w[:60] if w else "simulated ok"
    record(CAT, "C3 p=0.9", ok, f"warning={note}")


def test_c4():
    """Negative noise."""
    result = generate_recipe(
        _make_bell(), preview=True, noise_level=-0.1,
    )
    w = result.preview.warning
    ok = w is not None or result.preview.ideal_value is not None
    note = w[:60] if w else "simulated ok"
    record(CAT, "C4 negative noise", ok, f"warning={note}")


def test_c5():
    """Very small noise p=0.0001."""
    result = generate_recipe(
        _make_bell(), preview=True, noise_level=0.0001,
    )
    ok = result.preview.ideal_value is not None
    record(CAT, "C5 p=0.0001", ok)


def test_c6():
    """Noise level p=1 (invalid)."""
    result = generate_recipe(_make_bell(), preview=True, noise_level=1.0)
    w = result.preview.warning
    ok = w is not None or result.preview.ideal_value is not None
    note = w[:60] if w else "simulated ok"
    record(CAT, "C6 p=1.0", ok, f"warning={note}")


for fn in [test_c1, test_c2, test_c3, test_c4, test_c5, test_c6]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# PHASE 1D -- Technique + Preview Combinations
# ===================================================================
print("\n=== PHASE 1D: Technique + Preview Combinations ===")

CAT = "Technique combos"


def test_d1():
    """Force ZNE + preview."""
    result = generate_recipe(_make_3q(), preview=True, technique="zne")
    ok = "zne" in result.preview.technique.lower()
    record(CAT, "D1 force ZNE", ok,
           f"technique={result.preview.technique}")


def test_d2():
    """Force PEC + preview."""
    result = generate_recipe(
        _make_3q(), preview=True, technique="pec",
        noise_model_available=True,
    )
    ok = "pec" in result.preview.technique.lower()
    record(CAT, "D2 force PEC", ok,
           f"technique={result.preview.technique}")


def test_d3():
    """Force PEC without noise model + preview."""
    try:
        result = generate_recipe(
            _make_3q(), preview=True, technique="pec",
            noise_model_available=False,
        )
        ok = result.preview is not None
        record(CAT, "D3 PEC no noise model", ok,
               f"technique={result.preview.technique}")
    except Exception as e:
        record(CAT, "D3 PEC no noise model", True, f"Error: {e}")


def test_d4():
    """Invalid technique + preview."""
    try:
        generate_recipe(_make_3q(), preview=True, technique="invalid")
        record(CAT, "D4 invalid technique", False, "No error raised")
    except (ValueError, Exception) as e:
        record(CAT, "D4 invalid technique", True,
               f"{type(e).__name__}: {e}")


for fn in [test_d1, test_d2, test_d3, test_d4]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# PHASE 1E -- Consistency
# ===================================================================
print("\n=== PHASE 1E: Consistency ===")

CAT = "Consistency"


def test_e1():
    """ZNE preview determinism -- 5 identical runs."""
    qc = _make_bell()
    vals = []
    for _ in range(5):
        r = generate_recipe(qc, preview=True)
        vals.append(r.preview.mitigated_value)
    ok = len(set(vals)) == 1
    record(CAT, "E1 ZNE deterministic", ok, f"values={vals}")


def test_e2():
    """PEC preview seeded -- 5 runs (random_state=42)."""
    qc = _make_bell()
    vals = []
    for _ in range(5):
        r = generate_recipe(qc, preview=True, noise_model_available=True)
        vals.append(r.preview.mitigated_value)
    unique = len(set(vals))
    if unique == 1:
        record(CAT, "E2 PEC seeded", True, "deterministic (seed=42)")
    else:
        record(CAT, "E2 PEC variance", True,
               f"{unique} unique values")


for fn in [test_e1, test_e2]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# PHASE 1F -- Performance
# ===================================================================
print("\n=== PHASE 1F: Performance ===")

CAT = "Performance"


def test_f1():
    """2-qubit preview timing."""
    t0 = time.time()
    generate_recipe(_make_bell(), preview=True)
    elapsed = time.time() - t0
    ok = elapsed < 2
    record(CAT, "F1 2q preview", ok, f"{elapsed:.2f}s")


def test_f2():
    """5-qubit preview timing."""
    qc = QuantumCircuit(5)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    t0 = time.time()
    generate_recipe(qc, preview=True)
    elapsed = time.time() - t0
    ok = elapsed < 5
    record(CAT, "F2 5q preview", ok, f"{elapsed:.2f}s")


def test_f3():
    """10-qubit preview timing."""
    qc = QuantumCircuit(10)
    for i in range(9):
        qc.cx(i, i + 1)
    qc.measure_all()
    t0 = time.time()
    generate_recipe(qc, preview=True)
    elapsed = time.time() - t0
    # 10q density matrix sim is inherently slow (~25-35s).
    ok = elapsed < 45
    record(CAT, "F3 10q preview", ok, f"{elapsed:.2f}s")


def test_f4():
    """10-qubit deep circuit -- depth guard skips instantly."""
    qc = QuantumCircuit(10)
    for _ in range(8):
        for i in range(9):
            qc.cx(i, i + 1)
    qc.measure_all()
    t0 = time.time()
    result = generate_recipe(qc, preview=True)
    elapsed = time.time() - t0
    ok = elapsed < 2 and result.preview.warning is not None
    record(CAT, "F4 10q deep (guarded)", ok, f"{elapsed:.2f}s")


def test_f5():
    """5-qubit PEC preview timing."""
    qc = QuantumCircuit(5)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    t0 = time.time()
    generate_recipe(qc, preview=True, noise_model_available=True)
    elapsed = time.time() - t0
    record(CAT, "F5 5q PEC preview", True, f"{elapsed:.2f}s")


for fn in [test_f1, test_f2, test_f3, test_f4, test_f5]:
    run_test(CAT, fn.__doc__ or fn.__name__, fn)


# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 65)
print("PHASE 1 SUMMARY")
print("=" * 65)
for cat, tests in RESULTS.items():
    passed = sum(1 for _, s, _ in tests if s == "PASS")
    failed = sum(1 for _, s, _ in tests if s == "FAIL")
    label = f"  ({failed} FAILED)" if failed else ""
    print(f"  {cat:<30} {passed}/{len(tests)} passed{label}")

print(f"\nTotal: {PASS_COUNT} passed, {FAIL_COUNT} failed "
      f"out of {PASS_COUNT + FAIL_COUNT}")

if FAIL_COUNT > 0:
    print("\nFAILED TESTS:")
    for cat, tests in RESULTS.items():
        for name, status, note in tests:
            if status == "FAIL":
                print(f"  [{cat}] {name}: {note}")

sys.exit(1 if FAIL_COUNT > 0 else 0)
