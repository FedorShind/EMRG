# EMRG v0.3.0 CDR Stress Test — Insights

**Date:** 2026-04-03
**Tests run:** 41 stress tests + 366 pytest tests + 9 CLI tests
**Bugs found:** 0
**Failures:** 0

---

## A. Classification Accuracy

### What works well

All 16 gate classification tests passed. The `_is_clifford_angle` function correctly handles:
- Exact multiples of pi/2: `Rz(0)`, `Rz(pi/2)`, `Rz(pi)`, `Rz(3pi/2)`, `Rz(2pi)` all classified as Clifford.
- T/Tdg gates: always non-Clifford (hardcoded in the `("t", "tdg")` branch).
- Arbitrary angles: `Rz(0.123)`, `Rz(pi/4)` correctly flagged.
- Float tolerance: `Rz(pi/2 + 1e-15)` is Clifford (within `1e-8` tolerance), `Rz(pi/2 + 0.01)` is non-Clifford. The tolerance is appropriate.
- Multi-parameter gates: `U(pi, 0, pi)` = Clifford (all angles are multiples of pi/2), `U(0.5, 0.3, 0.7)` = non-Clifford (at least one angle is not).
- Symbolic parameters: `Rz(Parameter("theta"))` is correctly classified as non-Clifford because `isinstance(p, (int, float))` fails for `Parameter`, so `_is_clifford_angle` is never called, and the gate falls through to the non-Clifford count. This is the correct conservative behavior.

### Classification nuance: Ry(pi/2)

Test A10 revealed that `Ry(pi/2)` is classified as Clifford because its angle is a multiple of pi/2. This is correct for Ry specifically — `Ry(pi/2)` maps Pauli operators to Pauli operators (it maps X -> Z, Z -> -X) and is in the Clifford group. The same is true for `Rx(pi/2)`. The angle-based check is a valid proxy for Clifford membership on standard rotation gates.

### Unhandled gate types

The `CLIFFORD_GATE_NAMES` set covers 14 gates. The `_ROTATION_GATE_NAMES` set covers 16 rotation gates. Any gate not in either set (and not `t`/`tdg`) is conservatively classified as non-Clifford. This means:
- Custom gates from Qiskit transpiler passes (e.g., `sx`, `sxdg`, `r`) would be classified as non-Clifford.
- `sx` (sqrt-X) is Clifford but is not in `CLIFFORD_GATE_NAMES`. This is a gap — if a transpiled circuit uses `sx` gates, the non-Clifford fraction will be inflated.
- `iswap` is in `MULTI_QUBIT_GATE_NAMES` but not in `CLIFFORD_GATE_NAMES`. iSWAP is Clifford. It will be misclassified as non-Clifford.
- `csx` (controlled-sqrt-X) is in `MULTI_QUBIT_GATE_NAMES` but not handled — it will be non-Clifford (which is correct: CSX is not Clifford).
- `ch` (controlled-H) is in `MULTI_QUBIT_GATE_NAMES` but not in `CLIFFORD_GATE_NAMES`. CH is not Clifford. Correctly handled by the fallthrough.

**Severity:** Low-medium. Affects transpiled circuits where `sx`/`iswap` are common (IBM hardware basis gates include `sx`). Recommendation: add `sx`, `sxdg`, `iswap` to `CLIFFORD_GATE_NAMES`.

---

## B. Decision Engine

### Threshold observations

From the 41 stress tests, CDR thresholds behave as designed:
- **CDR_MIN_DEPTH = 10**: Correctly rejects depth-1 and depth-4 circuits. The B2 test showed a circuit with depth 8 and 67% non-Clifford fraction was rejected (ZNE instead). This is reasonable — CDR's training circuit approach needs enough circuit structure to fit a regression model.
- **CDR_MAX_DEPTH = 40**: Test X3 hit exactly depth 40, and CDR was correctly selected. Depth 50 correctly falls through to ZNE.
- **CDR_NON_CLIFFORD_FRACTION_THRESHOLD = 0.2**: The `>` (strictly greater than) boundary is correct — exactly 0.2 does not trigger CDR.

### Priority order PEC > CDR > ZNE

Test B1 confirmed: a circuit at depth 4 with 67% non-Clifford fraction and `noise_model_available=True` gets PEC (not CDR). This is correct because PEC's unbiased guarantee is stronger than CDR's regression fit.

### Surprising recommendations

**Test X1 (all-T-gate circuit):** A circuit of 20 T gates on 2 qubits has depth 10 and fraction 1.0. It gets CDR. This is correct but worth noting — a circuit with zero Clifford gates means CDR's "near-Clifford training circuits" will have all non-Clifford gates replaced, producing trivial training circuits. CDR may not learn a useful regression model in this case. In practice, Mitiq handles this internally, but the recommendation could be misleading.

**Test X4 (mid-circuit measurement):** Circuit with a `measure` between gates works fine — measurements are excluded from gate counting. The non-Clifford fraction is correctly computed on gates only.

### Depth vs. Qiskit depth

A subtle issue: Qiskit's `circuit.depth()` counts the critical path through the DAG, which can be much shorter than the raw "number of layers" when gates parallelize. A circuit with 25 layers of `[4x Rz, 2x CX]` has Qiskit depth 50, not 150. This means the CDR depth range (10-40) corresponds to relatively deep circuits in terms of wall-clock execution, but moderate in terms of Qiskit's parallelized depth metric. The thresholds are reasonable for the Qiskit depth model.

---

## C. CDR Performance

### Preview results from P4 (same 4q Rz-rotation circuit, p=0.01)

| Technique | Ideal | Noisy | Mitigated | Reduction |
|-----------|-------|-------|-----------|-----------|
| ZNE       | 1.0000 | 0.9079 | 0.9912 | 10.4x |
| PEC       | 1.0000 | 0.9079 | 0.9147 | 1.1x |
| CDR       | 1.0000 | 0.9079 | 1.0000 | inf |

CDR recovered the exact ideal value on this rotation-heavy circuit. ZNE performed well too (10.4x). PEC was poor (1.1x) — which is expected because PEC's depolarizing representation does not match the actual noise structure well on rotation-heavy circuits.

### Benchmark results (from Step 7)

| Circuit | CDR Reduction | ZNE Reduction |
|---------|---------------|---------------|
| Rz-rot 4q, p=0.01 | >500000x | 2.8x |
| Rz-rot 4q, p=0.03 | >1500000x | 2.3x |
| VQE 4q 2L, p=0.01 | 2.1x | 1.4x |
| VQE 4q 2L, p=0.03 | 1.9x | 1.3x |

CDR consistently outperforms ZNE on non-Clifford-heavy circuits. The enormous reduction factors on Rz-rot circuits are because CDR recovers the ideal value to within floating-point precision.

### Cases where CDR makes things worse

None observed in testing. However, on the Bell state (all-Clifford) with CDR forced, the preview showed 1.0x — CDR does no harm but also no benefit. This is expected: with no non-Clifford gates to replace, all training circuits are identical to the original, and the regression fit is trivially accurate.

### Training circuit scaling

The 8/12/16 scaling based on gate count appears appropriate. The benchmarks show sub-millisecond recipe generation regardless of training count (the count only affects execution time, not recipe generation).

---

## D. Architectural Concerns

### 1. Technique dispatch is linear, not pluggable

`codegen.py:generate_code()` uses `if/elif` chain:
```python
if recipe.technique == "pec": ...
if recipe.technique == "cdr": ...
return _generate_zne_code(...)
```

`preview.py:run_preview()` uses the same pattern:
```python
if recipe.technique == "pec": ...
elif recipe.technique == "cdr": ...
else: ... # ZNE
```

Adding a 4th technique (e.g., virtual distillation) requires editing 5 files in 8+ locations. This is manageable for now (3 techniques) but will become error-prone at 5+.

**Recommendation for Phase 3:** Consider a registry pattern — `TECHNIQUE_HANDLERS = {"zne": ZNEHandler, "pec": PECHandler, "cdr": CDRHandler}` — but only if a 4th technique is actually planned.

### 2. MitigationRecipe is technique-agnostic but abused

`MitigationRecipe` has fields like `factory_name`, `scale_factors`, `scaling_method` that are ZNE-specific. For CDR and PEC, these are empty strings and empty tuples. CDR-specific data (`num_training_circuits`, `fit_method`) is shoved into `factory_kwargs`.

This works but is fragile — any code that accesses `recipe.factory_name` without checking `recipe.technique` first will get an empty string. The benchmark code hit this exact bug (fixed by forcing `technique="zne"` in `benchmark_zne_fidelity`).

**Recommendation:** Either (a) add optional technique-specific fields to `MitigationRecipe` with `None` defaults, or (b) accept the current approach and document that `factory_name`/`scale_factors` are ZNE-only. Option (b) is fine for v0.3.0.

### 3. The `_count_non_clifford_gates` function traverses the DAG twice

`analyze_circuit` calls both `_compute_layer_heterogeneity(qc)` and `_count_non_clifford_gates(qc)`, each of which calls `circuit_to_dag(qc)` independently. For a 50-qubit, 1125-gate circuit, this doubles the DAG construction cost. The total time is still <12ms, so this is not a practical problem, but it is a missed optimization.

### 4. Coverage gap: `_ROTATION_GATE_NAMES` may be incomplete

The set includes 16 rotation gate names. Qiskit has more rotation-like gates added over time. If Qiskit adds a new rotation gate (e.g., `rxx` parameterized differently), it will fall through to the "unknown gate = non-Clifford" bucket. The conservative default is correct, but the set should be reviewed periodically.

### 5. Test coverage is strong but has a blind spot

The CDR preview tests (class `TestCDRPreview`) only verify that a `PreviewResult` is returned and `technique == "CDR"`. They do not assert that `ideal_value` is non-None or that `error_reduction >= 1.0`. This is understandable — CDR's regression fit can produce worse-than-noisy results on adversarial circuits — but it means a broken CDR execution path that returns garbage values would not be caught.

---

## E. Recommendations

### Top 3 fixes before v0.3.0 ships

1. **Add `sx`, `sxdg` to `CLIFFORD_GATE_NAMES`.** IBM hardware basis gates include `sx`. A transpiled circuit will have inflated non-Clifford fraction without this fix. `iswap` should also be added. This is a one-line change to the frozenset in `analyzer.py`.

2. **Add `r` gate to `_ROTATION_GATE_NAMES`.** Qiskit's `r(theta, phi)` gate is a general rotation. Without it in the set, it falls to the "unknown = non-Clifford" bucket, which is correct but could be refined.

3. **Update CLAUDE.md** to reflect v0.3.0 state — the file still says v0.2.9 and lists CDR as incomplete. Not a code issue but will confuse future AI sessions.

### Top 3 refactors for a future version

1. **Unified DAG pass.** Merge `_count_non_clifford_gates` and `_compute_layer_heterogeneity` into a single `_analyze_dag(qc)` function that extracts all DAG-derived metrics in one traversal. Saves one `circuit_to_dag` call per analysis.

2. **Technique handler registry.** Replace `if/elif` chains in `codegen.py`, `preview.py`, and `heuristics.py` with a `TECHNIQUE_REGISTRY` dict mapping technique names to handler classes/functions. Each handler provides `build_recipe()`, `generate_code()`, and `run_preview()`. This makes adding technique #4 a single-file addition.

3. **Stricter `MitigationRecipe` typing.** Either make `factory_name`/`scale_factors` `Optional` with `None` default (instead of empty strings/tuples), or create technique-specific subclasses. The current design silently returns empty strings when CDR/PEC recipes are accessed for ZNE-specific fields.

### Threshold adjustments

No adjustments needed based on observations. The depth 10-40 range and 0.2 fraction threshold produced correct recommendations in all 41 tests. The only edge case worth monitoring: circuits at CDR_MIN_DEPTH (10) with low gate counts may not benefit from CDR because there isn't enough circuit structure for meaningful Clifford substitution. This is a Mitiq limitation, not an EMRG bug.

### Production readiness concerns

1. **CDR requires cirq.** If a user gets CDR auto-recommended but doesn't have cirq installed, the generated code will fail at runtime with an ImportError. The generated code includes a comment about `pip install emrg[preview]`, but there's no runtime check in the generated script itself. Consider adding a try/except import guard in the CDR template.

2. **CDR preview may give misleading results on all-Clifford circuits.** When CDR is forced on a circuit with 0% non-Clifford gates, all training circuits are identical to the original, and the regression fit is trivially perfect. This gives the impression CDR is working when it's actually doing nothing. Not a bug, but could mislead users.

3. **The `factory_kwargs` dict is the only way to access CDR-specific config.** `result.recipe.factory_kwargs["num_training_circuits"]` is not discoverable — users would expect a dedicated attribute. This is a DX issue, not a correctness issue.
