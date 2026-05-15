# EMRG Benchmarks

EMRG benchmarks are for reproducibility first. They measure the current
heuristics against a fixed internal corpus and produce machine-readable JSON
that can be scored, compared, and rerun on another machine.

## Philosophy

Benchmark results should not tune the policy by accident. Run the baseline
first, save the JSON, then compare any candidate policy against that baseline.
Treat the internal corpus as a calibration set. Any policy change that looks
better there still needs holdout validation before it becomes a release claim.

The runner records skipped cases explicitly. A skipped simulation is not a pass.
External datasets are optional and stay outside the repository.

## Quick Run

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py --quick --output benchmarks\results\quick.json
.\.venv\Scripts\python.exe benchmarks\score_results.py benchmarks\results\quick.json
```

Use the quick run for smoke checks. It includes small quality cases and
larger speed-only cases, so it verifies the JSON schema without turning normal
development into a long simulation run.

## Full Baseline

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py --policy benchmarks\policies\default-v050.json --output benchmarks\results\baseline-v050.json
.\.venv\Scripts\python.exe benchmarks\score_results.py benchmarks\results\baseline-v050.json
```

The baseline output includes:

- package and platform versions
- git commit
- policy path, SHA-256, and policy data
- per-case speed timings
- per-case quality status and values
- selected recipe data from `MitigationRecipe.to_dict()`
- aggregate summary fields

## Candidate Comparison

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py --policy path\to\candidate.json --output benchmarks\results\candidate.json
.\.venv\Scripts\python.exe benchmarks\score_results.py benchmarks\results\candidate.json --baseline benchmarks\results\baseline-v050.json
```

Do not treat a higher calibration score as release proof. A candidate policy
needs a holdout run before changing `DEFAULT_POLICY` or updating public
performance claims.

## External QASM

External circuits are optional and speed-only by default:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py --external-qasm-dir path\to\qasm --include-speed --output benchmarks\results\external.json
```

Recommended external sources:

- MQT Bench for generated benchmark coverage across algorithm families and
  qubit counts.
- QASMBench for OpenQASM kernels and common algorithms.
- SupermarQ for scalable application-level benchmarks.
- VeriQBench for additional OpenQASM circuit coverage.

Do not commit external datasets or generated result JSON files. If an external
QASM file fails to parse, the runner records that file as a failed case and
continues.

## Interpreting Results

Quality scores reward smaller mitigated error, but they cap the benefit from
one extreme win. Failures are penalized heavily. Skipped quality cases are
reported separately because missing optional dependencies or intentionally
speed-only cases should not look like successful mitigation.

Stochastic methods can vary between runs. Use `--seed` and `--repeats` for
reproducibility, and compare candidates on both aggregate score and per-family
behavior.

## Limits

The default runner avoids expensive density-matrix simulations. Larger circuits
are still useful for analysis, recommendation, and code-generation timings, but
they are not proof of mitigation quality. Real hardware benchmarks remain a
separate release task.
