"""Tests for the benchmark harness and scoring utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from emrg import DEFAULT_POLICY
from emrg.policy import load_policy, policy_to_dict


def _minimal_result(cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "emrg_version": "0.5.0",
        "commit": "test",
        "timestamp_utc": "2026-05-15T00:00:00Z",
        "environment": {},
        "policy": {"path": "test.json", "sha256": "0" * 64, "data": {}},
        "config": {
            "seed": 1234,
            "repeats": 1,
            "quick": True,
            "include_speed": True,
            "include_quality": True,
            "max_qubits": 5,
        },
        "cases": cases,
        "summary": {},
    }


def _case(
    case_id: str,
    *,
    family: str = "unit",
    quality: dict | None = None,
    overhead: float = 3.0,
) -> dict:
    return {
        "case_id": case_id,
        "family": family,
        "num_qubits": 2,
        "depth": 3,
        "total_gate_count": 2,
        "multi_qubit_gate_count": 1,
        "non_clifford_fraction": 0.0,
        "noise_model": "depolarizing",
        "noise_level": 0.01,
        "observable": "Z0",
        "selected_recipe": {
            "technique": "zne",
            "estimated_overhead": overhead,
            "factory_name": "LinearFactory",
        },
        "speed": {
            "analyze_ms_median": 0.1,
            "recommend_ms_median": 0.1,
            "generate_code_ms_median": 0.1,
            "generate_recipe_ms_median": 0.3,
        },
        "quality": quality
        if quality is not None
        else {
            "ideal": 1.0,
            "noisy": 0.8,
            "mitigated": 0.95,
            "noisy_error": 0.2,
            "mitigated_error": 0.05,
            "error_reduction": 4.0,
            "runtime_seconds": 0.01,
            "status": "passed",
            "skip_reason": None,
            "failure": None,
        },
    }


def test_benchmark_corpus_has_unique_case_ids() -> None:
    from benchmarks.circuits import build_corpus

    cases = build_corpus(seed=1234, quick=False)
    case_ids = [case.case_id for case in cases]

    assert len(case_ids) == len(set(case_ids))
    assert {"bell_2q_zz_p001", "vqe_4q_2layers_z0_p001"}.issubset(case_ids)


def test_default_v050_policy_snapshot_matches_default_policy() -> None:
    snapshot = Path("benchmarks/policies/default-v050.json")

    loaded = load_policy(snapshot)

    assert policy_to_dict(loaded) == policy_to_dict(DEFAULT_POLICY)


def test_quick_benchmark_runner_writes_schema_json(tmp_path: Path) -> None:
    output = tmp_path / "quick.json"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_benchmark.py",
            "--quick",
            "--include-speed",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["policy"]["sha256"]
    assert data["config"]["quick"] is True
    assert data["cases"]

    first_case = data["cases"][0]
    for key in (
        "case_id",
        "family",
        "num_qubits",
        "depth",
        "selected_recipe",
        "speed",
        "quality",
    ):
        assert key in first_case


def test_external_qasm_failures_are_recorded_per_file(tmp_path: Path) -> None:
    output = tmp_path / "external.json"
    external_dir = tmp_path / "qasm"
    external_dir.mkdir()
    (external_dir / "bad.qasm").write_text("OPENQASM 2.0; invalid", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_benchmark.py",
            "--quick",
            "--include-speed",
            "--external-qasm-dir",
            str(external_dir),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(output.read_text(encoding="utf-8"))
    external_cases = [
        case for case in data["cases"] if case["family"] == "external_qasm"
    ]
    assert external_cases
    assert external_cases[0]["quality"]["status"] == "failed"
    assert external_cases[0]["quality"]["failure"]


def test_score_results_handles_quality_statuses_and_missing_fields() -> None:
    from benchmarks.score_results import score_document

    doc = _minimal_result(
        [
            _case("passed"),
            _case(
                "failed",
                quality={
                    "ideal": None,
                    "noisy": None,
                    "mitigated": None,
                    "noisy_error": None,
                    "mitigated_error": None,
                    "error_reduction": None,
                    "runtime_seconds": None,
                    "status": "failed",
                    "skip_reason": None,
                    "failure": "boom",
                },
            ),
            _case(
                "skipped",
                quality={
                    "ideal": None,
                    "noisy": None,
                    "mitigated": None,
                    "noisy_error": None,
                    "mitigated_error": None,
                    "error_reduction": None,
                    "runtime_seconds": None,
                    "status": "skipped",
                    "skip_reason": "optional dependency unavailable",
                    "failure": None,
                },
            ),
            _case(
                "worse",
                quality={
                    "ideal": 1.0,
                    "noisy": 0.8,
                    "mitigated": 0.7,
                    "noisy_error": 0.2,
                    "mitigated_error": 0.3,
                    "error_reduction": 0.67,
                    "runtime_seconds": 0.01,
                    "status": "passed",
                    "skip_reason": None,
                    "failure": None,
                },
            ),
            _case("missing", quality={}),
        ]
    )

    scored = score_document(doc)

    per_case = {case["case_id"]: case for case in scored["cases"]}
    assert per_case["failed"]["score"] < per_case["passed"]["score"]
    assert per_case["skipped"]["score"] == 0
    assert per_case["worse"]["score"] < 0
    assert per_case["missing"]["score"] == 0
    assert scored["summary"]["failure_rate"] > 0
    assert "unit" in scored["families"]
