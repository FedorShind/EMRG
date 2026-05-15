"""Reproducible benchmark runner for EMRG."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.circuits import BenchmarkCase, build_corpus
from emrg import __version__, generate_recipe
from emrg.analyzer import CircuitFeatures, analyze_circuit
from emrg.cli import _load_circuit
from emrg.codegen import generate_code
from emrg.heuristics import MitigationRecipe, recommend
from emrg.policy import DEFAULT_POLICY, RecipePolicy, load_policy, policy_to_dict

DEFAULT_POLICY_PATH = Path("benchmarks/policies/default-v050.json")

warnings.filterwarnings("ignore", category=UserWarning, module=r"mitiq\..*")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMRG benchmarks.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-quality", action="store_true")
    parser.add_argument("--include-speed", action="store_true")
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--external-qasm-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_qubits <= 0:
        parser.error("--max-qubits must be positive")
    if not args.include_quality and not args.include_speed:
        args.include_quality = True
        args.include_speed = True
    return args


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": _package_version("qiskit"),
        "mitiq": _package_version("mitiq"),
        "cirq": _package_version("cirq"),
        "numpy": _package_version("numpy"),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _console_safe(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_policy(path: Path) -> RecipePolicy:
    if path.exists():
        return load_policy(path)
    if path == DEFAULT_POLICY_PATH:
        return DEFAULT_POLICY
    return load_policy(path)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    if math.isinf(value):
        return 1e12 if value > 0 else -1e12
    return value


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _time_call(callable_obj, repeats: int) -> float:
    times: list[float] = []
    callable_obj()
    for _ in range(repeats):
        started = time.perf_counter()
        callable_obj()
        times.append((time.perf_counter() - started) * 1000)
    return float(statistics.median(times))


def _empty_speed() -> dict[str, float | None]:
    return {
        "analyze_ms_median": None,
        "recommend_ms_median": None,
        "generate_code_ms_median": None,
        "generate_recipe_ms_median": None,
    }


def _skipped_quality(reason: str) -> dict[str, Any]:
    return {
        "ideal": None,
        "noisy": None,
        "mitigated": None,
        "noisy_error": None,
        "mitigated_error": None,
        "error_reduction": None,
        "runtime_seconds": None,
        "status": "skipped",
        "skip_reason": reason,
        "failure": None,
    }


def _failed_quality(message: str) -> dict[str, Any]:
    return {
        "ideal": None,
        "noisy": None,
        "mitigated": None,
        "noisy_error": None,
        "mitigated_error": None,
        "error_reduction": None,
        "runtime_seconds": None,
        "status": "failed",
        "skip_reason": None,
        "failure": message,
    }


def _features_to_case_base(
    case: BenchmarkCase,
    features: CircuitFeatures,
    recipe: MitigationRecipe,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "num_qubits": features.num_qubits,
        "depth": features.depth,
        "total_gate_count": features.total_gate_count,
        "multi_qubit_gate_count": features.multi_qubit_gate_count,
        "non_clifford_fraction": features.non_clifford_fraction,
        "noise_model": case.noise_model,
        "noise_level": case.noise_level,
        "observable": case.observable,
        "selected_recipe": recipe.to_dict(),
    }


def _measure_speed(
    circuit,
    features: CircuitFeatures,
    recipe: MitigationRecipe,
    policy: RecipePolicy,
    case: BenchmarkCase,
    repeats: int,
) -> dict[str, float | None]:
    return {
        "analyze_ms_median": _time_call(
            lambda: analyze_circuit(
                circuit, noise_model_available=case.noise_model_available
            ),
            repeats,
        ),
        "recommend_ms_median": _time_call(
            lambda: recommend(features, policy=policy),
            repeats,
        ),
        "generate_code_ms_median": _time_call(
            lambda: generate_code(recipe, features),
            repeats,
        ),
        "generate_recipe_ms_median": _time_call(
            lambda: generate_recipe(
                circuit,
                noise_model_available=case.noise_model_available,
                policy=policy,
            ),
            repeats,
        ),
    }


def _quality_status_from_warning(warning: str | None) -> str:
    if warning is None:
        return "passed"
    lower = warning.lower()
    if lower.startswith("preview skipped") or "not installed" in lower:
        return "skipped"
    if "requires cirq" in lower or "no module named" in lower:
        return "skipped"
    return "failed"


def _measure_quality_once(
    circuit,
    recipe: MitigationRecipe,
    case: BenchmarkCase,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    from emrg.preview import run_preview

    started = time.perf_counter()
    try:
        result = run_preview(
            circuit,
            recipe,
            noise_level=case.noise_level,
            observable=case.observable,
        )
    except ImportError as exc:
        return _skipped_quality(str(exc))
    except Exception as exc:
        return _failed_quality(str(exc))
    runtime = time.perf_counter() - started

    if result.ideal_value is None:
        status = _quality_status_from_warning(result.warning)
        if status == "skipped":
            return _skipped_quality(result.warning or "simulation skipped")
        return _failed_quality(result.warning or "simulation did not produce values")

    return {
        "ideal": _finite_or_none(result.ideal_value),
        "noisy": _finite_or_none(result.noisy_value),
        "mitigated": _finite_or_none(result.mitigated_value),
        "noisy_error": _finite_or_none(result.noisy_error),
        "mitigated_error": _finite_or_none(result.mitigated_error),
        "error_reduction": _finite_or_none(result.error_reduction),
        "runtime_seconds": round(runtime, 6),
        "status": "passed",
        "skip_reason": None,
        "failure": None,
    }


def _measure_quality(
    circuit,
    recipe: MitigationRecipe,
    case: BenchmarkCase,
    *,
    include_quality: bool,
    max_qubits: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    if not include_quality:
        return _skipped_quality("quality disabled")
    if case.speed_only:
        return _skipped_quality("speed-only benchmark case")
    if not case.run_quality_by_default:
        return _skipped_quality("quality disabled for this case by default")
    if circuit.num_qubits > max_qubits:
        return _skipped_quality(
            f"case has {circuit.num_qubits} qubits; max quality qubits is {max_qubits}"
        )

    observations = [
        _measure_quality_once(circuit, recipe, case, seed + index)
        for index in range(repeats)
    ]
    first_non_pass = next(
        (item for item in observations if item["status"] != "passed"),
        None,
    )
    if first_non_pass is not None:
        return first_non_pass

    noisy_errors = [item["noisy_error"] for item in observations]
    mitigated_errors = [item["mitigated_error"] for item in observations]
    reductions = [item["error_reduction"] for item in observations]
    runtimes = [item["runtime_seconds"] for item in observations]
    mitigated_values = [item["mitigated"] for item in observations]

    return {
        "ideal": observations[0]["ideal"],
        "noisy": observations[0]["noisy"],
        "mitigated": _median(mitigated_values),
        "noisy_error": _median(noisy_errors),
        "mitigated_error": _median(mitigated_errors),
        "error_reduction": _median(reductions),
        "runtime_seconds": _median(runtimes),
        "status": "passed",
        "skip_reason": None,
        "failure": None,
        "repeats": observations,
        "mitigated_error_std": (
            float(statistics.pstdev(mitigated_errors))
            if len(mitigated_errors) > 1
            else 0.0
        ),
    }


def run_case(
    case: BenchmarkCase,
    *,
    policy: RecipePolicy,
    include_speed: bool,
    include_quality: bool,
    max_qubits: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    circuit = case.build()
    features = analyze_circuit(
        circuit,
        noise_model_available=case.noise_model_available,
    )
    recipe = recommend(features, policy=policy)
    result = _features_to_case_base(case, features, recipe)
    result["speed"] = (
        _measure_speed(circuit, features, recipe, policy, case, repeats)
        if include_speed
        else _empty_speed()
    )
    result["quality"] = _measure_quality(
        circuit,
        recipe,
        case,
        include_quality=include_quality,
        max_qubits=max_qubits,
        seed=seed,
        repeats=repeats,
    )
    return result


def _external_case_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in path.stem.lower())
    return f"external_{safe_stem}_{digest}"


def _run_external_case(
    path: Path,
    *,
    policy: RecipePolicy,
    include_speed: bool,
    repeats: int,
) -> dict[str, Any]:
    case_id = _external_case_id(path)
    try:
        circuit = _load_circuit(str(path))
        features = analyze_circuit(circuit)
        recipe = recommend(features, policy=policy)
    except Exception as exc:
        return {
            "case_id": case_id,
            "family": "external_qasm",
            "num_qubits": None,
            "depth": None,
            "total_gate_count": None,
            "multi_qubit_gate_count": None,
            "non_clifford_fraction": None,
            "noise_model": None,
            "noise_level": None,
            "observable": None,
            "selected_recipe": {},
            "speed": _empty_speed(),
            "quality": _failed_quality(f"{path.name}: {exc}"),
        }

    case = BenchmarkCase(
        case_id=case_id,
        family="external_qasm",
        build=lambda: circuit,
        stress_target="external_speed",
        run_quality_by_default=False,
        speed_only=True,
    )
    result = _features_to_case_base(case, features, recipe)
    result["speed"] = (
        _measure_speed(circuit, features, recipe, policy, case, repeats)
        if include_speed
        else _empty_speed()
    )
    result["quality"] = _skipped_quality("external QASM benchmarks are speed-only")
    return result


def run_external_qasm_dir(
    directory: Path,
    *,
    policy: RecipePolicy,
    include_speed: bool,
    repeats: int,
) -> list[dict[str, Any]]:
    if not directory.exists():
        return [
            {
                "case_id": "external_qasm_dir_missing",
                "family": "external_qasm",
                "num_qubits": None,
                "depth": None,
                "total_gate_count": None,
                "multi_qubit_gate_count": None,
                "non_clifford_fraction": None,
                "noise_model": None,
                "noise_level": None,
                "observable": None,
                "selected_recipe": {},
                "speed": _empty_speed(),
                "quality": _failed_quality(
                    f"External QASM directory not found: {directory}"
                ),
            }
        ]

    paths = sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in {".qasm"}
    )
    if not paths:
        return []
    return [
        _run_external_case(
            path,
            policy=policy,
            include_speed=include_speed,
            repeats=repeats,
        )
        for path in paths
    ]


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    quality = [case.get("quality", {}) for case in cases]
    speed_values = [
        case.get("speed", {}).get("generate_recipe_ms_median")
        for case in cases
        if case.get("speed", {}).get("generate_recipe_ms_median") is not None
    ]
    reductions = [
        case.get("quality", {}).get("error_reduction")
        for case in cases
        if case.get("quality", {}).get("error_reduction") is not None
    ]
    techniques: dict[str, int] = {}
    for case in cases:
        technique = case.get("selected_recipe", {}).get("technique", "unknown")
        techniques[technique] = techniques.get(technique, 0) + 1

    return {
        "total_cases": len(cases),
        "quality_passed": sum(item.get("status") == "passed" for item in quality),
        "quality_failed": sum(item.get("status") == "failed" for item in quality),
        "quality_skipped": sum(item.get("status") == "skipped" for item in quality),
        "median_generate_recipe_ms": _median(speed_values),
        "p95_generate_recipe_ms": _percentile(speed_values, 0.95),
        "median_error_reduction": _median(reductions),
        "p25_error_reduction": _percentile(reductions, 0.25),
        "selected_techniques": techniques,
    }


def build_output(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    policy = _load_policy(args.policy)
    policy_path = args.policy
    policy_data = policy_to_dict(policy)
    policy_sha = _sha256(policy_path) if policy_path.exists() else None

    cases = [
        run_case(
            case,
            policy=policy,
            include_speed=args.include_speed,
            include_quality=args.include_quality,
            max_qubits=args.max_qubits,
            seed=args.seed,
            repeats=args.repeats,
        )
        for case in build_corpus(seed=args.seed, quick=args.quick)
    ]
    if args.external_qasm_dir is not None:
        cases.extend(
            run_external_qasm_dir(
                args.external_qasm_dir,
                policy=policy,
                include_speed=args.include_speed,
                repeats=args.repeats,
            )
        )

    return {
        "schema_version": 1,
        "emrg_version": __version__,
        "commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "environment": _environment(),
        "policy": {
            "path": str(policy_path),
            "sha256": policy_sha,
            "data": policy_data,
        },
        "config": {
            "seed": args.seed,
            "repeats": args.repeats,
            "quick": args.quick,
            "include_speed": args.include_speed,
            "include_quality": args.include_quality,
            "max_qubits": args.max_qubits,
        },
        "cases": cases,
        "summary": summarize(cases),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_output(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=args.json_indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = output["summary"]
    print(f"Wrote {_console_safe(args.output)}")
    print(
        "Cases: {total_cases} | quality passed/failed/skipped: "
        "{quality_passed}/{quality_failed}/{quality_skipped}".format(**summary)
    )
    if summary["median_generate_recipe_ms"] is not None:
        print(f"Median generate_recipe: {summary['median_generate_recipe_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
