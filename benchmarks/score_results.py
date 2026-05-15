"""Score EMRG benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EPSILON = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score EMRG benchmark results.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


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


def _estimated_overhead(case: dict[str, Any]) -> float:
    value = case.get("selected_recipe", {}).get("estimated_overhead", 1.0)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return 1.0
    return max(1.0, float(value))


def _instability_penalty(quality: dict[str, Any]) -> float:
    std = quality.get("mitigated_error_std")
    mitigated_error = quality.get("mitigated_error")
    if not isinstance(std, int | float) or not isinstance(mitigated_error, int | float):
        return 0.0
    scale = max(abs(float(mitigated_error)), EPSILON)
    return 0.1 * math.log1p(abs(float(std)) / scale)


def score_case(case: dict[str, Any]) -> float:
    quality = case.get("quality") or {}
    status = quality.get("status")
    overhead = _estimated_overhead(case)

    if status == "failed":
        return -10.0
    if status == "skipped":
        return 0.0

    noisy_error = quality.get("noisy_error")
    mitigated_error = quality.get("mitigated_error")
    if not isinstance(noisy_error, int | float) or not isinstance(
        mitigated_error, int | float
    ):
        return 0.0

    noisy_error = abs(float(noisy_error))
    mitigated_error = abs(float(mitigated_error))
    if mitigated_error > noisy_error:
        base = -1.0
    else:
        ratio = min(noisy_error / max(mitigated_error, EPSILON), 1_000.0)
        base = math.log1p(ratio)

    base -= 0.05 * math.log1p(overhead)
    base -= _instability_penalty(quality)
    return float(base)


def _speed_values(cases: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for case in cases:
        speed = case.get("speed") or {}
        value = speed.get("generate_recipe_ms_median")
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _error_reductions(cases: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for case in cases:
        value = (case.get("quality") or {}).get("error_reduction")
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _overheads(cases: list[dict[str, Any]]) -> list[float]:
    return [_estimated_overhead(case) for case in cases]


def _technique_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for case in cases:
        technique = case.get("selected_recipe", {}).get("technique", "unknown")
        distribution[technique] = distribution.get(technique, 0) + 1
    return distribution


def _family_scores(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case.get("family", "unknown")].append(case)

    families: dict[str, dict[str, Any]] = {}
    for family, family_cases in grouped.items():
        scores = [float(case["score"]) for case in family_cases]
        families[family] = {
            "case_count": len(family_cases),
            "median_score": _median(scores),
            "p25_score": _percentile(scores, 0.25),
        }
    return families


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(case["score"]) for case in cases]
    quality_statuses = [(case.get("quality") or {}).get("status") for case in cases]
    speed_values = _speed_values(cases)
    reductions = _error_reductions(cases)
    final_score = 0.0
    if scores:
        final_score = float(_median(scores) + 0.25 * _percentile(scores, 0.25))

    total = len(cases) or 1
    return {
        "final_score": final_score,
        "case_count": len(cases),
        "pass_rate": quality_statuses.count("passed") / total,
        "failure_rate": quality_statuses.count("failed") / total,
        "skip_rate": quality_statuses.count("skipped") / total,
        "median_speed_ms": _median(speed_values),
        "p95_speed_ms": _percentile(speed_values, 0.95),
        "median_error_reduction": _median(reductions),
        "p25_error_reduction": _percentile(reductions, 0.25),
        "median_estimated_overhead": _median(_overheads(cases)),
        "selected_techniques": _technique_distribution(cases),
    }


def score_document(document: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for case in document.get("cases", []):
        scored_case = dict(case)
        scored_case["score"] = score_case(scored_case)
        cases.append(scored_case)

    return {
        "schema_version": 1,
        "source": {
            "emrg_version": document.get("emrg_version"),
            "commit": document.get("commit"),
            "policy": document.get("policy", {}),
            "config": document.get("config", {}),
        },
        "summary": _summary(cases),
        "families": _family_scores(cases),
        "cases": cases,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_summary(scored: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    summary = scored["summary"]
    print(f"Final score: {summary['final_score']:.4f}")
    print(
        "Quality pass/fail/skip: "
        f"{summary['pass_rate']:.1%}/"
        f"{summary['failure_rate']:.1%}/"
        f"{summary['skip_rate']:.1%}"
    )
    if summary["median_speed_ms"] is not None:
        print(
            "Speed median/p95: "
            f"{summary['median_speed_ms']:.3f} ms / "
            f"{summary['p95_speed_ms']:.3f} ms"
        )
    if summary["median_error_reduction"] is not None:
        print(
            "Error reduction median/p25: "
            f"{summary['median_error_reduction']:.3f}x / "
            f"{summary['p25_error_reduction']:.3f}x"
        )
    print(f"Median overhead: {summary['median_estimated_overhead']:.3f}")
    print(f"Technique distribution: {summary['selected_techniques']}")

    if baseline is not None:
        delta = summary["final_score"] - baseline["summary"]["final_score"]
        print(f"Delta vs baseline: {delta:+.4f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scored = score_document(_load_json(args.results))
    scored_baseline = None
    if args.baseline is not None:
        scored_baseline = score_document(_load_json(args.baseline))

    _print_summary(scored, scored_baseline)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(scored, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
