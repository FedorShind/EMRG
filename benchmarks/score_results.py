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


def _skip_reason(case: dict[str, Any]) -> str:
    reason = (case.get("quality") or {}).get("skip_reason")
    if not isinstance(reason, str):
        return ""
    return reason.lower()


def _is_speed_only_skip(case: dict[str, Any]) -> bool:
    reason = _skip_reason(case)
    return "speed-only" in reason


def _score_reason(case: dict[str, Any]) -> str:
    quality = case.get("quality") or {}
    status = quality.get("status")

    if status == "failed":
        return "failed"
    if status == "skipped":
        if _is_speed_only_skip(case):
            return "speed_only"
        return "quality_skipped"

    noisy_error = quality.get("noisy_error")
    mitigated_error = quality.get("mitigated_error")
    if not isinstance(noisy_error, int | float) or not isinstance(
        mitigated_error, int | float
    ):
        return "missing_quality_metrics"

    noisy_error = abs(float(noisy_error))
    mitigated_error = abs(float(mitigated_error))
    if mitigated_error > noisy_error:
        return "worse_than_noisy"
    return "quality_improved"


def _contributes_to_final(reason: str) -> bool:
    return reason in {
        "failed",
        "missing_quality_metrics",
        "worse_than_noisy",
        "quality_improved",
    }


def score_case(case: dict[str, Any]) -> float:
    quality = case.get("quality") or {}
    reason = _score_reason(case)
    overhead = _estimated_overhead(case)

    if reason == "failed":
        return -10.0
    if reason in {"speed_only", "quality_skipped", "missing_quality_metrics"}:
        return 0.0

    noisy_error = abs(float(quality["noisy_error"]))
    mitigated_error = abs(float(quality["mitigated_error"]))
    if reason == "worse_than_noisy":
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
        scores = [
            float(case["score"])
            for case in family_cases
            if case.get("score_contributes_to_final")
        ]
        families[family] = {
            "case_count": len(family_cases),
            "score_case_count": len(scores),
            "median_score": _median(scores),
            "p25_score": _percentile(scores, 0.25),
        }
    return families


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(case["score"]) for case in cases if case.get("score_contributes_to_final")
    ]
    quality_statuses = [(case.get("quality") or {}).get("status") for case in cases]
    score_reasons = [case.get("score_reason") for case in cases]
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
        "score_case_count": len(scores),
        "quality_improved_case_count": score_reasons.count("quality_improved"),
        "worse_than_noisy_case_count": score_reasons.count("worse_than_noisy"),
        "missing_quality_metric_case_count": score_reasons.count(
            "missing_quality_metrics"
        ),
        "quality_skipped_case_count": score_reasons.count("quality_skipped"),
        "speed_only_case_count": score_reasons.count("speed_only"),
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
        score_reason = _score_reason(scored_case)
        scored_case["score"] = score_case(scored_case)
        scored_case["score_reason"] = score_reason
        scored_case["score_contributes_to_final"] = _contributes_to_final(score_reason)
        scored_case["estimated_overhead"] = _estimated_overhead(scored_case)
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
        "Scored quality cases: "
        f"{summary['score_case_count']}/{summary['case_count']} "
        f"(improved={summary['quality_improved_case_count']}, "
        f"worse={summary['worse_than_noisy_case_count']}, "
        f"missing={summary['missing_quality_metric_case_count']})"
    )
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
    print("Case breakdown:")
    for case in scored["cases"]:
        quality = case.get("quality") or {}
        reduction = quality.get("error_reduction")
        reduction_text = (
            f"{reduction:.3f}x" if isinstance(reduction, int | float) else "-"
        )
        print(
            "  {case_id}: {family} {technique} {status} "
            "score={score:.4f} reduction={reduction} overhead={overhead:.3f} "
            "reason={reason}".format(
                case_id=case.get("case_id"),
                family=case.get("family"),
                technique=(case.get("selected_recipe") or {}).get(
                    "technique", "unknown"
                ),
                status=quality.get("status", "unknown"),
                score=float(case.get("score", 0.0)),
                reduction=reduction_text,
                overhead=float(case.get("estimated_overhead", 1.0)),
                reason=case.get("score_reason"),
            )
        )

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
