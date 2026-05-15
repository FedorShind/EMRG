"""Controlled random search for EMRG benchmark policy JSON files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_benchmark
from benchmarks.score_results import score_document
from emrg.policy import RecipePolicy, load_policy, policy_to_dict

SCALE_FACTOR_OPTIONS: tuple[tuple[float, ...], ...] = (
    (1.0, 1.5, 2.0),
    (1.0, 2.0, 3.0),
    (1.0, 1.25, 1.5, 2.0),
    (1.0, 1.5, 2.0, 2.5),
    (1.0, 1.5, 2.0, 2.5, 3.0),
)
SCALING_METHOD_OPTIONS = ("fold_global", "fold_gates_at_random")
PROFILE_FACTORY_OPTIONS: dict[str, tuple[str, ...]] = {
    "shallow": ("LinearFactory", "RichardsonFactory"),
    "moderate": ("LinearFactory", "RichardsonFactory", "PolyFactory"),
    "heterogeneous": ("RichardsonFactory", "PolyFactory"),
    "deep": ("RichardsonFactory", "PolyFactory"),
}

SEARCH_SPACE: dict[str, Any] = {
    "cdr": {
        "min_depth": [6, 8, 10, 12],
        "max_depth": [35, 40, 45, 50],
        "min_non_clifford_fraction": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "training_circuits": [(6, 10, 14), (8, 12, 16), (10, 14, 20)],
    },
    "pec": {
        "max_depth": [20, 25, 30],
        "max_overhead": [250.0, 500.0, 1000.0],
        "samples": [(80, 300), (100, 500), (150, 600)],
    },
    "composite": {
        "min_depth": [10, 15, 20],
        "max_depth": [25, 30, 35],
        "max_combined_overhead": [250.0, 500.0, 1000.0],
    },
    "zne_depths": {
        "shallow_max_depth": [15, 18, 20, 22],
        "moderate_max_depth": [35, 40, 45, 50, 55],
    },
    "zne_profiles": {
        "scale_factors": [list(item) for item in SCALE_FACTOR_OPTIONS],
        "scaling_methods": list(SCALING_METHOD_OPTIONS),
        "factories": PROFILE_FACTORY_OPTIONS,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search EMRG policy JSON files.")
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args(argv)

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_qubits <= 0:
        parser.error("--max-qubits must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def _json_dumps(data: Any, *, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, sort_keys=True) + "\n"


def _write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(data, indent=indent), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_search_key(data: dict[str, Any]) -> str:
    comparable = copy.deepcopy(data)
    comparable["name"] = "<candidate>"
    payload = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_profile_factory(profile: dict[str, Any], factory: str) -> None:
    profile["factory"] = factory
    if factory == "PolyFactory":
        profile["factory_kwargs"] = {"order": 2}
    else:
        profile.pop("factory_kwargs", None)


def _mutate_cdr(data: dict[str, Any], rng: random.Random) -> None:
    cdr = data["techniques"]["cdr"]
    min_depth = rng.choice(SEARCH_SPACE["cdr"]["min_depth"])
    max_depth = rng.choice(
        [value for value in SEARCH_SPACE["cdr"]["max_depth"] if value >= min_depth]
    )
    small, medium, large = rng.choice(SEARCH_SPACE["cdr"]["training_circuits"])
    cdr["min_depth"] = min_depth
    cdr["max_depth"] = max_depth
    cdr["min_non_clifford_fraction"] = rng.choice(
        SEARCH_SPACE["cdr"]["min_non_clifford_fraction"]
    )
    cdr["training_circuits"]["small"] = small
    cdr["training_circuits"]["medium"] = medium
    cdr["training_circuits"]["large"] = large


def _mutate_pec(data: dict[str, Any], rng: random.Random) -> None:
    pec = data["techniques"]["pec"]
    min_samples, max_samples = rng.choice(SEARCH_SPACE["pec"]["samples"])
    pec["max_depth"] = rng.choice(SEARCH_SPACE["pec"]["max_depth"])
    pec["max_overhead"] = rng.choice(SEARCH_SPACE["pec"]["max_overhead"])
    pec["min_samples"] = min_samples
    pec["max_samples"] = max_samples


def _mutate_composite(data: dict[str, Any], rng: random.Random) -> None:
    composite = data["techniques"]["composite"]
    min_depth = rng.choice(SEARCH_SPACE["composite"]["min_depth"])
    max_depth = rng.choice(
        [
            value
            for value in SEARCH_SPACE["composite"]["max_depth"]
            if value >= min_depth
        ]
    )
    composite["min_depth"] = min_depth
    composite["max_depth"] = max_depth
    composite["max_combined_overhead"] = rng.choice(
        SEARCH_SPACE["composite"]["max_combined_overhead"]
    )


def _mutate_zne_depths(data: dict[str, Any], rng: random.Random) -> None:
    zne = data["techniques"]["zne"]
    shallow_max = rng.choice(SEARCH_SPACE["zne_depths"]["shallow_max_depth"])
    moderate_min = rng.choice(
        sorted({max(1, shallow_max - 2), shallow_max, shallow_max + 2})
    )
    moderate_max = rng.choice(
        [
            value
            for value in SEARCH_SPACE["zne_depths"]["moderate_max_depth"]
            if value >= moderate_min
        ]
    )
    zne["shallow"]["max_depth"] = shallow_max
    zne["moderate"]["min_depth"] = moderate_min
    zne["moderate"]["max_depth"] = moderate_max
    zne["deep"]["min_depth"] = moderate_max + 1


def _mutate_zne_profiles(data: dict[str, Any], rng: random.Random) -> None:
    zne = data["techniques"]["zne"]
    profile_names = list(PROFILE_FACTORY_OPTIONS)
    for profile_name in rng.sample(profile_names, rng.randint(1, len(profile_names))):
        profile = zne[profile_name]
        _set_profile_factory(profile, rng.choice(PROFILE_FACTORY_OPTIONS[profile_name]))
        profile["scale_factors"] = list(rng.choice(SCALE_FACTOR_OPTIONS))
        profile["scaling_method"] = rng.choice(SCALING_METHOD_OPTIONS)


def _validate_monotonic_scale_factors(data: dict[str, Any]) -> None:
    for profile_name in PROFILE_FACTORY_OPTIONS:
        profile = data["techniques"]["zne"][profile_name]
        factors = profile["scale_factors"]
        if factors != sorted(set(factors)):
            raise ValueError(
                f"{profile_name} scale factors must be unique and monotonic"
            )
        if factors[0] != 1.0:
            raise ValueError(f"{profile_name} scale factors must start with 1.0")
        if profile["factory"] == "PolyFactory":
            order = profile.get("factory_kwargs", {}).get("order", 2)
            if len(factors) <= order:
                raise ValueError(
                    f"{profile_name} PolyFactory needs more factors than order"
                )


def validate_search_policy(data: dict[str, Any]) -> RecipePolicy:
    """Validate a generated search policy beyond the project schema."""
    policy = RecipePolicy.from_dict(data)
    _validate_monotonic_scale_factors(data)
    return policy


def generate_candidate_policy(
    base_policy_data: dict[str, Any],
    *,
    rng: random.Random,
    index: int,
) -> dict[str, Any]:
    """Return one validated candidate policy data object."""
    data = copy.deepcopy(base_policy_data)
    data["name"] = f"{base_policy_data.get('name', 'policy')}-candidate-{index:04d}"
    _mutate_cdr(data, rng)
    _mutate_pec(data, rng)
    _mutate_composite(data, rng)
    _mutate_zne_depths(data, rng)
    _mutate_zne_profiles(data, rng)
    validate_search_policy(data)
    return data


def _iter_candidates(
    base_policy_data: dict[str, Any],
    *,
    seed: int,
    trials: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = trials * 50
    while len(candidates) < trials and attempts < max_attempts:
        attempts += 1
        data = generate_candidate_policy(
            base_policy_data,
            rng=rng,
            index=len(candidates) + 1,
        )
        key = _canonical_search_key(data)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(data)
    return candidates


def _benchmark_args(policy: Path, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        policy=policy,
        output=Path("unused.json"),
        seed=args.seed,
        repeats=args.repeats,
        quick=args.quick,
        include_quality=True,
        include_speed=True,
        max_qubits=args.max_qubits,
        json_indent=args.json_indent,
        external_qasm_dir=None,
        split=args.split,
    )


def _run_benchmark(
    policy_path: Path, args: argparse.Namespace, output: Path
) -> dict[str, Any]:
    document = run_benchmark.build_output(_benchmark_args(policy_path, args))
    _write_json(output, document, indent=args.json_indent)
    return document


def _config_matches(document: dict[str, Any], args: argparse.Namespace) -> bool:
    config = document.get("config") or {}
    return {
        "seed": config.get("seed"),
        "repeats": config.get("repeats"),
        "quick": config.get("quick"),
        "include_speed": config.get("include_speed"),
        "include_quality": config.get("include_quality"),
        "max_qubits": config.get("max_qubits"),
        "split": config.get("split", "all"),
    } == {
        "seed": args.seed,
        "repeats": args.repeats,
        "quick": args.quick,
        "include_speed": True,
        "include_quality": True,
        "max_qubits": args.max_qubits,
        "split": args.split,
    }


def _load_or_create_baseline(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, bool]:
    if args.baseline is not None and args.baseline.exists():
        document = _load_json(args.baseline)
        if _config_matches(document, args):
            scored = score_document(document)
            _write_json(
                output_dir / "baseline-score.json", scored, indent=args.json_indent
            )
            return document, scored, args.baseline, False

    baseline_path = output_dir / "baseline-search.json"
    document = _run_benchmark(args.base_policy, args, baseline_path)
    scored = score_document(document)
    _write_json(output_dir / "baseline-score.json", scored, indent=args.json_indent)
    return document, scored, baseline_path, True


def _candidate_record(
    *,
    candidate_id: str,
    policy_path: Path,
    result_path: Path,
    score_path: Path,
    scored: dict[str, Any],
    baseline_score: float,
) -> dict[str, Any]:
    summary = scored["summary"]
    return {
        "candidate_id": candidate_id,
        "status": "passed",
        "policy_path": str(policy_path),
        "result_path": str(result_path),
        "score_path": str(score_path),
        "score": summary["final_score"],
        "delta_vs_baseline": summary["final_score"] - baseline_score,
        "failure_rate": summary["failure_rate"],
        "skip_rate": summary["skip_rate"],
        "median_speed_ms": summary["median_speed_ms"],
        "p95_speed_ms": summary["p95_speed_ms"],
        "median_error_reduction": summary["median_error_reduction"],
        "p25_error_reduction": summary["p25_error_reduction"],
        "median_estimated_overhead": summary["median_estimated_overhead"],
        "selected_techniques": summary["selected_techniques"],
        "families": scored["families"],
        "failure": None,
    }


def _failed_record(
    candidate_id: str, policy_path: Path, exc: BaseException
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": "failed",
        "policy_path": str(policy_path),
        "result_path": None,
        "score_path": None,
        "score": None,
        "delta_vs_baseline": None,
        "failure_rate": 1.0,
        "skip_rate": None,
        "median_speed_ms": None,
        "p95_speed_ms": None,
        "median_error_reduction": None,
        "p25_error_reduction": None,
        "median_estimated_overhead": None,
        "selected_techniques": {},
        "families": {},
        "failure": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            item["status"] == "passed",
            item["score"] if isinstance(item["score"], int | float) else float("-inf"),
        ),
        reverse=True,
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "candidate_id",
        "status",
        "score",
        "delta_vs_baseline",
        "failure_rate",
        "skip_rate",
        "median_estimated_overhead",
        "policy_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow(
                {field: record.get(field) for field in fields} | {"rank": rank}
            )


def _write_markdown(
    path: Path, records: list[dict[str, Any]], baseline: dict[str, Any]
) -> None:
    lines = [
        "# Policy Search Summary",
        "",
        f"Baseline score: {baseline['summary']['final_score']:.4f}",
        "",
        "| Rank | Candidate | Status | Score | Delta | Fail | Skip | Overhead |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(records, start=1):
        score = record["score"]
        delta = record["delta_vs_baseline"]
        failure_rate = record["failure_rate"]
        skip_rate = record["skip_rate"]
        overhead = record["median_estimated_overhead"]
        lines.append(
            "| {rank} | {candidate} | {status} | {score} | {delta} | "
            "{failure} | {skip} | {overhead} |".format(
                rank=rank,
                candidate=record["candidate_id"],
                status=record["status"],
                score=f"{score:.4f}" if isinstance(score, int | float) else "-",
                delta=f"{delta:+.4f}" if isinstance(delta, int | float) else "-",
                failure=f"{failure_rate:.1%}"
                if isinstance(failure_rate, int | float)
                else "-",
                skip=f"{skip_rate:.1%}" if isinstance(skip_rate, int | float) else "-",
                overhead=f"{overhead:.3f}"
                if isinstance(overhead, int | float)
                else "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    args: argparse.Namespace,
    *,
    baseline_path: Path,
    generated_baseline: bool,
    baseline_scored: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    ranked = _sort_records(records)
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank

    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "trials_requested": args.trials,
        "trials_completed": len(records),
        "quick": args.quick,
        "split": args.split,
        "repeats": args.repeats,
        "max_qubits": args.max_qubits,
        "top_k": args.top_k,
        "base_policy": str(args.base_policy),
        "baseline": {
            "path": str(baseline_path),
            "generated_for_search_config": generated_baseline,
            "score": baseline_scored["summary"]["final_score"],
            "summary": baseline_scored["summary"],
        },
        "search_space": SEARCH_SPACE,
        "candidates": ranked,
        "top_candidates": ranked[: args.top_k],
    }
    _write_json(args.output_dir / "summary.json", summary, indent=args.json_indent)
    _write_csv(args.output_dir / "summary.csv", ranked)
    _write_markdown(args.output_dir / "summary.md", ranked, baseline_scored)


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = args.output_dir / "candidates"
    results_dir = args.output_dir / "results"
    scores_dir = args.output_dir / "scores"

    base_policy_data = policy_to_dict(load_policy(args.base_policy))
    _baseline_doc, baseline_scored, baseline_path, generated_baseline = (
        _load_or_create_baseline(args, args.output_dir)
    )
    baseline_score = baseline_scored["summary"]["final_score"]

    records: list[dict[str, Any]] = []
    for index, candidate_data in enumerate(
        _iter_candidates(base_policy_data, seed=args.seed, trials=args.trials),
        start=1,
    ):
        candidate_id = f"candidate-{index:04d}"
        policy_path = candidates_dir / f"{candidate_id}.json"
        result_path = results_dir / f"{candidate_id}.json"
        score_path = scores_dir / f"{candidate_id}.json"
        _write_json(policy_path, candidate_data, indent=args.json_indent)
        try:
            document = _run_benchmark(policy_path, args, result_path)
            scored = score_document(document)
            _write_json(score_path, scored, indent=args.json_indent)
            records.append(
                _candidate_record(
                    candidate_id=candidate_id,
                    policy_path=policy_path,
                    result_path=result_path,
                    score_path=score_path,
                    scored=scored,
                    baseline_score=baseline_score,
                )
            )
        except Exception as exc:  # noqa: BLE001 - candidate failure must not stop search.
            records.append(_failed_record(candidate_id, policy_path, exc))
            continue

    _write_summary(
        args,
        baseline_path=baseline_path,
        generated_baseline=generated_baseline,
        baseline_scored=baseline_scored,
        records=records,
    )
    return _load_json(args.output_dir / "summary.json")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_search(args)
    baseline_score = summary["baseline"]["score"]
    generated_baseline = summary["baseline"]["generated_for_search_config"]
    baseline_source = "generated" if generated_baseline else "provided"
    print(f"Wrote {args.output_dir}")
    print(f"Baseline score: {baseline_score:.4f} ({baseline_source})")
    if summary["top_candidates"]:
        top = summary["top_candidates"][0]
        score = top["score"]
        delta = top["delta_vs_baseline"]
        print(
            "Best candidate: "
            f"{top['candidate_id']} score={score:.4f} delta={delta:+.4f} "
            f"policy={top['policy_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
