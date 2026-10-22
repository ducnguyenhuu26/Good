"""Summarize all CLSM diagnostic effects without promoting them to claims."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def hierarchical_interval(records: list[dict], seed: int = 7319,
                          draws: int = 2000) -> tuple[float, float] | None:
    """Paired hierarchy: resample training seeds, then independent worlds."""
    by_seed: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_seed[int(row["train_seed"])][int(row["world_seed"])].append(float(row["effect"]))
    if len(by_seed) < 5 or any(len(worlds) < 20 for worlds in by_seed.values()):
        return None
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        seed_means = []
        for selected_seed in rng.choice(seeds, size=len(seeds), replace=True):
            world_means = np.asarray([np.mean(v) for v in by_seed[int(selected_seed)].values()])
            seed_means.append(float(rng.choice(world_means, size=len(world_means), replace=True).mean()))
        values.append(np.mean(seed_means))
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_run(path: Path, claim: str) -> tuple[dict, Path] | None:
    """Return evaluator metadata only when it belongs to the current checkpoint.

    Evaluator directories persist across reruns. Comparing their recorded
    checkpoint hash with the canonical checkpoint prevents stale results from
    being silently pooled after retraining the same seed.
    """
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_dir = path.parent.parent
    checkpoint = seed_dir / "checkpoint_best.pt"
    expected = summary.get("checkpoint_sha256")
    if not checkpoint.is_file() or not expected or _sha256(checkpoint) != expected:
        return None
    if claim == "C1":
        pooled = seed_dir.parent.parent / "pooled_mlp" / seed_dir.name / "checkpoint_best.pt"
        pooled_expected = summary.get("pooled_checkpoint_sha256")
        if not pooled.is_file() or not pooled_expected or _sha256(pooled) != pooled_expected:
            return None
    return summary, seed_dir


def _records(results_dir: Path):
    excluded_stale = 0
    records = []
    for path in results_dir.glob("*/*/clsm/*/*/scores.csv"):
        kind = path.parent.name.split("_diagnostic_")[0]
        if kind not in ("forecast", "composition"):
            continue
        claim = "C2" if kind == "forecast" else "C1"
        verified = _verified_run(path, claim)
        if verified is None:
            excluded_stale += 1
            continue
        summary, seed_dir = verified
        probe_hash = summary.get("public_probe_sha256", summary.get("probe_sha256"))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                effect = row["skill"] if kind == "forecast" else row["pooled_minus_clsm"]
                key = ("C2", int(row["horizon"])) if kind == "forecast" else ("C1", int(row["held_out_code"]))
                records.append({"claim": key[0], "slice": key[1], "env": path.parts[-5],
                       "track": path.parts[-6], "train_seed": int(seed_dir.name),
                       "world_seed": int(row["world_seed"]), "effect": float(effect),
                       "source": str(path), "probe_hash": probe_hash,
                       "protocol": {"probes": summary.get("probes"),
                                    "samples": summary.get("samples"),
                                    "horizons": summary.get("horizons"),
                                    "adaptation_steps": row.get("adaptation_steps")}})
    for path in results_dir.glob("*/*/clsm/*/memory_diagnostic_*/memory_adaptation.csv"):
        verified = _verified_run(path, "C3")
        if verified is None:
            excluded_stale += 1
            continue
        summary, seed_dir = verified
        protocol = {"component_budget": summary.get("component_budget"),
                    "probe_hash": summary.get("probe_sha256"),
                    "budgets": summary.get("budgets"),
                    "matched_compute_scratch": summary.get("matched_compute_scratch"),
                    "schedule_condition": summary.get("schedule_condition", "in_family")}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append({"claim": "C3", "slice": int(row["budget"]), "env": path.parts[-5],
                       "track": path.parts[-6], "train_seed": int(seed_dir.name),
                       "world_seed": int(row["world_seed"]), "effect": float(row["gain"]),
                       "source": str(path),
                       "probe_hash": summary.get("probe_sha256"), "protocol": protocol,
                       "inference_degraded": max(float(row["reuse_max_dropped_mass"]),
                                                  float(row["scratch_max_dropped_mass"])) > .001})
    return records, excluded_stale


def summarize(results_dir: Path) -> dict:
    records, excluded_stale = _records(results_dir)
    grouped = defaultdict(list)
    for row in records:
        protocol_key = json.dumps(row.get("protocol", {}), sort_keys=True)
        grouped[(row["claim"], row["track"], row["env"], row["slice"],
                 protocol_key, row.get("probe_hash"))].append(row)
    effects = []
    for (claim, track, env, slice_value, protocol_key, probe_hash), rows in sorted(grouped.items()):
        interval = hierarchical_interval(rows)
        effects.append({"claim": claim, "track": track, "env": env, "slice": slice_value,
                        "protocol": json.loads(protocol_key), "probe_hash": probe_hash,
                        "mean_effect": float(np.mean([r["effect"] for r in rows])),
                        "train_seeds": len(set(r["train_seed"] for r in rows)),
                        "worlds": len(set((r["train_seed"], r["world_seed"]) for r in rows)),
                        "hierarchical_95pct_ci": interval,
                        "negative_effect_count": sum(r["effect"] < 0 for r in rows),
                        "inference_degraded": any(r.get("inference_degraded", False) for r in rows),
                        "status": "diagnostic_only"})
    report = {"status": "diagnostic_summary_not_confirmatory",
              "effects": effects, "records": len(records),
              "stale_or_unverified_run_directories_excluded": excluded_stale,
              "missing_requirements": ["five training seeds and twenty independent evaluation worlds",
                                       "complete matched baseline panel", "all environments",
                                       "passing long-world mixture-mass gate"]}
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "diagnostic_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"results_dir": str(args.results_dir)}))
        return
    result = summarize(args.results_dir)
    print(json.dumps({"status": result["status"], "records": result["records"],
                      "effects": len(result["effects"])}))


if __name__ == "__main__":
    main()
