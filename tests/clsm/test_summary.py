import csv
import hashlib
import json

from experiments.rwt.clsm.summarize import hierarchical_interval
from experiments.rwt.clsm.summarize import summarize


def test_hierarchical_interval_requires_independent_seeds_and_worlds():
    rows = [{"train_seed": 41, "world_seed": world, "effect": 1.}
            for world in range(20)]
    assert hierarchical_interval(rows) is None
    rows = [{"train_seed": seed, "world_seed": world, "effect": 1.}
            for seed in range(41, 46) for world in range(801, 821)]
    assert hierarchical_interval(rows, draws=100) == (1., 1.)


def _write_summary(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields[0]))
        writer.writeheader()
        writer.writerows(fields)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summarizer_excludes_stale_checkpoints_and_separates_ablation_budgets(tmp_path):
    results = tmp_path / "results"
    comp_seed = results / "comp" / "HalfCheetah-v5" / "clsm" / "41"
    pooled_seed = results / "comp" / "HalfCheetah-v5" / "pooled_mlp" / "41"
    comp_seed.mkdir(parents=True)
    pooled_seed.mkdir(parents=True)
    comp_checkpoint = comp_seed / "checkpoint_best.pt"
    pooled_checkpoint = pooled_seed / "checkpoint_best.pt"
    comp_checkpoint.write_bytes(b"current-clsm")
    pooled_checkpoint.write_bytes(b"current-pooled")
    row = {"world_seed": 801, "held_out_code": 3, "adaptation_steps": 4,
           "clsm_energy_score": 2., "pooled_energy_score": 1., "pooled_minus_clsm": -1.}
    current_dir = comp_seed / "composition_diagnostic_801"
    _write_summary(current_dir / "scores.csv", [row])
    (current_dir / "summary.json").write_text(json.dumps({
        "checkpoint_sha256": _hash(comp_checkpoint),
        "pooled_checkpoint_sha256": _hash(pooled_checkpoint),
        "probe_sha256": "fixed-probe-hash",
    }), encoding="utf-8")
    stale_dir = comp_seed / "composition_diagnostic_802"
    _write_summary(stale_dir / "scores.csv", [dict(row, world_seed=802)])
    (stale_dir / "summary.json").write_text(json.dumps({
        "checkpoint_sha256": "old-checkpoint", "pooled_checkpoint_sha256": _hash(pooled_checkpoint),
    }), encoding="utf-8")

    mem_seed = results / "mem" / "HalfCheetah-v5" / "clsm" / "41"
    checkpoint = mem_seed / "checkpoint_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"current-memory")
    for budget in (128, 256):
        run_dir = mem_seed / f"memory_diagnostic_801_components_{budget}"
        if budget == 128:
            run_dir = mem_seed / "memory_diagnostic_801"
        _write_summary(run_dir / "memory_adaptation.csv", [{
            "world_seed": 801, "budget": 16, "gain": float(budget),
            "reuse_max_dropped_mass": .002, "scratch_max_dropped_mass": .003,
        }])
        (run_dir / "summary.json").write_text(json.dumps({
            "checkpoint_sha256": _hash(checkpoint), "component_budget": budget,
            "probe_sha256": "fixed-probe-hash", "budgets": [16],
            "matched_compute_scratch": False,
        }), encoding="utf-8")

    report = summarize(results)
    c1 = [row for row in report["effects"] if row["claim"] == "C1"]
    c3 = [row for row in report["effects"] if row["claim"] == "C3"]
    assert len(c1) == 1 and c1[0]["mean_effect"] == -1.
    assert len(c3) == 2
    assert {row["protocol"]["component_budget"] for row in c3} == {128, 256}
    assert all(row["inference_degraded"] for row in c3)
    assert report["stale_or_unverified_run_directories_excluded"] == 1
