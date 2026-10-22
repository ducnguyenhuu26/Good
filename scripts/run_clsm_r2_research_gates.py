"""Sequential, fail-closed orchestration for the CLSM-R2 research gates.

The runner records unavailable scientific diagnostics as INCONCLUSIVE. It never
silently substitutes the existing stationary pilot for a composition/evolution
gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rwt/clsm_r2_gates"
STAGES = ("preflight", "A", "B", "C", "D", "E")
PUBLIC_ROOT = ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/data"
EVAL_ROOT = ROOT / "results/rwt/clsm_evaluator/comp/HalfCheetah-v5"
RUN_ROOT = ROOT / "results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def record(gate: str, status: str, commit: str, fingerprint: str,
           **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "gate": gate, "status": status, "source_commit": commit,
        "source_fingerprint": fingerprint, "architecture_revision": 2,
        "dataset_hashes": {}, "checkpoint_hashes": {}, "split_ids": {},
        "config": "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml",
        "training_seeds": [], "evaluation_worlds": [], "metrics": {},
        "confidence_intervals": {}, "runtime_seconds": {},
        "limitations": [], "next_action": "",
    }
    base.update(fields)
    return base


def run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, time.perf_counter() - started


def audit_preflight(out: Path, commit: str, fingerprint: str) -> dict[str, object]:
    pre = out / "preflight"
    pre.mkdir(parents=True, exist_ok=True)
    public_manifest_path = PUBLIC_ROOT / "manifest.json"
    eval_manifest_path = EVAL_ROOT / "manifest.json"
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8")) if public_manifest_path.exists() else {}
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8")) if eval_manifest_path.exists() else {}
    public_hashes: dict[str, str | None] = {}
    eval_hashes: dict[str, str | None] = {}
    public_ok = True
    eval_ok = True
    for split in ("train", "validation", "test"):
        entry = public_manifest.get("splits", {}).get(split, {})
        path = PUBLIC_ROOT / f"{split}.npz"
        actual = sha256(path)
        public_hashes[split] = actual
        public_ok &= actual is not None and actual == entry.get("public_sha256")
        e_entry = eval_manifest.get(split, {})
        e_path = EVAL_ROOT / f"{split}_oracle.npz"
        e_actual = sha256(e_path)
        eval_hashes[split] = e_actual
        eval_ok &= e_actual is not None and e_actual == e_entry.get("evaluator_sha256")
    checkpoint_path = RUN_ROOT / "checkpoint_r2.pt"
    training_report_path = RUN_ROOT / "training_report.json"
    checkpoint = sha256(checkpoint_path)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8")) if training_report_path.exists() else {}
    commands = [
        [sys.executable, "-m", "pytest", "tests/clsm", "-q"],
        [sys.executable, "scripts/evaluate_clsm_r2_effectiveness.py", "--env", "HalfCheetah-v5",
         "--config", "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml",
         "--output-dir", "results/rwt/clsm_r2_pilot", "--seed", "4100", "--max-steps", "32"],
    ]
    runtimes: dict[str, float] = {}
    test_rc, runtimes["tests"] = run_command(commands[0], pre / "pytest.log")
    eval_rc, runtimes["replay_16_32"] = run_command(commands[1], pre / "effectiveness_replay.log")
    replay_path = RUN_ROOT / "effectiveness_evaluation_steps_32.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() and eval_rc == 0 else {}
    report = {
        "source_commit": commit, "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short"), "source_fingerprint": fingerprint,
        "architecture_revision": 2, "public_manifest_sha256": sha256(public_manifest_path),
        "public_dataset_hashes": public_hashes, "evaluator_manifest_sha256": sha256(eval_manifest_path),
        "evaluator_dataset_hashes": eval_hashes, "checkpoint_sha256": checkpoint,
        "checkpoint_source_fingerprint": replay.get("source_fingerprint"),
        "checkpoint_training_dirty": replay.get("checkpoint_training_dirty"),
        "tests": {"command": commands[0], "exit_code": test_rc,
                  "summary": (pre / "pytest.log").read_text(encoding="utf-8", errors="replace").splitlines()[-1:]},
        "replay": {"command": commands[1], "exit_code": eval_rc,
                   "source_fingerprint": replay.get("source_fingerprint"),
                   "checkpoint_source_fingerprint": replay.get("checkpoint_source_fingerprint"),
                   "checkpoint_training_dirty": replay.get("checkpoint_training_dirty"),
                   "worlds": replay.get("worlds"),
                   "steps_per_world_max": replay.get("steps_per_world_max"),
                   "transitions": replay.get("transitions"), "metrics": replay.get("world_level_mean_nll", {}),
                   "paired_r2_minus_nominal": replay.get("r2_minus_nominal_nll"),
                   "world_bootstrap_95pct_ci": replay.get("paired_world_bootstrap_95pct_ci"),
                   "worlds_r2_better": replay.get("worlds_r2_better_than_nominal"),
                   "replay_16_path": str(RUN_ROOT / "effectiveness_evaluation_steps_16.json")},
        "data_boundary": {"public_and_evaluator_paths_separate": True,
                          "public_hashes_match_manifest": bool(public_ok),
                          "evaluator_hashes_match_manifest": bool(eval_ok),
                          "loader_rejects_hidden_columns_test": "tests/clsm/test_protocol.py"},
        "training_boundary": {"status": training_report.get("full_generalized_em"),
                              "label": "partition_profile_only"},
        "evolution_initialization": {"duration_scale": 20.0, "duration_shape": 1.4,
                                     "learned_evolution_evidence": False},
        "runtime_seconds": runtimes,
    }
    (pre / "source_and_data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    prereq = (test_rc == 0 and public_ok and eval_ok and checkpoint is not None and eval_rc == 0
              and replay.get("checkpoint_source_fingerprint") == fingerprint
              and replay.get("source_fingerprint") == fingerprint
              and replay.get("checkpoint_training_dirty") is False)
    # A preflight pass only certifies lineage/software/replay, never scientific C1-C3 success.
    status = "PASS" if prereq else "FAIL"
    return record("preflight", status, commit, fingerprint,
                  dataset_hashes={**{f"public_{k}": v for k, v in public_hashes.items()},
                                  **{f"evaluator_{k}": v for k, v in eval_hashes.items()}},
                  checkpoint_hashes={"r2_4100": checkpoint},
                  split_ids={k: v.get("world_ids", []) for k, v in public_manifest.get("splits", {}).items()},
                  evaluation_worlds=replay.get("worlds", []), metrics=report["replay"],
                  confidence_intervals={"paired_world_bootstrap_95pct": replay.get("paired_world_bootstrap_95pct_ci")},
                  runtime_seconds=runtimes,
                  limitations=["Existing test worlds contain only codes 0,1,2,4.",
                               "Existing training stage is partition_profile_only; full generalized EM is false.",
                               "Weibull scale=20 and shape=1.4 are initialization values, not learned-evolution evidence."],
                  next_action="Proceed to Gate A diagnostics." if prereq else "Repair the failed preflight prerequisite.")


def assess_gate(stage: str, commit: str, fingerprint: str) -> dict[str, object]:
    if stage == "A":
        return record("A", "INCONCLUSIVE", commit, fingerprint,
                      dataset_hashes={"existing_test": sha256(PUBLIC_ROOT / "test.npz")},
                      checkpoint_hashes={"pilot_checkpoint": sha256(RUN_ROOT / "checkpoint_r2.pt")},
                      split_ids={"available_test_codes": [0, 1, 2, 4]},
                      training_seeds=[4100], evaluation_worlds=10,
                      metrics={"existing_16_32_step_filter_diagnostic": "reproduced"},
                      limitations=[
                          "No held-out composition split for codes 3,5,6,7 is present.",
                          "No genuinely trained oracle-conditioned R2 decoder or matched MLP controls are present.",
                          "No joint 3x3 Fisher audit on mixed compositions is present.",
                          "The pilot checkpoint contains a nominal-anchor decoder and train-profiled partition; no full generalized EM."],
                      next_action="Build preregistered singleton/composition splits, train condition-matched Gate A models, then evaluate before considering Gate B.")
    return record(stage, "NOT_RUN", commit, fingerprint,
                  limitations=["Sequential prerequisite Gate A is INCONCLUSIVE."],
                  next_action="Blocked by the first unresolved prerequisite; no later gate was executed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", choices=STAGES)
    group.add_argument("--through", choices=("E",))
    parser.add_argument("--diagnostic-only", action="store_true", help="record a later gate as nonconfirmatory diagnostic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", help="continue an existing run; defaults to the latest run")
    args = parser.parse_args()
    commit = git("rev-parse", "HEAD")
    sys.path.insert(0, str(ROOT / "src"))
    from rwt.clsm.serialization import source_fingerprint
    fingerprint = source_fingerprint()
    if args.dry_run:
        dry_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-dry-run")
        dry_out = RESULTS / dry_id
        dry_out.mkdir(parents=True, exist_ok=True)
        for name in ("preflight", "gate_A", "gate_B", "gate_C", "gate_D", "gate_E"):
            (dry_out / name).mkdir(exist_ok=True)
        dry_manifest = {"run_id": dry_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                        "source_commit": commit, "source_fingerprint": fingerprint,
                        "architecture_revision": 2, "dry_run": True}
        (dry_out / "manifest.json").write_text(json.dumps(dry_manifest, indent=2), encoding="utf-8")
        (dry_out / "gates.json").write_text(json.dumps({"preflight": record(
            "preflight", "NOT_RUN", commit, fingerprint,
            limitations=["Dry run only; no model selection, training, or evaluation performed."],
            next_action="Run without --dry-run to execute the preflight.")}, indent=2), encoding="utf-8")
        files_to_hash = [PUBLIC_ROOT / f"{s}.npz" for s in ("train", "validation", "test")]
        files_to_hash += [EVAL_ROOT / f"{s}_oracle.npz" for s in ("train", "validation", "test")]
        files_to_hash += [PUBLIC_ROOT / "manifest.json", EVAL_ROOT / "manifest.json",
                          ROOT / "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml",
                          RUN_ROOT / "checkpoint_r2.pt"]
        plan = {"commands": [
            [sys.executable, "-m", "pytest", "tests/clsm", "-q"],
            [sys.executable, "scripts/evaluate_clsm_r2_effectiveness.py", "--env", "HalfCheetah-v5",
             "--config", "experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml",
             "--output-dir", "results/rwt/clsm_r2_pilot", "--seed", "4100", "--max-steps", "32"]],
            "source_commit": commit, "source_fingerprint": fingerprint,
            "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in files_to_hash},
            "expected_outputs": str(dry_out),
            "split_ids": json.loads((PUBLIC_ROOT / "manifest.json").read_text()).get("splits", {}) if (PUBLIC_ROOT / "manifest.json").exists() else {},
            "approximate_budget": {"tests": "about 50 seconds on this CPU", "replay": "about 150 seconds; 10 worlds x 32 transitions", "training": "none"}}
        (dry_out / "preflight" / "dry_run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return 0
    if args.run_id:
        out = RESULTS / args.run_id
        if not out.is_dir():
            parser.error(f"unknown run_id: {args.run_id}")
    elif args.through:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS / run_id
    else:
        candidates = sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.exists() else []
        if not candidates:
            parser.error("no prior gate run; execute --stage preflight or --through E first")
        out = candidates[-1]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("preflight", "gate_A", "gate_B", "gate_C", "gate_D", "gate_E"):
        (out / name).mkdir(exist_ok=True)
    manifest_path, gates_path = out / "manifest.json", out / "gates.json"
    manifest = {"run_id": out.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": commit, "source_fingerprint": fingerprint,
                "architecture_revision": 2, "diagnostic_only": bool(args.diagnostic_only)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gates = json.loads(gates_path.read_text(encoding="utf-8")) if gates_path.exists() else {}
    stages = STAGES[:6] if args.through else (args.stage,)
    for stage in stages:
        if stage == "preflight":
            result = audit_preflight(out, commit, fingerprint)
        else:
            prereq_stage = "preflight" if stage == "A" else chr(ord(stage) - 1)
            if gates.get(prereq_stage, {}).get("status") != "PASS":
                if not args.diagnostic_only:
                    parser.error(f"Gate {stage} requires a recorded PASS for {prereq_stage}")
                result = assess_gate(stage, commit, fingerprint)
                result["status"] = "INCONCLUSIVE"
            else:
                result = assess_gate(stage, commit, fingerprint)
        gates[stage] = result
        gates_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
        print(f"{stage}: {result['status']}")
        if result["status"] in ("FAIL", "INCONCLUSIVE"):
            if args.through:
                for later in STAGES[STAGES.index(stage) + 1:]:
                    gates.setdefault(later, record(later, "NOT_RUN", commit, fingerprint,
                                                   limitations=[f"Stopped after {stage}={result['status']} by sequential gate policy."],
                                                   next_action=f"Resolve {stage} before running {later}."))
                gates_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
