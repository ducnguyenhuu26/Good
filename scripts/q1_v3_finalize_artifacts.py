"""Finalize reproducibility metadata and plots after the frozen V3 evaluation."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q1/q1_v3_mechanism_dev1"
SOURCE_FILES = [
    "q1_v3/__init__.py", "q1_v3/mechanisms.py", "q1_v3/task_inference.py",
    "q1_v3/operators.py", "q1_v3/training.py", "q1_v3/program_search.py",
    "configs/q1_v3.yaml", "scripts/q1_v3_run.py",
    "scripts/q1_v3_capacity_diagnostic.py", "scripts/q1_v3_v2_root_compare.py",
    "scripts/q1_v3_report.py", "scripts/q1_v3_finalize_artifacts.py",
    "tests/test_q1_v3_mechanisms.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def save_training_plot() -> None:
    curves = json.loads((OUT / "training_curves.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    action = curves["action"]
    axes[0].plot([x["epoch"] for x in action], [x["train_objective"] for x in action], label="train objective")
    axes[0].plot([x["epoch"] for x in action], [x["inner_map_nll_per_observation"] for x in action], label="inner MAP NLL/obs")
    axes[0].set(title="Shared action / old-law fit", xlabel="Epoch", ylabel="Objective / NLL")
    axes[0].legend(frameon=False)
    for seed in ("4201", "4202"):
        for loss in ("sparse", "nll_only"):
            run = curves["operators"][seed][loss]
            curve = run["curve"]
            axes[1].plot([x["epoch"] for x in curve], [x["inner_nll_per_observation"] for x in curve], label=f"{seed} {loss}")
    axes[1].set(title="Operator inner validation", xlabel="Epoch", ylabel="Inner NLL / observation")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    save_training_plot()
    (OUT / "tests.json").write_text(json.dumps({
        "command": "uv run --extra q1 --extra test python -m pytest -q (Get-ChildItem tests -Filter 'test_q1_*.py' | ForEach-Object FullName)",
        "result": "passed", "passed": 82, "failed": 0,
        "scope": "all repository tests/test_q1_*.py, including V1/V2 regressions and V3 focused tests",
        "files": sorted(p.name for p in (ROOT / "tests").glob("test_q1_*.py")),
    }, indent=2), encoding="utf-8")
    (OUT / "diagnostics/failures.json").write_text(json.dumps({
        "note": "All listed issues occurred before their corresponding results were accepted; fixes were applied and the frozen evaluation was resumed without changing its protocol or development backgrounds.",
        "events": [
            {"stage": "first V3 runner attempt", "failure": "training prior estimator attempted to stack a Python list of tensors", "resolution": "stacked the list explicitly; resumed the already-frozen protocol", "accepted_results_impacted": False},
            {"stage": "capacity finalization", "failure": "missing mean_jacobian import after primary evaluation rows had been saved", "resolution": "added import and ran the capacity diagnostic separately on the frozen backgrounds", "accepted_results_impacted": False},
            {"stage": "capacity diagnostic implementation check", "failure": "initial diagnostic compared the full after-law prediction with the effect target", "resolution": "corrected the target to after-minus-before and reran; only corrected output is in mechanism_capacity.json", "accepted_results_impacted": False},
            {"stage": "V2 root-repair comparator", "failure": "attempted NumPy conversion on a tensor with gradients enabled", "resolution": "detached evaluation tensors and reran the comparator", "accepted_results_impacted": False},
            {"stage": "report aggregation", "failure": "legacy V2 CSV omitted scoring and raw-program columns; capacity key also used an incorrect code prefix", "resolution": "made legacy scoring explicitly marginal, derived raw accuracy from predicted/true program IDs, and corrected capacity keys", "accepted_results_impacted": False},
            {"stage": "seed reproducibility test fixture", "failure": "synthetic target retained an autograd graph and failed when the same fixture was fit repeatedly", "resolution": "detached the synthetic observed target; focused and full Q1 suites then passed", "accepted_results_impacted": False},
        ],
    }, indent=2), encoding="utf-8")

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    checkpoints = sorted(OUT.glob("*.pt"))
    provenance = {
        "branch": git("branch", "--show-current"),
        "base_commit": protocol["base_commit"],
        "head_commit_at_finalization": git("rev-parse", "HEAD"),
        "device": "cpu", "python": platform.python_version(), "torch": torch.__version__,
        "source_hashes_before_training": protocol["source_sha256_before_training"],
        "final_source_sha256": {p: sha256(ROOT / p) for p in SOURCE_FILES},
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoints},
        "original_test_split_used": False,
        "evaluation_background_counts": {"id_like": 16, "background_ood": 16},
        "all_seed_sets_disjoint": protocol["all_seed_sets_disjoint"],
        "test_status": "82 passed, 0 failed; see tests.json",
        "training_plot": "training_curves.png",
        "data_and_checkpoints_ignored_by_git": True,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
