"""Write provenance/config/split records for completed integration reproductions."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q1/q1_v5_integration_repro"
ZIP = Path.home() / "Downloads/Q1_V5_SOURCE.zip"
SEEDS = (20261024444, 20261024618)
NTRAIN = 40
NDEV = 16


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    code_files = [ROOT / x for x in (
        "q1_v5/__init__.py", "q1_v5/model.py", "q1_v5/joint_refit.py",
        "q1_v4/__init__.py", "q1_v4/stage_contrast.py", "scripts/q1_v5_experiment.py",
        "tests/test_q1_v5_model.py", "tests/test_stage_contrast.py",
        "q1_v5/identifiability.py", "scripts/q1_v5_identifiability.py",
        "tests/test_q1_v5_identifiability.py", "scripts/q1_v5_finalize_repro.py")]
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_files}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    except Exception:
        head, branch = None, None
    (OUT / "config.json").write_text(json.dumps({
        "method": "Q1 V5 supplied exploratory implementation",
        "train_backgrounds": NTRAIN, "development_backgrounds_per_regime": NDEV,
        "adaptation_budgets": [5, 10, 20, 50, 100], "old_support": 96,
        "query_transitions": 256, "dt": 0.05, "sigma": 0.0005,
        "program_count": 21, "protected_test_used": False,
        "inference_policy": "runner reports old MAP, old+EB, joint MAP and joint+EB diagnostics; no single deployable policy selected",
        "status": "exploratory integration reproduction; not Phase B",
    }, indent=2))
    manifest = {"manifest_type": "deterministic reconstruction of runner seeds; exploratory only",
                "protected_test_used": False, "seeds": {}}
    for seed in SEEDS:
        train = [{"background": i, "parameter_seed": seed + i * 97 + 33,
                  "old_trajectory_seed": seed + i * 811 + 191,
                  "after_trajectory_seeds": [seed + i * 811 + 90119 + j * 123 for j in range(4)]}
                 for i in range(NTRAIN)]
        dev = {}
        for regime, offset in (("IDLIKE_DEVELOPMENT", 0), ("BACKGROUND_OOD_DEVELOPMENT", 3_000_000)):
            dev[regime] = [{"background": i,
                            "parameter_seed": seed + 9_000_000 + offset + i * 1987,
                            "per_program_stage_pair_seeds": [seed + 9_000_000 + offset + i * 1987 + p * 7477 + 99
                                                               for p in range(21)]}
                           for i in range(NDEV)]
        manifest["seeds"][str(seed)] = {"train": train, "development": dev}
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "provenance.json").write_text(json.dumps({
        "repository": "ducnguyenhuu26/Good", "base_commit": "eff3d5c460f273917f132b37adf2d503ff558280",
        "working_branch": branch, "head_at_finalize": head,
        "source_zip_sha256": digest(ZIP), "source_hashes": source_hashes,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
        "reproduction_seeds": list(SEEDS), "protected_test_used": False,
    }, indent=2))
    (OUT / "tests.json").write_text(json.dumps({
        "focused_v5_and_stage_tests": {"command": "python -m pytest -q tests/test_q1_v5_model.py tests/test_stage_contrast.py tests/test_q1_v5_identifiability.py",
                                         "result": "15 passed"},
        "full_repository_tests": {"command": "python -m pytest -q tests",
                                  "result": "315 passed, 3 failed; unrelated Windows path expectation mismatches in tests/test_reproduction_commands.py"},
        "compileall": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
