"""Source and checkpoint provenance helpers for CLSM experiments."""
from __future__ import annotations

import subprocess
import hashlib
from pathlib import Path


SOURCE_PATHS = ("src/rwt/clsm", "experiments/rwt/clsm", "pyproject.toml")


def source_fingerprint() -> str:
    """Hash architecture/evaluator source bytes; documentation edits do not alter it."""
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                        text=True).strip())
    files = []
    for name in SOURCE_PATHS:
        path = root / name
        files.extend(path.rglob("*.py") if path.is_dir() else [path])
        if path.is_dir():
            files.extend(path.rglob("*.yaml"))
            files.extend(path.rglob("*.json"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def git_source_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain",
                                          "--untracked-files=normal", "--", *SOURCE_PATHS], text=True).strip())
    return commit, dirty


def verify_checkpoint_source(payload: dict, *, require_clean: bool = True) -> None:
    _, current_dirty = git_source_state()
    if payload.get("source_fingerprint") != source_fingerprint():
        raise ValueError("checkpoint architecture/evaluator source fingerprint differs from current checkout")
    if require_clean and (payload.get("source_dirty", True) or current_dirty):
        raise ValueError("checkpoint or current source is uncommitted; reproducible evaluation requires clean source")


def assert_deployable_checkpoint(payload: dict) -> None:
    """Keep privileged oracle diagnostics out of the deployable CLSM path."""
    if payload.get("role") == "oracle_diagnostic":
        raise ValueError("oracle_diagnostic checkpoint cannot be loaded as deployable_clsm_r2")
