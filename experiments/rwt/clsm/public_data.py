"""Strict public-only CLSM dataset loading and world-boundary validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rwt.clsm.train import PublicSequence
from experiments.rwt.clsm.build_datasets import PUBLIC_KEYS, _sha256


def load_public_split(data_dir: Path, split: str) -> tuple[list[PublicSequence], str]:
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown split")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["splits"][split]
    public_file = Path(entry["public_file"]).resolve()
    if public_file != (data_dir / f"{split}.npz").resolve():
        raise ValueError("manifest public path escaped declared data directory")
    digest = _sha256(public_file)
    if digest != entry["public_sha256"]:
        raise ValueError("public dataset hash mismatch")
    with np.load(public_file, allow_pickle=False) as arrays:
        if set(arrays.files) != set(PUBLIC_KEYS):
            raise ValueError("dataset contains nonpublic or missing columns")
        data = {key: arrays[key].copy() for key in PUBLIC_KEYS}
    n = len(data["obs"])
    if any(len(value) != n for value in data.values()) or n != entry["transitions"]:
        raise ValueError("manifest/array transition count mismatch")
    if not all(np.isfinite(data[key]).all() for key in ("obs", "action", "reward", "next_obs")):
        raise ValueError("nonfinite public transition")
    if not np.issubdtype(data["world_id"].dtype, np.integer):
        raise ValueError("world identifiers must be integers")
    ids = data["world_id"].astype(np.int64)
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, n]
    if n == 0 or len(set(ids[boundaries[:-1]].tolist())) != len(boundaries) - 1:
        raise ValueError("world identifier recurs across noncontiguous fragments")
    sequences = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if not np.array_equal(data["world_step"][start:end], np.arange(end - start)):
            raise ValueError("world steps are not contiguous from zero")
        if np.any(data["terminated"][start:end-1]) or np.any(data["truncated"][start:end-1]):
            raise ValueError("episode ending appears inside a world fragment")
        sequences.append(PublicSequence(torch.from_numpy(data["obs"][start:end]).to(torch.float64),
                                        torch.from_numpy(data["action"][start:end]).to(torch.float64),
                                        torch.from_numpy(data["next_obs"][start:end]).to(torch.float64)))
    if set(ids[boundaries[:-1]].tolist()) != set(entry["world_ids"]):
        raise ValueError("world IDs disagree with manifest")
    return sequences, digest
