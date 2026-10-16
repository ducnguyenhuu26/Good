"""Run a bounded CUDA, gradient, checkpoint, and resume preflight for Stage 3B."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from rwt.stage3b import StateDynamics

if __package__:
    from .run_stage3b_pilot import _train
else:
    from run_stage3b_pilot import _train


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    rng = np.random.default_rng(7801)
    state = rng.normal(size=(128, 17)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(128, 6)).astype(np.float32)
    next_state = state + rng.normal(0, 0.05, size=state.shape).astype(np.float32)
    reward = rng.normal(size=128).astype(np.float32)

    model = StateDynamics(hidden_dim=32)
    model.fit_normalization(state, action, next_state)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state_t = torch.as_tensor(state[:32], device=device)
    action_t = torch.as_tensor(action[:32], device=device)
    next_t = torch.as_tensor(next_state[:32], device=device)
    loss = model.loss(state_t, action_t, next_t)
    loss.backward()
    if not torch.isfinite(loss) or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("CUDA forward/backward produced nonfinite values")
    optimizer.step()
    torch.cuda.synchronize()

    train = {"state": state, "action": action, "next_state": next_state, "reward": reward}
    settings = {"hidden_dim": 16, "epochs": 2, "batch_size": 32, "learning_rate": 1e-3}
    with tempfile.TemporaryDirectory(prefix="rwt_stage3b_cuda_") as temporary:
        output = Path(temporary)
        started = perf_counter()
        trained, _, first = _train("pooled", 7801, train, train, None, None,
                                   settings, output, "cuda", False)
        resumed, _, second = _train("pooled", 7801, train, train, None, None,
                                    settings, output, "cuda", True)
        torch.cuda.synchronize()
        with torch.no_grad():
            probe = torch.as_tensor(state[:8], device=device)
            controls = torch.as_tensor(action[:8], device=device)
            prediction = trained.to(device)(probe, controls)
            resumed_prediction = resumed.to(device)(probe, controls)
        torch.testing.assert_close(prediction, resumed_prediction)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("resumed CUDA checkpoint prediction is nonfinite")
        if not (output / "checkpoints" / "pooled_7801_best.pt").is_file():
            raise AssertionError("CUDA training did not save a best checkpoint")
        print(json.dumps({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "forward_backward": "passed",
            "checkpoint_save_load": "passed",
            "resume": "passed",
            "resume_loaded_best_epoch": second["best_epoch"],
            "smoke_training_seconds": perf_counter() - started,
            "peak_vram_allocated_mb": max(first["peak_vram_allocated_mb"],
                                             second["peak_vram_allocated_mb"]),
            "prediction_shape": list(prediction.shape),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }, indent=2))


if __name__ == "__main__":
    main()
