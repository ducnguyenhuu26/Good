"""Compact plots and Markdown summaries for saved Q1 stage artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_report(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sections = ["# Q1 run report", "", f"Run directory: `{root.as_posix()}`", ""]
    m1_path = root / "m1" / "metrics.json"
    if m1_path.exists():
        m1 = json.loads(m1_path.read_text(encoding="utf-8"))
        budgets = list(m1["budgets"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(budgets, [m1["budgets"][n]["identity_accuracy"] for n in budgets], "o-", label="program ID")
        ax.plot(budgets, [m1["budgets"][n]["functional_class_accuracy"] for n in budgets], "o-", label="function class")
        ax.set(xlabel="Adaptation transitions N", ylabel="Oracle accuracy", ylim=(0, 1.02), title="M1 oracle discrimination")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "m1_oracle_accuracy.png", dpi=150)
        plt.close(fig)
        oracle_gate = m1.get("m1_oracle_gate", {}).get("status", "NOT RECORDED")
        sections.extend(["## M1", "", f"Oracle gate: **{oracle_gate}**.",
                         "Functional-class accuracy accounts for observationally equivalent programs.",
                         "![M1 oracle accuracy](figures/m1_oracle_accuracy.png)", ""])
    m2_path = root / "m2" / "metrics.json"
    if m2_path.exists():
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        curve = m2.get("learning_curve", [])
        if curve:
            fig, ax = plt.subplots(figsize=(6, 4))
            train_key = "train_mse_delta_v" if "train_mse_delta_v" in curve[0] else "train_mse_normalized"
            base_key = "context_free_train_mse_delta_v" if "context_free_train_mse_delta_v" in curve[0] else "context_free_train_mse_normalized"
            ax.plot([r["epoch"] for r in curve], [r[train_key] for r in curve], label="context train")
            ax.plot([r["epoch"] for r in curve], [r[base_key] for r in curve], label="context-free train")
            if "validation_mse_delta_v" in curve[0]:
                ax.plot([r["epoch"] for r in curve], [r["validation_mse_delta_v"] for r in curve], "--", label="context validation")
                ax.plot([r["epoch"] for r in curve], [r["context_free_validation_mse_delta_v"] for r in curve], "--", label="context-free validation")
            elif "validation_mse_normalized" in curve[0]:
                ax.plot([r["epoch"] for r in curve], [r["validation_mse_normalized"] for r in curve], "--", label="context validation")
                ax.plot([r["epoch"] for r in curve], [r["context_free_validation_mse_normalized"] for r in curve], "--", label="context-free validation")
            ax.set(xlabel="Epoch", ylabel="Delta-v MSE", title="M2 train and validation curves")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "m2_learning_curve.png", dpi=150)
            plt.close(fig)
        val = m2["validation"]
        sections.extend(["## M2", "", f"Validation MSE: context {val['context_mse']:.6g}; "
                         f"context-free {val['context_free_mse']:.6g}; relative improvement "
                         f"{val['relative_improvement_over_context_free']:.2%}.",
                         f"Context-swap response RMSE: {val['context_swap_response_rmse']:.6g}.",
                         "![M2 training curve](figures/m2_learning_curve.png)", ""])
    m3_path = root / "m3" / "metrics.json"
    if m3_path.exists():
        m3 = json.loads(m3_path.read_text(encoding="utf-8"))["validation"]
        distance_payload = m3.get("functional_distance_rmse_delta_v", m3.get("functional_distance_rmse"))
        if isinstance(distance_payload, dict):
            distance_payload = distance_payload["common_probe_rmse_delta_v"]
        matrix = np.asarray(distance_payload)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="magma")
        ax.set(xlabel="Code", ylabel="Code", title="M3 common-probe response distances")
        fig.colorbar(image, ax=ax, label="Physical delta-v RMSE")
        fig.tight_layout()
        fig.savefig(figures / "m3_functional_distance.png", dpi=150)
        plt.close(fig)
        sections.extend(["## M3", "", f"Average transfer MSE: {m3['average_operator_mse']:.6g}; "
                         f"persistence {m3['average_persistence_mse']:.6g}; wrong-code {m3['average_wrong_code_mse']:.6g}.",
                         "![M3 functional distance](figures/m3_functional_distance.png)", ""])
    m4_path = root / "m4" / "metrics.json"
    if m4_path.exists():
        m4 = json.loads(m4_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        regimes = ("composition_ood", "background_ood", "joint_ood")
        colors = {"composition_ood": "tab:blue", "background_ood": "tab:orange", "joint_ood": "tab:red"}
        for regime in regimes:
            rows = sorted(((int(key.split("N=")[1]), row) for key, row in m4["regimes"].items()
                           if key.startswith(regime + "|")), key=lambda pair: pair[0])
            if rows:
                axes[0].plot([n for n, _ in rows], [row["class_accuracy"] for _, row in rows], "o-",
                             label=regime, color=colors[regime])
                axes[1].plot([n for n, _ in rows], [row["learned_search_query_mse"] for _, row in rows], "o-",
                             label=regime, color=colors[regime])
        axes[0].set(xlabel="Adaptation transitions N", ylabel="Functional class accuracy", ylim=(0, 1.02))
        axes[1].set(xlabel="Adaptation transitions N", ylabel="Fresh-query delta-v MSE")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle("M4 held-out composition performance")
        fig.tight_layout()
        fig.savefig(figures / "m4_heldout_performance.png", dpi=150)
        plt.close(fig)
        sections.extend(["## M4", "", f"Length penalty selected on validation: {m4['length_penalty']:g}.",
                         f"Gate: **{m4['m4_gate']['status']}**; best direct-context query MSE gain "
                         f"{m4['m4_gate']['best_relative_query_mse_gain_vs_direct']:.2%}; mean held-out "
                         f"class accuracy {m4['m4_gate']['mean_composition_ood_class_accuracy']:.3f}.",
                         "![M4 held-out performance](figures/m4_heldout_performance.png)", ""])
    output = root / "report.md"
    output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output
