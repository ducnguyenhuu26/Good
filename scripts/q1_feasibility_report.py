"""Finalize A/B/C clustered comparisons, seed audit, tests, provenance, and decision."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.q1_feasibility_common import sha256_file,sha256_tree,write_json
from q1.diagnostics.feasibility_abc import mean_ci_by_background


def read_json(path):return json.loads(path.read_text(encoding="utf-8"))


def _ci(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return [m-1.96*se,m+1.96*se]


def _load_csv(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def _a_u4(a_rows):
    groups={}
    for row in a_rows:
        if json.loads(row["true_program"]) != [3]:continue
        key=(row["regime"],row["method"],int(row["budget"]))
        groups.setdefault(key,[]).append(row)
    out={}
    for (regime,method,budget),rows in groups.items():
        acc=np.asarray([int(r["functional_class_correct"]) for r in rows],float)
        ident=np.asarray([int(r["program_identity_correct"]) for r in rows],float)
        out[f"{regime}/{method}/N{budget}"]={"n_backgrounds":len(rows),
            "u4_functional_class_accuracy":float(acc.mean()),"u4_functional_class_accuracy_ci95_background":_ci(acc),
            "raw_program_identity_accuracy":float(ident.mean()),"raw_program_identity_accuracy_ci95_background":_ci(ident),
            "background_correct_counts":acc.astype(int).tolist()}
    return {"metric_scope":"true physical U4 program only; class and raw identity separate",
            "background_uncertainty":True,"results":out}


def _b_scales(b_rows):
    out={}
    for regime in ("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD"):
        for code in (2,3):
            data=[r for r in b_rows if r["regime"]==regime and int(r["physical_code"])==code and r["predictor"]=="context_ridge"]
            bybg={r["environment_id"]:r for r in data}
            vals={name:[float(r[name]) for r in bybg.values() if r[name] not in ("",None)]
                  for name in ("old_law_conditional_mse_delta_v_squared","effect_energy_delta_v_squared",
                               "effect_energy_to_old_model_error_ratio","speed_median",
                               "v_and_speedv_design_gram_condition")}
            out[f"{regime}/U{code+1}"]={name:{"mean":float(np.mean(v)) if v else None,"ci95_background":_ci(v),
                "median":float(np.median(v)) if v else None} for name,v in vals.items()}
            out[f"{regime}/U{code+1}"]["n_backgrounds"]=len(bybg)
    return {"metrics":out,"condition_metric":"condition number of Gram for [v, ||v||v] features"}


def _c_pairs(c_rows):
    by={}
    for r in c_rows:
        key=(r["regime"],int(r["physical_code"]),r["environment_id"],r["method"])
        by.setdefault(key,[]).append(r)
    bgmetrics={}
    for key,rows in by.items():
        reg,code,bg,method=key
        bgmetrics[(reg,code,bg,method)]={"effect_skill":float(np.mean([float(x["effect_skill"]) for x in rows])),
            "after_gain":float(np.mean([float(x["learned_after_gain"]) for x in rows if x["learned_after_gain"] not in ("",None)]))}
    output={}
    for regime in ("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD"):
        for code in range(4):
            backgrounds=sorted({k[2] for k in bgmetrics if k[0]==regime and k[1]==code})
            for left,right in (("direct_effect_context","m3_beta025"),
                               ("direct_effect_context","direct_effect_context_ablated"),
                               ("direct_effect_context","a2_physics_plugin_privileged"),
                               ("direct_effect_context_ablated","m3_beta025")):
                diffs=[]
                for bg in backgrounds:
                    a=bgmetrics.get((regime,code,bg,left));b=bgmetrics.get((regime,code,bg,right))
                    if a and b:diffs.append(a["effect_skill"]-b["effect_skill"])
                output[f"{regime}/U{code+1}/{left}_minus_{right}"]={
                    "n_backgrounds":len(diffs),"mean_paired_effect_skill_difference":float(np.mean(diffs)) if diffs else None,
                    "ci95_background":_ci(diffs),"background_differences":diffs}
    return {"paired_unit":"background; training seeds averaged within each background",
            "positive_difference_favors_left_method":True,"comparisons":output}


def _make_decision(a_all,b_metrics,b_scales,c_metrics,c_pairs):
    u4=a_all["results"]
    def a(reg,method,n):return u4[f"{reg}/{method}/N{n}"]["u4_functional_class_accuracy"]
    u4c=c_metrics["aggregates"]
    def c(reg,method):
        x=u4c[f"{reg}/code3/{method}"]
        return f"{x['mean_background_effect_skill']:.3f} [{x['effect_skill_ci95_background'][0]:.3f}, {x['effect_skill_ci95_background'][1]:.3f}]"
    id_ctx=b_metrics["metrics"]["IDLIKE_DEVELOPMENT/context_ridge/log_db_mse"]["mean"]
    id_int=b_metrics["metrics"]["IDLIKE_DEVELOPMENT/intercept_only/log_db_mse"]["mean"]
    ood_ctx=b_metrics["metrics"]["DEVELOPMENT_OOD/context_ridge/log_db_mse"]["mean"]
    ood_int=b_metrics["metrics"]["DEVELOPMENT_OOD/intercept_only/log_db_mse"]["mean"]
    b_id=b_scales["metrics"]["IDLIKE_DEVELOPMENT/U4"]["effect_energy_to_old_model_error_ratio"]["mean"]
    b_ood=b_scales["metrics"]["DEVELOPMENT_OOD/U4"]["effect_energy_to_old_model_error_ratio"]["mean"]
    lines=["# Q1 feasibility A/B/C decision", "",
        "**Overall status: DEVELOPMENT_DIAGNOSTIC — INCONCLUSIVE. This is not `Q1_PASS`.** The 16 existing validation backgrounds are `DEVELOPMENT_OOD`, not confirmatory. The new ID-like backgrounds are also development data.","",
        "## Evidence summary", "",
        "| Experiment | Result | What it supports / does not establish |",
        "|---|---|---|",
        f"| A: old-physics information | U4 class accuracy at N=100: privileged true-old A1 ID-like {a('IDLIKE_DEVELOPMENT','known_old_physics',100):.3f}, estimated-old A2 {a('IDLIKE_DEVELOPMENT','estimated_old_physics',100):.3f}; `DEVELOPMENT_OOD` A1 {a('DEVELOPMENT_OOD','known_old_physics',100):.3f}, A2 {a('DEVELOPMENT_OOD','estimated_old_physics',100):.3f}. | Old-law estimation is a material ID-like bottleneck in this privileged equation-aware diagnostic; A2 nearly approaches A1 on the inspected OOD backgrounds. A1 is not a fair learned baseline. |",
        f"| B: M2 representation | Ridge probe log(d,b) MSE vs intercept: ID-like {id_ctx:.4f} vs {id_int:.4f}; OOD {ood_ctx:.4f} vs {ood_int:.4f}. U4 effect-energy / frozen-M2 old-error ratio: ID-like {b_id:.4g}, OOD {b_ood:.4g}. | The linear context probe adds no ID-like value and modest OOD value. Failure does not prove M2 lacks nonlinear information. U4 is tiny relative to executor old-law error. |",
        f"| C: direct function-space effect | U4 mean skill (background 95% CI): direct G(c) ID-like {c('IDLIKE_DEVELOPMENT','direct_effect_context')}, context-free {c('IDLIKE_DEVELOPMENT','direct_effect_context_ablated')}, M3 {c('IDLIKE_DEVELOPMENT','m3_beta025')}; OOD direct G(c) {c('DEVELOPMENT_OOD','direct_effect_context')}, context-free {c('DEVELOPMENT_OOD','direct_effect_context_ablated')}, M3 {c('DEVELOPMENT_OOD','m3_beta025')}. | Direct G transfers better than M3 for U4, but context-free G matches or exceeds context-aware G. This does not show that G uses old-law information. Training used privileged paired simulator effects. |",
        "",
        "## Interpretation and limitations", "",
        "A fits the old law using only the same 96 noisy old-stage transitions used for context encoding. The solver knows the simulator equation and uses generic bounds; it receives no true parameter initialization or post-change observations. Its old-support residual is low, but estimated quadratic drag varies substantially from the reporting-only true value, and U4 ranking is weaker than known-old A1, especially on ID-like backgrounds. This supports an old-physics observability/estimation contribution, not a proof that ordinary few-shot inference can reach A1.",
        "",
        "B fits only the original train backgrounds with a 36/12 background inner split and refits on all 48 training backgrounds. The transition-statistic baseline and intercept are included. The M2 context ridge is slightly worse than the intercept on ID-like log(d,b) error and somewhat better on OOD; an expressive probe might recover information the linear probe misses. The very small U4 effect energy compared with frozen-M2 old-law error also limits absolute after-MSE movement.",
        "",
        "C trained only direct-effect heads on 40 original train backgrounds, selected best step on 8 inner train backgrounds, and used two fixed optimizer seeds with matched update counts and minibatch schedules. The paired simulator conditional-mean targets are privileged supervision stronger than opaque group labels. Evaluation used independent natural query trajectories common across all methods within each background/program. The head is single-intervention only; composition is untested.",
        "",
        "The A2 physics plug-in and C direct head are diagnostic references, not ordinary deployment predictors. Simulator parameters/program labels are used only for target construction or offline scoring. No M2/M3/M4 retraining or test/OOD split evaluation occurred. Protected legacy trees retain their before/after hashes.",
        "",
        "## Most supported next architecture decision", "",
        "Keep a direct function-space effect head as a research baseline against latent M3, but do not yet make it context-conditioned by default: the context-ablated head is as good or better here, and the A2 ID-like fit reveals old-law estimation difficulty. Do not redesign M2 or retrain M3/M4 from this run. A fresh, preregistered development study should first test whether a nonlinear probe of frozen `c_old` adds paired predictive value over transition statistics and context-free G on broader samples from the same declared training ranges. Any later acceptance claim needs a separately frozen untouched protocol.",
        "",
        "Evidence that would change this decision: (i) a context-aware direct head beats the context-free head on paired backgrounds in both regimes and improves prediction of held-out old-law parameters; (ii) a context-free head continues to match it, arguing that old-context conditioning is unnecessary on this task distribution; or (iii) the equation-aware A2 estimate closes the A1 gap on fresh development backgrounds, weakening the old-law observability explanation.",
        "",
        "## Artifact index", "",
        "See `a_system_id_metrics.json`, `a_per_background.csv`, `a_u4_identity_confusion.json`, `b_representation_metrics.json`, `b_per_background.csv`, `c_direct_effect_metrics.json`, `c_per_background.csv`, `c_training_curves.json`, `c_gradient_diagnostics.json`, `provenance.json`, and `tests.json` in this directory.",""]
    return "\n".join(lines)


def finalize(out_path="results/q1/feasibility_abc_v1"):
    out=ROOT/out_path
    new_files=["a_u4_metrics.json","b_effect_error_scales.json","c_paired_comparisons.json","seed_collision_audit.json","provenance.json","tests.json","decision.md"]
    exist=[x for x in new_files if (out/x).exists()]
    if exist:raise FileExistsError(f"write-once final report artifacts already exist: {exist}")
    cfg=yaml.safe_load((out/"config.resolved.yaml").read_text(encoding="utf-8"))
    protocol=read_json(out/"protocol_manifest.json")
    a_rows=_load_csv(out/"a_per_background.csv");b_rows=_load_csv(out/"b_per_background.csv");c_rows=_load_csv(out/"c_per_background.csv")
    a_all=read_json(out/"a_system_id_metrics.json");b_metrics=read_json(out/"b_representation_metrics.json");c_metrics=read_json(out/"c_direct_effect_metrics.json")
    a_u4=_a_u4(a_rows);b_scale=_b_scales(b_rows);c_pair=_c_pairs(c_rows)
    write_json(out/"a_u4_metrics.json",a_u4);write_json(out/"b_effect_error_scales.json",b_scale);write_json(out/"c_paired_comparisons.json",c_pair)
    # Audit parameter seeds against every existing saved split manifest without sampling/evaluating those splits.
    id_seeds=set(int(x) for x in protocol["idlike"]["parameter_seeds"])
    manifests=[];collisions={}
    for source in sorted((ROOT/"results/q1").rglob("split_manifest.json")):
        if out in source.parents:continue
        data=read_json(source);saved={int(x) for vals in data.get("environment_seeds",{}).values() for x in vals}
        overlap=sorted(id_seeds&saved)
        manifests.append({"path":source.relative_to(ROOT).as_posix(),"sha256":sha256_file(source),
            "seed_count":len(saved),"idlike_collisions":overlap})
        if overlap:collisions[source.relative_to(ROOT).as_posix()]=overlap
    if collisions:raise RuntimeError(f"new ID-like parameter seeds collide with saved split seeds: {collisions}")
    write_json(out/"seed_collision_audit.json",{"status":"PASS_NO_COLLISIONS","idlike_parameter_seeds":sorted(id_seeds),
        "saved_split_manifests_checked":manifests,"collisions":collisions,
        "split_contents_used_for_diagnostics":"only original train and existing validation; test seed metadata read solely for collision audit"})
    # Preserve hashes from the initial audit and check them once more.
    legacy=[ROOT/"results/q1/u4_root_cause_v1",ROOT/"results/q1/repair_v1_m3_effect_ablation"]
    a_status=read_json(out/"a_status.json")
    current={str(x.relative_to(ROOT)):sha256_tree(x) for x in legacy if x.exists()}
    if current!=a_status["protected_tree_hashes_before"]:raise RuntimeError("legacy result trees changed")
    # Exact executed command history, including fixed failures and successful alternatives.
    tests={"status":"PASS","toolchain":{"python":platform.python_version(),"numpy":np.__version__,
        "scipy":scipy.__version__,"torch":torch.__version__,"device":"cpu","uv":"uv sync --extra q1 --extra test completed"},
      "commands":[
        {"command":"uv sync --extra q1 --extra test","exit_code":0,"status":"passed"},
        {"command":"uv run --extra q1 --extra test pytest -q tests/test_q1_feasibility_abc.py tests/test_q1_simulator.py tests/test_q1_no_leakage.py","exit_code":1,"status":"failed: pytest console entry point omitted repository root from sys.path; no tests collected"},
        {"command":"uv run --extra q1 --extra test python -m pytest -q tests/test_q1_feasibility_abc.py tests/test_q1_simulator.py tests/test_q1_no_leakage.py","exit_code":0,"status":"18 passed"},
        {"command":"uv run --extra q1 python -m compileall -q q1 scripts","exit_code":0,"status":"passed"},
        {"command":"uv run --extra q1 --extra test python -m pytest -q tests/test_q1_model_contracts.py tests/test_q1_u4_crossfit.py","exit_code":0,"status":"16 passed"},
        {"command":"uv run --extra q1 python scripts/q1_feasibility_a.py --config configs/q1_feasibility_abc.yaml","exit_code":1,"status":"failed: A2 multistart metadata selected list.index on dicts containing NumPy arrays; fixed to retain selected integer index"},
        {"command":"uv run --extra q1 python scripts/q1_feasibility_a.py --config configs/q1_feasibility_abc.yaml","exit_code":0,"status":"complete: 32 background estimates, 6720 per-background rows; 21.8 s"},
        {"command":"uv run --extra q1 python scripts/q1_feasibility_b.py --config configs/q1_feasibility_abc.yaml","exit_code":0,"status":"complete: 512 per-background rows; 3.4 s"},
        {"command":"uv run --extra q1 python scripts/q1_feasibility_c.py --config configs/q1_feasibility_abc.yaml --device cpu --threads 1","exit_code":1,"status":"failed after four checkpoints saved: per-code scalar scale lacked output-dimension broadcast; all initial checkpoints preserved"},
        {"command":"uv run --extra q1 python scripts/q1_feasibility_c.py --config configs/q1_feasibility_abc.yaml --device cpu --threads 1","exit_code":0,"status":"complete: four retry-named checkpoints saved without overwriting first attempt; 1280 per-background rows; 8.1 s"}],
      "preserved_failures":"Detailed causes and corrections are retained here; failed attempt checkpoints remain local in checkpoints/ and are not used for final scoring."}
    write_json(out/"tests.json",tests)
    m2=ROOT/cfg["checkpoints"]["m2"];m3=ROOT/cfg["checkpoints"]["m3"]
    sources=["pyproject.toml","uv.lock","configs/q1_feasibility_abc.yaml","q1/diagnostics/feasibility_abc.py",
      "q1/direct_effect.py","scripts/q1_feasibility_a.py","scripts/q1_feasibility_b.py","scripts/q1_feasibility_c.py",
      "scripts/q1_feasibility_common.py","scripts/q1_feasibility_report.py","tests/test_q1_feasibility_abc.py"]
    status={k:read_json(out/f"{k}_status.json") for k in ("a","b","c")}
    prov={"status":"COMPLETE_DEVELOPMENT_DIAGNOSTIC","git_sha_at_task_start":"f7e17e5627299b54939bf13434f36b4a843838b3",
      "working_tree_was_modified_during_run":True,"source_sha256":{f:sha256_file(ROOT/f) for f in sources},
      "config_sha256":sha256_file(ROOT/"configs/q1_feasibility_abc.yaml"),
      "protocol_manifest_sha256":sha256_file(out/"protocol_manifest.json"),
      "original_split_manifest_sha256":sha256_file(ROOT/cfg["split_manifest"]),
      "all_saved_split_manifests_seed_audited":manifests,
      "checkpoints":{"m2":{"path":cfg["checkpoints"]["m2"],"sha256":sha256_file(m2)},
        "m3_beta025":{"path":cfg["checkpoints"]["m3"],"sha256":sha256_file(m3)},
        "direct_effect_c":{"selected":status["c"]["checkpoint_hashes"]}},
      "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "torch":torch.__version__,"pyyaml":yaml.__version__},
      "hardware":{"platform":platform.platform(),"processor":platform.processor(),"machine":platform.machine(),"device":"cpu","torch_threads":torch.get_num_threads()},
      "wall_seconds":{"A":status["a"]["wall_seconds"],"B":status["b"]["wall_seconds"],"C":status["c"]["wall_seconds"],
        "sum_A_B_C":sum(status[k]["wall_seconds"] for k in ("a","b","c"))},
      "protected_legacy_tree_hashes_before":a_status["protected_tree_hashes_before"],
      "protected_legacy_tree_hashes_after":current,
      "test_id_used":False,"test_background_ood_used":False,"joint_ood_used":False,
      "m2_m3_m4_trained":False,"direct_effect_training_seeds":cfg["c_direct_effect"]["optimizer_seeds"],
      "direct_effect_models_trained":True,"validation_label":"DEVELOPMENT_OOD_PREVIOUSLY_INSPECTED",
      "idlike_label":"IDLIKE_DEVELOPMENT_NOT_CONFIRMATORY"}
    write_json(out/"provenance.json",prov)
    (out/"decision.md").write_text(_make_decision(a_u4,b_metrics,b_scale,c_metrics,c_pair),encoding="utf-8")
    print(json.dumps({"status":"COMPLETE_DEVELOPMENT_DIAGNOSTIC","seed_collisions":0,
        "files":new_files,"wall_seconds":prov["wall_seconds"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",default="results/q1/feasibility_abc_v1");args=parser.parse_args()
    finalize(args.out)
