"""Bayesian optimisation of LightGBM, then untuned-vs-tuned comparison.

Step 1  Search: Optuna TPE, `--trials` trials (default 30), on the 80 train trips with
        3-fold GroupKFold by trip and every 8th fix (to keep each trial ~10 s). The study
        is stored in experiments/optuna_lightgbm.db and resumes if interrupted.
Step 2  Compare: untuned and tuned configurations are both re-run under Member 2's exact
        protocol (every 4th fix, 5-fold GroupKFold, same seed), then refit on all train
        and scored on all 10 student_test trips: CV Macro F1, student-test Top-1,
        student-test Macro F1, Log Loss, inference time.
Step 3  Final model: whichever has the higher CV Macro F1 (never chosen on student_test)
        is saved to artifacts/models/final/ for the pipeline.

Outputs: results/hpo/{trials.csv, best_params.json, untuned_vs_tuned.csv,
         param_importance.csv, search_space.json}, results/figures/hpo/*.png,
         artifacts/models/final/{model.joblib, feature_spec.json, selection.json}

Usage: python scripts/hpo.py [--trials 30]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import artifacts_dir, configs_dir, processed_dir, results_dir, set_global_seed  # noqa: E402
from pathsnap.eval.metrics import candidate_metrics, observation_metrics  # noqa: E402
from pathsnap.features.extractor import FEATURE_NAMES  # noqa: E402
from pathsnap.hpo.optuna_search import SEARCH_SPACE_DOC, UNTUNED_PARAMS, make_lgbm, run_study  # noqa: E402

MCFG = yaml.safe_load(open(configs_dir() / "model_config.yaml"))
SEED = int(MCFG["seed"])
OUT = results_dir() / "hpo"
FIG = results_dir() / "figures" / "hpo"


def full_protocol(params, tr, te, folds):
    """Member 2's protocol: 5-fold grouped CV on stride-4 train, then refit + student_test."""
    cv_f1, cv_ll, cv_top1 = [], [], []
    for tri, vai in folds:
        m = make_lgbm(params, SEED).fit(tr[FEATURE_NAMES].iloc[tri], tr["y"].iloc[tri])
        v = tr.iloc[vai]
        p = m.predict_proba(v[FEATURE_NAMES])[:, 1]
        cm = candidate_metrics(v["y"].to_numpy(), p)
        s = v[["trip_id", "obs_idx", "y"]].copy(); s["score"] = p
        cv_f1.append(cm["macro_f1"]); cv_ll.append(cm["log_loss"]); cv_top1.append(observation_metrics(s)["top1_acc"])
    t0 = time.perf_counter()
    model = make_lgbm(params, SEED).fit(tr[FEATURE_NAMES], tr["y"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = model.predict_proba(te[FEATURE_NAMES])[:, 1]
    pred_s = time.perf_counter() - t0
    cm = candidate_metrics(te["y"].to_numpy(), p)
    s = te[["trip_id", "obs_idx", "arc_id", "y"]].copy(); s["score"] = p
    om = observation_metrics(s)
    row = {"CV Macro F1": np.mean(cv_f1), "CV Macro F1 std": np.std(cv_f1, ddof=1), "CV Top-1": np.mean(cv_top1),
           "CV Log Loss": np.mean(cv_ll), "Student-Test Top-1 Accuracy": om["top1_acc"],
           "Student-Test Macro F1": cm["macro_f1"], "Student-Test Log Loss": cm["log_loss"],
           "Inference Time (ms / 1k candidates)": pred_s / len(te) * 1e6, "Fit time (s)": fit_s}
    return row, model, s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    set_global_seed(SEED)
    for p in (OUT, FIG, ROOT / "experiments"):
        p.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(processed_dir() / "features" / "train.parquet")
    te = pd.read_parquet(processed_dir() / "features" / "student_test.parquet")

    # ---- step 1: search
    search = full[full.obs_idx % 8 == 0].reset_index(drop=True)
    s_folds = list(GroupKFold(n_splits=3).split(search, search["y"], search["trip_id"]))
    storage = f"sqlite:///{ROOT / 'experiments' / 'optuna_lightgbm.db'}"
    t0 = time.perf_counter()
    study = run_study(search, s_folds, FEATURE_NAMES, a.trials, storage, "lightgbm_macro_f1", SEED)
    search_s = time.perf_counter() - t0
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration", "user_attrs"))
    trials.to_csv(OUT / "trials.csv", index=False)
    done = trials[trials.state == "COMPLETE"]
    best = dict(study.best_params)
    (OUT / "search_space.json").write_text(json.dumps({
        "sampler": "TPE (seed 42)", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "objective": "mean Macro F1 over 3-fold GroupKFold by trip, every 8th fix of the 80 train trips",
        "ranges": SEARCH_SPACE_DOC, "untuned_enqueued_as_trial_0": UNTUNED_PARAMS}, indent=2) + "\n")
    (OUT / "best_params.json").write_text(json.dumps({
        "best_params": best, "best_search_macro_f1": study.best_value, "n_trials": len(trials),
        "n_complete": int(len(done)), "n_pruned": int((trials.state == "PRUNED").sum()),
        "search_wall_clock_s_this_run": search_s}, indent=2) + "\n")
    print(f"search: {len(trials)} trials ({len(done)} complete), best {study.best_value:.4f} {best}", flush=True)

    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        pd.DataFrame({"param": list(imp), "importance": list(imp.values())}).to_csv(OUT / "param_importance.csv", index=False)
    except Exception as exc:  # importance needs >1 complete trial
        print("param importance skipped:", exc)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done.number, done.value, s=14, label="trial (3-fold CV Macro F1)")
    ax.plot(done.number, done.value.cummax(), color="C1", label="best so far")
    ax.axhline(done[done.number == 0].value.iloc[0] if (done.number == 0).any() else np.nan, color="grey",
               ls="--", label="untuned (trial 0)")
    ax.set(xlabel="trial", ylabel="Macro F1", title="Optuna TPE optimisation history (LightGBM)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "optimisation_history.png", dpi=140); plt.close(fig)
    if (OUT / "param_importance.csv").exists():
        pi = pd.read_csv(OUT / "param_importance.csv").sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(pi.param, pi.importance, color="#4C78A8")
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        fig.tight_layout(); fig.savefig(FIG / "param_importance.png", dpi=140); plt.close(fig)

    # ---- step 2: untuned vs tuned under Member 2's protocol
    tr = full[full.obs_idx % int(MCFG["obs_stride"]) == 0].reset_index(drop=True)
    folds = list(GroupKFold(n_splits=int(MCFG["cv_folds"])).split(tr, tr["y"], tr["trip_id"]))
    tuned_params = {**best}
    rows, models, scores = [], {}, {}
    for name, params in [("LightGBM untuned", UNTUNED_PARAMS), ("LightGBM tuned (Optuna)", tuned_params)]:
        row, model, sc = full_protocol(params, tr, te, folds)
        rows.append({"Model": name, **row, "params": json.dumps(params)})
        models[name], scores[name] = model, sc
        print(name, {k: round(v, 4) for k, v in row.items()}, flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "untuned_vs_tuned.csv", index=False)

    # ---- step 3: final model chosen on CV Macro F1 only
    win = comp.sort_values(["CV Macro F1", "CV Log Loss"], ascending=[False, True]).iloc[0]
    final = artifacts_dir() / "models" / "final"
    final.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[win.Model], final / "model.joblib", compress=3)
    shutil.copy(artifacts_dir() / "feature_spec.json", final / "feature_spec.json")
    (final / "selection.json").write_text(json.dumps({
        "selected_model": "LightGBM", "variant": win.Model, "params": json.loads(win.params),
        "rule": "higher 5-fold grouped CV Macro F1 (tie: lower CV Log Loss); student_test not used to choose",
        "cv_macro_f1": win["CV Macro F1"], "student_test_top1": win["Student-Test Top-1 Accuracy"],
        "student_test_macro_f1": win["Student-Test Macro F1"], "seed": SEED}, indent=2, default=float) + "\n")
    scores[win.Model].to_parquet(OUT / "final_student_test_scores.parquet", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, col in zip(ax, ["CV Macro F1", "Student-Test Top-1 Accuracy", "Student-Test Macro F1"]):
        axis.bar(["untuned", "tuned"], comp[col], color=["#9D9D9D", "#4C78A8"])
        lo = comp[col].min()
        axis.set(title=col, ylim=(max(0, lo - 0.03), min(1, comp[col].max() + 0.01)))
        for i, v in enumerate(comp[col]):
            axis.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("LightGBM: untuned vs Optuna-tuned (same folds, same student-test trips)")
    fig.tight_layout(); fig.savefig(FIG / "untuned_vs_tuned.png", dpi=140); plt.close(fig)
    print("FINAL:", win.Model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
