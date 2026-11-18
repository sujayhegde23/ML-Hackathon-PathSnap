"""Train, compare and select the candidate classifier.

Protocol (fixed before looking at any result):
  * Fit data: the 80 train trips, every `obs_stride`-th fix with all its candidates
    (consecutive 1 Hz fixes are near-duplicates, so this loses little and is ~4x faster).
  * 5-fold GroupKFold by trip_id: a trip is never split between training and validation,
    so scores measure performance on unseen trips / cities.
  * Each model is then refit on all training rows and scored on every fix of the 10
    student_test trips (no subsampling). student_test is never used to choose anything.
  * Selection rule: highest mean CV Macro F1; lowest mean CV Log Loss breaks ties.

Outputs
  results/models/cv_folds.csv, model_comparison.csv, feature_ablation.csv,
  permutation_importance.csv, candidate_scores_{train_oof,student_test}.parquet
  results/figures/models/*.png
  experiments/<timestamp>_<model>.json
  artifacts/models/<model>.joblib and artifacts/models/selected/{model.joblib,
  feature_spec.json, selection.json}

Usage: python scripts/train.py [--models LightGBM,MLP] [--skip-extras]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import artifacts_dir, configs_dir, processed_dir, results_dir, set_global_seed  # noqa: E402
from pathsnap.eval.metrics import candidate_metrics, observation_metrics, route_continuity, top1_rows  # noqa: E402
from pathsnap.features.extractor import FEATURE_GROUPS, FEATURE_NAMES  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_prediction  # noqa: E402
from pathsnap.models.registry import CONFIG_TEXT, FAMILY, make_model  # noqa: E402

MCFG = yaml.safe_load(open(configs_dir() / "model_config.yaml"))
SEED = int(MCFG["seed"])
RES = results_dir() / "models"
FIG = results_dir() / "figures" / "models"
ART = artifacts_dir() / "models"
EXP = ROOT / "experiments"
KEYS = ["trip_id", "obs_idx", "arc_id", "y"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def knn_subsample(df: pd.DataFrame) -> pd.DataFrame:
    """KNN keeps every training row and compares each query against all of them, so it
    is fitted on a random subset of whole fixes (all candidates of a kept fix)."""
    cap = int(MCFG["knn_max_train_rows"])
    if len(df) <= cap:
        return df
    keys = df[["trip_id", "obs_idx"]].drop_duplicates().sample(frac=cap / len(df), random_state=SEED)
    return df.merge(keys, on=["trip_id", "obs_idx"])


def fit_predict(name, train_df, eval_dfs, feats=FEATURE_NAMES):
    model = make_model(name, SEED)
    fit_df = knn_subsample(train_df) if name == "KNN" else train_df
    t0 = time.perf_counter()
    model.fit(fit_df[feats], fit_df["y"])
    fit_s = time.perf_counter() - t0
    preds, pred_s = [], []
    for d in eval_dfs:
        t0 = time.perf_counter()
        preds.append(model.predict_proba(d[feats])[:, 1])
        pred_s.append(time.perf_counter() - t0)
    return model, preds, fit_s, pred_s


def score(df, p):
    s = df[KEYS].copy()
    s["score"] = p
    m = candidate_metrics(s["y"].to_numpy(), p)
    m.update(observation_metrics(s))
    return m, s


def continuity(scored, graphs):
    top = top1_rows(scored).sort_values(["trip_id", "obs_idx"])
    return float(np.mean([route_continuity(g["arc_id"].to_numpy(), *graphs[t]) for t, g in top.groupby("trip_id")]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MCFG["models"]))
    ap.add_argument("--skip-extras", action="store_true", help="skip ablation and importance")
    a = ap.parse_args()
    set_global_seed(SEED)
    for p in (RES, FIG, ART, EXP):
        p.mkdir(parents=True, exist_ok=True)

    feat_dir = processed_dir() / "features"
    full = pd.read_parquet(feat_dir / "train.parquet")
    tr = full[full.obs_idx % int(MCFG["obs_stride"]) == 0].reset_index(drop=True)
    te = pd.read_parquet(feat_dir / "student_test.parquet")
    assert not set(tr.trip_id) & set(te.trip_id), "a student_test trip leaked into training"
    graphs = {}
    for ref in discover_trips(partitions=("student_test",)):
        t = load_trip_for_prediction(ref.trip_dir, ref.partition)
        graphs[t.trip_id] = (t.arcs[:, 0], t.arcs[:, 1])
    print(f"train rows {len(tr):,} ({tr.trip_id.nunique()} trips, stride {MCFG['obs_stride']}), "
          f"student_test rows {len(te):,}; positives {tr.y.mean():.3f}", flush=True)

    folds = list(GroupKFold(n_splits=int(MCFG["cv_folds"])).split(tr, tr["y"], tr["trip_id"]))
    for tri, vai in folds:
        assert not set(tr.trip_id.iloc[tri]) & set(tr.trip_id.iloc[vai]), "trip split across folds"

    fold_rows, summary, cms = [], [], {}
    # baseline: nearest candidate wins, no learning
    base_p = lambda d: 1.0 / (1.0 + d["dist_rank"].to_numpy(dtype=float))  # noqa: E731
    for k, (_, vai) in enumerate(folds):
        m, _ = score(tr.iloc[vai], base_p(tr.iloc[vai]))
        fold_rows.append(dict(model="NearestArc_baseline", fold=k, split="val", **m))
    m_te, sc = score(te, base_p(te))
    summary.append(dict(model="NearestArc_baseline", family="Rule (no learning)",
                        config="closest candidate wins", test=m_te, cont=continuity(sc, graphs), fit_s=0.0, ms_1k=0.0))

    oof_best = {}
    for name in a.models.split(","):
        print(f"== {name}", flush=True)
        oof = np.full(len(tr), np.nan)
        for k, (tri, vai) in enumerate(folds):
            t, v = tr.iloc[tri], tr.iloc[vai]
            samp = t.sample(n=min(len(t), int(MCFG["train_score_sample_rows"])), random_state=SEED)
            _, (pv, pt), fit_s, _ = fit_predict(name, t, [v, samp])
            oof[vai] = pv
            mv, _ = score(v, pv)
            mt = candidate_metrics(samp["y"].to_numpy(), pt)
            fold_rows.append(dict(model=name, fold=k, split="val", fit_s=fit_s, **mv))
            fold_rows.append(dict(model=name, fold=k, split="train", fit_s=fit_s, **mt))
            print(f"  fold {k}: val macroF1={mv['macro_f1']:.4f} top1={mv['top1_acc']:.4f} "
                  f"train macroF1={mt['macro_f1']:.4f} fit={fit_s:.1f}s", flush=True)
        oof_best[name] = oof
        model, (pte,), fit_s, (pred_s,) = fit_predict(name, tr, [te])
        m_te, sc = score(te, pte)
        cms[name] = confusion_matrix(te["y"], (pte >= 0.5).astype(int))
        summary.append(dict(model=name, family=FAMILY[name], config=CONFIG_TEXT[name], test=m_te,
                            cont=continuity(sc, graphs), fit_s=fit_s, ms_1k=pred_s / len(te) * 1e6))
        joblib.dump(model, ART / f"{name}.joblib", compress=3)
        sc.to_parquet(RES / f"student_test_scores_{name}.parquet", index=False)
        fr = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        (EXP / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.json").write_text(json.dumps({
            "model": name, "config": CONFIG_TEXT[name], "seed": SEED, "git_commit": git_hash(),
            "obs_stride": MCFG["obs_stride"], "folds": fr.to_dict("records"), "student_test": m_te,
            "fit_seconds_full": fit_s}, indent=2, default=float))
        if name == "LightGBM":
            cm = make_model("LightGBM", SEED)
            cm.fit(tr[FEATURE_NAMES], tr["y"], eval_set=[(tr[FEATURE_NAMES], tr["y"]), (te[FEATURE_NAMES], te["y"])],
                   eval_names=["train", "student_test"], eval_metric="binary_logloss")
            pd.DataFrame({k: v["binary_logloss"] for k, v in cm.evals_result_.items()}).rename_axis("iteration") \
                .to_csv(RES / "lightgbm_logloss_curve.csv")
        if name == "MLP":
            mlp = model[-1]
            pd.DataFrame({"train_loss": mlp.loss_curve_, "val_accuracy": mlp.validation_scores_}) \
                .rename_axis("epoch").to_csv(RES / "mlp_training_curve.csv")
        pd.DataFrame(fold_rows).to_csv(RES / "cv_folds.csv", index=False)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(RES / "cv_folds.csv", index=False)
    val = folds_df[folds_df.split == "val"].groupby("model")
    trn = folds_df[folds_df.split == "train"].groupby("model")
    rows = []
    for s in summary:
        m, learned = s["model"], s["model"] != "NearestArc_baseline"
        rows.append({
            "Model": m, "Algorithm Family": s["family"], "Key Configuration": s["config"],
            "CV Macro F1-score": val["macro_f1"].mean()[m],
            "Top-1 Accuracy": s["test"]["top1_acc"],
            "Route continuity": s["cont"],
            "Student-Test Macro F1-score": s["test"]["macro_f1"],
            "Log Loss": s["test"]["log_loss"] if learned else np.nan,
            "Inference Time": f"{s['ms_1k']:.2f} ms / 1k candidates",
            "CV Macro F1 std": val["macro_f1"].std()[m],
            "CV Top-1": val["top1_acc"].mean()[m],
            "CV Log Loss": val["log_loss"].mean()[m] if learned else np.nan,
            "Train Macro F1": trn["macro_f1"].mean()[m] if m in trn.groups else np.nan,
            "Student-Test Precision": s["test"]["precision"], "Student-Test Recall": s["test"]["recall"],
            "Student-Test Top-1 when covered": s["test"]["top1_when_covered"],
            "Candidate miss rate (student test)": s["test"]["candidate_miss_rate"],
            "Fit time (s)": s["fit_s"],
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(RES / "model_comparison.csv", index=False)
    learned = comp[comp.Model != "NearestArc_baseline"]
    best = learned.sort_values(["CV Macro F1-score", "CV Log Loss"], ascending=[False, True]).iloc[0]
    sel = ART / "selected"
    sel.mkdir(exist_ok=True)
    shutil.copy(ART / f"{best.Model}.joblib", sel / "model.joblib")
    shutil.copy(artifacts_dir() / "feature_spec.json", sel / "feature_spec.json")
    (sel / "selection.json").write_text(json.dumps({
        "selected_model": best.Model, "rule": MCFG["selection_rule"],
        "cv_macro_f1": best["CV Macro F1-score"], "cv_log_loss": best["CV Log Loss"],
        "student_test_top1": best["Top-1 Accuracy"], "student_test_macro_f1": best["Student-Test Macro F1-score"],
        "output": "predict_proba positive-class probability; not separately calibrated",
        "seed": SEED, "git_commit": git_hash()}, indent=2, default=float) + "\n")
    oof = tr[KEYS].copy()
    oof["score"] = oof_best[best.Model]
    oof.to_parquet(RES / "candidate_scores_train_oof.parquet", index=False)
    shutil.copy(RES / f"student_test_scores_{best.Model}.parquet", RES / "candidate_scores_student_test.parquet")
    np.savez(RES / "confusion_matrices.npz", **cms)
    pd.set_option("display.width", 250)
    print(comp.iloc[:, [0, 3, 4, 5, 6, 7, 8, 13]].round(4).to_string(index=False))
    print("SELECTED:", best.Model)
    make_figures(comp, folds_df, cms)
    if not a.skip_extras:
        ablation_and_importance(tr, folds, best.Model)
    return 0


def make_figures(comp, folds_df, cms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = comp.sort_values("CV Macro F1-score")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(c.Model, c["CV Macro F1-score"], xerr=c["CV Macro F1 std"].fillna(0), color="#4C78A8")
    ax.set(xlabel="CV Macro F1 (5-fold, grouped by trip)", title="Model comparison", xlim=(0.4, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for _, r in comp.iterrows():
        ax.scatter(r["Student-Test Macro F1-score"], r["Top-1 Accuracy"])
        ax.annotate(r.Model, (r["Student-Test Macro F1-score"], r["Top-1 Accuracy"]), fontsize=7)
    ax.set(xlabel="Student-test Macro F1", ylabel="Student-test Top-1 accuracy", title="Student-test results")
    fig.tight_layout(); fig.savefig(FIG / "student_test_scatter.png", dpi=140); plt.close(fig)

    l = comp.dropna(subset=["Train Macro F1"])
    x = np.arange(len(l))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, l["Train Macro F1"], 0.4, label="train folds")
    ax.bar(x + 0.2, l["CV Macro F1-score"], 0.4, label="validation folds")
    ax.set_xticks(x, l.Model, rotation=30, ha="right"); ax.set_ylim(0.4, 1.0); ax.legend()
    ax.set(title="Train vs validation Macro F1 (overfitting check)")
    fig.tight_layout(); fig.savefig(FIG / "train_vs_val.png", dpi=140); plt.close(fig)

    n = len(cms)
    if n:
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
        for ax, (name, cm) in zip(np.ravel(axes), cms.items()):
            ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7)
            ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
        for ax in np.ravel(axes)[n:]:
            ax.axis("off")
        fig.suptitle("Student-test confusion matrices (threshold 0.5)")
        fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=140); plt.close(fig)

    for fname, cols, title in [("lightgbm_logloss_curve.csv", ["train", "student_test"], "LightGBM log loss per iteration"),
                               ("mlp_training_curve.csv", ["train_loss", "val_accuracy"], "MLP training curve")]:
        p = RES / fname
        if p.exists():
            d = pd.read_csv(p)
            fig, ax = plt.subplots(figsize=(6, 4))
            for col in cols:
                ax.plot(d[col], label=col)
            ax.set(title=title, xlabel=d.columns[0]); ax.legend()
            fig.tight_layout(); fig.savefig(FIG / fname.replace(".csv", ".png"), dpi=140); plt.close(fig)


def ablation_and_importance(tr, folds, selected):
    """Ablation: drop one feature group, retrain LightGBM on the same folds (first 3, for
    time), record the drop. Importance: permutation importance of the selected model on
    fold 0's validation trips (never student_test)."""
    from sklearn.inspection import permutation_importance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for drop in [None, *FEATURE_GROUPS]:
        feats = [f for f in FEATURE_NAMES if drop is None or f not in FEATURE_GROUPS[drop]]
        f1s, t1s = [], []
        for tri, vai in folds[:3]:
            _, (p,), _, _ = fit_predict("LightGBM", tr.iloc[tri], [tr.iloc[vai]], feats)
            m, _ = score(tr.iloc[vai], p)
            f1s.append(m["macro_f1"]); t1s.append(m["top1_acc"])
        rows.append(dict(dropped_group=drop or "none (all features)", n_features=len(feats),
                         cv_macro_f1=np.mean(f1s), cv_top1=np.mean(t1s)))
        print("ablation", rows[-1], flush=True)
    ab = pd.DataFrame(rows)
    ab.to_csv(RES / "feature_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ab.dropped_group, ab.cv_top1, color="#F58518")
    ax.set(xlabel="CV Top-1 accuracy (LightGBM, 3 folds)", title="Feature-group ablation: lower = group matters more",
           xlim=(max(0, ab.cv_top1.min() - 0.05), 1.0))
    fig.tight_layout(); fig.savefig(FIG / "feature_ablation.png", dpi=140); plt.close(fig)

    tri, vai = folds[0]
    model, _, _, _ = fit_predict(selected, tr.iloc[tri], [])
    v = tr.iloc[vai]
    v = v.sample(n=min(len(v), 60_000), random_state=SEED)
    pi = permutation_importance(model, v[FEATURE_NAMES], v["y"], scoring="f1_macro", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean")
    imp.sort_values("importance_mean", ascending=False).to_csv(RES / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_std, color="#54A24B")
    ax.set(xlabel="drop in Macro F1 when shuffled", title=f"Permutation importance ({selected}, fold-0 validation)")
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=140); plt.close(fig)
    print(imp.sort_values("importance_mean", ascending=False).head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
