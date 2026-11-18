"""Decoder tuning and route-level comparison (Member 3).

1. Out-of-fold scores on train: the final model's settings are refit inside Member 2's
   5 trip-grouped folds; each validation trip is scored at full 1 Hz resolution by a
   model that never saw it.
2. Decoder tuning: grid over transition_beta_m x distance_weight on a seeded sample of
   train trips (low-fidelity-label trips excluded), objective = mean per-trip HMM Top-1.
   Chosen values are saved to <model dir>/decoder_config.json.
3. Student-test comparison (final model refit on all train, as in the pipeline):
     nearest arc | classifier only | classifier + HMM | nearest + HMM (distance-only)
   Metrics per trip: observation Top-1, continuity, arc-set F1 and normalised edit
   distance vs the reference route, decode time.
4. Where the HMM helps / hurts: per-trip fixed vs broken fixes, plus maps of the most
   improved and the most damaged stretch.

Outputs: results/routes/*.csv, results/figures/routes/*.png, maps/hmm_*.html,
         <model dir>/decoder_config.json and manifest.json

Usage: python scripts/evaluate_routes.py [--tune-trips 24]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.artifacts import load_artifacts, model_dir, write_manifest  # noqa: E402
from pathsnap.candidates.generator import generate_candidates  # noqa: E402
from pathsnap.config import configs_dir, maps_dir, processed_dir, results_dir, set_global_seed  # noqa: E402
from pathsnap.eval.route_metrics import collapse, route_metrics  # noqa: E402
from pathsnap.features.extractor import FEATURE_NAMES, build_candidate_features  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.hmm.viterbi import DecoderConfig, classifier_only, decode_route  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_training  # noqa: E402

OUT = results_dir() / "routes"
FIG = results_dir() / "figures" / "routes"
MCFG = yaml.safe_load(open(configs_dir() / "model_config.yaml"))
SEED = int(MCFG["seed"])
BETAS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
DIST_WEIGHTS = [0.0, 1.0]


def prepare(ref):
    """Prediction-side inputs (candidates, features) + evaluation-side truth (route, labels)."""
    trip = load_trip_for_training(ref.trip_dir, ref.partition)  # route used only for scoring below
    pt = project_trip(trip)
    g = build_road_graph(trip, pt)
    cands = generate_candidates(pt, g)
    fb = build_candidate_features(pt, g, cands)
    labels = dict(np.load(processed_dir() / "labels" / f"{trip.trip_id}.npz"))
    return dict(trip=trip, pt=pt, g=g, cands=cands, fb=fb, labels=labels)


def run_methods(d, scores, cfg, methods=("nearest", "classifier", "classifier+hmm", "nearest+hmm")):
    out = {}
    for m in methods:
        t0 = time.perf_counter()
        if m == "nearest":
            obs = [int(c.arc_ids[0]) if c.arc_ids.size else None for c in d["cands"]]
            route_join = None
        elif m == "classifier":
            obs = classifier_only(d["cands"], scores)
            route_join = None
        elif m == "classifier+hmm":
            dec = decode_route(d["pt"], d["g"], d["cands"], scores, cfg)
            obs, route_join = dec.observation_arc_ids, dec.predicted_arc_ids
        else:  # classic distance-only HMM: emission = Gaussian on distance (sigma from config)
            dist = np.concatenate([c.distances_m for c in d["cands"]]) if d["cands"] else np.zeros(0)
            like = np.exp(-0.5 * (dist / cfg.distance_sigma_m) ** 2)
            dec = decode_route(d["pt"], d["g"], d["cands"], like, replace(cfg, distance_weight=0.0))
            obs, route_join = dec.observation_arc_ids, dec.predicted_arc_ids
        ms = 1000 * (time.perf_counter() - t0)
        met = route_metrics(obs, collapse(obs), d["trip"], d["labels"])
        if route_join is not None:
            j = route_metrics(obs, route_join, d["trip"], d["labels"])
            met.update(joined_route_continuity=j["route_continuity"], joined_arc_set_f1=j["arc_set_f1"],
                       joined_norm_edit_distance=j["norm_edit_distance"])
        out[m] = dict(obs=obs, metrics=met, ms=ms)
    return out


def fit_final_like(X, y):
    """A fresh model with exactly the final model's settings."""
    from sklearn.base import clone
    return clone(load_artifacts()["model"]).fit(X, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-trips", type=int, default=24)
    a = ap.parse_args()
    set_global_seed(SEED)
    for p in (OUT, FIG, maps_dir()):
        p.mkdir(parents=True, exist_ok=True)
    arts = load_artifacts()
    base_cfg = DecoderConfig.load(configs_dir() / "decoder_config.yaml")
    print("model:", arts["model_version"], "from", arts["model_dir"], flush=True)

    # ---- 1. OOF scores for a sample of train trips (same folds as Member 2)
    full = pd.read_parquet(processed_dir() / "features" / "train.parquet")
    tr = full[full.obs_idx % int(MCFG["obs_stride"]) == 0].reset_index(drop=True)
    folds = list(GroupKFold(n_splits=int(MCFG["cv_folds"])).split(tr, tr["y"], tr["trip_id"]))
    low_fid = set(json.loads((results_dir() / "labels" / "manifest.json").read_text()).get("low_fidelity_trips", []))
    refs = {r.trip_id: r for r in discover_trips(partitions=("train",))}
    rng = np.random.default_rng(SEED)
    pool = sorted(t for t in refs if t not in low_fid)
    tune_ids = sorted(rng.choice(pool, size=min(a.tune_trips, len(pool)), replace=False).tolist())
    tune = {}
    t0 = time.perf_counter()
    for k, (tri, vai) in enumerate(folds):
        val_ids = [t for t in tr.trip_id.iloc[vai].unique() if t in tune_ids]
        if not val_ids:
            continue
        model = fit_final_like(tr[FEATURE_NAMES].iloc[tri], tr["y"].iloc[tri])
        for t in val_ids:
            d = prepare(refs[t])
            d["scores"] = model.predict_proba(d["fb"].X[FEATURE_NAMES])[:, 1]
            tune[t] = d
    print(f"OOF scores for {len(tune)} tuning trips in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- 2. decoder grid search on train trips
    rows = []
    for beta in BETAS:
        for dw in DIST_WEIGHTS:
            cfg = replace(base_cfg, transition_beta_m=beta, distance_weight=dw)
            for t, d in tune.items():
                r = run_methods(d, d["scores"], cfg, methods=("classifier+hmm",))["classifier+hmm"]
                rows.append(dict(trip_id=t, transition_beta_m=beta, distance_weight=dw, **r["metrics"], decode_ms=r["ms"]))
        print(f"beta={beta}: done", flush=True)
    for t, d in tune.items():  # classifier-only reference on the same trips
        r = run_methods(d, d["scores"], base_cfg, methods=("classifier",))["classifier"]
        rows.append(dict(trip_id=t, transition_beta_m=np.nan, distance_weight=np.nan, **r["metrics"], decode_ms=r["ms"]))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "decoder_tuning_per_trip.csv", index=False)
    agg = grid.dropna(subset=["transition_beta_m"]).groupby(["transition_beta_m", "distance_weight"]).agg(
        top1=("top1_acc", "mean"), continuity=("continuity", "mean"), arc_set_f1=("arc_set_f1", "mean"),
        norm_edit=("norm_edit_distance", "mean"), decode_ms=("decode_ms", "mean")).reset_index()
    agg.to_csv(OUT / "decoder_tuning.csv", index=False)
    best = agg.sort_values(["top1", "norm_edit"], ascending=[False, True]).iloc[0]
    tuned = replace(base_cfg, transition_beta_m=float(best.transition_beta_m), distance_weight=float(best.distance_weight))
    clf_ref = grid[grid.transition_beta_m.isna()].top1_acc.mean()
    print(f"best decoder: beta={tuned.transition_beta_m} dist_w={tuned.distance_weight} train top1={best.top1:.4f} "
          f"(classifier-only {clf_ref:.4f})", flush=True)
    mdir = model_dir()
    (mdir / "decoder_config.json").write_text(json.dumps({
        **asdict(tuned), "tuned_on": f"{len(tune)} train trips (out-of-fold scores), objective mean Top-1",
        "train_top1_hmm": float(best.top1), "train_top1_classifier_only": float(clf_ref)}, indent=2) + "\n")
    write_manifest(mdir)

    # ---- 3. student_test comparison with the final model (fit on all train)
    model = arts["model"]
    rows, per_obs = [], {}
    for ref in discover_trips(partitions=("student_test",)):
        d = prepare(ref)
        s = model.predict_proba(d["fb"].X[FEATURE_NAMES])[:, 1]
        res = run_methods(d, s, tuned)
        per_obs[ref.trip_id] = (d, {m: r["obs"] for m, r in res.items()})
        for m, r in res.items():
            rows.append(dict(trip_id=ref.trip_id, method=m, n_fixes=d["trip"].n_points, ms=r["ms"], **r["metrics"]))
        print(ref.trip_id, {m: round(r["metrics"]["top1_acc"], 4) for m, r in res.items()}, flush=True)
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "student_test_per_trip.csv", index=False)
    w = st.assign(correct=st.top1_acc * st.n_labelled)
    summ = w.groupby("method").agg(
        top1_acc=("correct", "sum"), n=("n_labelled", "sum"), continuity=("continuity", "mean"),
        arc_set_f1=("arc_set_f1", "mean"), norm_edit_distance=("norm_edit_distance", "mean"),
        joined_route_continuity=("joined_route_continuity", "mean"), joined_arc_set_f1=("joined_arc_set_f1", "mean"),
        joined_norm_edit_distance=("joined_norm_edit_distance", "mean"), ms_per_trip=("ms", "mean")).reset_index()
    summ["top1_acc"] = summ.top1_acc / summ.n
    order = ["nearest", "classifier", "classifier+hmm", "nearest+hmm"]
    summ = summ.set_index("method").loc[order].reset_index().drop(columns="n")
    summ.to_csv(OUT / "student_test_summary.csv", index=False)
    print(summ.round(4).to_string(index=False), flush=True)

    # ---- 4. where the HMM helps / hurts
    eff = []
    for t, (d, obs) in per_obs.items():
        lab, val = d["labels"]["arc_ids"], d["labels"]["valid"]
        fn, tn = d["trip"].arcs[:, 0], d["trip"].arcs[:, 1]

        def right(seq):
            p = np.array([-1 if x is None else x for x in seq])
            ok = (p >= 0) & (lab >= 0) & val
            return ok & (fn[np.maximum(p, 0)] == fn[np.maximum(lab, 0)]) & (tn[np.maximum(p, 0)] == tn[np.maximum(lab, 0)])
        c, h = right(obs["classifier"]), right(obs["classifier+hmm"])
        eff.append(dict(trip_id=t, fixed_by_hmm=int((~c & h).sum()), broken_by_hmm=int((c & ~h).sum()),
                        labelled=int((val & (lab >= 0)).sum()), _c=c, _h=h))
    effdf = pd.DataFrame([{k: v for k, v in e.items() if not k.startswith("_")} for e in eff])
    effdf["net_gain"] = effdf.fixed_by_hmm - effdf.broken_by_hmm
    effdf.to_csv(OUT / "hmm_effect_per_trip.csv", index=False)
    print(effdf.to_string(index=False), flush=True)
    make_figures(agg, summ, st, effdf)
    best_t = effdf.sort_values("net_gain", ascending=False).trip_id.iloc[0]
    worst_t = effdf.sort_values("broken_by_hmm", ascending=False).trip_id.iloc[0]
    for tag, t in [("improved", best_t), ("hurt", worst_t)]:
        e = next(x for x in eff if x["trip_id"] == t)
        key = (~e["_c"] & e["_h"]) if tag == "improved" else (e["_c"] & ~e["_h"])
        draw_example(tag, t, per_obs[t][0], per_obs[t][1], key)
    json.dump({"most_improved_trip": best_t, "most_hurt_trip": worst_t}, open(OUT / "examples.json", "w"), indent=2)
    return 0


def make_figures(agg, summ, st, effdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for dw, g in agg.groupby("distance_weight"):
        ax.plot(g.transition_beta_m, g.top1, "o-", label=f"distance_weight={dw:g}")
    ax.set(xscale="log", xlabel="transition beta (m)", ylabel="mean Top-1 (train, out-of-fold)",
           title="Decoder tuning on train trips")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "decoder_tuning.png", dpi=140); plt.close(fig)

    cols = [("top1_acc", "Top-1 accuracy"), ("continuity", "Route continuity"), ("arc_set_f1", "Arc-set F1"),
            ("norm_edit_distance", "Norm. edit distance (lower better)")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    colors = ["#9D9D9D", "#F58518", "#4C78A8", "#54A24B"]
    for axis, (c, title) in zip(ax, cols):
        axis.bar(summ.method, summ[c], color=colors)
        axis.set_title(title); axis.tick_params(axis="x", rotation=30)
        for i, v in enumerate(summ[c]):
            axis.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Student-test route comparison (10 trips)")
    fig.tight_layout(); fig.savefig(FIG / "method_comparison.png", dpi=140); plt.close(fig)

    p = st.pivot(index="trip_id", columns="method", values="top1_acc")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(p))
    ax.bar(x - 0.2, p["classifier"], 0.4, label="classifier only", color="#F58518")
    ax.bar(x + 0.2, p["classifier+hmm"], 0.4, label="classifier + HMM", color="#4C78A8")
    ax.set_xticks(x, p.index, rotation=45); ax.set_ylim(max(0, p.values.min() - 0.05), 1.0)
    ax.set(ylabel="Top-1 accuracy", title="Per-trip effect of HMM decoding (student test)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "hmm_effect_per_trip.png", dpi=140); plt.close(fig)


def draw_example(tag, trip_id, d, obs, key_mask):
    """Zoom on the 80-fix window with the most fixes the HMM fixed (or broke)."""
    import folium
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trip, g = d["trip"], d["g"]
    n = trip.n_points
    win = 80
    counts = np.convolve(key_mask.astype(int), np.ones(win, int), mode="valid") if n >= win else np.array([key_mask.sum()])
    i0 = int(np.argmax(counts)); i1 = min(n, i0 + win)
    xy = d["pt"].track_xy

    def arcs_xy(arc_ids):
        a, b = g.arc_endpoints_xy(np.array(arc_ids, dtype=np.int64))
        return a, b
    lab = d["labels"]["arc_ids"][i0:i1]
    ref_arcs = sorted({int(x) for x in lab if x >= 0})
    fig, ax = plt.subplots(figsize=(7, 7))
    # Arrows show driving direction; each method is shifted sideways a few metres so that a
    # road and its opposite-direction twin (identical geometry) are both visible.
    span = float(np.ptp(xy[i0:i1], axis=0).max()) + 1.0
    for shift, (arcs, color, lw, label) in zip((0.0, 1.0, -1.0), [
            (ref_arcs, "#54A24B", 3.0, "reference (offline analysis only)"),
            ([x for x in obs["classifier"][i0:i1] if x is not None], "#F58518", 2.0, "classifier only"),
            ([x for x in obs["classifier+hmm"][i0:i1] if x is not None], "#4C78A8", 2.0, "classifier + HMM")]):
        a, b = arcs_xy(sorted(set(arcs)))
        for k in range(len(a)):
            v = b[k] - a[k]
            nrm = np.hypot(*v) or 1.0
            off = shift * 0.02 * span * np.array([-v[1], v[0]]) / nrm
            ax.annotate("", xy=b[k] + off, xytext=a[k] + off,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=0.9, mutation_scale=14))
            ax.plot([a[k, 0] + off[0], b[k, 0] + off[0]], [a[k, 1] + off[1], b[k, 1] + off[1]], alpha=0)  # keep in view
        ax.plot([], [], color=color, lw=lw, label=label)
    ax.scatter(xy[i0:i1, 0], xy[i0:i1, 1], s=8, c="k", zorder=5, label="GPS fixes")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best")
    ax.autoscale()
    ax.set(title=f"Trip {trip_id}: where the HMM {'helped' if tag == 'improved' else 'hurt'} (fixes {i0}-{i1})",
           xlabel="x (m)", ylabel="y (m)")
    fig.tight_layout(); fig.savefig(FIG / f"example_hmm_{tag}_{trip_id}.png", dpi=140); plt.close(fig)

    ll = trip.track[:, :2]
    m = folium.Map(location=[float(ll[i0:i1, 1].mean()), float(ll[i0:i1, 0].mean())], zoom_start=17, tiles="OpenStreetMap")
    for arcs, color, w, name in [(ref_arcs, "#54A24B", 8, "reference (offline analysis only)"),
                                 ([x for x in obs["classifier"][i0:i1] if x is not None], "#F58518", 5, "classifier only"),
                                 ([x for x in obs["classifier+hmm"][i0:i1] if x is not None], "#4C78A8", 3, "classifier + HMM")]:
        layer = folium.FeatureGroup(name=name)
        for arc in sorted(set(arcs)):
            pts = g.arc_polyline_lonlat([arc])
            folium.PolyLine([(lat, lon) for lon, lat in pts], color=color, weight=w, opacity=0.8).add_to(layer)
        layer.add_to(m)
    gps = folium.FeatureGroup(name="GPS fixes")
    for lon, lat in ll[i0:i1]:
        folium.CircleMarker((float(lat), float(lon)), radius=2, color="black", fill=True).add_to(gps)
    gps.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(maps_dir() / f"hmm_{tag}_{trip_id}.html"))


if __name__ == "__main__":
    raise SystemExit(main())
