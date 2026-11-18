"""Build candidates + features for every trip and attach training labels.

Prediction-side code (candidates/, features/) never sees labels. Labels are joined only
here, from Member 1's cache data/processed/labels/<trip_id>.npz (run `make labels` first).

Label of a candidate row:
  y         1 if the candidate has the same (from_node, to_node) as the reference arc.
            Some trips store the same road twice with identical endpoints
            (docs/DATA_NOTES.md section 6), so matching on endpoints avoids calling an
            identical road "wrong". The reverse arc (B->A) is NOT a match.
  y_strict  1 only if the arc id itself matches.
Fixes with valid == False (no route arc within 150 m) are dropped from the table.

Outputs
  data/processed/features/{train,student_test}.parquet
  results/candidates/summary.csv     per-trip candidate statistics and miss rate
  artifacts/feature_spec.json        exact feature order + config used

Usage: python scripts/build_features.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.candidates.generator import candidates_to_frame, generate_candidates, load_feature_config  # noqa: E402
from pathsnap.config import artifacts_dir, processed_dir, results_dir  # noqa: E402
from pathsnap.features.extractor import FEATURE_GROUPS, FEATURE_NAMES, build_candidate_features  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_prediction  # noqa: E402

CFG = load_feature_config()


def process(ref):
    t0 = time.perf_counter()
    trip = load_trip_for_prediction(ref.trip_dir, ref.partition)  # no .route on this side
    pt = project_trip(trip)
    g = build_road_graph(trip, pt)
    cands = generate_candidates(pt, g, CFG)
    fb = build_candidate_features(pt, g, cands, CFG)
    t1 = time.perf_counter()

    lab = np.load(processed_dir() / "labels" / f"{trip.trip_id}.npz")
    valid = lab["valid"] & (lab["arc_ids"] >= 0)
    true_arc = lab["arc_ids"][fb.observation_indices]
    ok = valid[fb.observation_indices]
    safe_true = np.where(true_arc >= 0, true_arc, 0)
    same_ends = (g.from_node[fb.candidate_arc_ids] == g.from_node[safe_true]) & \
                (g.to_node[fb.candidate_arc_ids] == g.to_node[safe_true])
    df = fb.X.copy()
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(1, "partition", ref.partition)
    df.insert(2, "obs_idx", fb.observation_indices.astype(np.int32))
    df.insert(3, "arc_id", fb.candidate_arc_ids.astype(np.int64))
    df.insert(4, "y", (same_ends & ok).astype(np.int8))
    df.insert(5, "y_strict", ((fb.candidate_arc_ids == true_arc) & ok).astype(np.int8))
    df.insert(6, "label_confidence", lab["confidence"][fb.observation_indices].astype(np.float32))
    df = df[ok].reset_index(drop=True)

    cf = candidates_to_frame(cands)
    covered = df.groupby("obs_idx")["y"].max()
    n_valid = int(valid.sum())
    summary = dict(
        partition=ref.partition, trip_id=trip.trip_id, n_points=trip.n_points, n_valid_labels=n_valid,
        mean_candidates=len(cf) / max(trip.n_points, 1),
        share_radius_widened=float(np.mean([c.search_radius_m > CFG["candidates"]["radii_m"][0] for c in cands])),
        share_fallback=float(np.mean([c.used_fallback for c in cands])),
        empty_sets=int(sum(c.arc_ids.size == 0 for c in cands)),
        candidate_miss_rate=float(1 - covered.sum() / n_valid) if n_valid else np.nan,
        y_vs_y_strict_diff=int((df["y"] != df["y_strict"]).sum()),
        positive_rate=float(df["y"].mean()) if len(df) else np.nan,
        nan_or_inf=int((~np.isfinite(fb.X.to_numpy())).sum()),
        seconds=round(t1 - t0, 2),
    )
    return ref.partition, df, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    refs = discover_trips(partitions=("train", "student_test"))
    refs.sort(key=lambda r: -sum(f.stat().st_size for f in Path(r.trip_dir).iterdir()))  # big first
    out = processed_dir() / "features"
    out.mkdir(parents=True, exist_ok=True)
    (results_dir() / "candidates").mkdir(parents=True, exist_ok=True)
    frames: dict[str, list] = {"train": [], "student_test": []}
    rows = []
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for part, df, s in pool.imap_unordered(process, refs):
            frames[part].append(df)
            rows.append(s)
            print(f"{s['trip_id']} {part:12s} pts={s['n_points']:6d} miss={s['candidate_miss_rate']:.4f} "
                  f"cands={s['mean_candidates']:.1f} {s['seconds']}s", flush=True)
    for part, fr in frames.items():
        pd.concat(fr).sort_values(["trip_id", "obs_idx", "dist_rank"]).reset_index(drop=True) \
            .to_parquet(out / f"{part}.parquet", index=False)
    summ = pd.DataFrame(rows).sort_values(["partition", "trip_id"])
    summ.to_csv(results_dir() / "candidates" / "summary.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    (artifacts_dir() / "feature_spec.json").write_text(json.dumps(
        {"feature_names": FEATURE_NAMES, "feature_groups": FEATURE_GROUPS, "config": CFG,
         "git_commit": commit}, indent=2) + "\n")
    for part in ("train", "student_test"):
        s = summ[summ.partition == part]
        miss = 1 - (s.n_valid_labels * (1 - s.candidate_miss_rate)).sum() / s.n_valid_labels.sum()
        print(f"{part}: trips={len(s)} valid fixes={s.n_valid_labels.sum():,} overall miss rate={miss:.4%} "
              f"mean cands={s.mean_candidates.mean():.1f} fallback fixes={(s.share_fallback * s.n_points).sum():.0f}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
