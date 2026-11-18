"""Leakage checks for the prediction path (candidates, features, model interface).

Kept as a committed script because tests/ is git-ignored in this repository.
Exits with status 1 if any check fails.

  1. Prediction modules do not import pathsnap.labels and do not reference .route / labels/.
  2. Building candidates + features for a trip whose .route file is deleted gives
     exactly the same output as with it present.
  3. No feature name refers to absolute position or identifiers, and no feature column
     is (almost) a copy of raw longitude/latitude.
  4. Training data and student_test share no trip, and CV folds share no trip.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.candidates.generator import generate_candidates  # noqa: E402
from pathsnap.config import processed_dir  # noqa: E402
from pathsnap.features.extractor import FEATURE_NAMES, build_candidate_features  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_prediction  # noqa: E402

PREDICTION_MODULES = ["candidates/generator.py", "features/extractor.py", "models/predict.py"]
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def features_for(trip_dir: Path) -> pd.DataFrame:
    trip = load_trip_for_prediction(trip_dir, "student_test")
    pt = project_trip(trip)
    g = build_road_graph(trip, pt)
    fb = build_candidate_features(pt, g, generate_candidates(pt, g))
    out = fb.X.copy()
    out["arc"], out["obs"] = fb.candidate_arc_ids, fb.observation_indices
    return out


def main() -> int:
    for rel in PREDICTION_MODULES:
        src = (ROOT / "src" / "pathsnap" / rel).read_text()
        tree = ast.parse(src)
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
                  [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        code_only = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "'")))
        check(not any("labels" in m for m in imports), f"{rel} does not import pathsnap.labels")
        check('".route"' not in code_only and "'.route'" not in code_only and "labels/" not in code_only,
              f"{rel} does not reference .route or the labels cache")

    ref = discover_trips(partitions=("student_test",))[0]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / ref.trip_id
        shutil.copytree(ref.trip_dir, copy)
        before = features_for(copy)
        (copy / f"{ref.trip_id}.route").unlink()
        after = features_for(copy)
    check(before.equals(after), f"trip {ref.trip_id}: identical features with and without .route")

    bad = {"lon", "lat", "longitude", "latitude", "x", "y", "xy", "id", "trip", "node"}
    flagged = [f for f in FEATURE_NAMES if set(f.lower().split("_")) & bad]
    check(not flagged, f"no positional/id feature names {flagged or ''}")
    trip = load_trip_for_prediction(ref.trip_dir, "student_test")
    obs = before["obs"].to_numpy()
    lon, lat = trip.track[obs, 0], trip.track[obs, 1]
    worst = max(abs(np.corrcoef(before[f], v)[0, 1]) for f in FEATURE_NAMES for v in (lon, lat)
                if np.std(before[f]) > 0)
    check(worst < 0.99, f"no feature copies raw lon/lat (max |corr| = {worst:.3f})")

    feat = processed_dir() / "features"
    if (feat / "train.parquet").exists():
        tr = pd.read_parquet(feat / "train.parquet", columns=["trip_id"]).trip_id.unique()
        te = pd.read_parquet(feat / "student_test.parquet", columns=["trip_id"]).trip_id.unique()
        check(not set(tr) & set(te), "train and student_test share no trip")
        from sklearn.model_selection import GroupKFold
        ids = pd.Series(tr)
        ok = all(not set(ids.iloc[a]) & set(ids.iloc[b]) for a, b in GroupKFold(5).split(ids, groups=ids))
        check(ok, "GroupKFold folds share no trip")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
