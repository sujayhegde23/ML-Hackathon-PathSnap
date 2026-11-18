"""Contract checks for the decoder and the end-to-end pipeline (Member 3).
Kept as a committed script because tests/ is git-ignored. Exits 1 on any failure.

  1. Prediction-path modules (hmm, pipeline, artifacts) never import pathsnap.labels.
  2. predict_trip gives identical output with and without the trip's .route file.
  3. Reloading the saved artifacts reproduces the same prediction.
  4. Output schema: one arc per fix, valid arc ids, finite lon/lat in range.
  5. Misaligned scores are rejected instead of silently decoded.
  6. A time gap > max_gap_s forces a chain break exactly there.
"""

from __future__ import annotations

import ast
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.artifacts import load_artifacts  # noqa: E402
from pathsnap.candidates.generator import generate_candidates  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.hmm.viterbi import decode_route  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_prediction  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402

failures: list[str] = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    for rel in ["hmm/viterbi.py", "pipeline.py", "artifacts.py"]:
        tree = ast.parse((ROOT / "src" / "pathsnap" / rel).read_text())
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
               [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        check(not any("labels" in m for m in mods), f"{rel} does not import pathsnap.labels")

    ref = discover_trips(partitions=("student_test",))[4]
    arts = load_artifacts()
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / ref.trip_id
        shutil.copytree(ref.trip_dir, copy)
        a = predict_trip(copy, arts)
        (copy / f"{ref.trip_id}.route").unlink()
        b = predict_trip(copy, arts)
    check(a.predicted_arc_ids == b.predicted_arc_ids and a.observation_arc_ids == b.observation_arc_ids,
          f"trip {ref.trip_id}: identical prediction with and without .route")

    c = predict_trip(ref.trip_dir, load_artifacts())
    check(c.predicted_arc_ids == a.predicted_arc_ids, "reloaded artifacts reproduce the prediction")

    trip = load_trip_for_prediction(ref.trip_dir)
    obs = [x for x in a.observation_arc_ids if x is not None]
    poly = np.array(a.polyline_lonlat)
    check(len(a.observation_arc_ids) == trip.n_points, "one predicted arc per GPS fix")
    check(all(0 <= x < trip.n_arcs for x in obs + a.predicted_arc_ids), "all arc ids exist in the trip network")
    check(poly.size > 0 and np.isfinite(poly).all() and (np.abs(poly[:, 0]) <= 180).all()
          and (np.abs(poly[:, 1]) <= 90).all(), "polyline is finite lon/lat")
    check(isinstance(a.to_dict()["polyline_lonlat"][0], list), "to_dict() is JSON-ready")

    pt = project_trip(trip)
    g = build_road_graph(trip, pt)
    cands = generate_candidates(pt, g)
    n_rows = sum(cs.arc_ids.size for cs in cands)
    try:
        decode_route(pt, g, cands, np.ones(n_rows - 1), arts["decoder"])
        check(False, "misaligned scores are rejected")
    except ValueError:
        check(True, "misaligned scores are rejected")

    k = trip.n_points // 2
    track = trip.track.copy()
    track[k:, 2] += 10_000.0  # inject a ~3 h gap before fix k
    gap_trip = type(trip)(trip.trip_id, trip.partition, track, trip.nodes, trip.arcs, None)
    pt2 = project_trip(gap_trip)
    dec = decode_route(pt2, g, cands, np.full(n_rows, 0.5), arts["decoder"])
    check(k in dec.break_indices, f"time gap at fix {k} starts a new chain (breaks: {dec.break_indices[:5]})")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
