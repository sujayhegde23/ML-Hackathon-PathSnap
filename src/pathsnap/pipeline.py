"""End-to-end prediction for one trip folder (contract H). Prediction-only: it reads
<trip>.track, <trip>.nodes and <trip>.arcs, never <trip>.route.

    from pathsnap.pipeline import load_artifacts, predict_trip
    arts = load_artifacts()                    # once (e.g. at API start-up)
    result = predict_trip("/path/to/00000052", arts)
    payload = result.to_dict()                 # JSON-ready

Steps: load -> project to metres -> road graph -> candidates (Member 2) -> features
(Member 2) -> classifier probabilities (selected/tuned model) -> HMM/Viterbi decoding ->
route polyline in longitude/latitude for display.

Errors: a missing folder raises FileNotFoundError; malformed files raise the loader's
TripFormatError (pathsnap.io.loader). Both carry a readable message for the API.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pathsnap.artifacts import load_artifacts
from pathsnap.candidates.generator import generate_candidates
from pathsnap.features.extractor import build_candidate_features
from pathsnap.geo.projection import project_trip
from pathsnap.graph.network import build_road_graph
from pathsnap.hmm.viterbi import decode_route
from pathsnap.io.loader import load_trip_for_prediction
from pathsnap.models.predict import predict_candidate_scores

__all__ = ["PredictionResult", "load_artifacts", "predict_trip", "predict"]

_ARTIFACTS: dict | None = None


@dataclass
class PredictionResult:
    trip_id: str
    predicted_arc_ids: list[int]                 # drivable route, trip-local arc ids, in order
    observation_arc_ids: list[int | None]        # one predicted arc per GPS fix
    polyline_lonlat: list[tuple[float, float]]   # predicted route for display
    route_segments_lonlat: list[list[tuple[float, float]]]  # same, split at decoder breaks
    gps_lonlat: list[tuple[float, float]]        # the input GPS fixes (for display)
    model_version: str
    latency_ms: float
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["polyline_lonlat"] = [list(p) for p in self.polyline_lonlat]
        d["route_segments_lonlat"] = [[list(p) for p in seg] for seg in self.route_segments_lonlat]
        d["gps_lonlat"] = [list(p) for p in self.gps_lonlat]
        return d


def _segments(route, obs_arcs, breaks):
    """Split the route arc list at the decoder's break points."""
    if not breaks:
        return [route]
    starts = {obs_arcs[b] for b in breaks if b < len(obs_arcs) and obs_arcs[b] is not None}
    segs, cur = [], []
    for k, arc in enumerate(route):
        if k and arc in starts and cur:
            segs.append(cur); cur = []
            starts.discard(arc)
        cur.append(arc)
    if cur:
        segs.append(cur)
    return segs


def predict_trip(trip_dir: str | Path, artifacts: dict | None = None, partition: str = "eval") -> PredictionResult:
    global _ARTIFACTS
    t0 = time.perf_counter()
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"trip folder not found: {trip_dir}")
    if artifacts is None:
        _ARTIFACTS = _ARTIFACTS or load_artifacts()
        artifacts = _ARTIFACTS
    timings = {}
    trip = load_trip_for_prediction(trip_dir, partition)
    pt = project_trip(trip)
    g = build_road_graph(trip, pt)
    timings["load_graph_ms"] = 1000 * (time.perf_counter() - t0)
    t1 = time.perf_counter()
    cands = generate_candidates(pt, g)
    fb = build_candidate_features(pt, g, cands)
    scores = predict_candidate_scores(fb, artifacts)
    timings["candidates_features_model_ms"] = 1000 * (time.perf_counter() - t1)
    dec = decode_route(pt, g, cands, scores, artifacts["decoder"])
    timings["decode_ms"] = dec.diagnostics["decode_ms"]
    route = [int(a) for a in dec.predicted_arc_ids]
    segs = _segments(route, dec.observation_arc_ids, dec.break_indices)
    latency = 1000 * (time.perf_counter() - t0)
    return PredictionResult(
        trip_id=trip.trip_id,
        predicted_arc_ids=route,
        observation_arc_ids=dec.observation_arc_ids,
        polyline_lonlat=g.arc_polyline_lonlat(route),
        route_segments_lonlat=[g.arc_polyline_lonlat(s) for s in segs],
        gps_lonlat=[(float(x), float(y)) for x, y in trip.track[:, :2]],
        model_version=artifacts["model_version"],
        latency_ms=latency,
        diagnostics={**{k: v for k, v in dec.diagnostics.items() if k != "decode_ms"}, **timings,
                     "n_candidates": int(len(fb.X)), "n_arcs_in_network": int(trip.n_arcs),
                     "mean_candidates_per_fix": float(len(fb.X) / max(trip.n_points, 1)),
                     "break_indices": dec.break_indices},
    )


predict = predict_trip
