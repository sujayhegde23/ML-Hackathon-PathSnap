"""Candidate generation: for each GPS fix, the road arcs it could plausibly be on.

Why not just take the nearest arc? GPS noise (median ~4 m, 95th percentile ~13 m, see
docs/DATA_NOTES.md section 5) is the same size as the gap between parallel roads, and
two-way roads are stored as two arcs at exactly the same distance. So we keep several
nearby arcs and let the classifier score them.

How it works, per fix:
  1. ``RoadGraph.arcs_within`` (Member 1) returns arcs whose *segment* lies within r of
     the fix, nearest first. Segment search matters: node-only search misses long roads
     whose end nodes are both far away (results/eda/candidate_probe.json).
  2. If fewer than ``min_candidates`` are found, widen r (50 -> 100 -> 200 -> 400 m).
     Needed for offset trips such as 00000054.
  3. If still nothing, fall back to the ``fallback_k`` nearest arcs anywhere.
  4. Keep at most ``max_candidates``, nearest first (ties broken by arc id).

This module never reads ``.route`` or labels. It works on a trip loaded with
``load_trip_for_prediction``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.config import configs_dir

log = logging.getLogger(__name__)


@dataclass
class CandidateSet:
    """Contract D (team plan section 4.5)."""

    trip_id: str
    observation_index: int
    arc_ids: np.ndarray      # (K,) trip-local arc ids, nearest first
    distances_m: np.ndarray  # (K,) exact point-to-segment distance, metres
    search_radius_m: float   # final radius used; inf when the fallback was used
    used_fallback: bool


def load_feature_config(path=None) -> dict:
    import yaml

    with open(path or configs_dir() / "feature_config.yaml") as f:
        return yaml.safe_load(f)


def generate_candidates(projected_trip, road_graph, config: dict | None = None) -> list[CandidateSet]:
    """One CandidateSet per GPS fix, in observation order (empty set if none found)."""
    cfg = (config or load_feature_config())["candidates"]
    radii = [float(r) for r in cfg["radii_m"]]
    k_min, k_max, k_fb = int(cfg["min_candidates"]), int(cfg["max_candidates"]), int(cfg["fallback_k"])
    pts = projected_trip.track_xy
    tid = projected_trip.trip_id
    out: list[CandidateSet] = []
    if road_graph.n_arcs == 0:
        log.warning("trip %s has an empty road network: every candidate set is empty", tid)
        empty = np.empty(0, dtype=np.int64)
        return [CandidateSet(tid, i, empty, np.empty(0), np.inf, True) for i in range(len(pts))]

    for i, p in enumerate(pts):
        arcs = np.empty(0, dtype=np.int64)
        used_r = np.inf
        for r in radii:
            arcs = road_graph.arcs_within(p, r)
            used_r = r
            if arcs.size >= k_min:
                break
        fallback = False
        if arcs.size == 0:
            arcs, _ = road_graph.nearest_arcs(p, k=k_fb)
            used_r, fallback = np.inf, True
        d = road_graph.point_to_arc_distances(np.repeat(p[None, :], arcs.size, 0), arcs)
        order = np.lexsort((arcs, d))[:k_max]
        out.append(CandidateSet(tid, i, arcs[order].astype(np.int64), d[order], used_r, fallback))
    return out


def candidates_to_frame(candidate_sets: list[CandidateSet]) -> pd.DataFrame:
    """Flatten to one row per (fix, candidate arc). Row order: fix, then distance."""
    if not candidate_sets:
        return pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", "dist_m", "radius_m", "used_fallback"])
    sizes = np.array([c.arc_ids.size for c in candidate_sets])
    return pd.DataFrame({
        "trip_id": candidate_sets[0].trip_id,
        "obs_idx": np.repeat([c.observation_index for c in candidate_sets], sizes).astype(np.int32),
        "arc_id": np.concatenate([c.arc_ids for c in candidate_sets]).astype(np.int64),
        "dist_m": np.concatenate([c.distances_m for c in candidate_sets]),
        "radius_m": np.repeat([c.search_radius_m for c in candidate_sets], sizes),
        "used_fallback": np.repeat([c.used_fallback for c in candidate_sets], sizes),
    })
