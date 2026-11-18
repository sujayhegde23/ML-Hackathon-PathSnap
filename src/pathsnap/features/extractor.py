"""Feature extraction: describe each (GPS fix, candidate arc) pair with numbers.

Rule: every feature is RELATIVE to the road network (distance, direction agreement,
rank among candidates...). No longitude/latitude, no x/y, no arc/node/trip ids, and
nothing from ``.route`` or labels. That is what lets a model trained on 80 trips work
on the evaluator's unseen cities.

Feature groups (also used for the ablation experiment):

geometry  dist_m                 metres from the fix to the arc segment. Main signal.
          proj_t                 where along the arc the closest point falls (0=start, 1=end).
          end_dist_min_m         distance to the nearer end node; near 0 = at a junction,
                                 where the vehicle may already be on the next arc.
          log_arc_len            long arcs collect more fixes; very short arcs are rarely
                                 the label.
ranking   dist_rank              0 = nearest candidate.
          dist_gap_to_best       how much further than the nearest candidate (metres).
          dist_ratio_to_best     the same, as a ratio.
          n_cands                many candidates = dense area = harder decision.
          n_close_cands          candidates within 5 m of the best: genuine ambiguity.
heading   cos_heading            cos(direction of travel - arc direction). +1 = same way,
                                 -1 = opposite. The ONLY feature that separates a two-way
                                 road's two arcs, which sit at identical distance.
          heading_valid          0 when the vehicle is ~stopped (heading meaningless) or
                                 the arc has zero length.
context   ctx_dist_mean          mean distance of the neighbouring fixes (i-3..i+3) to this
                                 same arc: a real road fits the whole local track, a side
                                 street only touches it.
          ctx_dist_max           worst neighbour distance.
          ctx_gap_to_best        ctx_dist_mean minus the best candidate's ctx_dist_mean.
network   from_in_deg, to_out_deg  junction complexity at the arc's ends.
          has_reverse_twin       1 on two-way roads.
          local_arc_density      arcs within 50 m: how crowded the area is.
temporal  speed_mps              from real elapsed time (sub-second and repeated
                                 timestamps exist), clipped to 0..70 m/s.
          dt_prev_s              time since previous fix, clipped to 0..600 s.
          long_gap               1 after a gap longer than MAX_DT_GAP_S (60 s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.config import MAX_DT_GAP_S

FEATURE_GROUPS: dict[str, list[str]] = {
    "geometry": ["dist_m", "proj_t", "end_dist_min_m", "log_arc_len"],
    "ranking": ["dist_rank", "dist_gap_to_best", "dist_ratio_to_best", "n_cands", "n_close_cands"],
    "heading": ["cos_heading", "heading_valid"],
    "context": ["ctx_dist_mean", "ctx_dist_max", "ctx_gap_to_best"],
    "network": ["from_in_deg", "to_out_deg", "has_reverse_twin", "local_arc_density"],
    "temporal": ["speed_mps", "dt_prev_s", "long_gap"],
}
FEATURE_NAMES: list[str] = [f for names in FEATURE_GROUPS.values() for f in names]


@dataclass
class FeatureBatch:
    """Contract E (team plan section 4.5). Row r of X, candidate_arc_ids and
    observation_indices all describe the same (fix, candidate) pair."""

    X: pd.DataFrame
    candidate_arc_ids: np.ndarray
    observation_indices: np.ndarray
    feature_names: list[str]


def reverse_twin(arcs: np.ndarray, n_nodes: int) -> np.ndarray:
    """For each arc A->B, the id of an arc B->A, or -1."""
    key = arcs[:, 0] * n_nodes + arcs[:, 1]
    rkey = arcs[:, 1] * n_nodes + arcs[:, 0]
    order = np.argsort(key, kind="stable")
    pos = np.clip(np.searchsorted(key[order], rkey), 0, max(len(key) - 1, 0))
    hit = key[order][pos] == rkey if len(key) else np.zeros(0, bool)
    return np.where(hit, order[pos], -1)


def _per_obs_min(values, obs, n):
    out = np.full(n, np.inf)
    np.minimum.at(out, obs, values)
    return out[obs]


def build_candidate_features(projected_trip, road_graph, candidate_sets, config: dict | None = None) -> FeatureBatch:
    if config is None:
        from pathsnap.candidates.generator import load_feature_config
        config = load_feature_config()
    cfg = config["features"]
    P = projected_trip.track_xy
    t = projected_trip.source.track[:, 2]
    N = len(P)
    sizes = np.array([c.arc_ids.size for c in candidate_sets], dtype=np.int64)
    obs = np.repeat(np.arange(N), sizes)
    arc = np.concatenate([c.arc_ids for c in candidate_sets]).astype(np.int64) if len(obs) else np.zeros(0, np.int64)
    f: dict[str, np.ndarray] = {}
    if len(obs) == 0:
        X = pd.DataFrame({n: np.zeros(0, np.float32) for n in FEATURE_NAMES})
        return FeatureBatch(X, arc, obs, list(FEATURE_NAMES))

    # geometry
    d, tt = road_graph.segment_distance(P[obs], arc)
    a_xy, b_xy = road_graph.arc_endpoints_xy(arc)
    L = road_graph.arc_lengths_m[arc]
    f["dist_m"] = d
    f["proj_t"] = tt
    f["end_dist_min_m"] = np.minimum(np.hypot(*(P[obs] - a_xy).T), np.hypot(*(P[obs] - b_xy).T))
    f["log_arc_len"] = np.log1p(L)

    # ranking (candidate lists are already sorted nearest first)
    best = _per_obs_min(d, obs, N)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    f["dist_rank"] = np.arange(len(obs)) - np.repeat(starts, sizes)
    f["dist_gap_to_best"] = d - best
    f["dist_ratio_to_best"] = (d + 1.0) / (best + 1.0)
    f["n_cands"] = np.repeat(sizes, sizes)
    close = ((d - best) < cfg["close_cand_margin_m"]).astype(np.int64)
    per_obs_close = np.bincount(obs, weights=close, minlength=N)
    f["n_close_cands"] = per_obs_close[obs]

    # heading: same convention as RoadGraph.arc_bearings_rad = arctan2(dy, dx)
    k = int(cfg["heading_half_window"])
    idx = np.arange(N)
    lo, hi = np.clip(idx - k, 0, N - 1), np.clip(idx + k, 0, N - 1)
    move = P[hi] - P[lo]
    move_len = np.hypot(move[:, 0], move[:, 1])
    gps_bearing = np.arctan2(move[:, 1], move[:, 0])
    arc_bearing = road_graph.arc_bearings_rad[arc]
    valid = (move_len[obs] >= cfg["min_move_for_heading_m"]) & np.isfinite(arc_bearing)
    f["cos_heading"] = np.where(valid, np.cos(gps_bearing[obs] - np.nan_to_num(arc_bearing)), 0.0)
    f["heading_valid"] = valid.astype(float)

    # context: neighbouring fixes measured against the same arc
    c = int(cfg["context_half_window"])
    ctx = np.stack([road_graph.segment_distance(P[np.clip(obs + o, 0, N - 1)], arc)[0]
                    for o in range(-c, c + 1) if o != 0], axis=1)
    f["ctx_dist_mean"] = ctx.mean(axis=1)
    f["ctx_dist_max"] = ctx.max(axis=1)
    f["ctx_gap_to_best"] = f["ctx_dist_mean"] - _per_obs_min(f["ctx_dist_mean"], obs, N)

    # network
    twin = reverse_twin(road_graph.arcs, road_graph.n_nodes)
    f["from_in_deg"] = road_graph.node_in_degree[road_graph.from_node[arc]].astype(float)
    f["to_out_deg"] = road_graph.node_out_degree[road_graph.to_node[arc]].astype(float)
    f["has_reverse_twin"] = (twin[arc] >= 0).astype(float)
    within = (d <= cfg["density_radius_m"]).astype(np.int64)
    f["local_arc_density"] = np.bincount(obs, weights=within, minlength=N)[obs]

    # temporal (guard zero / negative / huge time steps)
    span = t[hi] - t[lo]
    speed = np.where(span > 0, move_len / np.where(span > 0, span, 1.0), 0.0)
    f["speed_mps"] = np.clip(speed, 0, 70)[obs]
    dt_prev = np.concatenate([[0.0], np.diff(t)])
    f["dt_prev_s"] = np.clip(dt_prev, 0, 600)[obs]
    f["long_gap"] = (dt_prev > MAX_DT_GAP_S).astype(float)[obs]

    X = pd.DataFrame({n: np.asarray(f[n], dtype=np.float32) for n in FEATURE_NAMES})
    return FeatureBatch(X, arc, obs, list(FEATURE_NAMES))
