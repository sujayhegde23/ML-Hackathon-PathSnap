"""HMM + Viterbi route reconstruction.

Why: the classifier scores every GPS fix on its own, so its top choices can jump between
roads that are not connected (route continuity of the classifier alone is ~0.83 on
student_test). An HMM chooses one candidate per fix for the WHOLE trip at once, trading
off "this road fits this fix" against "you can actually drive from the previous road to
this one in the time available".

Hidden states   : at fix i, the candidate arcs of fix i (Member 2's CandidateSet).
Emission score  : emission_weight * log(p_i(arc))         p = classifier probability
                  - distance_weight * 0.5 * (d / sigma)^2  optional Gaussian distance term
                  (distance_weight defaults to 0 because the classifier already uses distance)
Transition score: -|route_m - straight_m| / beta_m          (Newson & Krumm, 2009)
                  straight_m = metres between fix i-1 and fix i
                  route_m    = driving distance between the two map-matched positions:
                    same arc  : |t_b - t_a| * length
                    otherwise : rest of arc a + shortest path (end of a -> start of b)
                                + part of arc b up to the match
                  A transition is impossible (-inf) when no path exists within the
                  search cutoff (route_factor * straight_m + route_slack_m).
Decoding        : Viterbi in log space (sums instead of products, no underflow):
                    V_i(b) = emission_i(b) + max_a [ V_{i-1}(a) + transition(a, b) ]
                  with back-pointers, then backtracking from the best final state.
Breaks          : the chain restarts (new independent segment) when
                    * the time gap to the previous fix exceeds max_gap_s (60 s),
                    * a fix has no candidates (only possible on an empty network), or
                    * every transition from the previous fix is impossible.
                  Break positions are returned in DecodeResult.break_indices.
Route output    : per-fix arcs are collapsed (repeats removed) and consecutive arcs
                  that do not touch are joined with the shortest-path arcs between them,
                  so predicted_arc_ids is a drivable sequence inside each segment.

Never reads .route or labels.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from pathsnap.config import MAX_DT_GAP_S, configs_dir

NEG_INF = -np.inf


@dataclass
class DecoderConfig:
    emission_weight: float = 1.0
    distance_weight: float = 0.0
    distance_sigma_m: float = 5.0
    transition_beta_m: float = 5.0
    route_factor: float = 2.0
    route_slack_m: float = 100.0
    max_gap_s: float = MAX_DT_GAP_S
    min_prob: float = 1e-6

    @classmethod
    def load(cls, path=None) -> "DecoderConfig":
        import json
        from pathlib import Path

        import yaml

        from pathsnap.config import artifacts_dir

        if path is None:  # tuned config saved next to the final model wins over the default
            tuned = artifacts_dir() / "models" / "final" / "decoder_config.json"
            path = tuned if tuned.exists() else configs_dir() / "decoder_config.yaml"
        path = Path(path)
        data = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DecodeResult:
    """Contract G (team plan section 4.5)."""

    predicted_arc_ids: list[int]            # drivable route, trip-local arc ids, in order
    observation_arc_ids: list[int | None]   # one arc per GPS fix (None if no candidates)
    break_indices: list[int]                # fix indices where a new chain started (excl. 0)
    diagnostics: dict = field(default_factory=dict)


class _PathCache:
    """Bounded Dijkstra from a node, cached per source node. Distances in metres."""

    def __init__(self, graph):
        self.indptr = graph.out_indptr.tolist()
        self.indices = graph.out_indices.tolist()
        self.to_node = graph.to_node.tolist()
        self.arcs_from = graph.from_node.tolist()
        self.length = graph.arc_lengths_m.tolist()
        self.cache: dict[int, tuple[float, dict, dict]] = {}
        self.runs = 0

    def search(self, src: int, cutoff: float) -> tuple[dict, dict]:
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        cutoff = max(cutoff, 2 * hit[0] if hit else 0.0, 200.0)  # grow geometrically, better reuse
        self.runs += 1
        dist, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        indptr, indices, to_node, length = self.indptr, self.indices, self.to_node, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            for k in range(indptr[u], indptr[u + 1]):
                arc = indices[k]
                nd = d + length[arc]
                if nd <= cutoff:
                    v = to_node[arc]
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        pred[v] = arc
                        heapq.heappush(heap, (nd, v))
        self.cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, src: int, dst: int, cutoff: float) -> list[int] | None:
        """Arcs of the shortest path src -> dst (node ids), or None if beyond cutoff."""
        if src == dst:
            return []
        dist, pred = self.search(src, cutoff)
        if dst not in dist:
            return None
        out, v = [], dst
        while v != src:
            arc = pred[v]
            out.append(arc)
            v = self.arcs_from[arc]
        return out[::-1]


def _split_scores(candidate_sets, candidate_scores):
    sizes = [c.arc_ids.size for c in candidate_sets]
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    if scores.size != sum(sizes):
        raise ValueError(f"{scores.size} scores for {sum(sizes)} candidate rows: rows are misaligned")
    return np.split(scores, np.cumsum(sizes)[:-1]) if sizes else []


def decode_route(projected_trip, road_graph, candidate_sets, candidate_scores,
                 decoder_config: DecoderConfig | None = None) -> DecodeResult:
    """candidate_scores: flat array aligned with the candidate rows in order
    (exactly FeatureBatch row order / predict_candidate_scores output)."""
    cfg = decoder_config or DecoderConfig.load()
    t_start = time.perf_counter()
    g = road_graph
    P = projected_trip.track_xy
    T = projected_trip.source.track[:, 2]
    N = len(candidate_sets)
    per_obs = _split_scores(candidate_sets, candidate_scores)
    cache = _PathCache(g)
    from_node, to_node, L = g.from_node, g.to_node, g.arc_lengths_m

    # emissions and projection positions per fix
    emis, tpos = [], []
    for i, c in enumerate(candidate_sets):
        if c.arc_ids.size == 0:
            emis.append(np.empty(0)); tpos.append(np.empty(0)); continue
        e = cfg.emission_weight * np.log(np.clip(per_obs[i], cfg.min_prob, 1.0))
        if cfg.distance_weight:
            e = e - cfg.distance_weight * 0.5 * (c.distances_m / cfg.distance_sigma_m) ** 2
        emis.append(e)
        tpos.append(g.segment_distance(np.repeat(P[i][None, :], c.arc_ids.size, 0), c.arc_ids)[1])

    def transition(i):
        a, b = candidate_sets[i - 1].arc_ids, candidate_sets[i].arc_ids
        ta, tb = tpos[i - 1], tpos[i]
        straight = float(np.hypot(*(P[i] - P[i - 1])))
        cutoff = cfg.route_factor * straight + cfg.route_slack_m
        route = np.full((a.size, b.size), np.inf)
        tail = (1.0 - ta) * L[a]                       # rest of arc a
        head = tb * L[b]                               # start of arc b up to the match
        b_from = from_node[b].tolist()
        for j, (arc_a, src) in enumerate(zip(a.tolist(), to_node[a].tolist())):
            dist, _ = cache.search(src, cutoff)
            mid = np.fromiter((dist.get(n, np.inf) for n in b_from), float, b.size)
            route[j] = tail[j] + mid + head
            same = b == arc_a
            if same.any():
                route[j, same] = np.abs(tb[same] - ta[j]) * L[arc_a]
        out = -np.abs(route - straight) / cfg.transition_beta_m
        out[~np.isfinite(route) | (route > cutoff + L[a][:, None] + L[b][None, :])] = NEG_INF
        return out

    obs_arc: list[int | None] = [None] * N
    breaks: list[int] = []
    n_impossible_breaks = n_gap_breaks = n_empty = 0

    def backtrack(seg_start, score, back):
        k = int(np.argmax(score))
        idx = seg_start + len(back)
        obs_arc[idx] = int(candidate_sets[idx].arc_ids[k])
        for step in range(len(back) - 1, -1, -1):
            k = int(back[step][k])
            idx -= 1
            obs_arc[idx] = int(candidate_sets[idx].arc_ids[k])

    seg_start, score, back = None, None, []
    had_segment = False
    for i in range(N):
        if candidate_sets[i].arc_ids.size == 0:
            n_empty += 1
            if seg_start is not None:
                backtrack(seg_start, score, back)
            seg_start, score, back = None, None, []
            continue
        restart = seg_start is None
        if restart and had_segment:
            breaks.append(i)
        if not restart and T[i] - T[i - 1] > cfg.max_gap_s:
            restart, n_gap_breaks = True, n_gap_breaks + 1
        if not restart:
            tr = transition(i)
            total = score[:, None] + tr
            best_prev = np.argmax(total, axis=0)
            best = total[best_prev, np.arange(total.shape[1])]
            if not np.isfinite(best).any():
                restart, n_impossible_breaks = True, n_impossible_breaks + 1
            else:
                score = best + emis[i]
                back.append(best_prev)
        if restart:
            if seg_start is not None:
                backtrack(seg_start, score, back)
                breaks.append(i)
            seg_start, score, back = i, emis[i].copy(), []
            had_segment = True
    if seg_start is not None:
        backtrack(seg_start, score, back)

    route, unjoined = build_route(obs_arc, breaks, g, cache, P, cfg)
    return DecodeResult(
        predicted_arc_ids=route,
        observation_arc_ids=obs_arc,
        break_indices=breaks,
        diagnostics=dict(n_observations=N, n_breaks=len(breaks), gap_breaks=n_gap_breaks,
                         impossible_transition_breaks=n_impossible_breaks, empty_candidate_fixes=n_empty,
                         unjoined_route_steps=unjoined, dijkstra_runs=cache.runs,
                         decode_ms=1000 * (time.perf_counter() - t_start)),
    )


def build_route(obs_arc, breaks, g, cache, P, cfg) -> tuple[list[int], int]:
    """Collapse repeats and join consecutive arcs with shortest-path arcs (within segments)."""
    route: list[int] = []
    unjoined = 0
    break_set = set(breaks)
    prev_arc, prev_i = None, None
    for i, arc in enumerate(obs_arc):
        if arc is None:
            prev_arc = None
            continue
        if prev_arc is not None and arc != prev_arc and i not in break_set:
            src, dst = int(g.to_node[prev_arc]), int(g.from_node[arc])
            if src != dst:
                straight = float(np.hypot(*(P[i] - P[prev_i])))
                path = cache.path_arcs(src, dst, cfg.route_factor * straight + cfg.route_slack_m)
                if path is None:
                    unjoined += 1
                else:
                    route.extend(path)
        if not route or route[-1] != arc:
            route.append(arc)
        prev_arc, prev_i = arc, i
    return route, unjoined


def classifier_only(candidate_sets, candidate_scores) -> list[int | None]:
    """Baseline: highest-scoring candidate per fix, no sequence model."""
    out: list[int | None] = []
    for c, s in zip(candidate_sets, _split_scores(candidate_sets, candidate_scores)):
        out.append(int(c.arc_ids[int(np.argmax(s))]) if c.arc_ids.size else None)
    return out
