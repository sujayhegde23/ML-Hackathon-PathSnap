"""Observation-level reference labels from the ordered reference route.

The ``.route`` file lists the arcs a vehicle traversed, in order. It does not list one
arc per GPS fix: the median ratio of route arcs to fixes is 0.19, so roughly five fixes
share an arc. A per-observation reference arc therefore has to be inferred.

The inference here is a monotone alignment. Each GPS fix is assigned a position in the
route sequence, and the positions are constrained to never move backwards, because a
vehicle does not teleport to an earlier part of its route. Two costs drive the choice:

Distance
    How far the fix sits from the candidate arc, in metres.

Position prior
    How far the candidate position along the route is from where the distance already
    travelled says it should be. This is what stops the alignment from jumping to a
    distant arc that happens to be close to the fix, which is common where a route
    doubles back or crosses itself.

The alignment is dynamic programming over ``(observation, route position)`` with a
running prefix minimum, so the whole sequence costs ``O(N * R)`` time and memory
linear in the route length.

Labels carry two caveats. First, the route itself is supplied reference data, and
section 6 of ``docs/DATA_NOTES.md`` records where it is inconsistent. Second, an
observation whose nearest route arc is very far away cannot be labelled at all; those
are marked invalid rather than assigned a plausible-looking arc.

This module is for building training and development labels only. It must not be
imported by the prediction path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathsnap.geo.projection import haversine_m, project_trip
from pathsnap.io.loader import Trip

#: Weight of the absolute position prior. Measured on the supplied trips, any positive
#: weight lowered label fidelity, because GPS distance travelled does not track route
#: distance once the route loops or the two scales differ. Kept at 0 and left in the
#: signature so the comparison can be reproduced.
DEFAULT_POSITION_WEIGHT = 0.0

#: Charge in metres per route position advanced between consecutive observations.
#: See ``results/labels/weight_sweep.csv`` for the sweep behind this value.
DEFAULT_SKIP_PENALTY = 0.0

#: A fix further than this from every route arc is treated as unalignable.
DEFAULT_MAX_ALIGN_DISTANCE_M = 150.0

#: Scale of the distance term inside the confidence score, metres.
CONFIDENCE_DISTANCE_SCALE_M = 10.0

#: Scale of the margin term inside the confidence score, metres.
CONFIDENCE_MARGIN_SCALE_M = 4.0

#: Route positions this close to the chosen one are ignored when measuring ambiguity.
#: Adjacent arcs are always a few metres away and are not a competing explanation: the
#: vehicle is passing from one to the other. Only a distant stretch of the route that
#: also fits the fix is real ambiguity.
MARGIN_WINDOW_POSITIONS = 8


@dataclass(frozen=True)
class LabelResult:
    """Per-observation reference labels for one trip.

    ``arc_ids``
        Trip-local arc id assigned to each observation, or ``-1`` where the observation
        could not be aligned.
    ``route_positions``
        Index in ``trip.route`` that each observation was aligned to, or ``-1``. This is
        what preserves repeated visits: two observations on the same arc at different
        points in the route keep different positions.
    ``confidence``
        A heuristic alignment quality score in ``[0, 1]``. It is not a calibrated
        probability and should not be read as one. It combines how far the fix sits from
        the chosen arc with how clearly the next-best route arc was beaten.
    ``valid``
        True where a label was assigned. False marks an observation that had no route
        arc within ``max_align_distance_m``.
    """

    trip_id: str
    arc_ids: np.ndarray
    route_positions: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    dist_to_label_m: np.ndarray
    margin_m: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return int(self.arc_ids.shape[0])

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid))

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence[self.valid])) if self.n_valid else 0.0


def _route_geometry(trip: Trip, projected) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projected start points, end points and lengths for each route arc."""
    route = trip.route
    origin = projected.nodes_xy[trip.arcs[route, 0]]
    target = projected.nodes_xy[trip.arcs[route, 1]]
    lengths = np.linalg.norm(target - origin, axis=1)
    return origin, target, lengths


def _distances_to_route(
    points: np.ndarray, origin: np.ndarray, target: np.ndarray, chunk: int = 4096
) -> np.ndarray:
    """Point-to-segment distance from every fix to every route arc.

    Computed in chunks so a long trip with a long route does not have to materialise the
    full ``N x R`` matrix at once.
    """
    delta = target - origin
    squared = np.einsum("ij,ij->i", delta, delta)
    safe = np.where(squared > 0.0, squared, 1.0)

    n_points = points.shape[0]
    out = np.empty((n_points, origin.shape[0]), dtype=np.float64)
    for start in range(0, n_points, chunk):
        block = points[start : start + chunk]
        offset = block[:, None, :] - origin[None, :, :]
        parameter = np.einsum("mkj,kj->mk", offset, delta) / safe[None, :]
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = origin[None, :, :] + parameter[..., None] * delta[None, :, :]
        out[start : start + chunk] = np.linalg.norm(block[:, None, :] - closest, axis=2)
    return out


def _monotone_alignment(
    cost: np.ndarray,
    prior: np.ndarray,
    skip_penalty: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Align observations to non-decreasing route positions.

    ``cost`` is ``(N, R)`` distance in metres and ``prior`` is ``(N, R)`` position
    penalty. ``skip_penalty`` charges a fixed number of metres per route position
    advanced between consecutive observations, which discourages the alignment from
    jumping ahead over arcs it has no evidence for.

    Returns the chosen position per observation and the total cost.

    The recurrence is::

        best[i, k] = cost[i, k] + prior[i, k]
                     + min_{k' <= k} ( best[i-1, k'] + skip_penalty * (k - k') )

    The skip term separates into ``skip_penalty * k`` outside the minimum and
    ``-skip_penalty * k'`` inside it, so the inner minimum is still a running prefix
    minimum and each row costs ``O(R)``.
    """
    n_points, n_positions = cost.shape
    total = cost + prior
    offsets = skip_penalty * np.arange(n_positions, dtype=np.float64)

    back = np.zeros((n_points, n_positions), dtype=np.int32)
    best = total[0].copy()

    for i in range(1, n_points):
        candidates = best - offsets
        running_min = np.minimum.accumulate(candidates)

        # Position that achieves the running minimum, tracked with a running maximum
        # over the indices where a new minimum starts.
        improves = np.empty(n_positions, dtype=bool)
        improves[0] = True
        improves[1:] = candidates[1:] < running_min[:-1]
        positions = np.where(improves, np.arange(n_positions), 0)
        running_arg = np.maximum.accumulate(positions)

        back[i] = running_arg
        best = total[i] + offsets + running_min

    positions = np.empty(n_points, dtype=np.int64)
    positions[-1] = int(np.argmin(best))
    for i in range(n_points - 1, 0, -1):
        positions[i - 1] = back[i, positions[i]]
    return positions, float(best[positions[-1]])


def derive_observation_labels(
    trip: Trip,
    *,
    position_weight: float = DEFAULT_POSITION_WEIGHT,
    skip_penalty: float = DEFAULT_SKIP_PENALTY,
    max_align_distance_m: float = DEFAULT_MAX_ALIGN_DISTANCE_M,
) -> LabelResult:
    """Assign each GPS observation a reference arc from the ordered route."""
    n_points = trip.n_points
    empty_i = np.full(n_points, -1, dtype=np.int64)
    empty_f = np.full(n_points, np.nan, dtype=np.float64)

    if trip.route is None or trip.route.size == 0 or n_points == 0:
        return LabelResult(
            trip_id=trip.trip_id,
            arc_ids=empty_i,
            route_positions=empty_i.copy(),
            confidence=np.zeros(n_points, dtype=np.float64),
            valid=np.zeros(n_points, dtype=bool),
            dist_to_label_m=empty_f,
            margin_m=empty_f.copy(),
            diagnostics={"reason": "no reference route", "route_length": int(trip.n_route)},
        )

    projected = project_trip(trip)
    origin, target, lengths = _route_geometry(trip, projected)
    distances = _distances_to_route(projected.track_xy, origin, target)

    # Distance already travelled, and the matching position along the route.
    leg = haversine_m(
        trip.track[:-1, 0], trip.track[:-1, 1], trip.track[1:, 0], trip.track[1:, 1]
    )
    travelled = np.concatenate([[0.0], np.cumsum(leg)])
    route_midpoint = np.concatenate([[0.0], np.cumsum(lengths)])[:-1] + lengths / 2.0
    prior = position_weight * np.abs(route_midpoint[None, :] - travelled[:, None])

    positions, total = _monotone_alignment(distances, prior, skip_penalty)

    rows = np.arange(n_points)
    dist_to_label = distances[rows, positions]

    # Margin to the best route arc that is a genuinely different explanation: not the
    # chosen one, not a neighbouring position along the route, and not a duplicate row
    # describing the same arc.
    route = trip.route
    chosen_arc = route[positions]
    offset = np.abs(np.arange(route.size)[None, :] - positions[:, None])

    # Two rows can describe the same physical arc (identical endpoints under different
    # ids). They are the same road segment, so neither competes with the other.
    arc_geometry = trip.arcs[route]
    chosen_geometry = trip.arcs[chosen_arc]
    same_direction = np.all(arc_geometry[None, :, :] == chosen_geometry[:, None, :], axis=2)
    reversed_direction = np.all(
        arc_geometry[None, :, ::-1] == chosen_geometry[:, None, :], axis=2
    )
    same_road = same_direction | reversed_direction

    excluded = (
        (offset <= MARGIN_WINDOW_POSITIONS)
        | (route[None, :] == chosen_arc[:, None])
        | same_road
    )
    masked = np.where(excluded, np.inf, distances)
    second = masked.min(axis=1)
    margin = np.where(np.isfinite(second), second - dist_to_label, np.inf)

    valid = np.isfinite(dist_to_label) & (dist_to_label <= max_align_distance_m)

    confidence = np.zeros(n_points, dtype=np.float64)
    if np.any(valid):
        distance_term = np.exp(
            -np.clip(dist_to_label[valid], 0.0, None) / CONFIDENCE_DISTANCE_SCALE_M
        )
        finite_margin = np.where(np.isfinite(margin[valid]), margin[valid], 50.0)
        margin_term = 1.0 - np.exp(-np.clip(finite_margin, 0.0, None) / CONFIDENCE_MARGIN_SCALE_M)
        confidence[valid] = distance_term * margin_term

    arc_ids = np.where(valid, trip.route[positions], -1).astype(np.int64)
    route_positions = np.where(valid, positions, -1).astype(np.int64)

    monotone = bool(np.all(np.diff(positions) >= 0))
    diagnostics = {
        "reason": "aligned",
        "route_length": int(trip.route.size),
        "position_weight": float(position_weight),
        "skip_penalty": float(skip_penalty),
        "max_align_distance_m": float(max_align_distance_m),
        "n_invalid": int(n_points - np.sum(valid)),
        "monotone": monotone,
        "total_cost": total,
        "median_dist_to_label_m": float(np.median(dist_to_label[valid]))
        if np.any(valid)
        else float("nan"),
        "share_within_10m": float(np.mean(dist_to_label[valid] <= 10.0))
        if np.any(valid)
        else 0.0,
        "n_distinct_positions": int(np.unique(route_positions[valid]).size) if np.any(valid) else 0,
    }

    return LabelResult(
        trip_id=trip.trip_id,
        arc_ids=arc_ids,
        route_positions=route_positions,
        confidence=confidence,
        valid=valid,
        dist_to_label_m=dist_to_label,
        margin_m=margin,
        diagnostics=diagnostics,
    )


def nearest_route_arc_baseline(trip: Trip) -> np.ndarray:
    """Naive comparison: each fix takes its nearest route arc, ignoring order.

    Order can go backwards here, and repeated arcs are not distinguished. It exists so
    the cost of the ordering constraint can be measured.
    """
    if trip.route is None or trip.route.size == 0 or trip.n_points == 0:
        return np.full(trip.n_points, -1, dtype=np.int64)

    projected = project_trip(trip)
    origin, target, _ = _route_geometry(trip, projected)
    distances = _distances_to_route(projected.track_xy, origin, target)
    return np.argmin(distances, axis=1).astype(np.int64)


def compare_with_baseline(labels: LabelResult, baseline_positions: np.ndarray) -> dict:
    """Summarise where the ordered alignment and the naive baseline disagree."""
    both = labels.route_positions >= 0
    if not np.any(both):
        return {"n_compared": 0}

    agree = labels.route_positions[both] == baseline_positions[both]
    disagreement = np.flatnonzero(both)[~agree]
    gap = np.abs(
        labels.route_positions[both][~agree] - baseline_positions[both][~agree]
    )
    return {
        "n_compared": int(np.sum(both)),
        "n_agree": int(np.sum(agree)),
        "agreement": float(np.mean(agree)),
        "n_disagree": int(disagreement.size),
        "median_position_gap": float(np.median(gap)) if gap.size else 0.0,
        "max_position_gap": float(gap.max()) if gap.size else 0.0,
        "first_disagreements": disagreement[:10].tolist(),
    }
