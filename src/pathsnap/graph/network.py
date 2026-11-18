"""Road-network graph: geometry, topology and bounded shortest paths.

A trip's network is read as a directed graph. Arc ``i`` runs from node
``arcs[i, 0]`` to node ``arcs[i, 1]``, and its row index is the arc id that the
reference route and, later, the predictions refer to. Nothing in this module
renumbers arcs: the original trip-local ids are the only ids used.

Distances are metres in the trip's local frame (see :mod:`pathsnap.geo.projection`).
The graph keeps a spatial index so that nearby arcs can be found without scanning
the whole network, which matters because a single trip can carry several hundred
thousand arcs.

Self loops and zero-length arcs
    ``from_node == to_node`` and repeated coordinates both occur. Such an arc gets
    length ``0`` and a bearing of ``NaN``.

Bounded search
    Shortest paths are always bounded by a distance cutoff. A search that stops at a few
    hundred metres costs far less than one that walks 300 thousand arcs.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pathsnap.geo.projection import LocalMetricProjection, ProjectedTrip
from pathsnap.io.loader import Trip

#: Spacing of the auxiliary points laid along each arc for the spatial index, metres.
SAMPLE_SPACING_M = 20.0


@dataclass(frozen=True)
class GraphStats:
    """Summary counts for one trip's network, used by EDA and the report."""

    trip_id: str
    n_nodes: int
    n_arcs: int
    n_components: int
    largest_component_nodes: int
    largest_component_share: float
    n_isolated_nodes: int
    mean_out_degree: float
    max_out_degree: int
    n_dead_end_nodes: int
    n_self_loops: int
    n_zero_length_arcs: int
    total_length_km: float
    median_arc_length_m: float

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "trip_id": self.trip_id,
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_components": self.n_components,
            "largest_component_nodes": self.largest_component_nodes,
            "largest_component_share": self.largest_component_share,
            "n_isolated_nodes": self.n_isolated_nodes,
            "mean_out_degree": self.mean_out_degree,
            "max_out_degree": self.max_out_degree,
            "n_dead_end_nodes": self.n_dead_end_nodes,
            "n_self_loops": self.n_self_loops,
            "n_zero_length_arcs": self.n_zero_length_arcs,
            "total_length_km": self.total_length_km,
            "median_arc_length_m": self.median_arc_length_m,
        }


class RoadGraph:
    """Directed road-network graph for a single trip."""

    def __init__(
        self,
        *,
        trip_id: str,
        nodes_xy: np.ndarray,
        arcs: np.ndarray,
        projection: LocalMetricProjection,
    ) -> None:
        self.trip_id = trip_id
        self.nodes_xy = np.ascontiguousarray(nodes_xy, dtype=np.float64)
        self.arcs = np.ascontiguousarray(arcs, dtype=np.int64)
        self.projection = projection

        self.from_node = self.arcs[:, 0].copy()
        self.to_node = self.arcs[:, 1].copy()

        origin = self.nodes_xy[self.from_node]
        target = self.nodes_xy[self.to_node]
        delta = target - origin
        self.arc_lengths_m = np.linalg.norm(delta, axis=1)

        # A bearing is undefined for a zero-length arc, so it is reported as NaN.
        bearings = np.arctan2(delta[:, 1], delta[:, 0])
        bearings[self.arc_lengths_m <= 0.0] = np.nan
        self.arc_bearings_rad = bearings

        self._build_adjacency()
        self._build_components()
        self._build_spatial_index()

    # -- construction ---------------------------------------------------------------

    def _build_adjacency(self) -> None:
        n_nodes = self.nodes_xy.shape[0]
        order = np.argsort(self.from_node, kind="stable")
        counts = np.bincount(self.from_node, minlength=n_nodes)
        self.out_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.out_indices = order.astype(np.int64)

        in_order = np.argsort(self.to_node, kind="stable")
        in_counts = np.bincount(self.to_node, minlength=n_nodes)
        self.in_indptr = np.concatenate([[0], np.cumsum(in_counts)]).astype(np.int64)
        self.in_indices = in_order.astype(np.int64)

        self.node_out_degree = counts.astype(np.int64)
        self.node_in_degree = in_counts.astype(np.int64)

    def _build_components(self) -> None:
        n_nodes = self.nodes_xy.shape[0]
        if n_nodes == 0:
            self.n_components = 0
            self.component_labels = np.empty(0, dtype=np.int64)
            return

        rows = np.concatenate([self.from_node, self.to_node])
        cols = np.concatenate([self.to_node, self.from_node])
        data = np.ones(rows.size, dtype=np.int8)
        mat = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        n_components, labels = connected_components(mat, directed=False, return_labels=True)
        self.n_components = int(n_components)
        self.component_labels = labels.astype(np.int64)

    def _build_spatial_index(self) -> None:
        """Index auxiliary points laid along each arc.

        A midpoint index has to be queried with a radius inflated by the longest arc
        in the network before it is guaranteed to be complete. With arcs reaching over
        a kilometre, that turns a 50 metre query into a 700 metre one. Sampling along
        each arc keeps the query radius close to the search radius.
        """
        n_arcs = self.arcs.shape[0]
        if n_arcs == 0:
            self._sample_points = np.empty((0, 2))
            self._sample_arc = np.empty(0, dtype=np.int64)
            self._sample_tree = None
            self._query_margin = 0.0
            return

        counts = np.maximum(2, np.ceil(self.arc_lengths_m / SAMPLE_SPACING_M).astype(np.int64) + 1)
        total = int(counts.sum())
        arc_of_sample = np.repeat(np.arange(n_arcs, dtype=np.int64), counts)

        starts = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        index_in_arc = np.arange(total, dtype=np.int64) - starts
        span = np.maximum(counts[arc_of_sample] - 1, 1)
        t = index_in_arc / span

        origin = self.nodes_xy[self.from_node[arc_of_sample]]
        delta = self.nodes_xy[self.to_node[arc_of_sample]] - origin
        samples = origin + t[:, None] * delta

        self._sample_points = samples
        self._sample_arc = arc_of_sample
        self._sample_tree = cKDTree(samples)
        self._query_margin = SAMPLE_SPACING_M / 2.0

    # -- basic accessors ------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.arcs.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes_xy.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    def arcs_from_node(self, node: int) -> np.ndarray:
        return self.out_indices[self.out_indptr[node] : self.out_indptr[node + 1]]

    def arcs_into_node(self, node: int) -> np.ndarray:
        return self.in_indices[self.in_indptr[node] : self.in_indptr[node + 1]]

    def arc_endpoints_xy(self, arc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        return self.nodes_xy[self.from_node[arc_ids]], self.nodes_xy[self.to_node[arc_ids]]

    # -- point to segment geometry ---------------------------------------------------

    def segment_distance(
        self, points_xy: np.ndarray, arc_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Distance from each point to its paired arc, with the projection parameter.

        ``points_xy`` is ``(M, 2)`` and ``arc_ids`` is ``(M,)``; the two are paired row
        by row. Returns ``(distance_m, t)`` where ``t`` in ``[0, 1]`` locates the
        closest point along the arc.
        """
        points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        arc_ids = np.asarray(arc_ids, dtype=np.int64).reshape(-1)
        if points_xy.shape[0] != arc_ids.shape[0]:
            raise ValueError(
                f"points and arc ids must pair up: {points_xy.shape[0]} != {arc_ids.shape[0]}"
            )

        origin = self.nodes_xy[self.from_node[arc_ids]]
        delta = self.nodes_xy[self.to_node[arc_ids]] - origin
        squared = np.einsum("ij,ij->i", delta, delta)

        offset = points_xy - origin
        numerator = np.einsum("ij,ij->i", offset, delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(squared > 0.0, numerator / np.where(squared > 0.0, squared, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)

        closest = origin + t[:, None] * delta
        distance = np.linalg.norm(points_xy - closest, axis=1)
        return distance, t

    def point_to_arc_distances(self, points_xy: np.ndarray, arc_ids: np.ndarray) -> np.ndarray:
        return self.segment_distance(points_xy, arc_ids)[0]

    def project_onto_arc(self, point_xy: np.ndarray, arc_id: int) -> tuple[np.ndarray, float]:
        """Return the closest point on ``arc_id`` and its distance."""
        distance, t = self.segment_distance(
            np.asarray(point_xy, dtype=float).reshape(1, 2), np.array([arc_id])
        )
        origin = self.nodes_xy[self.from_node[arc_id]]
        delta = self.nodes_xy[self.to_node[arc_id]] - origin
        return origin + t[0] * delta, float(distance[0])

    # -- spatial search ---------------------------------------------------------------

    def arcs_within(self, point_xy: np.ndarray, radius_m: float) -> np.ndarray:
        """Arc ids whose segment lies within ``radius_m`` of the point, nearest first."""
        if self._sample_tree is None:
            return np.empty(0, dtype=np.int64)
        point_xy = np.asarray(point_xy, dtype=float).reshape(2)
        hits = self._sample_tree.query_ball_point(point_xy, radius_m + self._query_margin)
        if not hits:
            return np.empty(0, dtype=np.int64)

        candidates = np.unique(self._sample_arc[np.asarray(hits, dtype=np.int64)])
        distance, _ = self.segment_distance(np.repeat(point_xy[None, :], candidates.size, 0), candidates)
        keep = distance <= radius_m
        candidates, distance = candidates[keep], distance[keep]
        return candidates[np.argsort(distance, kind="stable")]

    def nearest_arcs(self, point_xy: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` nearest arcs by exact segment distance, nearest first."""
        if self._sample_tree is None or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0)
        point_xy = np.asarray(point_xy, dtype=float).reshape(2)

        # Widen the sample query until at least k distinct arcs are covered.
        margin = self._query_margin
        radius = max(25.0, margin)
        candidates = np.empty(0, dtype=np.int64)
        for _ in range(8):
            hits = self._sample_tree.query_ball_point(point_xy, radius + margin)
            if hits:
                candidates = np.unique(self._sample_arc[np.asarray(hits, dtype=np.int64)])
                if candidates.size >= k:
                    break
            radius *= 2.0
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0)

        distance, _ = self.segment_distance(np.repeat(point_xy[None, :], candidates.size, 0), candidates)
        order = np.argsort(distance, kind="stable")[:k]
        return candidates[order], distance[order]

    # -- shortest paths ---------------------------------------------------------------

    def node_distance(
        self, source: int, target: int, cutoff_m: float
    ) -> float | None:
        """Shortest directed distance between two nodes, bounded by ``cutoff_m``."""
        if source == target:
            return 0.0
        if not np.isfinite(cutoff_m) or cutoff_m < 0.0:
            return None

        best: dict[int, float] = {source: 0.0}
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > best.get(node, np.inf):
                continue
            if node == target:
                return distance
            if distance >= cutoff_m:
                continue
            start, stop = self.out_indptr[node], self.out_indptr[node + 1]
            for arc in self.out_indices[start:stop]:
                nxt = int(self.to_node[arc])
                candidate = distance + self.arc_lengths_m[arc]
                if candidate >= cutoff_m:
                    continue
                if candidate < best.get(nxt, np.inf):
                    best[nxt] = candidate
                    heapq.heappush(heap, (candidate, nxt))
        return None

    def shortest_path_distance(
        self, from_arc_id: int, to_arc_id: int, cutoff_m: float
    ) -> float | None:
        """Network distance from the end of one arc to the start of another.

        This is the connecting distance between two consecutive assignments, which is
        what a transition probability needs. The arcs themselves are not counted.
        """
        if from_arc_id == to_arc_id:
            return 0.0
        if from_arc_id < 0 or to_arc_id < 0:
            return None
        if from_arc_id >= self.n_arcs or to_arc_id >= self.n_arcs:
            return None
        return self.node_distance(
            int(self.to_node[from_arc_id]), int(self.from_node[to_arc_id]), cutoff_m
        )

    def reachable_arcs(self, from_node: int, cutoff_m: float) -> Iterator[tuple[int, float]]:
        """Yield ``(arc_id, distance_to_arc_start)`` for every arc reachable from a node."""
        best: dict[int, float] = {from_node: 0.0}
        heap: list[tuple[float, int]] = [(0.0, from_node)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > best.get(node, np.inf):
                continue
            start, stop = self.out_indptr[node], self.out_indptr[node + 1]
            for arc in self.out_indices[start:stop]:
                yield int(arc), distance
                nxt = int(self.to_node[arc])
                candidate = distance + self.arc_lengths_m[arc]
                if candidate >= cutoff_m:
                    continue
                if candidate < best.get(nxt, np.inf):
                    best[nxt] = candidate
                    heapq.heappush(heap, (candidate, nxt))

    # -- geometry output --------------------------------------------------------------

    def arc_polyline_lonlat(self, arc_ids: np.ndarray | list[int]) -> list[tuple[float, float]]:
        """Convert an ordered arc sequence into a ``(lon, lat)`` polyline.

        Consecutive arcs that share a node contribute one vertex instead of two, so the
        result has no duplicated points.
        """
        arc_ids = np.asarray(arc_ids, dtype=np.int64).reshape(-1)
        if arc_ids.size == 0:
            return []

        origin = self.nodes_xy[self.from_node[arc_ids]]
        target = self.nodes_xy[self.to_node[arc_ids]]
        vertices = np.empty((arc_ids.size * 2, 2), dtype=np.float64)
        vertices[0::2] = origin
        vertices[1::2] = target
        lonlat = self.projection.unproject(vertices[:, 0], vertices[:, 1])

        keep = np.ones(lonlat.shape[0], dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(lonlat, axis=0)) > 1e-12, axis=1)
        return [(float(lon), float(lat)) for lon, lat in lonlat[keep]]

    def stats(self) -> GraphStats:
        n_nodes = self.n_nodes
        largest = 0
        n_isolated = 0
        if n_nodes:
            counts = np.bincount(self.component_labels, minlength=self.n_components)
            largest = int(counts.max()) if counts.size else 0
            n_isolated = int(np.sum((self.node_out_degree == 0) & (self.node_in_degree == 0)))
        self_loops = int(np.sum(self.from_node == self.to_node))
        return GraphStats(
            trip_id=self.trip_id,
            n_nodes=n_nodes,
            n_arcs=self.n_arcs,
            n_components=self.n_components,
            largest_component_nodes=largest,
            largest_component_share=float(largest / n_nodes) if n_nodes else 0.0,
            n_isolated_nodes=n_isolated,
            mean_out_degree=float(np.mean(self.node_out_degree)) if n_nodes else 0.0,
            max_out_degree=int(self.node_out_degree.max()) if n_nodes else 0,
            n_dead_end_nodes=int(np.sum(self.node_out_degree == 0)),
            n_self_loops=self_loops,
            n_zero_length_arcs=int(np.sum(self.arc_lengths_m <= 0.0)),
            total_length_km=float(self.arc_lengths_m.sum() / 1000.0),
            median_arc_length_m=float(np.median(self.arc_lengths_m)) if self.n_arcs else 0.0,
        )


# ------------------------------------------------------------------------------------
# Module-level interface, as agreed in the shared contract
# ------------------------------------------------------------------------------------


def build_road_graph(trip: Trip, projected: ProjectedTrip | None = None) -> RoadGraph:
    """Build the directed graph for one trip."""
    if projected is None:
        from pathsnap.geo.projection import project_trip

        projected = project_trip(trip)
    return RoadGraph(
        trip_id=trip.trip_id,
        nodes_xy=projected.nodes_xy,
        arcs=trip.arcs,
        projection=projected.projection,
    )


def point_to_arc_distances(
    points_xy: np.ndarray, arc_ids: np.ndarray, graph: RoadGraph
) -> np.ndarray:
    """Distance in metres from each point to its paired arc."""
    return graph.point_to_arc_distances(points_xy, arc_ids)


def shortest_path_distance(
    from_arc_id: int, to_arc_id: int, graph: RoadGraph, cutoff_m: float
) -> float | None:
    """Network distance between two arcs, or ``None`` beyond the cutoff."""
    return graph.shortest_path_distance(from_arc_id, to_arc_id, cutoff_m)


def arc_polyline(arc_ids: np.ndarray | list[int], graph: RoadGraph) -> list[tuple[float, float]]:
    """Ordered ``(lon, lat)`` polyline for a sequence of arcs."""
    return graph.arc_polyline_lonlat(arc_ids)
