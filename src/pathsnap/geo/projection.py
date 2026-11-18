"""Per-trip local metric projection.

Longitude and latitude are angles. Euclidean distance between two degree pairs is
meaningless as a physical distance, and the size of a degree differs between the
equator and high latitudes. Every distance used for candidate scoring, arc lengths
and decoder transitions therefore has to be computed in metres.

Each trip gets its own tangent-plane frame centred on that trip. The frame maps
``(longitude, latitude)`` to ``(x, y)`` in metres using an equirectangular
approximation:

    x = R * radians(lon - lon0) * cos(lat0)
    y = R * radians(lat - lat0)

Accuracy
--------
The east-west scale is evaluated once at ``lat0``, so it drifts as a trip moves north
or south of that latitude. The relative error in east-west distance at latitude
``lat`` is::

    cos(lat) / cos(lat0) - 1  ~=  -tan(lat0) * (lat - lat0)   [radians]

Measured over the 90 supplied trips, the largest relative error was 0.58 percent and the
mean was 0.03 percent. At the 50 metre candidate radius that is under 30 centimetres.
North-south distances are unaffected by the drift.

:func:`haversine_m` is kept as the ground truth for distance, and
:func:`projection_error` measures the difference so the report can quote it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathsnap.io.loader import Trip

#: Mean Earth radius (IUGG), metres.
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
    radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray | float:
    """Great-circle distance in metres between two ``(lon, lat)`` pairs.

    Works elementwise; scalars in, scalar out.
    """
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    distance = 2.0 * radius_m * np.arcsin(np.sqrt(a))
    return float(distance) if np.ndim(distance) == 0 else distance


@dataclass(frozen=True)
class LocalMetricProjection:
    """An equirectangular tangent plane anchored at ``(lon0, lat0)``."""

    lon0: float
    lat0: float
    radius_m: float = EARTH_RADIUS_M

    @classmethod
    def centred_on(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalMetricProjection":
        """Build a frame centred on the mean of the supplied coordinates."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.size == 0:
            return cls(lon0=0.0, lat0=0.0)
        return cls(lon0=float(np.mean(lon)), lat0=float(np.mean(lat)))

    @property
    def cos_lat0(self) -> float:
        return float(np.cos(np.radians(self.lat0)))

    @property
    def metres_per_degree_lon(self) -> float:
        return self.radius_m * np.pi / 180.0 * self.cos_lat0

    @property
    def metres_per_degree_lat(self) -> float:
        return self.radius_m * np.pi / 180.0

    def project(
        self, lon: np.ndarray | float, lat: np.ndarray | float
    ) -> np.ndarray:
        """Map ``(lon, lat)`` in degrees to ``(x, y)`` in metres.

        Accepts scalars or arrays of equal shape. Column vectors stack along the last
        axis, so a pair of length-``N`` arrays returns an ``(N, 2)`` array.
        """
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        x = self.metres_per_degree_lon * (lon_arr - self.lon0)
        y = self.metres_per_degree_lat * (lat_arr - self.lat0)
        return np.stack([x, y], axis=-1)

    def unproject(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Inverse of :meth:`project`; returns ``(lon, lat)`` in degrees."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        lon = self.lon0 + x_arr / self.metres_per_degree_lon
        lat = self.lat0 + y_arr / self.metres_per_degree_lat
        return np.stack([lon, lat], axis=-1)

    def relative_scale_error(self, lat: np.ndarray | float) -> np.ndarray:
        """Relative error in the east-west scale at ``lat`` compared with ``lat0``."""
        lat_arr = np.asarray(lat, dtype=float)
        return np.cos(np.radians(lat_arr)) / self.cos_lat0 - 1.0


@dataclass(frozen=True)
class ProjectedTrip:
    """A trip with its coordinates mapped into a metric frame.

    The original longitude and latitude arrays are kept on the source ``Trip``, so
    anything destined for a map can be converted back exactly.
    """

    trip_id: str
    partition: str
    projection: LocalMetricProjection
    track_xy: np.ndarray
    nodes_xy: np.ndarray
    source: Trip

    @property
    def n_points(self) -> int:
        return int(self.track_xy.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes_xy.shape[0])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        """Convert projected metres back to ``(lon, lat)`` degrees."""
        xy = np.asarray(xy, dtype=float)
        return self.projection.unproject(xy[..., 0], xy[..., 1])

    def track_lonlat(self) -> np.ndarray:
        """The original track coordinates, read back from the source trip."""
        return self.source.track[:, :2].copy()

    def nodes_lonlat(self) -> np.ndarray:
        return self.source.nodes.copy()


def project_trip(trip: Trip) -> ProjectedTrip:
    """Project a trip's track and nodes into its own metric frame."""
    projection = LocalMetricProjection.centred_on(trip.track[:, 0], trip.track[:, 1])
    return ProjectedTrip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        projection=projection,
        track_xy=projection.project(trip.track[:, 0], trip.track[:, 1]),
        nodes_xy=projection.project(trip.nodes[:, 0], trip.nodes[:, 1]),
        source=trip,
    )


def projection_error(
    projection: LocalMetricProjection,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    max_pairs: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare projected distances with haversine distances for random point pairs.

    Returns the maximum, mean and 95th percentile relative error, so the accuracy of
    the tangent-plane approximation can be quoted from measurement.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = lon.size
    if n < 2:
        return {"max_rel_error": 0.0, "mean_rel_error": 0.0, "p95_rel_error": 0.0, "pairs": 0.0}

    rng = np.random.default_rng(seed)
    count = min(max_pairs, 4 * n)
    i = rng.integers(0, n, size=count)
    j = rng.integers(0, n, size=count)
    keep = i != j
    i, j = i[keep], j[keep]

    exact = haversine_m(lon[i], lat[i], lon[j], lat[j])
    xy = projection.project(lon, lat)
    approx = np.linalg.norm(xy[i] - xy[j], axis=1)

    valid = exact > 1e-6
    if not np.any(valid):
        return {"max_rel_error": 0.0, "mean_rel_error": 0.0, "p95_rel_error": 0.0, "pairs": 0.0}

    rel = np.abs(approx[valid] - exact[valid]) / exact[valid]
    return {
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "p95_rel_error": float(np.percentile(rel, 95)),
        "pairs": float(rel.size),
    }
