"""Raw PathSnap trip loading, structural validation and the canonical ``Trip`` object.

The four file formats were checked against the supplied dataset:

===========  ======================================================================
``.track``   three whitespace-separated columns: ``longitude latitude timestamp_s``
``.nodes``   two columns: ``longitude latitude``; row index is the node id
``.arcs``    two integer columns: ``from_node_id to_node_id``; row index is the arc id
``.route``   one integer column: arc ids in traversal order
===========  ======================================================================

Node and arc identifiers are trip-local: row ``i`` of this trip's ``.nodes`` file is
node ``i`` for this trip only, and nothing here assumes a global identifier space.

The module has two loading modes:

``load_trip_for_training``
    Reads ``.route``. Use it for label derivation on permitted partitions.
``load_trip_for_prediction``
    Skips ``.route`` and raises :class:`GroundTruthAccessError` if anything on this
    path asks for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pathsnap.config import MAX_DT_GAP_S, PARTITIONS, data_root

TRACK_COLUMNS = 3
NODES_COLUMNS = 2
ARCS_COLUMNS = 2
ROUTE_COLUMNS = 1

TRACK_SUFFIX = ".track"
NODES_SUFFIX = ".nodes"
ARCS_SUFFIX = ".arcs"
ROUTE_SUFFIX = ".route"

PREDICTION_SUFFIXES = (TRACK_SUFFIX, NODES_SUFFIX, ARCS_SUFFIX)
GROUND_TRUTH_SUFFIXES = (ROUTE_SUFFIX,)

ERROR = "error"
WARNING = "warning"
INFO = "info"


class TripFormatError(ValueError):
    """Raised when a trip file is structurally invalid and cannot be parsed."""


class GroundTruthAccessError(RuntimeError):
    """Raised when a prediction-time code path attempts to read reference data."""


# --------------------------------------------------------------------------------------
# Canonical data objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Trip:
    """One validated trip.

    Attributes
    ----------
    trip_id
        Trip identifier, taken from the trip directory name.
    partition
        ``"train"``, ``"student_test"`` or ``"eval"``.
    track
        ``(N, 3)`` float64 array of ``[longitude, latitude, timestamp_seconds]``.
    nodes
        ``(V, 2)`` float64 array of ``[longitude, latitude]``. Row index is the node id.
    arcs
        ``(E, 2)`` int64 array of ``[from_node_id, to_node_id]``. Row index is the arc id.
    route
        ``(R,)`` int64 array of arc ids in traversal order, or ``None`` when the trip was
        loaded in prediction mode.
    """

    trip_id: str
    partition: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None = None

    @property
    def n_points(self) -> int:
        return int(self.track.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arcs.shape[0])

    @property
    def n_route(self) -> int:
        return 0 if self.route is None else int(self.route.shape[0])

    @property
    def has_route(self) -> bool:
        return self.route is not None

    @property
    def timestamps(self) -> np.ndarray:
        return self.track[:, 2]


@dataclass(frozen=True)
class TripRef:
    """A discovered trip on disk, without its payload loaded."""

    trip_id: str
    partition: str
    trip_dir: Path


@dataclass(frozen=True)
class Issue:
    """One validation finding."""

    code: str
    severity: str
    count: int
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR


@dataclass
class TripValidation:
    """Structural and semantic summary for a single trip."""

    trip_id: str
    partition: str
    n_points: int = 0
    n_nodes: int = 0
    n_arcs: int = 0
    n_route: int = 0
    loaded: bool = True
    issues: tuple[Issue, ...] = ()

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == WARNING)

    @property
    def ok(self) -> bool:
        """True when the trip parsed and has no ERROR-severity finding."""
        return self.loaded and not self.errors

    def codes(self) -> tuple[str, ...]:
        return tuple(i.code for i in self.issues)

    def to_row(self) -> dict[str, object]:
        """Flatten to a CSV-friendly row."""
        row: dict[str, object] = {
            "trip_id": self.trip_id,
            "partition": self.partition,
            "n_points": self.n_points,
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_route": self.n_route,
            "loaded": self.loaded,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "codes": "|".join(self.codes()),
        }
        for issue in self.issues:
            row[f"count_{issue.code}"] = issue.count
        return row


# --------------------------------------------------------------------------------------
# Primitive readers
# --------------------------------------------------------------------------------------


def _read_whitespace(
    path: Path,
    n_columns: int,
    name: str,
    *,
    forbidden_suffixes: Sequence[str] = (),
) -> np.ndarray:
    """Read a headerless whitespace-separated matrix of ``n_columns`` columns.

    ``forbidden_suffixes`` blocks the reference file at the reader. A prediction-time
    caller passes ``GROUND_TRUTH_SUFFIXES``, so asking this helper for ``.route``
    raises straight away.
    """
    if path.suffix in forbidden_suffixes:
        raise GroundTruthAccessError(
            f"refusing to read {path.suffix!r} on a prediction-time code path ({path})"
        )
    if not path.is_file():
        raise TripFormatError(f"{name}: file not found: {path}")
    if path.stat().st_size == 0:
        return np.empty((0, n_columns), dtype=np.float64)

    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise TripFormatError(f"{name}: could not parse {path.name}: {exc}") from exc

    if data.size == 0:
        return np.empty((0, n_columns), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != n_columns:
        found = data.shape[1] if data.ndim == 2 else data.shape
        raise TripFormatError(
            f"{name}: expected {n_columns} columns in {path.name}, found {found}"
        )
    return data


def _as_int(data: np.ndarray, name: str) -> np.ndarray:
    """Convert to int64, rejecting non-finite and non-integral values."""
    if data.size == 0:
        return np.empty((0, data.shape[1]), dtype=np.int64)
    if not np.all(np.isfinite(data)):
        raise TripFormatError(f"{name}: contains non-finite values")
    integral = np.rint(data)
    if not np.array_equal(integral, data):
        bad = int(np.sum(integral != data))
        raise TripFormatError(f"{name}: expected integer ids, {bad} non-integral value(s)")
    return integral.astype(np.int64)


def _trip_files(trip_dir: Path, trip_id: str) -> dict[str, Path]:
    return {
        TRACK_SUFFIX: trip_dir / f"{trip_id}{TRACK_SUFFIX}",
        NODES_SUFFIX: trip_dir / f"{trip_id}{NODES_SUFFIX}",
        ARCS_SUFFIX: trip_dir / f"{trip_id}{ARCS_SUFFIX}",
        ROUTE_SUFFIX: trip_dir / f"{trip_id}{ROUTE_SUFFIX}",
    }


# --------------------------------------------------------------------------------------
# Loading modes
# --------------------------------------------------------------------------------------


def load_trip_for_training(
    trip_dir: str | os.PathLike[str],
    partition: str = "train",
    *,
    require_route: bool = True,
) -> Trip:
    """Load a trip including its reference route (training / label derivation only)."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    files = _trip_files(trip_dir, trip_id)

    track = _read_whitespace(files[TRACK_SUFFIX], TRACK_COLUMNS, "track")
    nodes = _read_whitespace(files[NODES_SUFFIX], NODES_COLUMNS, "nodes")
    arcs = _as_int(_read_whitespace(files[ARCS_SUFFIX], ARCS_COLUMNS, "arcs"), "arcs")

    route: np.ndarray | None = None
    if files[ROUTE_SUFFIX].is_file():
        route = _as_int(
            _read_whitespace(files[ROUTE_SUFFIX], ROUTE_COLUMNS, "route"), "route"
        ).reshape(-1)
    elif require_route:
        raise TripFormatError(f"route: file not found: {files[ROUTE_SUFFIX]}")

    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def load_trip_for_prediction(
    trip_dir: str | os.PathLike[str],
    partition: str = "eval",
) -> Trip:
    """Load a trip for prediction. Never reads ``.route``.

    The ``forbidden_suffixes`` argument is the enforcement point: reference files are
    refused at the reader, so a future edit cannot quietly reintroduce ground truth
    into the inference path.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    files = _trip_files(trip_dir, trip_id)

    def read(suffix: str, n_columns: int, name: str) -> np.ndarray:
        return _read_whitespace(
            files[suffix],
            n_columns,
            name,
            forbidden_suffixes=GROUND_TRUTH_SUFFIXES,
        )

    track = read(TRACK_SUFFIX, TRACK_COLUMNS, "track")
    nodes = read(NODES_SUFFIX, NODES_COLUMNS, "nodes")
    arcs = _as_int(read(ARCS_SUFFIX, ARCS_COLUMNS, "arcs"), "arcs")

    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=None,
    )


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def discover_trips(
    root: str | os.PathLike[str] | None = None,
    partitions: Iterable[str] | None = None,
) -> list[TripRef]:
    """Find every trip directory under ``root``.

    A directory is a trip when it contains ``<dirname>.track``. Partition is the first
    path component under ``root`` that names a known partition.
    """
    base = Path(root) if root is not None else data_root()
    wanted = tuple(partitions) if partitions is not None else PARTITIONS

    refs: list[TripRef] = []
    for partition in wanted:
        partition_dir = base / partition
        if not partition_dir.is_dir():
            continue
        for candidate in sorted(p for p in partition_dir.iterdir() if p.is_dir()):
            if (candidate / f"{candidate.name}{TRACK_SUFFIX}").is_file():
                refs.append(TripRef(candidate.name, partition, candidate))
    return refs


def load_partition(
    partition: str,
    root: str | os.PathLike[str] | None = None,
    *,
    mode: str = "training",
) -> list[Trip]:
    """Load every trip in one partition. ``mode`` is ``"training"`` or ``"prediction"``."""
    loader = load_trip_for_training if mode == "training" else load_trip_for_prediction
    return [
        loader(ref.trip_dir, ref.partition)
        for ref in discover_trips(root, partitions=(partition,))
    ]


def safe_load_trip(
    trip_dir: str | os.PathLike[str],
    partition: str,
    *,
    mode: str = "training",
) -> tuple[Trip | None, TripValidation]:
    """Load and validate one trip, converting parse failures into a validation record.

    This is the dataset-level entry point: a single malformed trip is reported rather
    than aborting the whole run.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    loader = load_trip_for_training if mode == "training" else load_trip_for_prediction

    try:
        trip = loader(trip_dir, partition)
    except (TripFormatError, GroundTruthAccessError) as exc:
        return None, TripValidation(
            trip_id=trip_id,
            partition=partition,
            loaded=False,
            issues=(Issue("E_LOAD", ERROR, 1, str(exc)),),
        )

    return trip, validate_trip(trip)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _issue(code: str, severity: str, count: int, message: str) -> Issue:
    return Issue(code=code, severity=severity, count=int(count), message=message)


def validate_trip(
    trip: Trip,
    *,
    gap_threshold_s: float = MAX_DT_GAP_S,
) -> TripValidation:
    """Check one trip for structural and semantic problems.

    Errors mean the trip cannot be used downstream. Warnings are recorded anomalies
    that are known properties of the supplied data and are expected to occur.
    """
    issues: list[Issue] = []

    if trip.n_points == 0:
        issues.append(_issue("E_TRACK_EMPTY", ERROR, 0, "track file has no observations"))
    if trip.n_nodes == 0:
        issues.append(_issue("E_NODES_EMPTY", ERROR, 0, "nodes file is empty"))
    if trip.n_arcs == 0:
        issues.append(_issue("E_ARCS_EMPTY", ERROR, 0, "arcs file is empty"))

    # --- finiteness and coordinate plausibility ---
    for name, array in (("track", trip.track), ("nodes", trip.nodes)):
        if array.size and not np.all(np.isfinite(array)):
            n_bad = int(np.sum(~np.isfinite(array)))
            issues.append(
                _issue("E_NONFINITE", ERROR, n_bad, f"{name} contains {n_bad} non-finite values")
            )

    if trip.track.size:
        lon, lat = trip.track[:, 0], trip.track[:, 1]
        n_bad = int(np.sum((np.abs(lat) > 90.0) | (np.abs(lon) > 180.0)))
        if n_bad:
            issues.append(
                _issue("E_COORD_RANGE", ERROR, n_bad, f"{n_bad} track coordinates out of range")
            )

    # --- identifier ranges ---
    if trip.n_arcs and trip.n_nodes:
        out_of_range = (trip.arcs < 0) | (trip.arcs >= trip.n_nodes)
        n_bad = int(out_of_range.any(axis=1).sum())
        if n_bad:
            issues.append(
                _issue(
                    "E_ARC_NODE_RANGE",
                    ERROR,
                    n_bad,
                    f"{n_bad} arcs reference node ids outside [0, {trip.n_nodes})",
                )
            )

    if trip.route is not None and trip.n_arcs:
        if trip.route.size == 0:
            issues.append(_issue("W_ROUTE_EMPTY", WARNING, 0, "route file has no arcs"))
        else:
            n_bad = int(np.sum((trip.route < 0) | (trip.route >= trip.n_arcs)))
            if n_bad:
                issues.append(
                    _issue(
                        "E_ROUTE_RANGE",
                        ERROR,
                        n_bad,
                        f"{n_bad} route entries outside arc id range [0, {trip.n_arcs})",
                    )
                )

    # --- duplicate arc rows (identical from/to pairs) ---
    if trip.n_arcs > 1:
        n_unique = int(np.unique(trip.arcs, axis=0).shape[0])
        n_dup = trip.n_arcs - n_unique
        if n_dup:
            issues.append(
                _issue(
                    "W_DUPLICATE_ARCS",
                    WARNING,
                    n_dup,
                    f"{n_dup} arc rows duplicate an existing (from, to) pair",
                )
            )

    # --- timestamps and sampling ---
    if trip.n_points > 1:
        dt = np.diff(trip.timestamps)

        n_back = int(np.sum(dt < 0))
        if n_back:
            issues.append(
                _issue("E_TIME_DECREASING", ERROR, n_back, f"{n_back} timestamps decrease")
            )

        n_dup_t = int(np.sum(dt == 0))
        if n_dup_t:
            issues.append(
                _issue("W_TIME_DUPLICATE", WARNING, n_dup_t, f"{n_dup_t} duplicate timestamps")
            )

        n_gap = int(np.sum(dt > gap_threshold_s))
        if n_gap:
            issues.append(
                _issue(
                    "W_DT_GAP",
                    WARNING,
                    n_gap,
                    f"{n_gap} gaps exceed {gap_threshold_s:g} s "
                    f"(largest {float(np.max(dt)):.0f} s)",
                )
            )

        # --- duplicate consecutive coordinates ---
        same = (np.diff(trip.track[:, 0]) == 0) & (np.diff(trip.track[:, 1]) == 0)
        n_same = int(same.sum())
        if n_same:
            issues.append(
                _issue("W_GPS_DUPLICATE", WARNING, n_same, f"{n_same} repeated coordinate fixes")
            )

    # --- reference route structure ---
    if trip.route is not None and trip.route.size > 1 and trip.n_arcs:
        in_range = np.all((trip.route >= 0) & (trip.route < trip.n_arcs))
        if in_range:
            seq = trip.arcs[trip.route]
            n_break = int(np.sum(seq[:-1, 1] != seq[1:, 0]))
            if n_break:
                issues.append(
                    _issue(
                        "W_ROUTE_CHAIN_BREAK",
                        WARNING,
                        n_break,
                        f"{n_break} consecutive route arcs are not connected in the graph",
                    )
                )

        n_unique = int(np.unique(trip.route).shape[0])
        if n_unique < trip.route.size:
            issues.append(
                _issue(
                    "W_ROUTE_REVISIT",
                    WARNING,
                    trip.route.size - n_unique,
                    f"route revisits {trip.route.size - n_unique} arc(s); order must be preserved",
                )
            )

    return TripValidation(
        trip_id=trip.trip_id,
        partition=trip.partition,
        n_points=trip.n_points,
        n_nodes=trip.n_nodes,
        n_arcs=trip.n_arcs,
        n_route=trip.n_route,
        loaded=True,
        issues=tuple(issues),
    )


def validate_partition(
    partition: str,
    root: str | os.PathLike[str] | None = None,
    *,
    mode: str = "training",
) -> list[TripValidation]:
    """Validate every trip in one partition, never raising on a single bad trip."""
    results = []
    for ref in discover_trips(root, partitions=(partition,)):
        _, validation = safe_load_trip(ref.trip_dir, ref.partition, mode=mode)
        results.append(validation)
    return results
