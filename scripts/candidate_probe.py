"""Measure how well candidate search finds the reference arc.

Two questions decide the candidate design, and both are answered here by measurement:

1. Does searching by nearby nodes find the reference arc, or does the search have to work
   at arc level? A node search takes nodes within a radius and keeps the arcs attached to
   them, which misses a long segment whose endpoints are both outside the radius even
   though the segment itself passes metres away.
2. How often does a fixed radius return nothing at all? That decides whether the radius
   has to adapt.

Truth is the observation-level label from :mod:`pathsnap.labels.truth`, so the comparison
is against the same labels the classifier will be trained on.

Outputs
-------
``results/eda/candidate_probe.csv``    one row per sampled observation set
``results/eda/candidate_probe.json``   aggregate coverage and empty-set rates

Usage
-----
    python scripts/candidate_probe.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import DEFAULT_SEED, data_root, processed_dir, results_dir, set_global_seed  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_training  # noqa: E402

#: Radius used for the comparison, metres.
RADIUS_M = 50.0

#: Observations sampled per trip for the node-versus-arc comparison.
SAMPLE_PER_TRIP = 250


def load_labels(trip_id: str) -> dict[str, np.ndarray] | None:
    path = processed_dir() / "labels" / f"{trip_id}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def node_based_candidates(graph, point_xy: np.ndarray, radius_m: float) -> set[int]:
    """Arcs attached to nodes within ``radius_m``. The naive alternative to arc search."""
    nodes = np.flatnonzero(
        np.linalg.norm(graph.nodes_xy - point_xy[None, :], axis=1) <= radius_m
    )
    found: set[int] = set()
    for node in nodes:
        found.update(graph.arcs_from_node(int(node)).tolist())
        found.update(graph.arcs_into_node(int(node)).tolist())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", nargs="*", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    set_global_seed(DEFAULT_SEED)
    root = Path(args.root) if args.root else data_root()
    refs = discover_trips(root, partitions=args.partitions)
    if not refs:
        print(f"no trips found under {root}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(DEFAULT_SEED)
    rows: list[dict] = []
    started = time.perf_counter()

    for index, ref in enumerate(refs, 1):
        labels = load_labels(ref.trip_id)
        if labels is None:
            continue

        trip = load_trip_for_training(ref.trip_dir, ref.partition)
        projected = project_trip(trip)
        graph = build_road_graph(trip, projected)

        truth = labels["arc_ids"]
        valid = (labels["valid"]) & (truth >= 0)
        positions = np.flatnonzero(valid)
        if positions.size == 0:
            continue

        # Empty-set rate, measured two ways. Over every observation, and over the
        # labelled ones only. The first answers "would prediction fail here", the second
        # answers "does the search find the label", and they differ where labels are
        # marked invalid.
        empty_all = 0
        for i in range(trip.n_points):
            if graph.arcs_within(projected.track_xy[i], RADIUS_M).size == 0:
                empty_all += 1
        empty = 0
        for i in positions:
            if graph.arcs_within(projected.track_xy[i], RADIUS_M).size == 0:
                empty += 1

        # Node search against arc search on a sample, where both are cheap enough to compare.
        sample = rng.choice(
            positions, size=min(SAMPLE_PER_TRIP, positions.size), replace=False
        )
        node_hit = 0
        arc_hit = 0
        for i in sample:
            point = projected.track_xy[i]
            reference = int(truth[i])
            if reference in node_based_candidates(graph, point, RADIUS_M):
                node_hit += 1
            if reference in graph.arcs_within(point, RADIUS_M).tolist():
                arc_hit += 1

        rows.append(
            {
                "partition": ref.partition,
                "trip_id": ref.trip_id,
                "n_observations": int(trip.n_points),
                "n_labelled": int(positions.size),
                "n_sampled": int(sample.size),
                "node_search_hits": node_hit,
                "arc_search_hits": arc_hit,
                "node_coverage": node_hit / sample.size,
                "arc_coverage": arc_hit / sample.size,
                "n_empty_at_50m_all": int(empty_all),
                "empty_share_all": empty_all / trip.n_points,
                "n_empty_at_50m_labelled": int(empty),
                "empty_share_labelled": empty / positions.size,
            }
        )

        if index % 15 == 0 or index == len(refs):
            print(f"  probed {index}/{len(refs)}")

    out_dir = results_dir() / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    node_cov = np.array([r["node_coverage"] for r in rows])
    arc_cov = np.array([r["arc_coverage"] for r in rows])
    empty_all = np.array([r["empty_share_all"] for r in rows])
    empty_labelled = np.array([r["empty_share_labelled"] for r in rows])
    worst_empty = rows[int(np.argmax(empty_all))]

    summary = {
        "radius_m": RADIUS_M,
        "sample_per_trip": SAMPLE_PER_TRIP,
        "trips": len(rows),
        "node_search_coverage": {
            "overall": float(node_cov.sum() / len(rows)),
            "min": float(node_cov.min()),
            "median": float(np.median(node_cov)),
        },
        "arc_search_coverage": {
            "overall": float(arc_cov.sum() / len(rows)),
            "min": float(arc_cov.min()),
            "median": float(np.median(arc_cov)),
        },
        "empty_at_radius": {
            "trips_with_any_empty": int(sum(1 for r in rows if r["n_empty_at_50m_all"])),
            "worst_trip": worst_empty["trip_id"],
            "worst_share_all": float(worst_empty["empty_share_all"]),
            "worst_share_labelled": float(worst_empty["empty_share_labelled"]),
            "median_share_all": float(np.median(empty_all)),
            "median_share_labelled": float(np.median(empty_labelled)),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (out_dir / "candidate_probe.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"node-based search coverage: {summary['node_search_coverage']['overall']:.3f}")
    print(f"arc-level search coverage:  {summary['arc_search_coverage']['overall']:.3f}")
    print(
        f"empty at {RADIUS_M:.0f} m: {summary['empty_at_radius']['trips_with_any_empty']} trips, "
        f"worst {summary['empty_at_radius']['worst_trip']} at "
        f"{100 * summary['empty_at_radius']['worst_share_all']:.1f} percent of all observations"
    )
    print(f"runtime: {summary['runtime_seconds']:.1f} s")
    print(f"wrote {out_dir / 'candidate_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
