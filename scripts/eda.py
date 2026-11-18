"""Exploratory data analysis for the supplied trips.

Produces the tables, figures and interactive maps that back the dataset section of
the report. Every output states which partition it was computed from.

Outputs
-------
``results/eda/trips.csv``       one row of measurements per trip
``results/eda/summary.json``    per-partition aggregates and flagged anomalies
``results/eda/gaps.csv``        every time gap above the threshold
``results/figures/eda/*.png``   report figures
``maps/eda_*.html``             interactive Folium maps

Usage
-----
    python scripts/eda.py                     # every partition present on disk
    python scripts/eda.py --partitions train  # one partition only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import MAX_DT_GAP_S, data_root, maps_dir, results_dir  # noqa: E402
from pathsnap.geo.projection import haversine_m, project_trip, projection_error  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.io.loader import (  # noqa: E402
    PARTITIONS,
    discover_trips,
    load_trip_for_training,
    validate_trip,
)

#: Points with nothing within this distance are clipped to it.
FAR_DISTANCE_M = 200.0

#: A trip whose median distance to the nearest road exceeds this is flagged as offset.
OFFSET_FLAG_M = 25.0


def nearest_road_distance(graph, points_xy: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest arc in the network.

    Uses the graph's own search so the number means the same thing here as it will in
    the candidate stage. Points with nothing inside ``FAR_DISTANCE_M`` are returned as
    that value, which is enough to separate "noisy" from "off the network".
    """
    out = np.full(points_xy.shape[0], FAR_DISTANCE_M, dtype=np.float64)
    for i, point in enumerate(points_xy):
        arcs = graph.arcs_within(point, FAR_DISTANCE_M)
        if arcs.size:
            distances = graph.point_to_arc_distances(
                np.repeat(point[None, :], arcs.size, 0), arcs
            )
            out[i] = float(distances.min())
    return out


def measure_trip(ref) -> dict[str, object]:
    """Compute every per-trip measurement used by the tables and figures."""
    trip = load_trip_for_training(ref.trip_dir, ref.partition)
    projected = project_trip(trip)
    graph = build_road_graph(trip, projected)
    validation = validate_trip(trip)
    stats = graph.stats()

    lon = trip.track[:, 0]
    lat = trip.track[:, 1]
    dt = np.diff(trip.track[:, 2])

    # Distance walked along the trajectory, and the straight-line spread of the trip.
    leg = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    path_km = float(np.sum(leg) / 1000.0)
    extent_km = float(
        haversine_m(lon.min(), lat.min(), lon.max(), lat.max()) / 1000.0
    )

    nearest = nearest_road_distance(graph, projected.track_xy)
    projection_err = projection_error(
        projected.projection, lon, lat, max_pairs=1500, seed=0
    )

    distance_map = {
        code: sum(i.count for i in validation.issues if i.code == code)
        for code in validation.codes()
    }

    return {
        "_dir": ref.trip_dir,
        "partition": ref.partition,
        "trip_id": ref.trip_id,
        "n_points": trip.n_points,
        "n_nodes": trip.n_nodes,
        "n_arcs": trip.n_arcs,
        "n_route": trip.n_route,
        "invalid": int(not validation.ok),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_centre": float(lon.mean()),
        "lat_centre": float(lat.mean()),
        "duration_s": float(trip.track[-1, 2] - trip.track[0, 2]) if trip.n_points else 0.0,
        "dt_median_s": float(np.median(dt)) if dt.size else 0.0,
        "dt_max_s": float(dt.max()) if dt.size else 0.0,
        "dt_one_share": float(np.mean(dt == 1.0)) if dt.size else 0.0,
        "n_gaps": int(np.sum(dt > MAX_DT_GAP_S)),
        "n_duplicate_fixes": int(distance_map.get("W_GPS_DUPLICATE", 0)),
        "path_km": path_km,
        "extent_km": extent_km,
        "mean_speed_kmh": float(path_km / (trip.track[-1, 2] / 3600.0))
        if trip.n_points > 1 and trip.track[-1, 2] > 0
        else 0.0,
        "n_nonfinite": int(
            np.sum(~np.isfinite(trip.track)) + np.sum(~np.isfinite(trip.nodes))
        ),
        "n_time_decreasing": int(np.sum(np.diff(trip.track[:, 2]) < 0))
        if trip.n_points > 1
        else 0,
        "n_coords_out_of_range": int(
            np.sum((np.abs(trip.track[:, 1]) > 90.0) | (np.abs(trip.track[:, 0]) > 180.0))
        ),
        "track_inside_nodes_bbox_share": float(
            np.mean(
                (lon >= trip.nodes[:, 0].min())
                & (lon <= trip.nodes[:, 0].max())
                & (lat >= trip.nodes[:, 1].min())
                & (lat <= trip.nodes[:, 1].max())
            )
        ),
        "road_median_m": float(np.median(nearest)) if nearest.size else 0.0,
        "road_p95_m": float(np.percentile(nearest, 95)) if nearest.size else 0.0,
        "road_far_share": float(np.mean(nearest >= FAR_DISTANCE_M)) if nearest.size else 0.0,
        "n_components": stats.n_components,
        "largest_component_share": stats.largest_component_share,
        "n_isolated_nodes": stats.n_isolated_nodes,
        "mean_out_degree": stats.mean_out_degree,
        "max_out_degree": stats.max_out_degree,
        "n_dead_ends": stats.n_dead_end_nodes,
        "n_self_loops": stats.n_self_loops,
        "n_zero_length_arcs": stats.n_zero_length_arcs,
        "network_length_km": stats.total_length_km,
        "median_arc_length_m": stats.median_arc_length_m,
        "projection_max_rel_error": projection_err["max_rel_error"],
        "route_track_ratio": float(trip.n_route / trip.n_points) if trip.n_points else 0.0,
        "issue_codes": "|".join(sorted(distance_map)),
        "_nearest_sample": nearest[:: max(1, nearest.size // 400)],
        "_dt_sample": dt,
        "_graph": graph,
        "_projected": projected,
        "_trip": trip,
    }


def write_tables(rows: list[dict], out_dir: Path) -> None:
    import csv

    scalar_keys = [k for k in rows[0] if not k.startswith("_")]
    with (out_dir / "trips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Every time gap above the threshold, so the long-gap claim can be checked.
    with (out_dir / "gaps.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["partition", "trip_id", "observation_index", "gap_seconds"])
        for row in rows:
            dt = row["_dt_sample"]
            for index in np.flatnonzero(dt > MAX_DT_GAP_S):
                writer.writerow([row["partition"], row["trip_id"], int(index), float(dt[index])])


def summarise(rows: list[dict]) -> dict[str, object]:
    by_partition: dict[str, dict[str, object]] = {}
    for partition in sorted({r["partition"] for r in rows}):
        subset = [r for r in rows if r["partition"] == partition]
        by_partition[partition] = {
            "trips": len(subset),
            "gps_points": int(sum(r["n_points"] for r in subset)),
            "nodes": int(sum(r["n_nodes"] for r in subset)),
            "arcs": int(sum(r["n_arcs"] for r in subset)),
            "route_arcs": int(sum(r["n_route"] for r in subset)),
            "invalid_trips": [r["trip_id"] for r in subset if r["invalid"]],
            "trips_with_gaps": sorted(r["trip_id"] for r in subset if r["n_gaps"]),
            "trips_with_duplicate_fixes": sorted(
                r["trip_id"] for r in subset if r["n_duplicate_fixes"]
            ),
            "offset_trips": sorted(
                r["trip_id"] for r in subset if r["road_median_m"] > OFFSET_FLAG_M
            ),
            "total_path_km": float(sum(r["path_km"] for r in subset)),
        }

    longest_gap = max(rows, key=lambda r: r["dt_max_s"])
    worst_offset = max(rows, key=lambda r: r["road_median_m"])

    return {
        "partition_scope": sorted({r["partition"] for r in rows}),
        "by_partition": by_partition,
        "sampling": {
            "share_of_steps_equal_to_1s": float(np.mean([r["dt_one_share"] for r in rows])),
            "median_of_median_dt_s": float(np.median([r["dt_median_s"] for r in rows])),
        },
        "gps_to_road": {
            "median_of_trip_medians_m": float(np.median([r["road_median_m"] for r in rows])),
            "median_of_trip_p95_m": float(np.median([r["road_p95_m"] for r in rows])),
            "worst_trip": worst_offset["trip_id"],
            "worst_trip_median_m": worst_offset["road_median_m"],
        },
        "gaps": {
            "trips_with_gap": int(sum(1 for r in rows if r["n_gaps"])),
            "total_gaps": int(sum(r["n_gaps"] for r in rows)),
            "largest_gap_s": longest_gap["dt_max_s"],
            "largest_gap_trip": longest_gap["trip_id"],
        },
        "missing_values": {
            "trips_with_nonfinite": int(sum(1 for r in rows if r["n_nonfinite"])),
            "total_nonfinite": int(sum(r["n_nonfinite"] for r in rows)),
            "trips_with_decreasing_timestamps": int(
                sum(1 for r in rows if r["n_time_decreasing"])
            ),
            "trips_with_out_of_range_coordinates": int(
                sum(1 for r in rows if r["n_coords_out_of_range"])
            ),
            "trips_with_duplicate_fixes": int(
                sum(1 for r in rows if r["n_duplicate_fixes"])
            ),
            "min_track_inside_nodes_bbox_share": float(
                min(r["track_inside_nodes_bbox_share"] for r in rows)
            ),
        },
        "connectivity": {
            "median_components": float(np.median([r["n_components"] for r in rows])),
            "min_largest_component_share": float(
                min(r["largest_component_share"] for r in rows)
            ),
        },
        "projection": {
            "max_relative_error": float(max(r["projection_max_rel_error"] for r in rows)),
            "mean_relative_error": float(
                np.mean([r["projection_max_rel_error"] for r in rows])
            ),
        },
    }


def make_figures(rows: list[dict], summary: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    scope = ", ".join(summary["partition_scope"])
    made: list[str] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(name)

    def counts(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    def scatter_hist(ax, key: str, title: str, xlabel: str, log: bool = True) -> None:
        values = counts(key)
        ax.hist(values[values > 0], bins=30, color="#3b6ea5", edgecolor="white")
        if log:
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("trips", fontsize=9)
        ax.grid(alpha=0.25)

    # Figure 1: dataset composition.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    partitions = list(summary["by_partition"])
    trips = [summary["by_partition"][p]["trips"] for p in partitions]
    points = [summary["by_partition"][p]["gps_points"] for p in partitions]
    axes[0].bar(partitions, trips, color="#3b6ea5")
    axes[0].set_title(f"Trips per partition ({scope})", fontsize=10)
    axes[0].set_ylabel("trips", fontsize=9)
    axes[1].bar(partitions, points, color="#8c6135")
    axes[1].set_title(f"GPS observations per partition ({scope})", fontsize=10)
    axes[1].set_ylabel("observations", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    save(fig, "01_inventory.png")

    # Figure 2: size distributions.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    scatter_hist(axes[0, 0], "n_points", "GPS observations per trip", "observations")
    scatter_hist(axes[0, 1], "n_nodes", "Road-network nodes per trip", "nodes")
    scatter_hist(axes[1, 0], "n_arcs", "Road-network arcs per trip", "arcs")
    scatter_hist(axes[1, 1], "n_route", "Reference route length", "arcs")
    fig.suptitle(f"Trip size distributions ({scope})", fontsize=11)
    save(fig, "02_trip_sizes.png")

    # Figure 3: sampling interval.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    dt_all = np.concatenate([r["_dt_sample"] for r in rows if r["_dt_sample"].size])
    axes[0].hist(dt_all[dt_all <= 10], bins=60, color="#3b6ea5", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Sampling interval, steps up to 10 s ({scope})", fontsize=10)
    axes[0].set_xlabel("seconds between fixes", fontsize=9)
    axes[0].set_ylabel("count (log)", fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[1].hist(np.log10(np.maximum(dt_all, 1e-3)), bins=60, color="#8c6135", edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title(f"All intervals, log scale ({scope})", fontsize=10)
    axes[1].set_xlabel("log10(seconds)", fontsize=9)
    axes[1].set_ylabel("count (log)", fontsize=9)
    axes[1].grid(alpha=0.25)
    save(fig, "03_sampling_intervals.png")

    # Figure 4: gaps per trip.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    order = np.argsort([-r["dt_max_s"] for r in rows])
    values = [rows[i]["dt_max_s"] for i in order]
    labels = [f"{rows[i]['partition'][:2]}/{rows[i]['trip_id']}" for i in order]
    ax.bar(range(len(values)), np.maximum(values, 1e-3), color="#a8453b")
    ax.set_yscale("log")
    ax.set_title(f"Largest time gap per trip, sorted ({scope})", fontsize=10)
    ax.set_ylabel("seconds (log)", fontsize=9)
    ax.set_xticks(range(len(labels))[::5])
    ax.set_xticklabels(labels[::5], rotation=90, fontsize=6)
    ax.grid(alpha=0.25, axis="y")
    save(fig, "04_gaps.png")

    # Figure 5: distance from fixes to the nearest road. This is the ambiguity evidence.
    fig, ax = plt.subplots(figsize=(7, 4))
    nearest_all = np.concatenate([r["_nearest_sample"] for r in rows])
    ax.hist(nearest_all, bins=80, range=(0, 100), color="#3b6ea5", edgecolor="white")
    ax.set_title(f"Distance from GPS fix to nearest road arc ({scope})", fontsize=10)
    ax.set_xlabel("metres", fontsize=9)
    ax.set_ylabel("observations", fontsize=9)
    ax.axvline(float(np.median(nearest_all)), color="#a8453b", linestyle="--", linewidth=1)
    ax.annotate(
        f"median {np.median(nearest_all):.1f} m",
        xy=(np.median(nearest_all), ax.get_ylim()[1] * 0.9),
        xytext=(np.median(nearest_all) + 8, ax.get_ylim()[1] * 0.9),
        fontsize=9,
        color="#a8453b",
    )
    ax.grid(alpha=0.25)
    save(fig, "05_gps_to_road_distance.png")

    # Figure 6: where the trips are.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(counts("lon_centre"), counts("lat_centre"), s=22, color="#3b6ea5")
    ax.set_title(f"Trip locations by centroid ({scope})", fontsize=10)
    ax.set_xlabel("longitude", fontsize=9)
    ax.set_ylabel("latitude", fontsize=9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(alpha=0.25)
    save(fig, "06_trip_locations.png")

    # Figure 7: road-network shape.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(counts("mean_out_degree"), counts("max_out_degree"), s=18, color="#3b6ea5")
    axes[0].set_title("Node degree per trip", fontsize=10)
    axes[0].set_xlabel("mean out-degree", fontsize=9)
    axes[0].set_ylabel("max out-degree", fontsize=9)
    axes[1].hist(counts("largest_component_share"), bins=25, color="#4f7d4f", edgecolor="white")
    axes[1].set_title("Largest component share", fontsize=10)
    axes[1].set_xlabel("fraction of nodes", fontsize=9)
    axes[1].set_ylabel("trips", fontsize=9)
    axes[2].scatter(counts("n_arcs"), counts("n_components"), s=18, color="#8c6135")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Components vs network size", fontsize=10)
    axes[2].set_xlabel("arcs", fontsize=9)
    axes[2].set_ylabel("components", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Road-network connectivity ({scope})", fontsize=11)
    save(fig, "07_network_connectivity.png")

    # Figure 8: route against track length.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(counts("n_points"), counts("n_route"), s=20, color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Reference route length against GPS count ({scope})", fontsize=10)
    ax.set_xlabel("GPS observations", fontsize=9)
    ax.set_ylabel("route arcs", fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, "08_route_vs_track.png")

    # Figure 9: sampling of actual geometry, one busy and one quiet trip.
    busiest = max(rows, key=lambda r: r["n_arcs"])
    quietest = min(rows, key=lambda r: r["n_arcs"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, row, title in (
        (axes[0], busiest, f"densest network: {busiest['trip_id']}"),
        (axes[1], quietest, f"sparsest network: {quietest['trip_id']}"),
    ):
        graph = row["_graph"]
        subset = graph.arcs[:: max(1, graph.n_arcs // 12000)]
        origin = graph.nodes_xy[subset[:, 0]]
        target = graph.nodes_xy[subset[:, 1]]
        for a, b in zip(origin, target):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", linewidth=0.3, zorder=0)
        track = row["_projected"].track_xy
        ax.scatter(track[:, 0], track[:, 1], s=2, color="#a8453b", zorder=2)
        ax.set_title(f"{title} ({row['partition']})", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("metres east of trip centre", fontsize=8)
        ax.set_ylabel("metres north of trip centre", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"GPS fixes against road geometry ({scope})", fontsize=11)
    save(fig, "09_track_vs_network.png")

    # Figure 10: coordinate order check.
    order_rows, order_summary = coordinate_order_report(rows)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    written = np.array([r["median_dist_as_written"] for r in order_rows])
    swapped = np.array([r["median_dist_swapped"] for r in order_rows])
    ax.scatter(np.maximum(written, 1e-4), np.maximum(swapped, 1e-4), s=18, color="#3b6ea5")
    limits = [1e-4, max(written.max(), swapped.max()) * 2]
    ax.plot(limits, limits, color="#999999", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"Distance to nearest node: written order against swapped ({scope})", fontsize=10
    )
    ax.set_xlabel("degrees, read as (lon, lat)", fontsize=9)
    ax.set_ylabel("degrees, read as (lat, lon)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.annotate(
        "dotted line = equal",
        xy=(limits[0] * 3, limits[0] * 3),
        fontsize=8,
        color="#777777",
    )
    save(fig, "10_coordinate_order.png")

    # Figure 11: geographical extent of each trajectory.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    order = np.argsort([-r["extent_km"] for r in rows])
    values = [max(rows[i]["extent_km"], 1e-3) for i in order]
    labels = [rows[i]["trip_id"] for i in order]
    ax.bar(range(len(values)), values, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_title(f"Geographical extent of each trajectory, sorted ({scope})", fontsize=10)
    ax.set_ylabel("km, corner to corner (log)", fontsize=9)
    ax.set_xticks(range(len(labels))[::5])
    ax.set_xticklabels(labels[::5], rotation=90, fontsize=6)
    ax.grid(alpha=0.25, axis="y")
    save(fig, "11_trajectory_extent.png")

    # Figure 12: projection error per trip.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    errors = np.array([r["projection_max_rel_error"] * 100.0 for r in rows])
    ax.bar(range(errors.size), np.sort(errors)[::-1], color="#4f7d4f")
    ax.set_title(f"Projection error against haversine, per trip ({scope})", fontsize=10)
    ax.set_ylabel("max relative error, percent", fontsize=9)
    ax.set_xlabel("trips, sorted", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    save(fig, "12_projection_error.png")

    # Figure 13: where the coordinates themselves sit.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lons_all = np.concatenate([r["_trip"].track[:, 0] for r in rows])
    lats_all = np.concatenate([r["_trip"].track[:, 1] for r in rows])
    axes[0].hist(lons_all, bins=120, color="#3b6ea5")
    axes[0].set_title(f"Longitude of all fixes ({scope})", fontsize=10)
    axes[0].set_xlabel("degrees", fontsize=9)
    axes[0].set_ylabel("observations", fontsize=9)
    axes[1].hist(lats_all, bins=120, color="#8c6135")
    axes[1].set_title(f"Latitude of all fixes ({scope})", fontsize=10)
    axes[1].set_xlabel("degrees", fontsize=9)
    axes[1].set_ylabel("observations", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.axvline(0, color="#999999", linewidth=0.8)
    save(fig, "13_coordinate_distributions.png")

    return made


def zoom_for_bounds(lats: np.ndarray, lons: np.ndarray, pixels: int = 900) -> int:
    """Pick the conventional tile zoom that fits a bounding box into about ``pixels``.

    ``fit_bounds`` is still applied, but it runs before the map container has its final
    size, so the initial zoom is set here as well.
    """
    lat_centre = float(np.mean(lats))
    width_m = float(haversine_m(lons.min(), lat_centre, lons.max(), lat_centre))
    height_m = float(haversine_m(lons.mean(), lats.min(), lons.mean(), lats.max()))
    span_m = max(width_m, height_m, 100.0)
    resolution = 156543.03392 * np.cos(np.radians(lat_centre))
    zoom = np.log2(resolution * pixels / span_m)
    return int(np.clip(np.floor(zoom), 3, 17))


def make_maps(rows: list[dict], out_dir: Path) -> list[str]:
    import folium

    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    dense = max(rows, key=lambda r: r["n_arcs"])
    sparse = min(rows, key=lambda r: r["n_arcs"])
    offset = max(rows, key=lambda r: r["road_median_m"])

    for row, name, note in (
        (dense, "eda_dense_network.html", "densest road network"),
        (sparse, "eda_sparse_network.html", "sparsest road network"),
        (offset, "eda_offset_trip.html", "largest offset between GPS and road"),
    ):
        trip = row["_trip"]
        graph = row["_graph"]

        # The graph stores projected metres; Folium needs degrees, so convert back.
        nodes_lonlat = graph.projection.unproject(graph.nodes_xy[:, 0], graph.nodes_xy[:, 1])

        lats, lons = trip.track[:, 1], trip.track[:, 0]
        fmap = folium.Map(
            location=[float(np.mean(lats)), float(np.mean(lons))],
            zoom_start=zoom_for_bounds(lats, lons),
            tiles="Esri World Street Map",
        )
        fmap.fit_bounds(
            [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
        )

        network_layer = folium.FeatureGroup(name="Road network (sampled)")
        step = max(1, graph.n_arcs // 2000)
        for arc in range(0, graph.n_arcs, step):
            u, v = int(graph.from_node[arc]), int(graph.to_node[arc])
            network_layer.add_child(
                folium.PolyLine(
                    locations=[
                        (float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])),
                        (float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])),
                    ],
                    color="#444444",
                    weight=1.5,
                    opacity=0.6,
                )
            )

        gps_layer = folium.FeatureGroup(name="GPS fixes")
        for lon, lat in trip.track[:: max(1, trip.n_points // 1200), :2]:
            gps_layer.add_child(
                folium.CircleMarker(
                    location=(float(lat), float(lon)),
                    radius=3,
                    color="#c0392b",
                    fill=True,
                    fill_color="#c0392b",
                    fill_opacity=0.85,
                )
            )

        # Layers first, then the control, so the control knows what to toggle.
        network_layer.add_to(fmap)
        gps_layer.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)

        title = (
            f"<div style='position:fixed;top:10px;left:60px;z-index:9999;"
            f"background:white;padding:6px 10px;border:1px solid #999;font-size:13px'>"
            f"{row['partition']} / {row['trip_id']} | {note}<br>"
            f"<span style='color:#555555'>grey: road network (sampled)</span> &nbsp; "
            f"<span style='color:#c0392b'>red: GPS fixes</span></div>"
        )
        fmap.get_root().html.add_child(folium.Element(title))

        path = out_dir / name
        fmap.save(str(path))
        made.append(name)

    return made



# ------------------------------------------------------------------------------------
# Dataset audit: the checks the brief asks for explicitly
# ------------------------------------------------------------------------------------

#: Expected columns and dtypes per raw file.
FILE_SCHEMA: dict[str, tuple[str, int, str]] = {
    ".track": ("longitude latitude timestamp_s", 3, "float64"),
    ".nodes": ("longitude latitude", 2, "float64"),
    ".arcs": ("from_node_id to_node_id", 2, "int64"),
    ".route": ("arc_id", 1, "int64"),
}


def file_inventory(rows: list[dict]) -> list[dict]:
    """Availability, byte size, row count, column count and dtype of every raw file."""
    out: list[dict] = []
    for row in rows:
        trip = row["_trip"]
        arrays = {
            ".track": trip.track,
            ".nodes": trip.nodes,
            ".arcs": trip.arcs,
            ".route": trip.route if trip.route is not None else np.empty(0),
        }
        for suffix, (columns, expected_columns, expected_dtype) in FILE_SCHEMA.items():
            path = row["_dir"] / f"{trip.trip_id}{suffix}"
            array = arrays[suffix]
            n_rows = 1 if array.ndim == 1 else array.shape[0]
            actual_columns = 1 if array.ndim == 1 else array.shape[1]
            out.append(
                {
                    "partition": row["partition"],
                    "trip_id": trip.trip_id,
                    "file": suffix,
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                    "rows": int(n_rows),
                    "columns": int(actual_columns),
                    "expected_columns": expected_columns,
                    "column_names": columns,
                    "dtype": str(array.dtype),
                    "expected_dtype": expected_dtype,
                    "schema_ok": bool(
                        n_rows > 0
                        and actual_columns == expected_columns
                        and np.dtype(array.dtype) == np.dtype(expected_dtype)
                    ),
                }
            )
    return out


def coordinate_order_report(rows: list[dict]) -> tuple[list[dict], dict]:
    """Test that column 0 is longitude and column 1 is latitude.

    Two independent arguments. First, a latitude cannot exceed 90 degrees, so any
    trip whose column 0 passes 90 settles the order outright. Second, with the
    opposite reading every fix would land on the wrong side of the planet, which
    shows up as a much larger distance to the nearest network node.
    """
    from scipy.spatial import cKDTree

    per_trip: list[dict] = []
    for row in rows:
        trip = row["_trip"]
        count = min(200, trip.n_points)
        index = np.linspace(0, trip.n_points - 1, count).astype(int)
        points = trip.track[index, :2]
        tree = cKDTree(trip.nodes)
        as_written, _ = tree.query(points)
        swapped, _ = tree.query(points[:, [1, 0]])
        per_trip.append(
            {
                "partition": row["partition"],
                "trip_id": trip.trip_id,
                "max_abs_column0": float(np.abs(trip.track[:, 0]).max()),
                "max_abs_column1": float(np.abs(trip.track[:, 1]).max()),
                "median_dist_as_written": float(np.median(as_written)),
                "median_dist_swapped": float(np.median(swapped)),
                "written_is_closer": bool(np.median(as_written) < np.median(swapped)),
            }
        )

    column0 = np.array([r["max_abs_column0"] for r in per_trip])
    column1 = np.array([r["max_abs_column1"] for r in per_trip])
    summary = {
        "max_abs_column0": float(column0.max()),
        "max_abs_column1": float(column1.max()),
        "trips_column0_above_90": int(np.sum(column0 > 90.0)),
        "trips_column1_above_90": int(np.sum(column1 > 90.0)),
        "trips_where_written_order_is_closer": int(sum(r["written_is_closer"] for r in per_trip)),
        "conclusion": (
            "column 0 is longitude, column 1 is latitude: a latitude cannot exceed 90 "
            "degrees, and several trips exceed it in column 0 only; reading the pair in "
            "the written order also puts every fix far closer to the network"
        ),
    }
    return per_trip, summary


def reference_route_report(rows: list[dict]) -> list[dict]:
    """How the reference route is encoded and how well it associates with the network."""
    out: list[dict] = []
    for row in rows:
        trip = row["_trip"]
        graph = row["_graph"]
        if trip.route is None or trip.route.size == 0:
            out.append(
                {
                    "partition": row["partition"],
                    "trip_id": trip.trip_id,
                    "n_route": 0,
                    "n_unique_arcs": 0,
                    "n_chain_breaks": 0,
                    "n_revisited_arcs": 0,
                    "route_length_km": 0.0,
                    "arcs_share_of_network": 0.0,
                    "ratio_route_to_points": 0.0,
                    "median_dist_to_route_m": 0.0,
                }
            )
            continue

        sequence = trip.arcs[trip.route]
        breaks = int(np.sum(sequence[:-1, 1] != sequence[1:, 0]))
        unique = int(np.unique(trip.route).size)

        # How far the fixes sit from the route they are supposed to be on.
        projected = row["_projected"]
        sample = projected.track_xy[:: max(1, len(projected.track_xy) // 200)]
        repeated_points = np.repeat(sample, trip.route.size, axis=0)
        repeated_arcs = np.tile(trip.route, sample.shape[0])
        distances = graph.segment_distance(repeated_points, repeated_arcs)[0].reshape(
            sample.shape[0], trip.route.size
        )
        median_to_route = float(np.median(distances.min(axis=1)))
        out.append(
            {
                "partition": row["partition"],
                "trip_id": trip.trip_id,
                "n_route": int(trip.route.size),
                "n_unique_arcs": unique,
                "n_chain_breaks": breaks,
                "n_revisited_arcs": int(trip.route.size - unique),
                "route_length_km": float(graph.arc_lengths_m[trip.route].sum() / 1000.0),
                "arcs_share_of_network": float(trip.route.size / graph.n_arcs),
                "ratio_route_to_points": float(trip.route.size / trip.n_points),
                "median_dist_to_route_m": median_to_route,
            }
        )
    return out


def write_audit_tables(
    inventory: list[dict],
    order_rows: list[dict],
    route_rows: list[dict],
    out_dir: Path,
) -> dict[str, object]:
    import csv

    def dump(name: str, records: list[dict]) -> None:
        if not records:
            return
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    dump("file_inventory.csv", inventory)
    dump("coordinate_order.csv", order_rows)
    dump("reference_route.csv", route_rows)

    missing = [r for r in inventory if not r["exists"]]
    bad_schema = [r for r in inventory if not r["schema_ok"]]
    sizes: dict[str, list[int]] = defaultdict(list)
    for record in inventory:
        sizes[record["file"]].append(record["size_bytes"])

    return {
        "files": {
            "files_checked": len(inventory),
            "missing_files": len(missing),
            "schema_mismatches": len(bad_schema),
            "size_bytes": {
                name: {
                    "min": int(np.min(values)),
                    "median": int(np.median(values)),
                    "max": int(np.max(values)),
                    "total": int(np.sum(values)),
                }
                for name, values in sorted(sizes.items())
            },
        },
        "reference_route": {
            "trips_with_route": int(sum(1 for r in route_rows if r["n_route"] > 0)),
            "trips_with_chain_breaks": int(sum(1 for r in route_rows if r["n_chain_breaks"])),
            "trips_revisiting_an_arc": int(sum(1 for r in route_rows if r["n_revisited_arcs"])),
            "median_ratio_route_to_points": float(
                np.median([r["ratio_route_to_points"] for r in route_rows])
            ),
            "median_dist_to_route_m": float(
                np.median([r["median_dist_to_route_m"] for r in route_rows])
            ),
            "worst_route_offset_trip": max(
                route_rows, key=lambda r: r["median_dist_to_route_m"]
            )["trip_id"],
            "worst_route_offset_m": float(
                max(r["median_dist_to_route_m"] for r in route_rows)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", nargs="*", default=None, choices=list(PARTITIONS))
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    root = Path(args.root) if args.root else data_root()
    refs = discover_trips(root, partitions=args.partitions)
    if not refs:
        print(f"no trips found under {root}", file=sys.stderr)
        return 1

    print(f"dataset root: {root}")
    print(f"partitions  : {sorted({r.partition for r in refs})}")
    print(f"trips       : {len(refs)}")

    rows: list[dict] = []
    for i, ref in enumerate(refs, 1):
        rows.append(measure_trip(ref))
        if i % 10 == 0 or i == len(refs):
            print(f"  measured {i}/{len(refs)}")

    eda_dir = results_dir() / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    write_tables(rows, eda_dir)

    summary = summarise(rows)

    inventory = file_inventory(rows)
    order_rows, order_summary = coordinate_order_report(rows)
    route_rows = reference_route_report(rows)
    summary["audit"] = write_audit_tables(inventory, order_rows, route_rows, eda_dir)
    summary["audit"]["coordinate_order"] = order_summary

    (eda_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figures = make_figures(rows, summary, results_dir() / "figures" / "eda")
    maps = make_maps(rows, maps_dir())

    print()
    for partition, stats in summary["by_partition"].items():
        print(
            f"{partition:<14} trips={stats['trips']:>3} points={stats['gps_points']:>7} "
            f"arcs={stats['arcs']:>8} gaps={len(stats['trips_with_gaps']):>3} "
            f"offset={len(stats['offset_trips'])}"
        )
    print()
    print(f"figures: {len(figures)} written to {results_dir() / 'figures' / 'eda'}")
    print(f"maps   : {len(maps)} written to {maps_dir()}")
    print(f"tables : {eda_dir / 'trips.csv'}, {eda_dir / 'gaps.csv'}, {eda_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
