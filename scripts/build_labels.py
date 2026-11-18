"""Derive observation-level reference labels for every supplied trip.

The alignment in :mod:`pathsnap.labels.truth` has two tunable terms: an absolute position
prior and a per-step skip penalty. Both are swept here and the measured effect is recorded,
so the defaults follow from measurement.

Outputs
-------
``data/processed/labels/<trip_id>.npz``   cached labels, one file per trip
``results/labels/label_summary.csv``      per-trip label quality
``results/labels/weight_sweep.csv``       position weight and skip penalty sweep
``results/labels/baseline_comparison.csv`` ordered alignment against naive nearest arc
``results/labels/manifest.json``          how the cache was produced and what to watch for

Usage
-----
    python scripts/build_labels.py
    python scripts/build_labels.py --partitions train
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
from pathsnap.io.loader import discover_trips, load_trip_for_training  # noqa: E402
from pathsnap.labels.truth import (  # noqa: E402
    compare_with_baseline,
    derive_observation_labels,
    nearest_route_arc_baseline,
)

#: Trips used for the weight sweep, chosen to span the size range.
SWEEP_TRIPS = (
    "00000001",
    "00000003",
    "00000008",
    "00000021",
    "00000033",
    "00000042",
    "00000046",
    "00000054",
    "00000078",
    "00000092",
)

POSITION_WEIGHTS = (0.0, 0.05, 0.15, 0.5, 1.0)
SKIP_PENALTIES = (0.0, 0.5, 2.0, 8.0)

#: A trip is flagged when its labels are weaker than this.
LOW_FIDELITY_MEDIAN_DIST_M = 25.0
LOW_FIDELITY_CONFIDENCE = 0.35


def sweep_weights(trips: dict[str, object]) -> list[dict]:
    """Measure how each alignment term affects label fidelity."""
    records: list[dict] = []
    for trip_id in SWEEP_TRIPS:
        trip = trips.get(trip_id)
        if trip is None:
            continue
        baseline = nearest_route_arc_baseline(trip)

        for weight in POSITION_WEIGHTS:
            started = time.perf_counter()
            labels = derive_observation_labels(trip, position_weight=weight)
            elapsed = time.perf_counter() - started
            comparison = compare_with_baseline(labels, baseline)
            records.append(
                {
                    "trip_id": trip_id,
                    "term": "position_weight",
                    "value": weight,
                    "median_dist_to_label_m": labels.diagnostics["median_dist_to_label_m"],
                    "mean_confidence": labels.mean_confidence(),
                    "valid_fraction": labels.valid_fraction,
                    "agreement_with_naive": comparison["agreement"],
                    "seconds": elapsed,
                }
            )

        for penalty in SKIP_PENALTIES:
            started = time.perf_counter()
            labels = derive_observation_labels(trip, skip_penalty=penalty)
            elapsed = time.perf_counter() - started
            comparison = compare_with_baseline(labels, baseline)
            records.append(
                {
                    "trip_id": trip_id,
                    "term": "skip_penalty",
                    "value": penalty,
                    "median_dist_to_label_m": labels.diagnostics["median_dist_to_label_m"],
                    "mean_confidence": labels.mean_confidence(),
                    "valid_fraction": labels.valid_fraction,
                    "agreement_with_naive": comparison["agreement"],
                    "seconds": elapsed,
                }
            )
    return records


def summarise_sweep(records: list[dict]) -> list[dict]:
    """Collapse the sweep to one row per term value, averaged over trips."""
    out: list[dict] = []
    for term in sorted({r["term"] for r in records}):
        for value in sorted({r["value"] for r in records if r["term"] == term}):
            subset = [r for r in records if r["term"] == term and r["value"] == value]
            out.append(
                {
                    "term": term,
                    "value": value,
                    "trips": len(subset),
                    "mean_median_dist_to_label_m": float(
                        np.mean([r["median_dist_to_label_m"] for r in subset])
                    ),
                    "mean_confidence": float(np.mean([r["mean_confidence"] for r in subset])),
                    "mean_valid_fraction": float(np.mean([r["valid_fraction"] for r in subset])),
                    "mean_agreement_with_naive": float(
                        np.mean([r["agreement_with_naive"] for r in subset])
                    ),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", nargs="*", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-sweep", action="store_true", help="skip the weight sweep")
    args = parser.parse_args()

    set_global_seed(DEFAULT_SEED)
    root = Path(args.root) if args.root else data_root()
    refs = discover_trips(root, partitions=args.partitions)
    if not refs:
        print(f"no trips found under {root}", file=sys.stderr)
        return 1

    print(f"dataset root: {root}")
    print(f"trips       : {len(refs)}")

    trips = {}
    for ref in refs:
        trips[ref.trip_id] = load_trip_for_training(ref.trip_dir, ref.partition)

    label_dir = processed_dir() / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    out_dir = results_dir() / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    baseline_rows: list[dict] = []
    started_all = time.perf_counter()

    for index, ref in enumerate(refs, 1):
        trip = trips[ref.trip_id]
        started = time.perf_counter()
        labels = derive_observation_labels(trip)
        elapsed = time.perf_counter() - started

        np.savez_compressed(
            label_dir / f"{ref.trip_id}.npz",
            arc_ids=labels.arc_ids,
            route_positions=labels.route_positions,
            confidence=labels.confidence,
            valid=labels.valid,
            dist_to_label_m=labels.dist_to_label_m,
            margin_m=labels.margin_m,
        )

        baseline = nearest_route_arc_baseline(trip)
        comparison = compare_with_baseline(labels, baseline)

        low_fidelity = (
            not np.isfinite(labels.diagnostics["median_dist_to_label_m"])
            or labels.diagnostics["median_dist_to_label_m"] > LOW_FIDELITY_MEDIAN_DIST_M
            or labels.mean_confidence() < LOW_FIDELITY_CONFIDENCE
            or labels.valid_fraction < 0.99
        )

        summary_rows.append(
            {
                "partition": ref.partition,
                "trip_id": ref.trip_id,
                "n_points": labels.n_observations,
                "n_route": labels.diagnostics["route_length"],
                "n_valid": labels.n_valid,
                "valid_fraction": labels.valid_fraction,
                "n_invalid": labels.diagnostics["n_invalid"],
                "monotone": labels.diagnostics["monotone"],
                "n_distinct_positions": labels.diagnostics["n_distinct_positions"],
                "median_dist_to_label_m": labels.diagnostics["median_dist_to_label_m"],
                "share_within_10m": labels.diagnostics["share_within_10m"],
                "mean_confidence": labels.mean_confidence(),
                "alignment_cost": labels.diagnostics["total_cost"],
                "low_fidelity_trip": low_fidelity,
                "seconds": elapsed,
            }
        )
        baseline_rows.append(
            {
                "partition": ref.partition,
                "trip_id": ref.trip_id,
                "n_compared": comparison["n_compared"],
                "n_agree": comparison["n_agree"],
                "agreement": comparison["agreement"],
                "n_disagree": comparison["n_disagree"],
                "median_position_gap": comparison["median_position_gap"],
                "max_position_gap": comparison["max_position_gap"],
            }
        )

        if index % 10 == 0 or index == len(refs):
            print(f"  labelled {index}/{len(refs)}")

    total_seconds = time.perf_counter() - started_all

    def dump(name: str, records: list[dict]) -> None:
        if not records:
            return
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    dump("label_summary.csv", summary_rows)
    dump("baseline_comparison.csv", baseline_rows)

    sweep_summary: list[dict] = []
    if not args.no_sweep:
        print("sweeping alignment terms")
        sweep_summary = summarise_sweep(sweep_weights(trips))
        dump("weight_sweep.csv", sweep_summary)

    low = [r for r in summary_rows if r["low_fidelity_trip"]]
    monotone_all = all(r["monotone"] for r in summary_rows)
    agreement = np.array([r["agreement"] for r in baseline_rows if r["n_compared"]])

    manifest = {
        "generated_from": str(root),
        "trips": len(summary_rows),
        "seed": DEFAULT_SEED,
        "defaults": {"position_weight": 0.0, "skip_penalty": 0.0},
        "monotone_for_every_trip": bool(monotone_all),
        "total_observations": int(sum(r["n_points"] for r in summary_rows)),
        "total_valid": int(sum(r["n_valid"] for r in summary_rows)),
        "median_dist_to_label_m": float(
            np.median([r["median_dist_to_label_m"] for r in summary_rows])
        ),
        "mean_confidence": float(np.mean([r["mean_confidence"] for r in summary_rows])),
        "agreement_with_naive_median": float(np.median(agreement)),
        "low_fidelity_trips": [r["trip_id"] for r in low],
        "low_fidelity_note": (
            "these trips keep their labels but are flagged in the per-trip table; no trip is "
            "dropped, so evaluation still covers them"
        ),
        "runtime_seconds": total_seconds,
        "sweep": sweep_summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print(f"labelled {len(summary_rows)} trips, {manifest['total_valid']:,} valid observations")
    print(f"monotone for every trip: {monotone_all}")
    print(f"median distance to label: {manifest['median_dist_to_label_m']:.2f} m")
    print(f"mean confidence: {manifest['mean_confidence']:.3f}")
    print(f"agreement with naive nearest arc (median): {manifest['agreement_with_naive_median']:.3f}")
    if low:
        print(f"low fidelity trips flagged: {[r['trip_id'] for r in low]}")
    print(f"runtime: {total_seconds:.1f} s")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
