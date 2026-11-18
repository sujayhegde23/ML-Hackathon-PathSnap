"""Validate every supplied trip and write a machine-readable summary.

Usage
-----
    python scripts/validate_dataset.py [--mode training|prediction]

Outputs
-------
``results/validation/trips.csv``     one row per trip
``results/validation/summary.json``  per-partition aggregates and notable findings

The run never aborts on a single bad trip: parse failures are recorded as an
``E_LOAD`` issue and reported alongside everything else.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import data_root, results_dir  # noqa: E402
from pathsnap.io.loader import discover_trips, safe_load_trip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("training", "prediction"),
        default="training",
        help="training mode reads .route; prediction mode must not",
    )
    parser.add_argument("--root", default=None, help="dataset root (defaults to config)")
    args = parser.parse_args()

    root = Path(args.root) if args.root else data_root()
    refs = discover_trips(root)
    if not refs:
        print(f"no trips found under {root}", file=sys.stderr)
        return 1

    validations = [safe_load_trip(ref.trip_dir, ref.partition, mode=args.mode)[1] for ref in refs]

    out_dir = results_dir() / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- per-trip table, with a stable column order across all rows ---
    rows = [v.to_row() for v in validations]
    base_columns = [
        "trip_id",
        "partition",
        "n_points",
        "n_nodes",
        "n_arcs",
        "n_route",
        "loaded",
        "ok",
        "n_errors",
        "n_warnings",
        "codes",
    ]
    counted = sorted({k for row in rows for k in row if k.startswith("count_")})
    columns = base_columns + counted

    with (out_dir / "trips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # --- aggregates ---
    by_partition: dict[str, dict[str, object]] = {}
    for partition in sorted({v.partition for v in validations}):
        subset = [v for v in validations if v.partition == partition]
        by_partition[partition] = {
            "trips": len(subset),
            "loaded": sum(v.loaded for v in subset),
            "ok": sum(v.ok for v in subset),
            "gps_points": sum(v.n_points for v in subset),
            "nodes": sum(v.n_nodes for v in subset),
            "arcs": sum(v.n_arcs for v in subset),
            "route_arcs": sum(v.n_route for v in subset),
        }

    issue_counts = Counter(i.code for v in validations for i in v.issues)
    affected = defaultdict(list)
    for v in validations:
        for issue in v.issues:
            affected[issue.code].append(f"{v.partition}/{v.trip_id}")

    summary = {
        "data_root": str(root),
        "mode": args.mode,
        "total_trips": len(validations),
        "trips_ok": sum(v.ok for v in validations),
        "trips_with_errors": [f"{v.partition}/{v.trip_id}" for v in validations if not v.ok],
        "by_partition": by_partition,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": {
            code: {
                "trips_affected": len(trips),
                "examples": trips[:8],
            }
            for code, trips in sorted(affected.items())
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # --- console report ---
    print(f"dataset root : {root}")
    print(f"mode         : {args.mode}")
    print(f"trips        : {len(validations)}")
    print()
    print(f"{'partition':<14}{'trips':>7}{'loaded':>8}{'ok':>5}{'points':>9}{'nodes':>10}{'arcs':>10}")
    for partition, stats in by_partition.items():
        print(
            f"{partition:<14}{stats['trips']:>7}{stats['loaded']:>8}{stats['ok']:>5}"
            f"{stats['gps_points']:>9}{stats['nodes']:>10}{stats['arcs']:>10}"
        )
    print()

    if issue_counts:
        print(f"{'code':<24}{'trips':>7}{'total':>9}")
        for code, count in sorted(issue_counts.items()):
            total = sum(i.count for v in validations for i in v.issues if i.code == code)
            print(f"{code:<24}{len(affected[code]):>7}{total:>9}")
    else:
        print("no issues recorded")

    errors = [v for v in validations if not v.ok]
    if errors:
        print()
        print(f"{len(errors)} trip(s) are unusable:")
        for v in errors:
            detail = "; ".join(i.message for i in v.errors)
            print(f"  {v.partition}/{v.trip_id}: {detail}")

    print()
    print(f"wrote {out_dir / 'trips.csv'}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
