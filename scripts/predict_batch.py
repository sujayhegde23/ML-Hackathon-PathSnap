"""Run the final pipeline on every trip of the given partitions and save the outputs.

Uses only pipeline.predict_trip (prediction-only), then scores each result separately
with eval.route_metrics.score_prediction (evaluation-only, reads .route).

Outputs: results/predictions/<partition>_routes.jsonl   predicted route + per-fix arcs
         results/predictions/summary.csv               latency and accuracy per trip

Usage: python scripts/predict_batch.py [--partitions student_test train]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import results_dir  # noqa: E402
from pathsnap.eval.route_metrics import score_prediction  # noqa: E402
from pathsnap.io.loader import discover_trips  # noqa: E402
from pathsnap.pipeline import load_artifacts, predict_trip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partitions", nargs="+", default=["student_test"])
    a = ap.parse_args()
    out = results_dir() / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    arts = load_artifacts()
    rows = []
    for part in a.partitions:
        with open(out / f"{part}_routes.jsonl", "w") as f:
            for ref in discover_trips(partitions=(part,)):
                r = predict_trip(ref.trip_dir, arts, part)
                f.write(json.dumps({"trip_id": r.trip_id, "model_version": r.model_version,
                                    "predicted_arc_ids": r.predicted_arc_ids,
                                    "observation_arc_ids": r.observation_arc_ids}) + "\n")
                s = score_prediction(ref.trip_dir, r)
                rows.append({"partition": part, "trip_id": r.trip_id, "n_fixes": len(r.observation_arc_ids),
                             "latency_ms": r.latency_ms, "decode_ms": r.diagnostics.get("decode_ms"),
                             "n_breaks": r.diagnostics["n_breaks"], **{k: v for k, v in s.items() if k != "trip_id"}})
                print(part, r.trip_id, f"{r.latency_ms:.0f} ms", round(s["observation_top1_accuracy"], 4), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.groupby("partition")[["latency_ms", "observation_top1_accuracy", "route_continuity",
                                   "arc_set_f1", "normalized_edit_distance"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
