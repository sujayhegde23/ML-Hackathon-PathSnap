"""Experiment: how large should the candidate search radius be?

For fixed radii (no widening) and for the adaptive policy in configs/feature_config.yaml,
measured on up to 300 labelled fixes per train trip:
  miss rate            share of fixes whose reference arc (same endpoints) is not a candidate
  candidates per fix   more = slower and a harder choice for the classifier
  ms per fix           search time
Node-only vs arc search was already measured by Member 1 (scripts/candidate_probe.py).

Outputs: results/candidates/radius_sweep.csv, results/figures/candidates/radius_sweep.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.candidates.generator import generate_candidates, load_feature_config  # noqa: E402
from pathsnap.config import processed_dir, results_dir, set_global_seed  # noqa: E402
from pathsnap.geo.projection import project_trip  # noqa: E402
from pathsnap.graph.network import build_road_graph  # noqa: E402
from pathsnap.io.loader import discover_trips, load_trip_for_prediction  # noqa: E402

RADII = [10, 20, 30, 50, 75, 100, 150]
PER_TRIP = 300


def main() -> int:
    set_global_seed()
    rng = np.random.default_rng(42)
    cfg = load_feature_config()
    acc = {r: dict(n=0, hit=0, cands=0, sec=0.0) for r in [*RADII, "adaptive"]}
    for ref in discover_trips(partitions=("train",)):
        trip = load_trip_for_prediction(ref.trip_dir, ref.partition)
        lab = np.load(processed_dir() / "labels" / f"{trip.trip_id}.npz")
        ok = np.flatnonzero(lab["valid"] & (lab["arc_ids"] >= 0))
        if ok.size == 0:
            continue
        idx = np.sort(rng.choice(ok, size=min(PER_TRIP, ok.size), replace=False))
        pt = project_trip(trip)
        g = build_road_graph(trip, pt)
        truth = lab["arc_ids"][idx]
        tkey = set(zip(g.from_node[truth], g.to_node[truth]))  # noqa: F841 (documentational)

        def is_hit(arcs, t):
            return bool(np.any((g.from_node[arcs] == g.from_node[t]) & (g.to_node[arcs] == g.to_node[t])))

        for r in RADII:
            t0 = time.perf_counter()
            found = [g.arcs_within(pt.track_xy[i], r) for i in idx]
            acc[r]["sec"] += time.perf_counter() - t0
            acc[r]["n"] += len(idx)
            acc[r]["hit"] += sum(is_hit(f, t) for f, t in zip(found, truth))
            acc[r]["cands"] += sum(f.size for f in found)
        sub = type(pt)(pt.trip_id, pt.partition, pt.projection, pt.track_xy[idx], pt.nodes_xy, pt.source)
        t0 = time.perf_counter()
        cs = generate_candidates(sub, g, cfg)
        acc["adaptive"]["sec"] += time.perf_counter() - t0
        acc["adaptive"]["n"] += len(idx)
        acc["adaptive"]["hit"] += sum(is_hit(c.arc_ids, t) for c, t in zip(cs, truth))
        acc["adaptive"]["cands"] += sum(c.arc_ids.size for c in cs)
    rows = [dict(radius_m=r, fixes=v["n"], miss_rate=1 - v["hit"] / v["n"], candidates_per_fix=v["cands"] / v["n"],
                 ms_per_fix=1000 * v["sec"] / v["n"]) for r, v in acc.items()]
    df = pd.DataFrame(rows)
    out = results_dir() / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "radius_sweep.csv", index=False)
    print(df.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = df[df.radius_m != "adaptive"]
    ad = df[df.radius_m == "adaptive"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fixed.radius_m.astype(float), fixed.miss_rate * 100, "o-", label="fixed radius")
    ax[0].axhline(ad.miss_rate * 100, color="C1", ls="--", label="adaptive 50→400 m")
    ax[0].set(xlabel="search radius (m)", ylabel="miss rate (%)", title="Reference arc missing from candidates")
    ax[0].legend()
    ax[1].plot(fixed.radius_m.astype(float), fixed.candidates_per_fix, "o-")
    ax[1].axhline(ad.candidates_per_fix, color="C1", ls="--")
    ax[1].set(xlabel="search radius (m)", ylabel="candidates per fix", title="Candidate count")
    fig.tight_layout()
    (results_dir() / "figures" / "candidates").mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir() / "figures" / "candidates" / "radius_sweep.png", dpi=140)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
