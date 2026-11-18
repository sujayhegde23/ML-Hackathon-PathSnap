"""Route-level metrics and the SEPARATE scoring function (evaluation only).

`score_prediction` is the only place in the prediction stack that reads `.route`. It is
called after prediction, server-side, and returns numbers only, never reference arcs.

Arc equality: two arcs count as the same road when they have the same (from_node,
to_node). Some trips store a road twice with identical endpoints (DATA_NOTES §6), so
matching on endpoints avoids calling an identical road wrong. The reverse direction is
a different arc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathsnap.eval.metrics import route_continuity


def _keys(arcs, from_node, to_node) -> list[tuple[int, int]]:
    arcs = np.asarray([a for a in arcs if a is not None and a >= 0], dtype=np.int64)
    return list(zip(from_node[arcs].tolist(), to_node[arcs].tolist()))


def collapse(seq):
    out = []
    for s in seq:
        if s is not None and (not out or out[-1] != s):
            out.append(s)
    return out


def observation_top1(pred_obs, label_arcs, valid, from_node, to_node) -> float:
    """Share of labelled fixes whose predicted arc is the reference arc (endpoint match).
    Fixes with no prediction count as wrong."""
    idx = np.flatnonzero(np.asarray(valid) & (np.asarray(label_arcs) >= 0))
    if idx.size == 0:
        return float("nan")
    pred = np.array([-1 if pred_obs[i] is None else pred_obs[i] for i in idx], dtype=np.int64)
    lab = np.asarray(label_arcs)[idx]
    ok = (pred >= 0) & (from_node[np.maximum(pred, 0)] == from_node[lab]) & (to_node[np.maximum(pred, 0)] == to_node[lab])
    return float(ok.mean())


def observation_continuity(pred_obs, from_node, to_node) -> float:
    """Member 2's route continuity on the per-fix sequence (same definition for all methods)."""
    seq = np.array(collapse(pred_obs), dtype=np.int64)
    return route_continuity(seq, from_node, to_node) if seq.size else float("nan")


def arc_set_f1(pred_arcs, ref_arcs, from_node, to_node) -> float:
    p, r = set(_keys(pred_arcs, from_node, to_node)), set(_keys(ref_arcs, from_node, to_node))
    if not p or not r:
        return 0.0
    tp = len(p & r)
    return 2 * tp / (len(p) + len(r))


def normalized_edit_distance(pred_arcs, ref_arcs, from_node, to_node) -> float:
    """Levenshtein distance between the two arc sequences (repeats collapsed), divided by
    the longer length. 0 = identical order, 1 = nothing in common."""
    a = collapse(_keys(pred_arcs, from_node, to_node))
    b = collapse(_keys(ref_arcs, from_node, to_node))
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ids = {k: n for n, k in enumerate(set(a) | set(b))}
    A = np.array([ids[k] for k in a]); B = np.array([ids[k] for k in b])
    m = len(B)
    j = np.arange(m + 1)
    prev = j.astype(np.int64)
    for i in range(1, len(A) + 1):
        sub = prev[:-1] + (A[i - 1] != B)          # substitute / match
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, sub)      # delete
        cur = np.minimum.accumulate(cur - j) + j     # insert (running minimum trick)
        prev = cur
    return float(prev[-1] / max(len(A), m))


def route_metrics(pred_obs, pred_route, trip, labels) -> dict:
    """All route-level metrics for one trip. `labels` is Member 1's LabelResult or its npz."""
    fn, tn = trip.arcs[:, 0], trip.arcs[:, 1]
    lab_arcs = labels["arc_ids"] if isinstance(labels, dict) else labels.arc_ids
    lab_valid = labels["valid"] if isinstance(labels, dict) else labels.valid
    return dict(
        top1_acc=observation_top1(pred_obs, lab_arcs, lab_valid, fn, tn),
        continuity=observation_continuity(pred_obs, fn, tn),
        route_continuity=route_continuity(np.array(collapse(pred_route), dtype=np.int64), fn, tn)
        if len(collapse(pred_route)) else float("nan"),
        arc_set_f1=arc_set_f1(pred_route, trip.route, fn, tn),
        norm_edit_distance=normalized_edit_distance(pred_route, trip.route, fn, tn),
        n_labelled=int(np.sum(np.asarray(lab_valid) & (np.asarray(lab_arcs) >= 0))),
    )


def score_prediction(trip_dir, result) -> dict:
    """Server-side accuracy of a PredictionResult (or its dict) against the trip's own
    reference route. Reads .route; returns only numbers. Never call it in prediction."""
    from pathsnap.io.loader import load_trip_for_training
    from pathsnap.labels.truth import derive_observation_labels

    trip = load_trip_for_training(Path(trip_dir), Path(trip_dir).parent.name)
    labels = derive_observation_labels(trip)
    r = result if isinstance(result, dict) else result.to_dict()
    m = route_metrics(r["observation_arc_ids"], r["predicted_arc_ids"], trip, labels)
    return {
        "trip_id": trip.trip_id,
        "observation_top1_accuracy": m["top1_acc"],
        "labelled_observations": m["n_labelled"],
        "route_continuity": m["continuity"],
        "arc_set_f1": m["arc_set_f1"],
        "normalized_edit_distance": m["norm_edit_distance"],
    }
