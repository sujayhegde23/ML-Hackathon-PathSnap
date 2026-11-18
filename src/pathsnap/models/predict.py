"""Contract F: the only thing Member 3's pipeline needs from the models.

    arts = load_model_artifacts()                       # once
    scores = predict_candidate_scores(feature_batch, arts)

scores[r] is the probability that candidate_arc_ids[r] is the right arc for fix
observation_indices[r]. Higher = more likely. Row order is never changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from pathsnap.config import artifacts_dir


def load_model_artifacts(model_dir: str | Path | None = None) -> dict:
    d = Path(model_dir) if model_dir else artifacts_dir() / "models" / "selected"
    spec = json.loads((d / "feature_spec.json").read_text())
    return {"model": joblib.load(d / "model.joblib"), "feature_names": spec["feature_names"],
            "selection": json.loads((d / "selection.json").read_text())}


def predict_candidate_scores(feature_batch, model_artifacts: dict) -> np.ndarray:
    names = model_artifacts["feature_names"]
    if list(feature_batch.feature_names) != list(names):
        raise ValueError(f"feature order mismatch:\n got  {feature_batch.feature_names}\n want {names}")
    if len(feature_batch.X) == 0:
        return np.zeros(0)
    p = model_artifacts["model"].predict_proba(feature_batch.X[names])[:, 1]
    return np.nan_to_num(p.astype(np.float64), nan=0.0)
