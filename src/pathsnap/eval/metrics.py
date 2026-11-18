"""Candidate- and observation-level metrics (team plan section 6). Evaluation only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Candidate-level: each row is 'is this candidate the true arc?'.
    Macro F1 averages the F1 of the positive and negative classes."""
    yhat = (p >= threshold).astype(int)
    return dict(
        macro_f1=f1_score(y, yhat, average="macro"),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    )


def top1_rows(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, fix)."""
    return df.loc[df.groupby(["trip_id", "obs_idx"], sort=False)[score_col].idxmax()]


def observation_metrics(df: pd.DataFrame, score_col: str = "score") -> dict:
    """Top-1 accuracy over every labelled fix. A fix whose true arc is not in its candidate
    list counts as wrong; top1_when_covered excludes those fixes."""
    n = df.groupby(["trip_id", "obs_idx"]).ngroups
    correct = int(top1_rows(df, score_col)["y"].sum())
    covered = int(df.groupby(["trip_id", "obs_idx"])["y"].max().sum())
    return dict(top1_acc=correct / n, top1_when_covered=correct / max(covered, 1),
                candidate_miss_rate=1 - covered / n)


def route_continuity(arc_seq: np.ndarray, from_node: np.ndarray, to_node: np.ndarray) -> float:
    """Share of consecutive (distinct) predicted arcs that connect: end of one == start of next."""
    s = arc_seq[np.concatenate([[True], arc_seq[1:] != arc_seq[:-1]])]
    if len(s) < 2:
        return 1.0
    return float(np.mean(to_node[s[:-1]] == from_node[s[1:]]))
