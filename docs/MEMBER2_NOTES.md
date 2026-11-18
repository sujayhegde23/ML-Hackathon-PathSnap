# Member 2: candidates, features, model comparison

Every number below was written by a script. The file it comes from is named next to it.

## How to reproduce

```bash
export PATHSNAP_DATA_ROOT="/path/to/Student folder"
make labels          # Member 1's labels (needed for training only)
make features        # candidates + features for 90 trips          (~20 s)
make radius-sweep    # radius experiment                            (~20 s)
make leakage         # leakage checks, must print "0 failure(s)"
make train           # 8 models + baseline, selection, ablation    (~10-15 min on 2 cores)
```

`make member2` runs all five steps in order.

## 1. The idea in one paragraph

GPS points are noisy, with a median of 4 m and a 95th percentile of about 13 m off the road
(DATA_NOTES §5). That is about the gap between parallel roads. So for each GPS point we first
list the roads it could be on. These are the **candidates**. We then describe each
(point, candidate) pair with numbers, the **features**. Finally a classifier gives each pair a
probability of being the true road. The HMM (Member 3) then picks one candidate per point so
that the chosen roads join up into a drivable route.

## 2. Candidate generation — `src/pathsnap/candidates/generator.py`

- We use Member 1's `RoadGraph.arcs_within`, which searches by **road segment**, not just by
  road end points. Searching by end points only misses long roads that pass close to the point
  while both their ends are far away (Member 1's `results/eda/candidate_probe.json`).
- **Adaptive radius:** we start at 50 m. If fewer than 3 candidates are found, we widen to 100,
  then 200, then 400 m. If there is still nothing, we take the 5 nearest roads anywhere. We keep
  the 24 nearest. These settings live in `configs/feature_config.yaml`.
- **Radius experiment** (`results/candidates/radius_sweep.csv`, 24,000 labelled train points):

  | Radius | Miss rate | Candidates per point |
  |---|---|---|
  | 10 m | 19.7% | 2.4 |
  | 20 m | 5.0% | 4.6 |
  | 30 m | 3.0% | 6.8 |
  | 50 m | 2.3% | 12.1 |
  | 100 m | 1.6% | 30.9 |
  | 150 m | 0.0% | 58.5 |
  | Adaptive (ours) | 1.1% | 11.8 |

  The adaptive policy halves the 50 m miss rate while keeping the same candidate count, because
  it widens only where needed (for example the offset trip 00000054). A fixed 150 m radius finds
  everything, but with 5× more candidates. Note that labels are only defined up to 150 m from
  the route, which is why 150 m reaches 0%.
- **Full run** (`results/candidates/summary.csv`): the train miss rate is 2.39% and the
  student_test miss rate is 0.08%. No point needed the fallback. Train misses come almost
  entirely from the known bad trips 00000054 and 00000033 (DATA_NOTES §6).
- Building candidates and features for all 90 trips takes about 17 s.

## 3. Features — `src/pathsnap/features/extractor.py`

There are 21 features in 6 groups. All of them are relative to the road: no longitude/latitude,
no ids. The reason for each feature is in the file's docstring. In short:

- **geometry:** distance to the road, position along it, distance to the nearest junction, road
  length.
- **ranking:** this candidate compared with the others (rank, extra distance over the best one,
  how many are close to the best).
- **heading:** cos(direction of travel − road direction), plus a "valid" flag for when the car is
  stopped.
- **context:** how close the 3 points before and the 3 after are to the same road.
- **network:** junction sizes, whether the road is two-way, how crowded the area is.
- **temporal:** speed (from real elapsed time), time since the previous point, long-gap flag.

**Why heading matters most:** about 92% of roads are stored twice, once for each direction
(A→B and B→A), at *exactly the same distance* from the point. Only direction of travel can tell
them apart. The ablation shows this: removing heading drops CV top-1 from 0.798 to 0.539
(`results/models/feature_ablation.csv`). Removing any other group costs at most 0.01.

**Label details** (`scripts/build_features.py`): a candidate counts as positive if it has the
same (start node, end node) as Member 1's reference arc. This matters because 60 trips store some
roads twice with identical end points, and picking either copy should not be marked wrong. The
reverse direction does not count as positive. Points Member 1 marked `valid = False` are left
out.

**Permutation importance** (`results/models/permutation_importance.csv`, LightGBM, measured on
fold-0 validation trips, never student_test). This is the drop in Macro F1 when one feature is
shuffled:

- `cos_heading`: 0.195
- `ctx_gap_to_best`: 0.116
- `proj_t`: 0.095
- `has_reverse_twin`: 0.046
- `dist_gap_to_best`: 0.028
- everything else: under 0.02

Raw distance looks unimportant here only because the ranking and context features already carry
the same information. Dropping the whole geometry group still costs 0.013 Macro F1.

## 4. Leakage checks — `scripts/check_leakage.py` (`make leakage`)

All checks pass:

- The prediction-path code does not import labels or read `.route`.
- Features are identical whether or not the `.route` file exists.
- No feature name refers to position or ids, and no feature tracks raw lon/lat (max |corr| 0.71).
- Train and student_test share no trip, and the CV folds share no trip.

This is a script rather than a file in `tests/` because `tests/` is git-ignored in this repo.

## 5. Model comparison — `results/models/model_comparison.csv`

Protocol: 80 train trips, every 4th GPS point (points 1 s apart are near-duplicates), 5-fold
GroupKFold by trip, so validation is always on unseen trips. Each model is then refit on all of
train and scored on every point of the 10 student_test trips. The selection rule was fixed
before training: **highest mean CV Macro F1, with lower CV Log Loss winning a tie.**

| Model | Family | CV Macro F1 | Student-test Top-1 | Route continuity | Student-test Macro F1 | Log loss |
|---|---|---|---|---|---|---|
| Nearest-arc baseline | rule | 0.750 | 0.541 | 0.421 | 0.776 | – |
| Logistic Regression | linear | 0.874 | 0.854 | 0.732 | 0.907 | 0.092 |
| Gaussian NB | probabilistic | 0.660 | 0.829 | 0.690 | 0.661 | 1.076 |
| KNN (k=25) | instance-based | 0.875 | 0.830 | 0.692 | 0.892 | 0.102 |
| Decision Tree | tree | 0.899 | 0.881 | 0.796 | 0.932 | 0.059 |
| Random Forest | bagging | 0.908 | 0.885 | 0.804 | 0.931 | 0.054 |
| HistGradientBoosting | boosting | 0.902 | 0.881 | 0.751 | 0.931 | 0.067 |
| **LightGBM (selected)** | **boosting** | **0.914** | **0.896** | **0.828** | **0.936** | **0.049** |
| MLP (64-32) | neural net | 0.908 | 0.881 | 0.799 | 0.931 | 0.057 |

Inference time and the remaining columns (CV std, train F1, precision/recall, fit time) are in
the CSV. Per-fold scores are in `results/models/cv_folds.csv`, and one JSON log per model is in
`experiments/`.

How to read the table:

- **The nearest-road rule gets only 54% top-1.** Two-way roads tie on distance, so it guesses.
  Every learned model beats it by about 30 points.
- **Gaussian NB has a decent top-1 but a poor Macro F1 and log loss.** It assumes features are
  independent, but distance, rank and context are strongly linked. Its probabilities come out
  overconfident, and at a 0.5 threshold it flags far too many positives (precision 0.29).
- **Overfitting check** (`results/figures/models/train_vs_val.png`): tree models score
  0.94–0.97 on training folds against about 0.90–0.91 on validation, a small gap.
  LightGBM's per-iteration log-loss curve is in `lightgbm_logloss_curve.png`.
- **Fold 0 is weaker for every model** (top-1 ≈ 0.6). Its validation trips include the
  low-quality-label trips, so the difficulty is in the data, not the model.
- **Why LightGBM suits the HMM:** it has the best ranking (top-1) and the best log loss, meaning
  its probabilities are the most trustworthy. The HMM multiplies these probabilities along the
  route, so honest probabilities matter. It has not been separately calibrated.

## 6. Handoff to Member 3 — `src/pathsnap/models/predict.py`

```python
arts = load_model_artifacts()                     # artifacts/models/selected/
fb = build_candidate_features(pt, g, generate_candidates(pt, g))
scores = predict_candidate_scores(fb, arts)       # (M,), same row order as fb
```

- `scores[r]` belongs to arc `fb.candidate_arc_ids[r]` at GPS point `fb.observation_indices[r]`.
- The function raises an error if the feature order differs from
  `artifacts/models/selected/feature_spec.json`.
- Verified: on a full trip it reproduces the saved scores exactly, and the whole trip takes
  0.7 s end to end.
- For HMM tuning use `results/models/candidate_scores_train_oof.parquet`. These are
  out-of-fold scores, so each trip was scored by a model that never saw it.
  `candidate_scores_student_test.parquet` holds the student_test scores.
- For Bayesian tuning (Member 3): LightGBM's settings are in
  `src/pathsnap/models/registry.py`. Tune with the same GroupKFold folds and the same Macro F1.

## 7. Known weaknesses

- **Stopped vehicles:** with no movement there is no heading, so the two directions of a road
  tie. The HMM fixes most of these from the points before and after.
- **Offset trips (00000054, 00000033):** the labels are unreliable there, which lowers fold 0's
  scores. They were kept rather than hand-removed.
- **Threshold 0.5** is used for Macro F1 only. The HMM uses the raw probabilities.

## 8. Viva prep (short answers)

1. *Why candidates instead of predicting the road directly?* Road ids are local to each trip, so
   there is no fixed list of classes. We turn the problem into "is this nearby road the right
   one: yes or no?", which works in any city.
2. *Why arc search, not node search?* Long roads pass close to points while their end nodes are
   far away. Member 1's probe found node search misses many true roads at 50 m, while arc search
   misses very few.
3. *Why an adaptive radius?* A fixed 50 m misses 2.3%. Fixed 150 m misses none but gives 58
   candidates per point. Adaptive gets 1.1% with about 12.
4. *Why no lat/lon features?* The evaluator's cities are unseen. Position would let the model
   memorise places instead of learning what a correct match looks like.
5. *What is GroupKFold and why?* It splits by trip, so one trip is never in both train and
   validation. Splitting by row would leak nearly identical neighbouring points into validation
   and inflate the scores.
6. *Why train on every 4th point?* Points 1 s apart are almost copies. Using every 4th is about
   4× faster and loses almost no information.
7. *Which feature matters most?* Heading. Removing it drops top-1 from 0.80 to 0.54, because
   two-way roads tie on distance.
8. *Why LightGBM?* It had the highest CV Macro F1 under the rule we fixed beforehand, and the
   lowest log loss, which is good for the HMM.
9. *Is there overfitting?* The train/validation gap is about 0.03–0.06 Macro F1 for the boosted
   trees, which is small. See `train_vs_val.png`.
10. *Why is Gaussian NB bad on F1 but OK on top-1?* Its independence assumption gives badly
    scaled probabilities: too many land above 0.5. But it still ranks the true road first fairly
    often.
