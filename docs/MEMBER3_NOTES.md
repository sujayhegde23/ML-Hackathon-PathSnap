# Member 3: Bayesian tuning, HMM/Viterbi, route evaluation, pipeline

Every number below comes from a file a script wrote. The file is named next to it.

## How to reproduce

Run Member 2's steps (`make member2`) first, then:

```bash
make hpo             # Optuna tuning + untuned vs tuned              (~8 min)
make routes          # decoder tuning + route comparison             (~3 min)
make check-pipeline  # contract checks, must print "0 failure(s)"
make predict-batch   # final pipeline on all 90 trips + scores
```

`make member3` runs all four steps in order.

## 1. Bayesian hyperparameter optimisation — `src/pathsnap/hpo/optuna_search.py`, `scripts/hpo.py`

- **Model tuned:** LightGBM, Member 2's selected model.
- **Method:** Optuna's TPE sampler (seed 42). TPE is Bayesian: it learns from earlier trials
  which settings score well and tries more settings like them. A median pruner stops a trial
  early when its first folds are already worse than the median trial.
- **Objective:** mean candidate-level Macro F1 over 3-fold GroupKFold by trip, using every 8th
  GPS point of the 80 train trips. student_test is never used here.
- **Trial 0 is the untuned configuration**, so the search always contains the baseline it has to
  beat.
- **Search space** (`results/hpo/search_space.json`):

  | Hyperparameter | Range |
  |---|---|
  | num_leaves | 15–255 (log scale) |
  | learning_rate | 0.01–0.2 (log scale) |
  | n_estimators | 100–800 |
  | min_child_samples | 10–300 (log scale) |
  | subsample | 0.5–1.0 |
  | colsample_bytree | 0.5–1.0 |
  | reg_lambda | 0.001–10 (log scale) |

- **Trials:** 29 in total; 16 ran to completion and 13 were pruned. The search took 407 s on
  2 CPU cores (`results/hpo/best_params.json`). The study is stored in
  `experiments/optuna_lightgbm.db`, so it can be resumed.
- **Best settings:** num_leaves 54, learning_rate 0.0198, n_estimators 400,
  min_child_samples 27, subsample 0.54, colsample_bytree 0.68, reg_lambda 1.41.
- **Most important hyperparameters** (fANOVA, `results/hpo/param_importance.csv`):
  n_estimators 0.34, colsample_bytree 0.28, learning_rate 0.12.

**Untuned vs tuned** (`results/hpo/untuned_vs_tuned.csv`). Both were run under Member 2's exact
protocol: every 4th GPS point, 5-fold grouped CV, same seed, same 10 student-test trips.

| Metric | Untuned | Tuned |
|---|---|---|
| CV Macro F1 | 0.9141 | **0.9145** |
| CV Log Loss | 0.0885 | **0.0839** |
| Student-test Top-1 | **0.8957** | 0.8919 |
| Student-test Macro F1 | 0.9363 | **0.9372** |
| Student-test Log Loss | 0.0487 | 0.0488 |
| Inference (ms / 1k candidates) | 7.8 | 8.5 |

**Honest reading:** tuning barely helped. The CV gain (+0.0004) is about a hundredth of the
fold-to-fold spread (std 0.037). The untuned settings were already close to the best, and
there is little left to gain from LightGBM settings on these features. The tuned model is used
as the final model because the rule fixed in advance says so (higher CV Macro F1, never chosen
on student_test). It also has the clearly better CV log loss, which matters for the HMM.

## 2. HMM + Viterbi — `src/pathsnap/hmm/viterbi.py`

1. **Hidden states:** at GPS point i, the candidate arcs of point i, taken from Member 2's
   `CandidateSet`. The arc id stays attached to each state throughout.
2. **Emission:**
   `log p(arc)` (the classifier probability, floored at 1e-6) − `w · 0.5 · (distance / 5 m)²`.
   The second term is an optional extra distance penalty; `w` was tuned to 1.
3. **Transition:** `−|route distance − straight-line distance| / beta`, following Newson &
   Krumm (2009).
   - The straight-line distance is between the two consecutive GPS points.
   - On the same arc, the route distance is how far the match moved along it.
   - Otherwise it is the rest of the first arc, plus the shortest driving path to the next arc,
     plus the part of the next arc up to the match.
   - Shortest paths come from a bounded Dijkstra, capped at `2 × straight-line + 100 m` and
     cached per start node. A path that can't be found within the cap counts as impossible
     (log probability −∞).
   - Switching to the opposite-direction arc of the same road costs a U-turn's worth of
     distance, so it is strongly penalised.
4. **Viterbi in log space:** `V_i(b) = emission_i(b) + max_a [V_{i−1}(a) + transition(a, b)]`,
   with back-pointers recorded at each step and a backtrack from the best final state. Adding
   logs instead of multiplying probabilities avoids underflow.
5. **Gaps and dead ends:** the chain restarts as a new, independent segment when:
   - the time gap is over 60 s,
   - a GPS point has no candidates (only possible on an empty network), or
   - every transition from the previous point is impossible.

   Restarts are returned in `break_indices`.
6. **Route output:** the per-point arcs are collapsed, and arcs that don't touch are joined by
   the shortest-path arcs between them. This gives a drivable route inside each segment.

**Decoder tuning** (`results/routes/decoder_tuning.csv`): we tried transition beta in
{1, 2, 5, 10, 20, 50} m, with the extra distance term on or off. This was on 24 random train
trips (low-fidelity-label trips excluded), scored out-of-fold: the model scoring a trip never
saw it. Top-1 falls steadily as beta grows, from 0.941 at 1 m to 0.921 at 50 m. The chosen
setting is **beta = 1 m, with the distance term on** (train Top-1 0.9406 vs 0.9056 for the
classifier alone). It is saved to `artifacts/models/final/decoder_config.json`. Note: 1 m is
the edge of the grid, so an even smaller beta might be slightly better.

## 3. Route-level results on student_test — `results/routes/student_test_summary.csv`

Uses the final tuned model trained on all 80 train trips, scored on the 10 student-test trips.

| Method | Top-1 | Continuity | Arc-set F1 | Norm. edit distance | Decode time per trip |
|---|---|---|---|---|---|
| Nearest arc | 0.541 | 0.421 | 0.579 | 0.667 | – |
| Classifier only | 0.892 | 0.826 | 0.898 | 0.351 | – |
| **Classifier + HMM** | **0.939** | **0.915** | **0.936** | **0.144** | 0.27 s |
| Nearest arc + HMM (distance only) | 0.916 | 0.914 | 0.932 | 0.164 | 0.25 s |

Notes on the table:

- Continuity, arc-set F1 and edit distance here are computed on the per-point arcs, so all
  four methods are measured identically.
- After the route is joined, the HMM route's continuity is **1.00**, its arc-set F1 is 0.978
  and its normalised edit distance is 0.075.
- The ML classifier is better than a distance-only HMM both alone and inside the HMM
  (0.939 vs 0.916).

**Where it helps and where it hurts** (`results/routes/hmm_effect_per_trip.csv`): across the 10
trips the HMM fixed 1,259 GPS points and broke 355. It improved every trip. Trip 00000077 shows
both effects most strongly: Top-1 went from 0.755 to 0.862, but 202 points also got worse.

- **Helps:**
  - when the vehicle is stopped or slow, so heading is useless and the classifier flips between
    a road and its opposite-direction twin;
  - on parallel roads, where one bad GPS point drags the classifier sideways.
- **Hurts:**
  - where the classifier is confidently wrong for several points in a row, the HMM produces a
    smooth route along the wrong road, because consistency makes the error stick;
  - near junctions, where the HMM sometimes moves to the next arc a point early or late.
- Pictures: `results/figures/routes/example_hmm_improved_*.png` and `example_hmm_hurt_*.png`.
  Interactive versions are in `maps/hmm_*.html`; their reference layer is for offline analysis
  only.

## 4. Final pipeline — `src/pathsnap/pipeline.py` (for Member 4)

```python
from pathsnap.pipeline import load_artifacts, predict_trip
arts = load_artifacts()                   # once at server start: model + feature spec + decoder
res = predict_trip(trip_dir, arts)        # reads .track/.nodes/.arcs only, never .route
res.to_dict()  # trip_id, predicted_arc_ids, observation_arc_ids, polyline_lonlat,
               # route_segments_lonlat, gps_lonlat, model_version, latency_ms, diagnostics

from pathsnap.eval.route_metrics import score_prediction
score_prediction(trip_dir, res)           # SEPARATE, server-side; reads .route; numbers only
```

- Use `route_segments_lonlat` to draw the route on the map. It is split at gaps, so the map
  doesn't draw a straight line across a break.
- Errors: a missing folder raises `FileNotFoundError`; malformed files raise `TripFormatError`.
- Artifacts live in `artifacts/models/final/`: `model.joblib`, `feature_spec.json`,
  `selection.json`, `decoder_config.json`, and `manifest.json` (file hashes, library versions,
  git commit). If `final/` is missing, the pipeline falls back to `artifacts/models/selected/`.
- **Speed:** 0.1–1.6 s per student-test trip end to end, 0.65 s on average
  (`results/predictions/summary.csv`).
- **Batch predictions** for all 90 trips are in `results/predictions/*_routes.jsonl`.
  - The student-test **per-trip average** Top-1 is 0.946. That differs from the 0.939 in
    section 3 only because 0.939 pools all GPS points before averaging.
  - Train-trip scores (0.932) are **not** a fair estimate, because the final model was trained
    on those trips. Use the CV and student-test numbers in the report.
- **Checks** (`make check-pipeline`, all pass):
  - nothing on the prediction side imports labels;
  - deleting `.route` changes nothing;
  - reloading the artifacts reproduces the prediction;
  - one arc per GPS point, all ids valid, valid lon/lat;
  - misaligned scores are rejected;
  - a 60 s+ gap starts a new chain.

## 5. Viva prep (short answers)

1. *What are the hidden states?* The candidate road arcs of each GPS point.
2. *Emission?* The log of the classifier's probability that this arc is right, plus a small
   penalty for distance.
3. *Transition?* How well the driving distance between two candidates matches the
   straight-line distance between the two GPS points. A big mismatch means an unlikely jump.
4. *Why Viterbi?* It finds the single best sequence for the whole trip exactly, in time linear
   in trip length, instead of trying every combination.
5. *Why log space?* Multiplying thousands of probabilities underflows to zero; adding logs
   doesn't.
6. *Gaps and disconnected roads?* Over 60 s, or no possible path within the cap, means the
   chain restarts as a new segment, recorded in `break_indices`.
7. *Did the HMM help?* Yes. Top-1 went from 0.892 to 0.939 and continuity from 0.826 to 0.915
   on student_test, and every trip improved. It still broke 355 points, where the classifier
   was confidently wrong several points in a row.
8. *Did tuning help?* Barely: CV Macro F1 went from 0.9141 to 0.9145, well inside the noise.
   We report that honestly rather than overselling it.
9. *How was beta chosen?* A grid search on 24 train trips with out-of-fold scores. Never on
   student_test, and never on evaluator data.
10. *Does prediction ever see the answer?* No. `predict_trip` uses the prediction loader, which
    refuses `.route`. A check deletes `.route` and gets an identical prediction.
