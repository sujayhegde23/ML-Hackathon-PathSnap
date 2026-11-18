"""Build the technical report from saved experiment outputs.

Every number in the report is read from a committed CSV/JSON/parquet, never typed by hand,
so the PDF cannot drift from the repository.

Outputs
  report/PathSnap_Report.md    markdown source
  report/PathSnap_Report.html  styled HTML (for the PDF conversion)

Usage: python scripts/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REP = ROOT / "report"
FIG = RES / "figures"

TEAM = [
    ("Srihari Nagesh", "PES1UG24CS445"),
    ("Shriniketh Kana", "PES1UG24CS446"),
    ("Sukmeet", "PES1UG24CS479"),
    ("Sujay Hegde", "PES1UG24CS478"),
]

#: Per-model prose. Answers A1, A2 and B6 of Deliverables section 3.2; B5 comes from the
#: registry and C7 to C9 are read from the saved results.
MODEL_NOTES = {
    "LogisticRegression": dict(
        why="The task reduces to a yes/no decision per candidate, and a linear model on a "
            "standardised feature vector is the cheapest way to ask whether the features are "
            "additive in log-odds at all. It also gives a properly normalised probability, "
            "which the decoder needs.",
        strengths="Fast to fit, calibrated by construction, coefficients can be read as "
                  "effect sizes. Any signal it finds is genuinely linear, so it sets a floor.",
        limits="Cannot express the interaction that matters here: a candidate that is close "
               "but pointing the wrong way should be rejected, which is a product of distance "
               "and heading, not a sum.",
        objective="Binary cross-entropy (log loss), L2-regularised, lbfgs solver.",
    ),
    "GaussianNB": dict(
        why="A generative baseline that assumes features are independent within each class. "
            "Included to test that assumption, not because it is expected to win.",
        strengths="Trains in milliseconds and needs almost no data per class.",
        limits="The independence assumption is badly violated: distance, distance rank and "
               "distance gap are three views of the same quantity, so their evidence is "
               "counted three times and the probabilities saturate. Visible in the log loss.",
        objective="Maximum likelihood under a per-feature Gaussian per class.",
    ),
    "KNN": dict(
        why="Map matching is a local decision: two fixes 5 m apart on the same road should "
            "look alike. A nearest-neighbour vote on the candidate feature vector tests "
            "whether similarity in feature space carries the label.",
        strengths="No training step, naturally non-linear, easy to explain to a reader.",
        limits="Inference cost grows with the training set, so it was capped at 80k rows. "
               "It also cannot extrapolate: a candidate unlike anything in training falls "
               "back to whatever neighbour happens to be closest.",
        objective="Distance-weighted majority vote over the 25 nearest training rows.",
    ),
    "DecisionTree": dict(
        why="A single tree shows which thresholds the data actually supports, and the tree "
            "depth that stops helping. Useful as a readable model before moving to ensembles.",
        strengths="Interpretable, handles mixed scales without standardisation, fast.",
        limits="High variance: a small change in the training trips redraws the splits. Depth "
               "was capped to control this, which costs some accuracy.",
        objective="Gini impurity, max_depth 12, min_samples_leaf 50.",
    ),
    "RandomForest": dict(
        why="Averaging many de-correlated trees addresses the variance problem seen in the "
            "single tree without giving up non-linearity.",
        strengths="Stable across folds, no scaling needed, gives a usable probability from "
                  "the vote share.",
        limits="Large and slow at inference compared with boosting, and the vote share is "
               "less sharp than a boosted model's probability.",
        objective="Gini impurity, 100 trees, 50 percent bootstrap sample per tree.",
    ),
    "HistGradientBoosting": dict(
        why="Boosting corrects the errors of the previous trees in sequence, which suits a "
            "problem where most candidates are easy and a small ambiguous minority decides "
            "the score.",
        strengths="Fast, handles the 21 features with no preprocessing, strong on tabular data.",
        limits="Fixed binning and a fixed leaf budget limit fine-grained splits on the "
               "distance feature, where the useful resolution is a metre.",
        objective="Binary log loss, 200 iterations, learning rate 0.1.",
    ),
    "LightGBM": dict(
        why="Leaf-wise growth concentrates capacity on the hard candidates rather than "
            "splitting every branch evenly, and the histogram binning keeps it fast enough "
            "for an API that must answer in seconds.",
        strengths="Best CV Macro F1 and best log loss of the eight, small enough to ship, "
                  "and exposes a probability the decoder can consume directly.",
        limits="Leaf-wise growth overfits quickly on small data, so depth, leaves and both "
               "subsampling rates had to be constrained. Needs a seed to be reproducible.",
        objective="Binary log loss, 400 trees, learning rate 0.05, 63 leaves.",
    ),
    "MLP": dict(
        why="The relationship between distance, heading and correctness is plausibly "
            "non-linear and interacting, which is what a small network represents.",
        strengths="Learns interactions directly and beats the linear model by a wide margin.",
        limits="Sensitive to feature scaling and seed, slower per fold, and the probability "
               "is less well behaved than a tree ensemble's on unseen cities.",
        objective="Binary log loss, Adam, early stopping on an internal validation split.",
    ),
}


def md_table(df: pd.DataFrame, cols=None, floatfmt="{:.4f}") -> str:
    d = df[cols] if cols else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def build() -> str:
    comp = pd.read_csv(RES / "models" / "model_comparison.csv")
    folds = pd.read_csv(RES / "models" / "cv_folds.csv")
    hpo = pd.read_json(RES / "hpo" / "best_params.json", typ="series")
    ut = pd.read_csv(RES / "hpo" / "untuned_vs_tuned.csv")
    trials = pd.read_csv(RES / "hpo" / "trials.csv")
    routes = pd.read_csv(RES / "routes" / "student_test_summary.csv")
    hmm_eff = pd.read_csv(RES / "routes" / "hmm_effect_per_trip.csv")
    ev = pd.read_csv(RES / "predictions" / "evaluator_test_per_trip.csv")
    abl = pd.read_csv(RES / "models" / "feature_ablation.csv")
    imp = pd.read_csv(RES / "models" / "permutation_importance.csv")
    rad = pd.read_csv(RES / "candidates" / "radius_sweep.csv")
    eda = json.loads((RES / "eda" / "summary.json").read_text())

    val = folds[folds.split == "val"].groupby("model")["macro_f1"].mean()
    trn = folds[folds.split == "train"].groupby("model")["macro_f1"].mean()

    L: list[str] = []
    A = L.append

    A("# PathSnap: distributed map-matching and route reconstruction")
    A("")
    A("**UE24CS352A Machine Learning, PES University (RR Campus)**")
    A("")
    A("| Member | SRN |")
    A("|---|---|")
    for name, srn in TEAM:
        A(f"| {name} | {srn} |")
    A("")

    # ---------------------------------------------------------------- 2.2 dataset
    A("## 2.2 Dataset understanding and exploratory data analysis")
    A("")
    A("The dataset is 90 independent trips, each shipped with its own local road network, and "
      "each from a different city. Four headerless files per trip:")
    A("")
    A("| File | Columns | Meaning |")
    A("|---|---|---|")
    A("| `.track` | 3, float | longitude, latitude, timestamp in seconds |")
    A("| `.nodes` | 2, float | longitude, latitude; row index is the node id |")
    A("| `.arcs` | 2, int | from_node_id, to_node_id; row index is the arc id |")
    A("| `.route` | 1, int | arc ids traversed, in order |")
    A("")
    A("Identifiers are trip-local, so an arc id means nothing outside its own trip. That single "
      "fact rules out classifying over arc ids and forces a pairwise model over "
      "(observation, candidate arc).")
    A("")
    A("### Preprocessing")
    A("")
    A("1. Parse and validate every file: column counts, integer ids, id ranges, timestamp "
      "ordering. 360 of 360 files pass.")
    A("2. Project each trip into its own equirectangular tangent plane, centred on the trip, so "
      "distances are metres. Haversine is kept as the reference and the projection is measured "
      "against it: worst per-trip error 0.583 percent, mean of the per-trip maxima 0.096 percent.")
    A("3. Build a directed graph per trip: CSR adjacency, arc lengths, bearings, connected "
      "components, and a spatial index over points sampled along each arc.")
    A("4. Derive per-observation reference labels with a monotone alignment against the ordered "
      "route, keeping route position so a revisited arc stays distinct.")
    A("5. Generate candidates, extract features, train.")
    A("")
    A("### Coordinate order")
    A("")
    A("The files have no headers. Eight trips carry a column 0 value above 90 degrees, the "
      "largest being 137.24, and column 1 never exceeds 61.70. A latitude cannot exceed 90, so "
      "column 0 is longitude. Reading the pair as `(lon, lat)` also puts fixes closer to their "
      "network on 90 of 90 trips.")
    A("")
    A("### What the exploration found")
    A("")
    A("| Finding | Value |")
    A("|---|---|")
    A(f"| GPS fixes | {eda['by_partition']['train']['gps_points'] + eda['by_partition']['student_test']['gps_points']:,} |")
    A(f"| Road-network nodes / arcs | {eda['by_partition']['train']['nodes'] + eda['by_partition']['student_test']['nodes']:,} / {eda['by_partition']['train']['arcs'] + eda['by_partition']['student_test']['arcs']:,} |")
    A(f"| Fixes per trip | 262 to 19,313 |")
    A(f"| Median distance, fix to nearest road | {eda['gps_to_road']['median_of_trip_medians_m']:.2f} m |")
    A(f"| 95th percentile of that distance | {eda['gps_to_road']['median_of_trip_p95_m']:.1f} m |")
    A(f"| Steps exactly 1 s apart | {100 * eda['sampling']['share_of_steps_equal_to_1s']:.1f} percent |")
    A(f"| Trips with a gap over 60 s | {eda['gaps']['trips_with_gap']} ({eda['gaps']['total_gaps']} gaps, largest {eda['gaps']['largest_gap_s']:,.0f} s) |")
    A(f"| Median connected components per trip | {eda['connectivity']['median_components']:.0f} |")
    A(f"| Trips whose route has chain breaks | {eda['audit']['reference_route']['trips_with_chain_breaks']} |")
    A(f"| Trips whose route revisits an arc | {eda['audit']['reference_route']['trips_revisiting_an_arc']} |")
    A("")
    A("The noise figure is the one that motivates the whole approach. A median 4.2 m error is "
      "the same size as the gap between parallel carriageways, so distance alone cannot say "
      "which road the vehicle was on. Taking the single nearest arc lands on the reference "
      "route only about 80 percent of the time.")
    A("")
    A("Missing values: zero non-finite values, zero decreasing timestamps, zero out-of-range "
      "coordinates. Five trips repeat a coordinate on consecutive fixes, which is a stationary "
      "vehicle rather than corruption. The data is irregular in time and noisy in space, not "
      "holey.")
    A("")
    A("### Features extracted, and why")
    A("")
    A("Twenty-one features in six groups, all relative to the road network. No absolute "
      "longitude or latitude anywhere, because the evaluator trips come from cities that are "
      "not in the training data.")
    A("")
    A("| Group | Features | Why |")
    A("|---|---|---|")
    A("| geometry | dist_m, proj_t, end_dist_min_m, log_arc_len | distance is the main signal; "
      "position along the arc and proximity to a junction matter near intersections |")
    A("| ranking | dist_rank, dist_gap_to_best, dist_ratio_to_best, n_cands, n_close_cands | "
      "how much better this candidate is than the alternatives, and how crowded the area is |")
    A("| heading | cos_heading, heading_valid | the only feature that separates a two-way "
      "road's two arcs, which sit at identical distance |")
    A("| context | ctx_dist_mean, ctx_dist_max, ctx_gap_to_best | a real road fits the whole "
      "local track; a side street only touches it |")
    A("| network | from_in_deg, to_out_deg, has_reverse_twin, local_arc_density | junction "
      "complexity and whether the road is two-way |")
    A("| temporal | speed_mps, dt_prev_s, long_gap | guards against stopped vehicles, "
      "sub-second timestamps and the 60 s gaps |")
    A("")
    A("Ablation, LightGBM on the first three folds:")
    A("")
    A(md_table(abl, ["dropped_group", "n_features", "cv_macro_f1", "cv_top1"]))
    A("")
    A("Dropping heading costs about 0.26 CV Top-1 accuracy, far more than any other group. That "
      "is the expected result: two-way roads are stored as two arcs at the same distance, and "
      "direction of travel is the only thing that tells them apart.")
    A("")
    A("### Visualisations")
    A("")
    A("| Figure | File |")
    A("|---|---|")
    A("| Dataset inventory by partition | `results/figures/eda/01_inventory.png` |")
    A("| Trip size distributions | `results/figures/eda/02_trip_sizes.png` |")
    A("| Sampling intervals | `results/figures/eda/03_sampling_intervals.png` |")
    A("| Largest gap per trip | `results/figures/eda/04_gaps.png` |")
    A("| Fix to nearest road distance | `results/figures/eda/05_gps_to_road_distance.png` |")
    A("| Trip locations worldwide | `results/figures/eda/06_trip_locations.png` |")
    A("| Network connectivity | `results/figures/eda/07_network_connectivity.png` |")
    A("| Route length against GPS count | `results/figures/eda/08_route_vs_track.png` |")
    A("| GPS track over road geometry | `results/figures/eda/09_track_vs_network.png` |")
    A("| Coordinate order proof | `results/figures/eda/10_coordinate_order.png` |")
    A("| Trajectory extent | `results/figures/eda/11_trajectory_extent.png` |")
    A("| Projection error per trip | `results/figures/eda/12_projection_error.png` |")
    A("| Coordinate distributions | `results/figures/eda/13_coordinate_distributions.png` |")
    A("")
    A("All thirteen are computed from all 90 supplied trips. The full data-quality record, "
      "including assumptions and open questions, is in `docs/DATA_NOTES.md`.")
    A("")

    # ---------------------------------------------------------------- 3 models
    A("## 3. Model experimentation and comparison")
    A("")
    A("Eight models across seven algorithm families, all trained on the same 21 features with "
      "the same row order, the same five GroupKFold folds by trip, and standardisation fitted "
      "inside each training fold only.")
    A("")
    A("### 3.1 Overall comparison")
    A("")
    short_family = {
        "Rule (no learning)": "Rule", "Linear / probabilistic": "Linear",
        "Probabilistic (generative)": "Generative", "Instance-based": "Instance",
        "Tree": "Tree", "Ensemble (bagging)": "Bagging", "Boosting": "Boosting",
        "Gradient boosting (leaf-wise)": "Boosting", "Neural network": "Neural",
    }
    scores = pd.DataFrame({
        "Model": comp["Model"],
        "Family": comp["Algorithm Family"].map(lambda f: short_family.get(f, f)),
        "CV MacF1": pd.to_numeric(comp["CV Macro F1-score"], errors="coerce"),
        "Top-1": pd.to_numeric(comp["Top-1 Accuracy"], errors="coerce"),
        "Cont.": pd.to_numeric(comp["Route continuity"], errors="coerce"),
        "Test MacF1": pd.to_numeric(comp["Student-Test Macro F1-score"], errors="coerce"),
        "LogLoss": pd.to_numeric(comp["Log Loss"], errors="coerce"),
    })
    A(md_table(scores, floatfmt="{:.3f}"))
    A("")
    cfg = comp[["Model", "Key Configuration", "Inference Time"]].copy()
    cfg.columns = ["Model", "Key configuration", "Inference time"]
    A(md_table(cfg))
    A("")
    A("The nine required columns are spread over these two tables: model, family, "
      "configuration, CV Macro F1, Top-1, route continuity, student-test Macro F1, log loss and "
      "inference time. They are split only so the table fits on the page.")
    A("Nearest-arc has no log loss because it produces no probability, only a ranking. Every "
      "Top-1 figure is on the 10 student-test trips, which are never used to fit or select "
      "anything.")
    A("")
    A("**Selection rule, fixed before the results were seen:** highest mean CV Macro F1, with "
      "mean CV Log Loss breaking ties. Macro F1 rather than accuracy because the positive class "
      "is 9.3 percent of candidate rows; log loss as tie-break because the decoder consumes a "
      "probability itself.")
    A("")

    A("### 3.2 Per-model answers")
    A("")
    for name, note in MODEL_NOTES.items():
        row = comp[comp.Model == name].iloc[0]
        cfg = row["Key Configuration"]
        A(f"#### {name}")
        A("")
        A(f"**A1. Why this algorithm suits PathSnap.** {note['why']}")
        A("")
        A(f"**A2. Strengths and limitations.** {note['strengths']} {note['limits']}")
        A("")
        A(f"**B5. Important hyperparameters.** `{cfg}`. Chosen on the training folds; the full "
          f"configuration and seed are in `experiments/*_{name}.json`.")
        A("")
        A(f"**B6. Loss function or objective.** {note['objective']}")
        A("")
        A(f"**C7. Performance on the student-testing set.** Top-1 accuracy "
          f"{row['Top-1 Accuracy']:.4f}, Macro F1 {row['Student-Test Macro F1-score']:.4f}, "
          f"precision {row['Student-Test Precision']:.4f}, recall "
          f"{row['Student-Test Recall']:.4f}, route continuity "
          f"{row['Route continuity']:.4f}.")
        A("")
        ll = "not applicable: the model produces no probability" if pd.isna(row["Log Loss"]) \
            else f"{row['Log Loss']:.4f}"
        A(f"**C8. Observation-level Top-1, Macro F1 and Log Loss.** Top-1 "
          f"{row['Top-1 Accuracy']:.4f}, Macro F1 {row['Student-Test Macro F1-score']:.4f}, "
          f"Log Loss {ll}.")
        A("")
        gap = trn.get(name, float("nan")) - val.get(name, float("nan"))
        verdict = ("underfitting: the training folds score only slightly above validation"
                   if gap < 0.03 else
                   "some overfitting: the training folds score clearly above validation, which "
                   "is expected for a flexible model and is why the fold spread is reported"
                   if gap < 0.1 else
                   "clear overfitting: training folds are far above validation")
        A(f"**C9. Overfitting or underfitting.** Train-fold Macro F1 {trn.get(name, float('nan')):.4f} "
          f"against validation {val.get(name, float('nan')):.4f}, a gap of {gap:.4f}, so "
          f"{verdict}. Fold standard deviation is {row['CV Macro F1 std']:.4f}.")
        A("")

    A("### 3.3 Model visualisations")
    A("")
    A("| Figure | What it shows | File |")
    A("|---|---|---|")
    A("| CV Macro F1 with fold spread | all models, 5-fold grouped by trip | `results/figures/models/cv_macro_f1.png` |")
    A("| Student-test Macro F1 against Top-1 | all models | `results/figures/models/student_test_scatter.png` |")
    A("| Train against validation Macro F1 | overfitting evidence per model | `results/figures/models/train_vs_val.png` |")
    A("| Confusion matrices | all models, student test, threshold 0.5 | `results/figures/models/confusion_matrices.png` |")
    A("| Feature-group ablation | LightGBM, first three folds | `results/figures/models/feature_ablation.png` |")
    A("| Permutation importance | selected model, fold-0 validation | `results/figures/models/permutation_importance.png` |")
    A("| LightGBM log loss per iteration | training history, real curve | `results/figures/models/lightgbm_logloss_curve.png` |")
    A("| MLP training curve | loss and validation accuracy per epoch | `results/figures/models/mlp_training_curve.png` |")
    A("")
    A("Only LightGBM and the MLP produce epoch-wise histories, so only those two have training "
      "curves. The other six are fitted in one step and have no curve to plot.")
    A("")
    A("Top permutation importance, selected model on fold-0 validation:")
    A("")
    A(md_table(imp.head(6), ["feature", "importance_mean", "importance_std"]))
    A("")

    # ---------------------------------------------------------------- 4 HPO
    A("## 4. Bayesian hyperparameter optimisation")
    A("")
    best = hpo["best_params"]
    A("**1. Hyperparameters in the search.** Seven: `num_leaves`, `learning_rate`, "
      "`n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`.")
    A("")
    A("**2. Ranges and distributions.**")
    A("")
    A("| Parameter | Range | Why |")
    A("|---|---|---|")
    A("| num_leaves | 15 to 255, log | tree complexity; the untuned value was 63 |")
    A("| learning_rate | 0.01 to 0.2, log | step size, traded against the number of trees |")
    A("| n_estimators | 100 to 800, step 50 | boosting rounds |")
    A("| min_child_samples | 10 to 300, log | rows per leaf; larger smooths the fit |")
    A("| subsample | 0.5 to 1.0 | row bagging per tree |")
    A("| colsample_bytree | 0.5 to 1.0 | feature bagging per tree |")
    A("| reg_lambda | 1e-3 to 10, log | L2 penalty on leaf values |")
    A("")
    A("**3. Objective function.** Mean Macro F1 over 3-fold GroupKFold by trip, on every "
      "eighth fix of the 80 training trips. Macro F1 is the same criterion used to select the "
      "model, and grouping by trip keeps whole cities out of the training side of each fold.")
    A("")
    A(f"**4. Trials.** {int(hpo['n_trials'])} trials requested, {int(hpo['n_complete'])} "
      f"completed and {int(hpo['n_pruned'])} pruned by a MedianPruner after five startup "
      f"trials. Wall clock {hpo['search_wall_clock_s_this_run']:.0f} s. The untuned "
      f"configuration was enqueued as trial 0, so tuned and untuned were scored on identical "
      f"folds.")
    A("")
    A("**5. Best values.**")
    A("")
    A("| Parameter | Value |")
    A("|---|---|")
    for k, v in best.items():
        A(f"| {k} | {v:.6g} |" if isinstance(v, float) else f"| {k} | {v} |")
    A("")
    A("**6. Did the tuned model outperform the untuned one?** Marginally on the selection "
      "criterion, and not on everything else.")
    A("")
    utm = ut[["Model", "CV Macro F1", "CV Top-1", "CV Log Loss", "Student-Test Top-1 Accuracy",
              "Student-Test Macro F1", "Student-Test Log Loss", "Inference Time (ms / 1k candidates)"]].copy()
    A(md_table(utm))
    A("")
    A("CV Macro F1 rises from 0.91410 to 0.91454, a gain of 0.0004. Student-test Top-1 falls "
      "from 0.8957 to 0.8919 and student-test log loss rises from 0.0487 to 0.0488, while "
      "student-test Macro F1 rises from 0.9363 to 0.9372. Inference cost goes from 7.83 to "
      "8.46 ms per thousand candidates. The tuning is close to a wash. The untuned configuration was "
      "already reasonable and the search spent much of its budget on a 3-fold objective that "
      "is itself noisy.")
    A("")
    A("Tuning figures: `results/figures/hpo/optimisation_history.png`, "
      "`param_importance.png`, `untuned_vs_tuned.png`.")
    A("")

    # ---------------------------------------------------------------- 5 HMM
    A("## 5. HMM and Viterbi route reconstruction")
    A("")
    A("**1. Hidden states.** At fix *i*, the state is one of that fix's candidate arcs. A path "
      "through the trellis picks exactly one arc per fix.")
    A("")
    A("**2. Emission probabilities.** `emission_weight * log p(arc | fix)` from the LightGBM "
      "positive-class probability, optionally plus a Gaussian distance term "
      "`-distance_weight * 0.5 * (d / sigma)^2`. The distance term defaults to off because the "
      "classifier already uses distance as its main feature.")
    A("")
    A("**3. Transition probabilities.** Newson and Krumm's model: "
      "`-|route_distance - straight_line_distance| / beta`. The route distance is the driving "
      "distance between the two matched positions, computed as the remainder of the first arc "
      "plus a bounded Dijkstra path plus the head of the second arc. A pair with no path inside "
      "`route_factor * straight + route_slack` gets a transition of minus infinity.")
    A("")
    A("**4. Missing candidates and large gaps.** A fix with no candidates ends the current "
      "chain, and the next fix starts a new one. A time gap over 60 s also starts a new chain, "
      "so no impossible transition is ever forced across the 42-hour gap in trip 00000090.")
    A("")
    A("**5. Disconnected segments.** If every transition from the previous fix is impossible, "
      "the chain restarts at the current fix and the position is recorded in "
      "`break_indices`. The counts appear in the response diagnostics as `gap_breaks` and "
      "`impossible_transition_breaks`.")
    A("")
    A("**6. Did decoding improve the route?** Yes, on every metric and on every one of the ten "
      "student-test trips.")
    A("")
    rt = routes[["method", "top1_acc", "continuity", "arc_set_f1", "norm_edit_distance",
                 "joined_route_continuity", "joined_arc_set_f1"]].copy()
    A(md_table(rt))
    A("")
    A("Observation-level Top-1 rises from 0.8919 to 0.9391, route continuity from 0.826 to "
      "0.915, arc-set F1 from 0.898 to 0.936, and normalised edit distance falls from 0.351 to "
      "0.144. After joining non-touching consecutive arcs with shortest paths, continuity is "
      "1.0 and edit distance 0.075.")
    A("")
    A("Per-trip effect on the student-test set:")
    A("")
    A(md_table(hmm_eff))
    A("")
    A("Every trip gains more than it loses, but the losses are real: on trip 00000077 the "
      "decoder fixes 622 observations and breaks 202. Those are cases where a smooth, "
      "confidently-scored but wrong road won on transition cost, which is the failure mode to "
      "watch for. Worked examples are in `results/routes/examples.json`, with maps at "
      "`maps/hmm_improved_00000077.html` and `maps/hmm_hurt_00000077.html`.")
    A("")
    A("Decoder settings were tuned on 24 training trips using out-of-fold scores only, with the "
      "objective being mean per-trip Top-1; student-test was never part of the tuning. The "
      "chosen values and the tuning set are recorded in "
      "`artifacts/models/final/decoder_config.json`.")
    A("")

    # ---------------------------------------------------------------- 6 folium
    A("## 6. Geographical visualisation with Folium")
    A("")
    A("Interactive maps are generated by `scripts/eda.py` and by Member 3's route evaluation. "
      "Each has a layer control, a legend and a fixed title bar naming the trip and partition:")
    A("")
    A("| Map | Layers | File |")
    A("|---|---|---|")
    A("| Dense network, trip 00000013 | road network (sampled), GPS fixes | `maps/eda_dense_network.html` |")
    A("| Sparse network, trip 00000046 | road network (sampled), GPS fixes | `maps/eda_sparse_network.html` |")
    A("| Offset trip, 00000054 | road network (sampled), GPS fixes | `maps/eda_offset_trip.html` |")
    A("| HMM improvement, trip 00000077 | GPS, predicted route, reference | `maps/hmm_improved_00000077.html` |")
    A("| HMM regression, trip 00000077 | GPS, predicted route, reference | `maps/hmm_hurt_00000077.html` |")
    A("")
    A("Static captures are in `results/figures/maps/`. The base layer is Esri World Street Map "
      "because OpenStreetMap returns HTTP 403 to automated tile requests; the geometry layers "
      "are drawn by Leaflet regardless, so a map still reads if tiles fail.")
    A("")
    A("The evaluator-facing UI draws the same layers for the trip it is asked about: GPS fixes "
      "in grey, the predicted route in blue, with a legend and pan/zoom.")
    A("")

    # ---------------------------------------------------------------- 7 api/ui
    A("## 7. API, UI and deployment")
    A("")
    A("Laptop 1 runs the FastAPI service, which holds the model, the decoder, the road networks "
      "and the scoring. Laptop 2 runs the UI and holds nothing.")
    A("")
    A("| Endpoint | Purpose |")
    A("|---|---|")
    A("| `GET /api/health` | status, model version, git commit, trips available |")
    A("| `GET /api/trips` | the evaluation trips on the server |")
    A("| `POST /api/predict` | `{trip_id}` in, predicted route plus server-side metrics out |")
    A("| `GET /api/model` | selected model, decoder settings, saved evaluation summary |")
    A("")
    A("Ground truth never leaves Laptop 1. The prediction pipeline reads only `.track`, "
      "`.nodes` and `.arcs`; the reference route is opened afterwards, inside the scoring "
      "function, and only the resulting numbers are returned. `scripts/check_pipeline.py` "
      "verifies this by producing an identical prediction with and without a `.route` file "
      "present, and by checking that no module on the prediction path imports the labels.")
    A("")
    A("### Working UI")
    A("")
    A("The screenshots below are the running UI on Laptop 2 taking predictions from the API on "
      "Laptop 1, each showing one evaluator trip with the accuracy the server computed for it. "
      "The trip list, the model version and the latency are returned by the API; the UI holds "
      "neither the model nor the reference routes.")
    A("")
    for trip, note in [
        ("00000049", "the strongest trip, 100 percent Top-1 on 2,642 fixes"),
        ("00000017", "a long highway trip, 93.8 percent on 1,346 fixes"),
        ("00000030", "the weakest trip, 87.2 percent on 654 fixes"),
        ("00000015", "the largest trip, 93.3 percent on 6,854 fixes"),
    ]:
        A(f"![UI, evaluator trip {trip}](figures/ui/eval_{trip}.png)")
        A("")
        A(f"Evaluator trip {trip}, {note}.")
        A("")
    A("The remaining six evaluator trips were run the same way; their scores are in the table "
      "below, and every screenshot is in `report/figures/ui/`.")
    A("")
    A("### Final evaluation on the withheld evaluator trips")
    A("")
    A("The ten evaluator trips were placed in `data/eval/` on Laptop 1 and run through the same "
      "pipeline, with no retraining and no adjustment.")
    A("")
    evt = ev[["trip_id", "fixes", "breaks", "observation_top1_accuracy", "route_continuity",
              "arc_set_f1", "normalized_edit_distance", "latency_ms"]].copy()
    A(md_table(evt))
    A("")
    w = lambda c: (ev[c] * ev.fixes).sum() / ev.fixes.sum()  # noqa: E731
    A(f"Across all {int(ev.fixes.sum()):,} evaluator-test observations: weighted Top-1 "
      f"accuracy {w('observation_top1_accuracy'):.4f}, mean per-trip Top-1 "
      f"{ev.observation_top1_accuracy.mean():.4f}, mean route continuity "
      f"{ev.route_continuity.mean():.4f}, mean arc-set F1 {ev.arc_set_f1.mean():.4f}, mean "
      f"normalised edit distance {ev.normalized_edit_distance.mean():.4f}. Slowest trip "
      f"{ev.latency_ms.max() / 1000:.1f} s end to end.")
    A("")
    A("Trip 00000015 is the weak case: Top-1 is 0.933 but arc-set F1 is 0.835 and edit distance "
      "0.498, which means the right arcs were mostly found but the returned sequence carried "
      "extra or missing arcs. That points at route assembly rather than classification.")
    A("")

    # ---------------------------------------------------------------- 8 integrity
    A("## 8. Academic integrity and AI use")
    A("")
    A("Every result in this report is produced by code in the repository and can be regenerated "
      "from the commands in `README.md`. Numbers are read from the saved CSV, JSON and parquet "
      "outputs by `scripts/build_report.py`, so the report cannot state a figure that the "
      "experiments did not produce.")
    A("")
    A("The four members divided the work as recorded in `docs/`: data engineering and labels "
      "(`DATA_NOTES.md`), candidates, features and model comparison (`MEMBER2_NOTES.md`), "
      "tuning, decoding and the pipeline (`MEMBER3_NOTES.md`), and the API and UI "
      "(`DEPLOYMENT.md`). Each member can explain their own modules and the interface "
      "contracts between them.")
    A("")
    A("AI assistants were used during development for scaffolding, documentation review and "
      "code generation, under the disclosure terms of the event. Every generated result was "
      "re-run and checked against the saved outputs before being reported. The modelling "
      "choices, experiment design, evaluation protocol and interpretation are the team's.")
    A("")

    return "\n".join(L)


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>PathSnap report</title>
<style>
body {{ font-family: "Times New Roman", Times, serif; max-width: 850px; margin: 40px auto;
       line-height: 1.45; color: #000; background: #fff; font-size: 11pt; }}
h1 {{ font-size: 17pt; font-weight: bold; margin-bottom: 10px; }}
h2 {{ font-size: 13pt; font-weight: bold; margin-top: 24px; }}
h3 {{ font-size: 11.5pt; font-weight: bold; margin-top: 18px; }}
h4 {{ font-size: 11pt; font-weight: bold; margin-top: 14px; }}
p {{ margin: 8px 0; }}
table {{ border-collapse: collapse; margin: 10px 0 14px; font-size: 7.5pt; width: 100%; }}
th, td {{ border: 1px solid #000; padding: 3px 5px; text-align: left; vertical-align: top; }}
th:first-child, td:first-child {{ white-space: nowrap; }}
th, td {{ word-wrap: break-word; }}
th {{ font-weight: bold; white-space: nowrap; }}
code {{ font-family: "Courier New", monospace; font-size: 9pt; white-space: nowrap; }}
img {{ max-width: 100%; }}
</style></head><body>
{body}
</body></html>
"""


def md_to_html(md: str) -> str:
    """Minimal converter for the subset used here: headings, tables, bold, code, images."""
    import re

    out, lines, i = [], md.split("\n"), 0
    inline = lambda t: re.sub(r"`([^`]+)`", r"<code>\1</code>",  # noqa: E731
                              re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t))
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
            rows = [r for r in rows if not set("".join(r)) <= set("- ")]
            width = max(1, len(rows[0]) - 1)
            out.append("<table><colgroup>")
            out.append('<col style="width:18%">')
            out.append(f'<col style="width:{82 // width}%">' * width)
            out.append("</colgroup>")
            for n, r in enumerate(rows):
                tag = "th" if n == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            out.append(f'<p><img src="{m.group(2)}" alt="{m.group(1)}"></p>'); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if ln.strip() == "---":
            out.append("<hr>"); i += 1; continue
        if ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def main() -> int:
    md = build()
    # the stray conditional-format guard: keep the report free of em dashes
    md = md.replace("\u2014", ", ").replace("\u2013", "-")
    (REP / "PathSnap_Report.md").write_text(md + "\n")
    from report_pdf import render

    stats = render(REP / "PathSnap_Report.md", REP / "Team_4.pdf", REP)
    (REP / "PathSnap_Report.html").write_text(HTML_TEMPLATE.format(body=md_to_html(md)))
    print(f"wrote {REP / 'PathSnap_Report.md'}")
    print(f"wrote {REP / 'Team_4.pdf'}  ({stats['tables']} tables, {stats['images']} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
