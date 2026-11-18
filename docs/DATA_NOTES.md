# Dataset notes: issues, assumptions and decisions

This file records what was checked against the supplied data and what was found, so the
decisions behind the pipeline can be traced back to evidence.

Numbers come from `scripts/eda.py` (tables in `results/eda/`, figures in
`results/figures/eda/`) or `scripts/validate_dataset.py` (`results/validation/`). Tables holding
a specific figure are named where it matters.

## 1. What is in the data

| Partition | Trips present | GPS fixes |
|---|---|---|
| `train` | 80 | 208,417 |
| `student_test` | 10 | 19,160 |
| `eval` | 0 (injected at grading time) | withheld |

Totals over the 90 supplied trips: 227,577 GPS fixes, 2,024,174 road-network nodes, 4,061,945
arcs, 360 raw files. Every trip directory holds exactly four files, named after the trip.

File sizes across the corpus:

| File | Smallest | Median | Largest | Total |
|---|---|---|---|---|
| `.track` | 8.3 KB | 48.5 KB | 626 KB | 7.5 MB |
| `.nodes` | 5.7 KB | 187 KB | 2.9 MB | 40.3 MB |
| `.arcs` | 4.7 KB | 183 KB | 3.9 MB | 46.2 MB |
| `.route` | 253 B | 1.4 KB | 12.1 KB | 0.18 MB |

## 2. What was verified

The brief warns that trips may not be identical. They are not, so each claim below is measured
per trip and stored in `results/eda/file_inventory.csv`.

- Columns and types. `.track` has 3 float columns, `.nodes` 2 float columns, `.arcs` 2 integer
  columns, `.route` 1 integer column. 360 of 360 files match. No file is missing.
- Row counts. GPS fixes range 262 to 19,313 per trip; nodes 299 to 155,003; arcs 649 to
  307,392; route arcs 59 to 2,266. No count is constant across trips.
- Identifier ranges. Every `arcs` entry indexes a node that exists in that trip's `.nodes`
  file, and every `.route` entry indexes an arc that exists in that trip's `.arcs` file, for
  all 90 trips.
- Missing values. Zero non-finite values anywhere in the dataset, in either `.track` or
  `.nodes`.
- Timestamp ordering. No trip has a decreasing timestamp. Five trips repeat a coordinate on
  consecutive fixes (six occurrences), which is a stationary vehicle, not a data error.
- Coordinate plausibility. Every GPS fix lies inside its own trip's node bounding box, for
  all 90 trips. No fix is out of range, and none is a placeholder zero.

## 3. Coordinate order

The files are undocumented. Assuming the wrong column order would silently place every trip in
the wrong hemisphere, so the order was tested two ways.

First, a latitude cannot exceed 90 degrees. Eight trips have a column 0 value above 90, the
largest being 137.24. Column 1 never exceeds 61.70. Column 0 is therefore longitude.

Second, the fix-to-nearest-node distance was computed with both readings. Reading the pair as
`(lon, lat)` gives a smaller median distance than `(lat, lon)` for 90 of 90 trips, by orders of
magnitude. See `results/figures/eda/10_coordinate_order.png`.

Conclusion: column 0 is longitude, column 1 is latitude, recorded in that order in every file.

## 4. Projection

Distances must be metric. Euclidean distance in degrees is meaningless, and the length of a
degree of longitude shrinks with latitude, so a single global scale would also be wrong.

Each trip is projected into its own equirectangular tangent plane centred on that trip:

```
x = R * radians(lon - lon0) * cos(lat0)
y = R * radians(lat - lat0)
```

Computing geodesic distance for every comparison would be more accurate but far slower when
each fix is compared against hundreds of candidate arcs. Haversine is kept as the reference and
every projection is checked against it. Measured over the 90 trips, the largest per-trip
relative error against haversine is 0.58 percent, and the mean of those per-trip maxima is 0.10
percent. At the 50 metre candidate radius in use that is under 30 centimetres, well below the
GPS noise described next.

The whole frame is per trip, so a model built on one city is not sensitive to absolute position.

## 5. GPS noise, sampling and gaps

Distance from a fix to the nearest road arc has a median of 4.2 metres per trip, with a median
95th percentile of 13.0 metres. Fixes are close to roads but not on them, and the noise is of
the same order as the spacing between parallel carriageways. This is the ambiguity the whole
pipeline exists to resolve, and it caps what a nearest-arc rule can achieve.

Sampling is nominally 1 Hz. Across the corpus 94.3 percent of consecutive steps are exactly one
second. Three trips carry sub-second timestamps, so speed is always computed from real elapsed
time rather than assuming a one-second step.

Gaps above 60 seconds occur in 23 trips, 70 gaps in total. The largest is 153,064 seconds, about
42 hours, in trip 00000090. Any transition across such a gap is meaningless, so the decoder
splits a trajectory at large gaps instead of forcing continuity. Every gap is listed in
`results/eda/gaps.csv`.

## 6. Reference route: how it is encoded, and its limits

`.route` is an ordered list of arc ids, not one entry per GPS fix. The median ratio of route
arcs to GPS fixes is 0.19, so around five fixes share an arc. This matters for label
derivation: a per-fix reference arc has to be inferred, and that inference is a source of
uncertainty in its own right.

The route is meant to be a walk through the graph, and mostly it is. Across 90 trips:

- 79 trips form a fully connected chain, where each arc ends where the next begins.
- 11 trips contain between one and three breaks, 14 breaks in total.
- 10 trips revisit an arc, so matching on arc id alone would collapse distinct visits. Route
  position is preserved instead.
- 60 trips contain duplicate arc rows, that is, two rows with the same endpoints. A route can
  legitimately point at either, so labels are compared as arcs, not as row positions.

- The route is reference information supplied with the data, not independently surveyed ground
  truth. The break analysis shows it is not self-consistent.
- GPS fixes do not lie on the route. The median distance from a fix to its nearest route arc is
  4.6 metres across the corpus, with 87 of 90 trips between 3 and 9 metres
  (`results/eda/reference_route.csv`).
- Trip 00000054 sits far outside the network: its median distance to the nearest road is
  83.9 metres, and 85.5 percent of its observations have no candidate arc within 50 metres at
  all. Candidate generation has to widen its radius and fall back to nearest arcs, or that trip
  produces nothing.
- Trip 00000033 is a different case. Its fixes are close to the network in general, with only
  0.8 percent of observations lacking a candidate within 50 metres, but its recorded route runs
  a median of 122.8 metres away. The route for that trip, not the network, is what does not line
  up. Trip 00000054 is offset in the other direction as well: 138.2 metres from its route and
  83.9 metres from the nearest road.

What follows for evaluation: labels derived from the route are reference labels with a
measurable uncertainty, not certainty. Accuracy figures should be read with that in mind, and
the label derivation step records a confidence value per observation rather than presenting
every label as equally reliable.

## 7. Partitions and leakage

Trips are read one directory at a time, and a trip never spans partitions because the trip
directory is the unit of loading. The 80 training trips are the only ones used for fitting
anything learned. The 10 student-test trips are used for development comparisons only. The
evaluator trips are not present in the supplied data and are never read during development.

Ground-truth isolation is enforced in code rather than by convention. `load_trip_for_prediction`
cannot read `.route`: the reader raises `GroundTruthAccessError` if a prediction-time caller
asks for it, and a test confirms that a deliberately corrupted `.route` file leaves prediction
loading unaffected while training loading fails.

## 8. Assumptions that shape map matching

1. A fix and the road it belongs to are at most a few tens of metres apart. True for most
   trips, false for the two offset trips, which is why the candidate search adapts.
2. A vehicle moves along the road network in continuous time. True except across the recorded
   gaps, which are handled explicitly.
3. Arc direction indicates permitted travel. The networks are mostly bidirectional, and the
   presence of a reverse arc is treated as a feature rather than a constraint.
4. Routing follows shortest path between consecutive assignments. This is an approximation:
   a driver can take a longer path between two fixes, and the transition model allows for it
   through a tolerance around the observed distance.
5. Trip-local identifiers carry no meaning across trips. Nothing in the pipeline compares node
   or arc ids between trips.

## 9. Open questions

- Whether the evaluator trips follow the same directory layout and file naming as the supplied
  ones. The loader assumes `<partition>/<trip_id>/<trip_id>.<ext>` and should be tested against
  the evaluator package before grading.
- Whether the two offset trips represent a projection difference in the source, a route error,
  or roads absent from the extracted network. The available data cannot separate these.
- How the evaluator computes accuracy for a route, since `.route` is a set of traversed arcs
  rather than a per-fix label. The team should confirm the intended comparison before the final
  evaluation.
