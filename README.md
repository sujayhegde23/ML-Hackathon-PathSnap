# PathSnap: distributed map-matching and route reconstruction

Course project for UE24CS352A Machine Learning, PES University (RR campus).

PathSnap reconstructs a vehicle's route by matching noisy GPS fixes onto a road network. Each
trip holds a trajectory and the local road network around it. The work is to decide, for every
fix, which road segment the vehicle was on, then return those segments as one coherent route.

The repository holds the source for data processing, feature engineering, model experiments,
Bayesian hyperparameter optimisation, HMM/Viterbi decoding, the FastAPI backend and the
evaluator UI.

## Repository layout

```text
.
├── configs/                  Run configuration (paths, partitions, thresholds)
├── docs/
│   └── DATA_NOTES.md         Data-quality record: issues, assumptions, decisions
├── data/
│   ├── raw/                  Supplied trip data (not committed)
│   ├── processed/            Cached derived artefacts
│   └── eval/                 Evaluator trips, injected at grading time
├── src/pathsnap/
│   ├── config.py             Paths, partitions, global seed
│   ├── io/loader.py          Raw trip parsing, validation, canonical Trip object
│   ├── geo/                  Per-trip metric projection
│   ├── graph/                Road-network graph, spatial index, bounded shortest paths
│   ├── labels/               Observation-level reference labels from the ordered route
│   ├── candidates/           Candidate-set generation
│   ├── features/             Candidate feature extraction
│   ├── models/               Model training and comparison
│   ├── hpo/                  Bayesian hyperparameter optimisation
│   ├── hmm/                  HMM emissions/transitions and Viterbi decoding
│   ├── eval/                 Metrics and experiment runners
│   ├── viz/                  Plots and Folium maps
│   ├── pipeline.py           End-to-end prediction
│   └── artifacts.py          Model/preprocessing persistence
├── api/                      FastAPI service
├── ui/                       Evaluator UI
├── scripts/                  Reproducible entry points
├── artifacts/                Model weights and configuration
├── experiments/              Experiment logs
├── results/                  Metrics, tables, figures
├── maps/                     Generated Folium maps
└── report/                   Technical report
```

## Dataset

The trip data comes from the faculty and is not committed. Put it where the repository can see
one directory containing the partition folders:

```text
data/raw/
├── train/                    80 trips
└── student_test/             10 trips
```

Or point `PATHSNAP_DATA_ROOT` at the directory wherever it sits.

Each trip directory holds four headerless whitespace-separated files, all named after the trip:

| File | Columns | Meaning |
|---|---|---|
| `.track` | `longitude latitude timestamp_s` | GPS observations for the trip |
| `.nodes` | `longitude latitude` | Network nodes; row index is the node id |
| `.arcs` | `from_node_id to_node_id` | Network arcs; row index is the arc id |
| `.route` | `arc_id` | Reference route: arc ids in traversal order |

Identifiers are trip-local. Row `i` of a trip's `.nodes` file is node `i` for that trip, and
nothing in the project compares identifiers across trips.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the default temporary or matplotlib configuration directories are not writable, point them
somewhere writable:

```bash
export TMPDIR=$PWD/.tmp
export MPLCONFIGDIR=$PWD/.mpl
```

The `Makefile` covers the common tasks:

```bash
make validate   # parse and validate every supplied trip, write results/validation/
make eda        # measurements, figures and interactive maps for the report
make recon      # dataset reconnaissance summary
```

## Implementation status

| Area | Owner | Status |
|---|---|---|
| Trip loading, validation, canonical `Trip` object | Member 1 | Done |
| Metric projection, road graph | Member 1 | Done |
| EDA, data-quality evidence, Folium maps | Member 1 | Done |
| Observation-level label derivation | Member 1 | Done |
| Candidates, features, model experiments | Member 2 | Next |
| Bayesian optimisation, HMM/Viterbi, pipeline | Member 3 | Planned |
| API, UI, deployment | Member 4 | Planned |

## Conventions

- Distances are in metres and coordinates in degrees, unless a name says `_xy` or `_m`.
- The prediction path does not read `.route`. The loader raises `GroundTruthAccessError` if a
  prediction-time caller asks for it.
- Model features never contain absolute longitude or latitude, because the evaluator trips come
  from cities that are not in the training set.
- `pathsnap.config.set_global_seed` fixes the random seeds.
