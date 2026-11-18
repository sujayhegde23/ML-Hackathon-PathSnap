# Two-laptop demo: runbook

- **Laptop 1** runs the API, which holds the model, the HMM, the road networks and the
  server-side scoring.
- **Laptop 2** runs the UI, which lists the evaluation trips, sends the chosen trip id to the
  API, and shows the returned route and results.

## Current wiring

Both laptops are on the same tailnet, so no hotspot or venue Wi-Fi is needed. The host runs
the API and the model; the presenting laptop only runs the UI.

| Role | Tailnet IP |
|---|---|
| Host (API, model, HMM, scoring) | `100.98.58.11` |
| Presenting laptop (UI only) | `100.84.49.97` |

Host, from the repository root:

```bash
PATHSNAP_CORS_ORIGINS='*' python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Presenting laptop, after `git pull`, from the repository root:

```bash
python3 ui/serve.py --api http://100.98.58.11:8000
```

Then open `http://localhost:5500`. Before that, confirm the host answers:

```bash
curl http://100.98.58.11:8000/api/health
```

`ui/serve.py` forwards every call to the host and prints each request and response, so the
presenting laptop shows the traffic as well. The host prints the same requests to its own
terminal and to `logs/api.log`.

## 0. Before the evaluator arrives (once)

1. **Put both laptops on the same network.** Venue/university Wi-Fi often blocks
   laptop-to-laptop traffic, so the fastest reliable option is a **phone hotspot that both
   laptops join**.
2. **Laptop 1** (the only one that needs Python packages):
   ```bash
   git clone https://github.com/sujayhegde23/ML-Hackathon-PathSnap.git && cd ML-Hackathon-PathSnap
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Laptop 1: evaluation trips.** Put each trip in `data/eval/<trip_id>/` containing
   `<trip_id>.track`, `.nodes` and `.arcs`. Add `<trip_id>.route` too if you want the server
   to compute accuracy; it never leaves Laptop 1. A different folder works via
   `PATHSNAP_TRIP_DIRS=/path/to/folder`.
   - Until the evaluation trips exist, the API lists the 10 student_test trips, so you can
     rehearse now.
   - That fallback needs `PATHSNAP_DATA_ROOT` pointing at the folder holding `train/` and
     `student_test/`.
4. **Laptop 2** needs only this repo's `ui/` folder and Python 3, with no packages. Without
   Python, a browser alone is enough (see step 2 below).

## 1. Laptop 1: start the API

```bash
export PATHSNAP_DATA_ROOT="/path/to/Student folder"   # only needed for the student_test fallback
make api          # same as: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- The terminal prints:
  - the model it loaded;
  - the trips it found;
  - the exact command to run on Laptop 2, with this laptop's IP address.
- **macOS:** click *Allow* when asked whether Python may accept incoming connections.
  **Windows:** allow access on *Private networks*.
- **Keep this terminal on screen.** Every request and a summary of its response
  (`REQUEST …` / `RESPONSE 200 …`) appear here and in `logs/api.log`. This is the "API logs
  on the laptop receiving the requests" the evaluator asks to see.

## 2. Laptop 2: start the UI

```bash
python3 ui/serve.py --api http://<LAPTOP-1-IP>:8000
```

1. Open **http://localhost:5500**. The 10 trips appear.
2. Click one. The route is drawn and the accuracy computed on Laptop 1 is shown.

Laptop 2's terminal also prints each API response.

No Python on Laptop 2? Double-click `ui/index.html`, type `http://<LAPTOP-1-IP>:8000` in the
API box, and press Connect. The API allows browser access from any origin.

## 3. If something fails

| Symptom | Fix |
|---|---|
| UI says "Cannot reach the API" | On Laptop 2, open `http://<IP>:8000/api/health`. If it times out, it's the network or firewall: allow Python on Laptop 1, or switch both laptops to a phone hotspot |
| "No trips found on the server" | Trip folders are not in `data/eval/` on Laptop 1, or `PATHSNAP_DATA_ROOT` is not set for the student_test fallback |
| Route shows on a blank background | No internet for map tiles. The route and GPS points are still drawn correctly (offline mode) |
| Wrong IP | Use the IP printed by `make api`, or run `ipconfig getifaddr en0` (macOS) / `ipconfig` (Windows) |

## 4. Screenshots for the report

- The UI with a trip selected: the map plus the evaluation card.
- Laptop 1's terminal showing the `REQUEST` / `RESPONSE 200` lines for that trip.

## API reference (`api/main.py`)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | status, model_version, git_sha, trips_available |
| GET | `/api/trips` | `{"trips": [{"trip_id", "partition", "n_fixes"}]}` |
| POST | `/api/predict` | body `{"trip_id": "00000052"}`. Returns predicted_arc_ids, observation_arc_ids, polyline_lonlat, route_segments_lonlat, gps_lonlat, model_version, latency_ms, diagnostics, evaluation (numbers only, or null), evaluation_note |
| GET | `/api/model` | model and decoder settings, saved evaluation summary |

Errors come back as `{"detail": {"code", "message"}}`. The codes are:

- `unknown_trip` (404)
- `trip_files_missing` (404)
- `malformed_trip` (422)
- `artifacts_unavailable` (503)
- `inference_failed` (500)

Reference arcs and reference geometry are never sent.
