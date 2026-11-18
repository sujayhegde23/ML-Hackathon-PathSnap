"""PathSnap API — runs on Laptop 1.

Start (from the repository root):
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000      # or: make api

Endpoints
    GET  /api/health    status, model version, git commit, number of trips available
    GET  /api/trips     trip ids the UI should list (the evaluation trips)
    POST /api/predict   {"trip_id": "..."} -> predicted route + server-side evaluation
    GET  /api/model     selected/tuned model, decoder settings, saved evaluation summary

How a request is processed
    1. The trip id must be one of the trip folders found on this laptop (no paths accepted).
    2. pathsnap.pipeline.predict_trip runs the full pipeline: candidates -> features ->
       tuned LightGBM -> HMM/Viterbi. It reads only .track/.nodes/.arcs.
    3. Separately, if this laptop holds the trip's reference route (.route in the trip
       folder), pathsnap.eval.route_metrics.score_prediction computes accuracy here, on
       the server. Only the numbers are returned; reference arcs never leave this laptop.
    4. Every request and a summary of every response is printed to this terminal and
       appended to logs/api.log, so the evaluator can watch the requests arrive.

Where trips are looked for (first match wins for a duplicated id)
    PATHSNAP_TRIP_DIRS (folders separated by ':' on macOS/Linux, ';' on Windows), else
    data/eval/ and <data root>/eval/. If none of those holds a trip, the student_test
    trips are listed instead so the system can be tested before the evaluation trips arrive.

Configuration (environment variables)
    PATHSNAP_DATA_ROOT     folder with train/ student_test/ (eval/) partitions
    PATHSNAP_TRIP_DIRS     explicit trip folders to serve
    PATHSNAP_CORS_ORIGINS  allowed browser origins, comma separated (default "*")
    PATHSNAP_API_HOST / PATHSNAP_API_PORT  when started with `python api/main.py`
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from pathsnap.artifacts import git_commit, load_artifacts  # noqa: E402
from pathsnap.config import data_root, results_dir  # noqa: E402
from pathsnap.io.loader import TripFormatError  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
if not log.handlers:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api.log")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False

STATE: dict = {"artifacts": None, "error": None, "git": "unknown"}


# --------------------------------------------------------------------------- trips
def _roots() -> list[Path]:
    env = os.environ.get("PATHSNAP_TRIP_DIRS")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    return [ROOT / "data" / "eval", data_root() / "eval"]


def _scan(root: Path, label: str, out: dict) -> None:
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / f"{d.name}.track").is_file():
                out.setdefault(d.name, (label, d))


def discover_trips() -> dict[str, tuple[str, Path]]:
    trips: dict[str, tuple[str, Path]] = {}
    for root in _roots():
        _scan(root, root.name, trips)
    if not trips:  # nothing injected yet: fall back to student_test for testing
        _scan(data_root() / "student_test", "student_test", trips)
    return trips


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


# --------------------------------------------------------------------------- app
def _lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    try:
        STATE["artifacts"] = load_artifacts()  # load once, reuse for every request
        STATE["git"] = git_commit()
        log.info(f"MODEL loaded {STATE['artifacts']['model_version']} from {STATE['artifacts']['model_dir']} "
                 f"in {1000 * (time.perf_counter() - t0):.0f} ms")
    except Exception as exc:  # keep serving /api/health so the problem is visible
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.error(f"MODEL failed to load: {STATE['error']}")
    trips = discover_trips()
    log.info(f"TRIPS {len(trips)} available: {', '.join(sorted(trips)) or '(none)'}")
    port = os.environ.get("PATHSNAP_API_PORT", "8000")
    for ip in _lan_ips():
        log.info(f"LAN  on Laptop 2 run:  python3 ui/serve.py --api http://{ip}:{port}")
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _error(status: int, code: str, message: str) -> HTTPException:
    log.warning(f"ERROR {status} {code}: {message}")
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class PredictRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=64, examples=["00000052"])


@app.get("/api/health")
def health():
    arts = STATE["artifacts"]
    return {"status": "ok" if arts else "degraded", "model_version": arts["model_version"] if arts else None,
            "git_sha": STATE["git"], "trips_available": len(discover_trips()), "error": STATE["error"]}


@app.get("/api/trips")
def trips():
    found = discover_trips()
    out = [{"trip_id": t, "partition": part, "n_fixes": _count_lines(d / f"{t}.track")}
           for t, (part, d) in sorted(found.items())]
    log.info(f"TRIPS listed {len(out)}")
    return {"trips": out}


@app.get("/api/model")
def model_info():
    arts = STATE["artifacts"]
    if not arts:
        raise _error(503, "artifacts_unavailable", STATE["error"] or "model not loaded")
    summary = {}
    for name, path in [("route_comparison_student_test", results_dir() / "routes" / "student_test_summary.csv"),
                       ("untuned_vs_tuned", results_dir() / "hpo" / "untuned_vs_tuned.csv")]:
        if path.exists():
            import pandas as pd
            summary[name] = json.loads(pd.read_csv(path).drop(columns=["params"], errors="ignore").to_json(orient="records"))
    dec = arts["decoder"]
    return {"model_version": arts["model_version"], "selection": arts.get("selection", {}),
            "decoder": {k: getattr(dec, k) for k in dec.__dataclass_fields__},
            "feature_names": arts["feature_names"], "saved_evaluation": summary}


@app.post("/api/predict")
def predict(req: PredictRequest):
    arts = STATE["artifacts"]
    if not arts:
        raise _error(503, "artifacts_unavailable", STATE["error"] or "model not loaded")
    found = discover_trips()
    if req.trip_id not in found:
        raise _error(404, "unknown_trip", f"trip '{req.trip_id}' is not available on this server")
    partition, trip_dir = found[req.trip_id]
    log.info(f"REQUEST predict trip={req.trip_id} partition={partition}")
    try:
        result = predict_trip(trip_dir, arts, partition)
    except (TripFormatError, ValueError) as exc:
        raise _error(422, "malformed_trip", str(exc))
    except FileNotFoundError as exc:
        raise _error(404, "trip_files_missing", str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("inference failed")
        raise _error(500, "inference_failed", f"{type(exc).__name__}: {exc}")

    # ---- separate, server-side scoring (reference data never leaves this laptop)
    evaluation, note = None, None
    if (trip_dir / f"{req.trip_id}.route").is_file():
        try:
            from pathsnap.eval.route_metrics import score_prediction
            t0 = time.perf_counter()
            evaluation = score_prediction(trip_dir, result)
            evaluation.pop("trip_id", None)
            evaluation["scoring_ms"] = 1000 * (time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            note = f"evaluation failed: {type(exc).__name__}: {exc}"
    else:
        note = "no reference route on the server for this trip, so accuracy cannot be computed"

    body = result.to_dict()
    body.update(partition=partition, evaluation=evaluation, evaluation_note=note)
    ev = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in (evaluation or {}).items())
    log.info(f"RESPONSE 200 trip={result.trip_id} fixes={len(result.observation_arc_ids)} "
             f"route_arcs={len(result.predicted_arc_ids)} breaks={result.diagnostics.get('n_breaks')} "
             f"latency_ms={result.latency_ms:.0f} model={result.model_version} | {ev or note}")
    log.info("RESPONSE summary " + json.dumps({
        "trip_id": result.trip_id, "model_version": result.model_version, "latency_ms": round(result.latency_ms, 1),
        "predicted_arc_ids[:10]": result.predicted_arc_ids[:10], "n_predicted_arcs": len(result.predicted_arc_ids),
        "evaluation": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (evaluation or {}).items()} or None}))
    return body


@app.get("/ui")
def ui_for_local_testing():
    """Convenience for testing on one laptop. In the demo the UI runs on Laptop 2 (ui/serve.py)."""
    return FileResponse(ROOT / "ui" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("PATHSNAP_API_HOST", "0.0.0.0"),
                port=int(os.environ.get("PATHSNAP_API_PORT", "8000")))
