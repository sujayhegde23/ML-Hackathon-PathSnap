"""PathSnap UI server — runs on Laptop 2. Needs only Python 3 (standard library).

    python3 ui/serve.py --api http://<Laptop-1-IP>:8000            # then open http://localhost:5500

It serves ui/index.html and forwards every /api/... call to the API on Laptop 1, printing
a summary of each API response in this terminal. The browser therefore talks only to
this laptop (no cross-origin setup), and the selected trip id still goes to the API,
which runs the whole pipeline and the scoring. Nothing is computed or stored here.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    api_base = "http://localhost:8000"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._forward("GET")
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._forward("POST")
        self.send_error(404)

    def _forward(self, method: str):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        req = urllib.request.Request(self.api_base + self.path, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                status, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, data = e.code, e.read()
        except Exception as e:  # noqa: BLE001
            status = 502
            data = json.dumps({"detail": {"code": "api_unreachable",
                                          "message": f"cannot reach the API at {self.api_base}: {e}"}}).encode()
        self._print_response(method, status, data, time.perf_counter() - t0, body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _print_response(self, method, status, data, secs, body):
        stamp = time.strftime("%H:%M:%S")
        try:
            obj = json.loads(data)
        except ValueError:
            obj = {}
        if self.path.startswith("/api/predict") and status == 200:
            ev = obj.get("evaluation") or {}
            print(f"{stamp} | API {method} {self.path} {json.loads(body or b'{}')} -> {status} in {secs:.2f}s | "
                  f"trip={obj.get('trip_id')} fixes={len(obj.get('observation_arc_ids', []))} "
                  f"route_arcs={len(obj.get('predicted_arc_ids', []))} model={obj.get('model_version')} | "
                  + (" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in ev.items())
                     or str(obj.get("evaluation_note"))), flush=True)
        else:
            short = json.dumps(obj)[:300]
            print(f"{stamp} | API {method} {self.path} -> {status} in {secs:.2f}s | {short}", flush=True)

    def log_message(self, fmt, *args):  # API calls are already printed above, with their results
        if not self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True, help="API base URL on Laptop 1, e.g. http://192.168.1.23:8000")
    ap.add_argument("--port", type=int, default=5500)
    a = ap.parse_args()
    Handler.api_base = a.api.rstrip("/")
    try:  # quick connectivity check before the evaluator arrives
        with urllib.request.urlopen(Handler.api_base + "/api/health", timeout=5) as r:
            print("API reachable:", r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: cannot reach {Handler.api_base}/api/health yet ({e}). Check the IP, the Wi-Fi and the "
              f"firewall on Laptop 1.")
    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"PathSnap UI on http://localhost:{a.port}  (forwarding /api to {Handler.api_base})  host={socket.gethostname()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
