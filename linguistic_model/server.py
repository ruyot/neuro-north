"""Local HTTP server for the mouse-driven linguistic pipeline simulator."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Dict
from urllib.parse import urlparse

from .decoder import DecoderError
from .simulator import PipelineSimulator


_WEB_ROOT = Path(__file__).with_name("web")


class SimulationHandler(SimpleHTTPRequestHandler):
    simulator = PipelineSimulator()
    lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(_WEB_ROOT), **kwargs)

    def _json(self, status: int, body: Dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/state":
            with self.lock:
                self._json(200, {"ok": True, "state": self.simulator.state()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/action":
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise DecoderError("request body is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise DecoderError("request body must be a JSON object")
            action = str(payload.pop("action", ""))
            with self.lock:
                self.simulator.dispatch(action, payload)
                state = self.simulator.state()
            self._json(200, {"ok": True, "state": state})
        except (DecoderError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Neuro North pipeline simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SimulationHandler)
    print("Pipeline simulator: http://%s:%d" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
