"""Local helper the userscript talks to. Listens on 127.0.0.1 only.

GET /health                      -> {"ok": true, "voices": [...]}
GET /analyze?v=VIDEO_ID[&start=1] -> {"state": ..., "progress": 0..1, "result": {...}}

Requests must carry the header `X-Speaker-Skip: 1`. Browsers can't add a custom header
to a cross-site request without a CORS preflight (which this server never approves), so
random web pages can't trigger downloads; Tampermonkey's GM_xmlhttpRequest can.
"""

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import pipeline, speakers
from .audio import check_video_id

HOST, PORT = "127.0.0.1", 8787

_jobs = {}
_jobs_lock = threading.Lock()
_work_lock = threading.Lock()  # analyze one video at a time
_last_seen = 0.0  # last time the userscript talked to us


def _touch():
    global _last_seen
    now = time.time()
    if now - _last_seen > 600:
        print(time.strftime("%F %T"), "userscript connected", flush=True)
    _last_seen = now


def snapshot():
    """Current state for the menu-bar item."""
    with _jobs_lock:
        jobs = {vid: dict(j) for vid, j in _jobs.items()}
    active = [(vid, j) for vid, j in jobs.items() if j.get("state") not in ("done", "error", None)]
    running = [x for x in active if x[1].get("state") != "queued"]
    errors = [(vid, j) for vid, j in jobs.items() if j.get("state") == "error"]
    return {
        "active": (running or active or [None])[0],
        "queued": max(0, len(active) - 1),
        "error": errors[-1] if errors else None,
        "last_seen": _last_seen,
        "analyzed": len(list(pipeline.RESULTS_DIR.glob("*.json"))),
        "voices": sorted(speakers.load_voiceprints()),
    }


def _set(vid, **kw):
    with _jobs_lock:
        _jobs.setdefault(vid, {}).update(kw)


def _run(vid):
    try:
        with _work_lock:
            result = pipeline.analyze_video(vid, status=lambda s, p: _set(vid, state=s, progress=round(p, 3)))
        _set(vid, state="done", progress=1.0, result=result)
    except Exception as e:  # report to the userscript instead of dying
        traceback.print_exc()
        _set(vid, state="error", message=str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e))


def status(vid, start=False):
    cached = pipeline.load_result(vid)
    if cached:
        return {"state": "done", "progress": 1.0, "result": cached}
    with _jobs_lock:
        job = dict(_jobs.get(vid, {}))
        if start and job.get("state") in (None, "error"):
            _jobs[vid] = job = {"state": "queued", "progress": 0.0}
            threading.Thread(target=_run, args=(vid,), daemon=True).start()
    return job or {"state": "idle"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("X-Speaker-Skip") != "1":
            return self._send(403, {"error": "missing X-Speaker-Skip header"})
        _touch()
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/health":
            return self._send(200, {"ok": True, "voices": sorted(speakers.load_voiceprints())})
        if url.path == "/analyze":
            try:
                vid = check_video_id(q.get("v", [""])[0])
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(200, status(vid, start=q.get("start", ["0"])[0] == "1"))
        return self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass  # keep the log to errors only


def start_server():
    """Start the HTTP server on a background thread."""
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(time.strftime("%F %T"), f"speaker-skip helper on http://{HOST}:{PORT}", flush=True)
    return httpd


def serve(menubar=True):
    start_server()
    if menubar:
        from . import menubar as mb
        mb.run()  # blocks in the macOS event loop; shows the status-bar icon
    else:
        threading.Event().wait()
