#!/usr/bin/env python3
"""The Studio: a local web UI for Jev's Bob Ross paintings.

Serves studio.html and the runs/ folder, and starts one bob_ross.py painting at a time
from a prompt typed in the browser. Standard library only.

  python3 studio_server.py                     # http://127.0.0.1:8000
  python3 studio_server.py --port 8080 --open  # another port, and open the browser
  python3 studio_server.py --daily-cap 1 --max-cost 0.25

Two styles. "Anything": a chat model on OpenRouter (default google/gemini-3.1-flash-lite) plans the prompt as
coloured shapes and Jev paints them. "Bob Ross landscape": the prompt steers Jev's composition
choices for the mountain-lake scene. Either way, Jev decides every block of the canvas.
Only listens on 127.0.0.1, and every paint request must carry this server's session token.
"""
import argparse
import datetime as dt
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
STUDIO_HTML = os.path.join(ROOT, "studio.html")
VIEWER_HTML = os.path.join(ROOT, "viewer.html")
META = "studio.json"   # written by this server into each run it starts
RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
PROMPT_MAX = 400

ABOUT = ("Describe anything. A chat model sketches it as flat coloured shapes, then Jev decides the colour "
         "of every block of the canvas from that sketch, big strokes first.")

MODES = [
    {"id": "anything", "label": "Anything",
     "hint": "A chat model plans the picture as shapes (a fraction of a cent), then Jev paints them."},
    {"id": "landscape", "label": "Bob Ross landscape",
     "hint": "Jev picks the time of day, mountain, clouds, trees and cabin from your words. Landscapes only."},
]
MODE_IDS = {m["id"] for m in MODES}


def has_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    try:
        with open(os.path.join(ROOT, ".env")) as f:
            return any(line.startswith("OPENROUTER_API_KEY=") and line.split("=", 1)[1].strip() for line in f)
    except OSError:
        return False


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_live(run_dir):
    """live.js is `window.__live = {...};` -- the painter's running numbers."""
    try:
        with open(os.path.join(run_dir, "live.js")) as f:
            text = f.read()
    except OSError:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def last_error(run_dir):
    """The last meaningful line of a failed run's log, for the gallery caption."""
    try:
        with open(os.path.join(run_dir, "log.txt"), errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return "The painter stopped without saying why."
    for line in reversed(lines):
        if not line.startswith(("Traceback", "File ", "^", "~")):
            return line[:300]
    return "The painter stopped without saying why."


class Studio:
    """One painting at a time, plus a gallery of everything the Studio has painted."""

    def __init__(self, args):
        self.args = args
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()  # re-entrant: start() reads the gallery while holding it
        self.proc = None
        self.current = None

    # ---- the current painting ----

    def poll(self):
        """Reap the painter if it has exited, and record how it ended."""
        with self.lock:
            if self.proc is None or self.proc.poll() is None:
                return
            run_dir = os.path.join(RUNS, self.current)
            meta = read_json(os.path.join(run_dir, META)) or {}
            meta["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            meta["exit_code"] = self.proc.returncode
            if self.proc.returncode != 0:
                meta["error"] = last_error(run_dir)
            write_json(os.path.join(run_dir, META), meta)
            self.proc, self.current = None, None

    def busy(self):
        self.poll()
        return self.proc is not None

    def start(self, prompt, size, mode="anything"):
        self.poll()
        spent = self.spent_today()  # outside the lock: it walks every run folder
        with self.lock:
            if self.proc is not None:
                raise StudioError(HTTPStatus.CONFLICT, "A painting is already at the easel. Wait for it to finish.")
            if self.args.daily_cap > 0 and spent >= self.args.daily_cap:
                raise StudioError(HTTPStatus.PAYMENT_REQUIRED,
                                  f"Today's cap of ${self.args.daily_cap:.2f} has been spent. Raise --daily-cap to paint more.")
            if not has_key():
                raise StudioError(HTTPStatus.SERVICE_UNAVAILABLE,
                                  "No OpenRouter key found. Put OPENROUTER_API_KEY=... in .env next to studio_server.py and restart.")
            # never let one painting take the rest of the day's budget past the cap
            max_cost = self.args.max_cost
            if self.args.daily_cap > 0:
                max_cost = min(max_cost, max(0.01, self.args.daily_cap - spent))

            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            run = f"studio-{stamp}-{secrets.token_hex(2)}"
            run_dir = os.path.join(RUNS, run)
            os.makedirs(run_dir)
            if os.path.exists(VIEWER_HTML):
                # the easel can show the viewer straight away; bob_ross.py overwrites it with the same file
                shutil.copyfile(VIEWER_HTML, os.path.join(run_dir, "index.html"))
            write_json(os.path.join(run_dir, META), {
                "prompt": prompt, "size": size, "mode": mode, "method": self.args.method, "max_cost": max_cost,
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            cmd = [sys.executable, os.path.join(ROOT, "bob_ross.py"), "--prompt", prompt, "--size", str(size),
                   "--method", self.args.method, "--max-cost", f"{max_cost:.4f}", "--run-dir", run_dir,
                   "--workers", str(self.args.workers)]
            if mode == "anything":
                cmd += ["--freeform", "--planner-model", self.args.planner_model]
            log = open(os.path.join(run_dir, "log.txt"), "w")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=env)
            log.close()  # the child has its own handle
            self.current = run
            print(f"[studio] painting {run} ({mode}): {size}px, cap ${max_cost:.2f}: {prompt!r}", flush=True)
            return run

    def stop(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()

    # ---- the gallery ----

    def runs(self):
        self.poll()
        out = []
        if not os.path.isdir(RUNS):
            return out
        for name in os.listdir(RUNS):
            run_dir = os.path.join(RUNS, name)
            meta = read_json(os.path.join(run_dir, META)) if os.path.isdir(run_dir) else None
            if not meta:
                continue  # only list paintings the Studio started
            result = read_json(os.path.join(run_dir, "run.json"))
            live = read_live(run_dir)
            running = name == self.current
            size = meta.get("size")
            image = None
            if result:
                for candidate in sorted(os.listdir(run_dir)):
                    if candidate.startswith("painting_x") and candidate.endswith(".png"):
                        image = candidate
                image = image or ("painting.png" if os.path.exists(os.path.join(run_dir, "painting.png")) else None)
            elif not running and os.path.exists(os.path.join(run_dir, "live.png")):
                image = "live.png"  # stopped part way: show what got painted
            cost = (result or {}).get("total_cost")
            if cost is None and live:
                cost = live.get("cost")
            accuracy = (result or {}).get("accuracy")
            out.append({
                "run": name,
                "prompt": meta.get("prompt"),
                "mode": meta.get("mode", "landscape"),
                "size": size,
                "created": meta.get("created"),
                "finished": not running,
                "error": meta.get("error"),
                "image": f"/runs/{name}/{image}" if image else None,
                "viewer": f"/runs/{name}/index.html",
                "cost": cost,
                "accuracy": accuracy,
            })
        out.sort(key=lambda r: r.get("created") or "", reverse=True)
        return out

    def spent_today(self):
        today = dt.date.today().isoformat()
        total = 0.0
        for r in self.runs():
            if (r.get("created") or "").startswith(today) and isinstance(r.get("cost"), (int, float)):
                total += r["cost"]
        return round(total, 6)


class StudioError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler(studio):
    class Handler(BaseHTTPRequestHandler):
        server_version = "JevStudio/1.0"

        def log_message(self, fmt, *args):
            if not self.path.startswith(("/api/status", "/api/runs", "/runs/")):
                sys.stderr.write("[http] " + (fmt % args) + "\n")

        # ---- helpers ----

        def send_json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_error_json(self, status, message):
            self.send_json({"error": message}, status)

        def send_file(self, path, no_store=False):
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except OSError:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if no_store else "max-age=60")
            self.end_headers()
            self.wfile.write(body)

        def local_origin(self):
            """Reject cross-site requests: the browser's Origin, when sent, must be this server."""
            origin = self.headers.get("Origin")
            if not origin:
                return True
            host = urlparse(origin).netloc
            return host == self.headers.get("Host")

        # ---- routes ----

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/studio", "/studio.html"):
                return self.send_file(STUDIO_HTML, no_store=True)
            if path == "/api/session":
                return self.send_json({
                    "token": studio.token,
                    "prompt_max": PROMPT_MAX,
                    "sizes": studio.args.sizes,
                    "default_size": studio.args.default_size,
                    "max_cost": studio.args.max_cost,
                    "daily_cap": studio.args.daily_cap,
                    "about": ABOUT,
                    "modes": MODES,
                    "default_mode": "anything",
                    "has_key": has_key(),
                })
            if path == "/api/status":
                busy = studio.busy()
                return self.send_json({
                    "busy": busy,
                    "run": studio.current if busy else None,
                    "spent_today": studio.spent_today(),
                    "daily_cap": studio.args.daily_cap,
                })
            if path == "/api/runs":
                return self.send_json({"runs": studio.runs()})
            if path.startswith("/runs/"):
                parts = [unquote(p) for p in path[len("/runs/"):].split("/")]
                if len(parts) != 2 or not all(RUN_ID.match(p) for p in parts) or parts[1].startswith("."):
                    return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
                live = parts[1] in ("live.png", "live.js", "replay.js", "index.html")
                return self.send_file(os.path.join(RUNS, parts[0], parts[1]), no_store=live)
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

        def do_POST(self):
            path = urlparse(self.path).path
            if not self.local_origin():
                return self.send_error_json(HTTPStatus.FORBIDDEN, "Requests from other sites are refused.")
            if self.headers.get("X-Studio-Token") != studio.token:
                return self.send_error_json(HTTPStatus.FORBIDDEN,
                                            "This page's session is out of date. Reload the page and try again.")
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self.send_error_json(HTTPStatus.BAD_REQUEST, "The request was not valid JSON.")

            if path == "/api/paint":
                prompt = " ".join(str(payload.get("prompt") or "").split())
                try:
                    size = int(payload.get("size"))
                except (TypeError, ValueError):
                    size = 0
                if not prompt:
                    return self.send_error_json(HTTPStatus.BAD_REQUEST, "Write a prompt first.")
                if len(prompt) > PROMPT_MAX:
                    return self.send_error_json(HTTPStatus.BAD_REQUEST, f"Keep the prompt under {PROMPT_MAX} characters.")
                if size not in studio.args.sizes:
                    return self.send_error_json(HTTPStatus.BAD_REQUEST, "Pick one of the canvas sizes offered.")
                mode = str(payload.get("mode") or "anything")
                if mode not in MODE_IDS:
                    return self.send_error_json(HTTPStatus.BAD_REQUEST, "Pick one of the styles offered.")
                try:
                    run = studio.start(prompt, size, mode)
                except StudioError as e:
                    return self.send_error_json(e.status, e.message)
                return self.send_json({"run": run, "viewer": f"/runs/{run}/index.html"})
            if path == "/api/stop":
                studio.stop()
                return self.send_json({"stopped": True})
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sizes", default="64,128,256", help="canvas sizes offered, comma separated (multiples of 16)")
    ap.add_argument("--default-size", type=int, default=128)
    ap.add_argument("--method", choices=("blocks", "pixels"), default="blocks",
                    help="blocks is about 5x cheaper than one decision per pixel")
    ap.add_argument("--max-cost", type=float, default=0.50, help="hard cap in USD for each painting")
    ap.add_argument("--daily-cap", type=float, default=2.00, help="stop accepting paintings after this much today (0 = no cap)")
    ap.add_argument("--planner-model", default="google/gemini-3.1-flash-lite",
                    help='OpenRouter model that plans "Anything" paintings as shapes')
    ap.add_argument("--workers", type=int, default=16, help="Jev requests in flight per painting (more = faster)")
    ap.add_argument("--open", action="store_true", help="open the Studio in your browser")
    args = ap.parse_args()
    args.sizes = sorted({int(s) for s in args.sizes.split(",") if s.strip()})
    if any(s % 16 for s in args.sizes):
        ap.error("every size must be a multiple of 16 (the blocks method starts from 16px blocks)")
    if args.default_size not in args.sizes:
        args.default_size = args.sizes[0]

    os.makedirs(RUNS, exist_ok=True)
    studio = Studio(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(studio))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"The Studio is open at {url}")
    print(f"  each painting capped at ${args.max_cost:.2f}, ${args.daily_cap:.2f} a day; method: {args.method}")
    if not has_key():
        print("  WARNING: no OPENROUTER_API_KEY in the environment or .env -- Paint will refuse until you add one")
    print("  Ctrl-C to close")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nclosing the Studio")
    finally:
        studio.stop()
        server.server_close()


if __name__ == "__main__":
    main()
