"""
Flask viewer — live two-panel dungeon observer.

Usage:
    python viewer/app.py        (from project root)
"""

import json
import queue
import random
import sys
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, abort, Response, request

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
RUNS_DIR = Path(__file__).parent.parent / "runs"

# ── Global SSE state (one run at a time) ─────────────────────────────────────
_event_queue: queue.Queue = queue.Queue()
_run_state: dict = {"active": False, "run_id": None}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Simulation control ────────────────────────────────────────────────────────

@app.route("/start", methods=["POST"])
def start_run():
    if _run_state["active"]:
        return jsonify({"error": "A run is already in progress"}), 409

    data = request.get_json(silent=True) or {}
    seed           = data.get("seed")           or random.randint(0, 9999)
    max_turns      = data.get("max_turns")      or 100
    dm_stale_turns = data.get("dm_stale_turns") or 5

    # Drain any stale events from a previous run
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except queue.Empty:
            break

    _run_state["active"] = True
    _run_state["run_id"] = None

    def _run():
        from run import simulate
        try:
            run_id, outcome, path = simulate(
                seed=seed,
                max_turns=max_turns,
                dm_stale_turns=dm_stale_turns,
                on_event=_event_queue.put,
            )
            _run_state["run_id"] = run_id
        finally:
            _run_state["active"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "seed": seed, "max_turns": max_turns})


# ── SSE stream ────────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("analysis", "analysis_error"):
                    break
            except queue.Empty:
                # Keep connection alive; stop if run finished and queue is empty
                if not _run_state["active"]:
                    break
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Run history & download ────────────────────────────────────────────────────

@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/download/<run_id>")
def download(run_id: str):
    path = _resolve_run_path(run_id)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_run_path(run_id: str) -> "Path | None":
    direct = RUNS_DIR / f"{run_id}.json"
    if direct.exists():
        return direct
    matches = list(RUNS_DIR.glob(f"{run_id}*.json"))
    return matches[0] if matches else None


def _list_runs() -> list:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            events  = data.get("events", [])
            end_evt = next((e for e in reversed(events) if e.get("phase") == "run_end"), None)
            runs.append({
                "run_id":       data.get("run_id", f.stem),
                "filename":     f.name,
                "outcome":      end_evt.get("outcome", "unknown") if end_evt else "unknown",
                "total_turns":  end_evt.get("total_turns", len(events)) if end_evt else len(events),
                "divergences":  sum(1 for e in events if e.get("divergences")),
            })
        except Exception:
            continue
    return runs


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
