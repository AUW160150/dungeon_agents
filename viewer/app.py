"""
Flask viewer — serves the legibility layer.

Usage:
    cd viewer && python app.py
    or from project root: python -m viewer.app
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, abort

app = Flask(__name__)

RUNS_DIR = Path(__file__).parent.parent / "runs"


def _list_runs():
    if not RUNS_DIR.exists():
        return []
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            events = data.get("events", [])
            end_event = next((e for e in reversed(events) if e.get("phase") == "run_end"), None)
            runs.append({
                "run_id":     data.get("run_id", f.stem),
                "filename":   f.name,
                "outcome":    end_event.get("outcome", "unknown") if end_event else "unknown",
                "total_turns": end_event.get("total_turns", len(events)) if end_event else len(events),
                "event_count": len(events),
            })
        except Exception:
            continue
    return runs


@app.route("/")
def index():
    runs = _list_runs()
    return render_template("index.html", runs=runs)


@app.route("/run/<run_id>")
def run_detail(run_id: str):
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        # Try full filename prefix match
        matches = list(RUNS_DIR.glob(f"{run_id}*.json"))
        if not matches:
            abort(404)
        path = matches[0]

    with open(path) as f:
        data = json.load(f)
    return render_template("run.html", run_id=data["run_id"], data_json=json.dumps(data))


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/api/run/<run_id>")
def api_run(run_id: str):
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        matches = list(RUNS_DIR.glob(f"{run_id}*.json"))
        if not matches:
            abort(404)
        path = matches[0]
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/download/<run_id>")
def download(run_id: str):
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        matches = list(RUNS_DIR.glob(f"{run_id}*.json"))
        if not matches:
            abort(404)
        path = matches[0]
    return send_file(path, as_attachment=True, download_name=path.name)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
