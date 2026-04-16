# Bug Log

## Main Bug — BFS Infinite Loop on Door Navigation

**File:** `agents.py` — `_bfs_next_step()`

**Symptom:** Game loop and browser both froze completely (no terminal output, no SSE events).
Occurred reliably after one agent picked up the key and the other was near/at the exit, or
whenever either agent had the door in `known_landmarks` and needed to navigate toward it.

**Root cause:** `_bfs_next_step` treated `'D'` (locked door) as impassable for all cells,
including the goal cell itself. When `known_landmarks["door"] = [r, c]` and `known_map[(r,c)] = 'D'`,
the BFS filtered out the goal on every expansion. With no bounds on the search space
(unknown cells are treated as passable), the BFS queue expanded infinitely across
all integer coordinates and never returned.

This correlated with key pickup because: after pickup, `"key"` is removed from
`known_landmarks`, leaving `"door"` as the primary navigation target. The next call
to `_format_observation()` triggered the infinite BFS.

**Fix:**
```python
# Before (hung forever when goal was a door cell):
if cell in ("#", "D"):
    continue

# After (door is only impassable when it is NOT the goal):
if cell == "#" or (cell == "D" and npos != goal):
    continue
```

Also added `MAX_VISITED = 400` cap as a safety net against any future
unreachable-goal scenarios (400 >> reachable cells on a 10×10 grid).

---

## Other Bugs Fixed This Session

### 1. Stale Analysis Event Breaking SSE Stream

**File:** `viewer/app.py`

**Symptom:** Browser appeared frozen during a new run while the game loop kept
executing normally in the terminal. Seemed to correlate with key pickup / successful
events (which triggered run completion and then analysis in a prior run).

**Root cause:** Post-run LLM analysis runs in a background thread (~10–30s after run
ends). If the user starts a new run before the old analysis thread finishes, the old
`{"type": "analysis", "run_id": "<old_id>"}` event fires into the shared
`_event_queue`. The SSE generator for the new run consumed it and hit the unguarded
`break` condition, closing the browser connection while the game loop kept running.

**Fix:**
1. Set `_run_state["run_id"]` immediately when `world_init` fires (start of run),
   not after `simulate()` returns.
2. Guard the SSE break with a run_id match:
```python
if event.get("type") in ("analysis", "analysis_error"):
    if event.get("run_id") == _run_state.get("run_id"):
        break
    # Stale event from a previous run — discard, keep streaming
```

---

### 2. Silent 30-Second Freezes from LLM Retry Loop

**Files:** `agents.py`, `dm_agent.py`, `analysis.py`

**Symptom:** Occasional ~30-second silences in the terminal with no output at all,
then either recovery or error.

**Root cause:** OpenAI Python SDK defaults to `max_retries=2`. With `timeout=10` per
attempt, a single failed or timed-out LLM request could take up to 30 seconds
(3 attempts × 10s). Combined with stdout buffering (see bug 3), no output appeared
during the entire retry window.

**Fix:** Set `max_retries=0` on all three OpenAI clients so failures surface
immediately rather than silently retrying:
```python
_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=12.0,
    max_retries=0,
)
```

---

### 3. stdout Buffering Hiding Debug Prints

**Files:** `agents.py`, `game_loop.py`

**Symptom:** Debug prints (e.g. `[agent A] calling LLM`) sometimes did not appear in
the terminal even though they were reached in code, making it look like the freeze
was earlier in the call stack than it actually was.

**Root cause:** Python's `print()` buffers stdout when the output is not a TTY
(e.g. when running under Flask's dev server). Prints would be held in the OS buffer
and only flushed in a burst, or not at all if the process was stuck.

**Fix:** Added `flush=True` to all diagnostic prints in the hot path:
```python
print(f"  [agent {self.agent_id}] calling LLM (turn {self.turn_count})…", flush=True)
print(f"  [dbg] T{turn} pre-decide agent={agent_id}", flush=True)
# etc.
```

---

### 4. Langfuse Flush Blocking the Game Loop Thread

**File:** `tracer.py`

**Symptom:** Game loop would pause at run end while Langfuse flushed buffered
observations to the remote server over the network.

**Fix:** Run the flush in a daemon background thread:
```python
threading.Thread(target=_lf.flush, daemon=True).start()
```

---

### 5. Post-Run Analysis Blocking the Game Loop Thread

**File:** `run.py`

**Symptom:** The SSE `run_end` event was delayed and the browser showed the run as
still active while LLM analysis (~10–30s) completed synchronously.

**Fix:** Moved LLM analysis into a background daemon thread. The trace is saved and
`simulate()` returns immediately; the analysis panel in the browser fills in
asynchronously when the background thread completes.
