# Dungeon Agents

A two-agent dungeon simulation built to explore structured observability and failure diagnosis in multi-agent LLM systems. The dungeon is not the point — the traces are.

**Model:** `gpt-4o-mini` (OpenAI function calling)  
**Framework:** Custom game loop, no LangChain/LangGraph  
**Observability:** Langfuse (optional) + structured local JSON traces  
**Viewer:** Flask + SSE + SVG, no frontend build step

→ **Demo:** [Watch a live run walkthrough (Loom)](https://www.loom.com/share/b1204499c75c4c7bac93688cdc09ded9)  
→ See [FAQ.md](FAQ.md) for common questions: how the agents navigate, what the DM does, how success/failure is classified, and what tools are called when.

---

## What it does

Two explorer agents (A and B) navigate a 10×10 fog-of-war dungeon. One must find a key, unlock a door, then both must reach the exit. Agents take turns, communicate with 1-turn message delay, and can query a Dungeon Master agent that sees the full map but works from a stale world snapshot.

Every agent decision is logged as a structured event capturing: what the agent believed, what was actually true, what action it took, what the result was, and where the two diverged. After each run, an LLM generates a structured incident report answering: what happened, why it happened, and what should change.

---

## Setup

```bash
# Create virtualenv and install dependencies
python3 -m venv dun
source dun/bin/activate
pip install openai flask python-dotenv langfuse

# Add credentials
cp .env.example .env
# Fill in OPENAI_API_KEY (required)
# Fill in LANGFUSE_* keys (optional — Langfuse is silently disabled if absent)
```

`.env` format:
```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

## Running

**Web viewer (recommended):**
```bash
python viewer/app.py
# Open http://localhost:5050
```

Set seed, max turns, and DM staleness in the UI, then click Run. The dungeon animates live, the trace log scrolls as events arrive, and a post-run analysis panel appears when the run finishes. Use **⏸ Pause** to freeze execution mid-run (no API calls consumed while paused) and **⏹ Stop** to end the run early and trigger analysis.

**CLI:**
```bash
python run.py                       # random seed
python run.py --seed 42             # deterministic
python run.py --runs 5              # batch
python run.py --max-turns 50 --dm-stale 10
```

Traces are saved to `runs/<run_id[:8]>.json`.

---

## Architecture

```
world.py        — DungeonWorld: grid, fog of war, item/door logic
agents.py       — Agent: LLM decision via function calling, cross-turn landmark/map state
dm_agent.py     — DungeonMaster: full-map view, configurable staleness
game_loop.py    — GameLoop: turn order, tool dispatch, cross-agent diagnostics
tracer.py       — Tracer: event logging, divergence detection, Langfuse spans
analysis.py     — post-run LLM compression + GPT incident report
run.py          — simulate(): wires everything together, emits SSE events
viewer/app.py   — Flask server, SSE stream, pause/stop endpoints
viewer/templates/index.html — two-panel live observer with run controls
AI_convo/       — full Claude Code conversation transcripts (submission requirement)
```

### Why no framework

LangChain/LangGraph would have obscured the parts that matter here: the event model, the divergence detection, and the boundary between what an agent believes and what is true. Custom tool dispatch means every decision point is explicit and traceable. The game loop is ~250 lines and easy to reason about.

### Why stateless LLM calls (with Python-side memory)

Each LLM call gets only the current observable state — no conversation history carried across turns. This keeps each call independently traceable and the context window predictable. However, the Python `Agent` object does accumulate cross-turn state: a `known_map` of every cell ever seen (used for BFS pathfinding around walls), `known_landmarks` (key/door/exit coordinates from DM or observation), and recent position history for loop detection. The LLM is stateless; the harness is not. This split is intentional — it means the agent's "memory" is explicit, inspectable, and logged, rather than buried in a growing context window.

---

## Event schema

Each event in the trace has this shape:

```json
{
  "run_id": "...",
  "event_id": "...",
  "turn": 12,
  "agent": "A",
  "phase": "action",
  "timestamp": "...",
  "action": { "tool": "move", "args": { "direction": "north" } },
  "result": { "success": true, "new_position": [3, 4] },
  "belief_state": {
    "position": [4, 4],
    "inventory": [],
    "visible_cells": { "3,4": ".", "4,3": "#", ... },
    "door_unlocked": false,
    "messages_received": []
  },
  "world_truth": {
    "agent_positions": { "A": [4, 4], "B": [7, 2] },
    "key_location": [2, 8],
    "door_unlocked": false,
    ...
  },
  "divergences": [],
  "latency_ms": 340,
  "extra": {
    "repeat_failures": 0,
    "coordination_gap": 14,
    "turns_stationary": 0,
    "messages_in_flight": { "A": [], "B": [] }
  }
}
```

### Schema design decisions

**`belief_state` vs `world_truth` at the same timestamp**

The key structural choice is pairing what the agent could see with what was actually true at the moment of decision — not after. `world_truth` is captured before the tool executes. This makes divergences meaningful: a position mismatch means the agent was navigating from a wrong belief, not from a stale snapshot caused by its own action. Without this, ~30% of divergences in early runs were false positives.

**`divergences` as a first-class field**

Divergences are computed inline at log time and stored on every event. Types detected:

| type | what it catches |
|---|---|
| `position_mismatch` | Agent believes it's somewhere it isn't |
| `inventory_mismatch` | Agent's belief about what it's carrying is wrong |
| `door_state_mismatch` | Agent thinks door is locked/unlocked, reality differs |
| `stale_key_belief` | Agent's visible cells show the key, but it's already been picked up |
| `coordination_gap` | No inter-agent message for N turns (threshold: 10) |
| `agent_stagnation` | Agent hasn't moved for N turns (threshold: 5) |

The last two are cross-agent diagnostics computed in the game loop and injected as extra divergences. They're the most interesting class of failure: locally consistent behaviour that's globally broken. An agent calling `look` from the same cell 50 turns in a row has correct beliefs at each step — but the trace should still surface it.

**DM interactions as separate events**

Dungeon Master interactions are logged as `agent: "DM"` events with their own divergence detection: `dm_stale_key_location`, `dm_stale_door_state`, `dm_stale_agent_position`. These capture the "reasonable decision, wrong information" failure class — an agent asked the DM where the key was, the DM gave a confident answer, and the answer was already false. The staleness of that answer (in turns) is recorded alongside whether the information was actually wrong.

**`extra` for diagnostics that don't fit the schema**

`coordination_gap` and `turns_stationary` are also stored as raw integers in `extra` so queries and the analysis layer can use them without parsing divergence arrays.

**Run-end event**

The final event in every trace has `phase: "run_end"` and an `outcome` field. Possible values:

| outcome | meaning |
|---|---|
| `success` | Both agents reached the exit |
| `turn_limit` | Max turn cap reached before success |
| `stopped` | User manually stopped the run via the viewer |
| `stuck` | Reserved; not currently emitted |

**What is not stored locally**

`llm_prompt` and `llm_response` (the raw LLM input/output strings) are passed to Langfuse as generation spans but are not written to the local JSON trace. This keeps trace files a manageable size. The prompt can be reconstructed from `belief_state` and the system prompt in `agents.py`.

---

## Legibility layer

The viewer is built around three questions from the spec: what happened, why, what should change.

**Left panel — trace log**

Events scroll in real time. Each entry shows turn, agent, tool, args, result. The left border signals failure type at a glance: yellow for belief/reality divergence, red for a repeated failing action. Coordination gap and stagnation divergences render with counts rather than raw JSON. DM interactions appear inline as question/answer pairs with a staleness badge that turns red when the DM's information was factually wrong.

**Right panel — live dungeon**

SVG grid with fading agent trails. The trail uses post-action positions so it accurately reflects where agents moved, not where they were before the action. Key and door cells update visually when picked up or unlocked. The DM badge pulses on interaction. Fog of war is not visualised — the agents don't have it, you do.

**Status bar — milestones**

Key found, door unlocked, and each agent reaching the exit are surfaced as timeline badges so you can tell at a glance how far a run got before failing.

**Analysis panel — post-run incident report**

After each run, the event log is compressed into a summary (tool usage, milestones, failures, messages sent, coordination gap, DM interactions) and fed to GPT with a post-mortem prompt. The model returns a structured JSON with four fields: `what_happened`, `why_it_happened`, `failure_type` (`single_agent / interaction / emergent / success`), and `what_should_change`. This appears at the bottom of the viewer after the run ends.

The failure type classification was deliberately kept to four categories matching the spec's diagnostic questions: single-agent decision failure, interaction failure between agents, failure emergent from system design, or success.

---

## Failure classes the traces surface

Three distinct failure modes appear in the included runs:

**1. Stagnation (single agent)**  
An agent calls `look` repeatedly from the same cell. Technically correct beliefs at each step — fog of war is consistent. Caught by `agent_stagnation` divergence when `turns_stationary >= 5`. The LLM analysis classifies this as `single_agent` and typically identifies it as a prompt/tool-design failure: the agent needed stronger pressure to move when it has no new information.

**2. Coordination gap (interaction)**  
Both agents explore independently and never message each other. One finds the key; the other wanders near the exit with no way to progress. Caught by `coordination_gap` divergence when no message has been exchanged for 10+ turns. The LLM analysis classifies this as `interaction` and suggests adding a required communication step or shared memory.

**3. Stale DM information (emergent)**  
An agent asks the DM where the key is. The DM answers based on a 5-turn-old snapshot — the key has already been picked up. The agent navigates to the reported location and finds nothing. Caught by `dm_stale_key_location` divergence on the DM event. The LLM analysis classifies this as `emergent` (the staleness is a system design parameter, not either agent's fault in isolation).

---

## Trace files

Saved to `runs/`. Each file contains the full event log plus the post-run analysis:

```json
{
  "run_id": "...",
  "events": [...],
  "analysis": {
    "what_happened": "...",
    "why_it_happened": "...",
    "failure_type": "single_agent",
    "what_should_change": "- ...\n- ...\n- ...",
    "trace_summary": "..."
  }
}
```

`trace_summary` is what was fed to the LLM — included for transparency and reproducibility.

---

## Files not committed

- `.env` — credentials
- `dun/` — virtualenv
- `__pycache__/`

---

## Submission

- `runs/` — 13 trace files (mix of successes, turn-limit failures, and interesting failure modes)
- `AI_convo/` — full Claude Code conversation transcripts showing the full development session
