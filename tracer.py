import os
import uuid
import time
import json
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Langfuse is optional — if keys are missing we still log locally
try:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"),
    )
    # v3 auth check — disables silently if creds are wrong
    if not _lf._tracing_enabled:
        raise RuntimeError("Langfuse tracing disabled (check credentials)")
    LANGFUSE_ENABLED = True
except Exception as e:
    print(f"[tracer] Langfuse unavailable: {e}")
    _lf = None
    LANGFUSE_ENABLED = False


class Tracer:
    """
    Records one structured event per agent turn.
    Also sends LLM spans to Langfuse when available.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.events: list[dict] = []
        # Langfuse v3: root span represents the full run
        self._lf_span = None

        if LANGFUSE_ENABLED:
            self._lf_span = _lf.start_observation(
                name="dungeon-run",
                as_type="span",
                input={"run_id": self.run_id},
                metadata={"run_id": self.run_id},
            )

    # ------------------------------------------------------------------
    # Core event logging
    # ------------------------------------------------------------------

    def log_turn(
        self,
        *,
        turn: int,
        agent: str,
        phase: str,                     # "decision" | "action" | "message_delivery"
        action: Optional[dict],         # {"tool": ..., "args": ...}
        result: Optional[dict],         # tool execution result
        belief_state: dict,             # what the agent can see / thinks is true
        world_truth: dict,              # actual world snapshot
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
        latency_ms: Optional[int] = None,
        extra_divergences: Optional[list] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        divergences = _detect_divergences(belief_state, world_truth, agent)
        if extra_divergences:
            divergences.extend(extra_divergences)

        event = {
            "run_id": self.run_id,
            "event_id": str(uuid.uuid4()),
            "turn": turn,
            "agent": agent,
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "result": result,
            "belief_state": belief_state,
            "world_truth": world_truth,
            "divergences": divergences,   # [] when beliefs match reality
            "latency_ms": latency_ms,
        }
        if extra:
            event["extra"] = extra

        self.events.append(event)

        # Send LLM generation span to Langfuse v3
        if LANGFUSE_ENABLED and llm_prompt and llm_response:
            obs = _lf.start_observation(
                name=f"turn-{turn}-agent-{agent}",
                as_type="generation",
                input=llm_prompt,
                output=llm_response,
                model="gpt-4o-mini",
                metadata={
                    "turn": turn,
                    "agent": agent,
                    "action": action,
                    "divergences": divergences,
                    "latency_ms": latency_ms,
                },
            )
            obs.end()

        return event

    def log_run_end(self, outcome: str, total_turns: int, metadata: Optional[dict] = None):
        """outcome: 'success' | 'turn_limit' | 'stuck' | 'stopped'"""
        summary = {
            "run_id": self.run_id,
            "event_id": str(uuid.uuid4()),
            "turn": total_turns,
            "agent": "system",
            "phase": "run_end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "total_turns": total_turns,
            "action": None,
            "result": None,
            "belief_state": {},
            "world_truth": {},
            "divergences": [],
            "latency_ms": None,
        }
        if metadata:
            summary["extra"] = metadata
        self.events.append(summary)

        if LANGFUSE_ENABLED and self._lf_span:
            self._lf_span.end()
            # Flush in background — sync flush blocks the game loop thread
            import threading
            threading.Thread(target=_lf.flush, daemon=True).start()

    # ------------------------------------------------------------------
    # DM interaction logging
    # ------------------------------------------------------------------

    def log_dm_interaction(
        self,
        *,
        turn: int,
        asker: str,
        question: str,
        dm_result: dict,
        current_truth: dict,
    ) -> dict:
        """
        Log a DM question/answer pair.
        Detects where the DM's stale snapshot diverges from current world truth
        — these are the "reasonable decision, wrong info" moments.
        """
        divergences = _detect_dm_divergences(dm_result["stale_snapshot"], current_truth)

        event = {
            "run_id":     self.run_id,
            "event_id":   str(uuid.uuid4()),
            "turn":       turn,
            "agent":      "DM",
            "phase":      "dm_response",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "action":     {"tool": "ask_dm", "asker": asker, "question": question},
            "result":     {"answer": dm_result["answer"]},
            "belief_state": {},
            "world_truth":  current_truth,
            "divergences":  divergences,
            "latency_ms":   dm_result["latency_ms"],
            "extra": {
                "actual_staleness":     dm_result["actual_staleness"],
                "configured_staleness": dm_result["configured_staleness"],
                "stale_snapshot":       dm_result["stale_snapshot"],
            },
        }
        self.events.append(event)

        if LANGFUSE_ENABLED:
            obs = _lf.start_observation(
                name=f"dm-turn-{turn}-asked-by-{asker}",
                as_type="generation",
                input=question,
                output=dm_result["answer"],
                model="gpt-4o-mini",
                metadata={
                    "turn": turn,
                    "asker": asker,
                    "actual_staleness": dm_result["actual_staleness"],
                    "divergences": divergences,
                },
            )
            obs.end()

        return event

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "events": self.events,
            }, f, indent=2)
        print(f"[tracer] Saved {len(self.events)} events → {path}")


# ------------------------------------------------------------------
# Divergence detection
# ------------------------------------------------------------------

def _detect_divergences(belief: dict, truth: dict, agent: str) -> list[dict]:
    """
    Compare what the agent believes vs world truth.
    Returns a list of divergence objects — empty list means beliefs are accurate.
    """
    divergences = []

    # Agent position belief vs truth
    believed_pos = belief.get("position")
    true_pos = truth["agent_positions"].get(agent)
    if believed_pos and true_pos and list(believed_pos) != list(true_pos):
        divergences.append({
            "type": "position_mismatch",
            "believed": believed_pos,
            "truth": true_pos,
        })

    # Inventory belief vs truth
    believed_inv = set(belief.get("inventory", []))
    true_inv = set(truth["inventories"].get(agent, []))
    if believed_inv != true_inv:
        divergences.append({
            "type": "inventory_mismatch",
            "believed": list(believed_inv),
            "truth": list(true_inv),
        })

    # Door state belief vs truth
    if "door_unlocked" in belief and belief["door_unlocked"] != truth["door_unlocked"]:
        divergences.append({
            "type": "door_state_mismatch",
            "believed": belief["door_unlocked"],
            "truth": truth["door_unlocked"],
        })

    # Key location — agent saw key at a cell, but key has moved
    # (detected when agent believes key exists but world says it's gone)
    believed_key_in_view = any(
        v == "K" for v in belief.get("visible_cells", {}).values()
    )
    key_actually_exists = truth["key_location"] is not None
    if believed_key_in_view and not key_actually_exists:
        divergences.append({
            "type": "stale_key_belief",
            "believed": "key visible",
            "truth": "key already picked up",
        })

    return divergences


def _detect_dm_divergences(stale: dict, truth: dict) -> list[dict]:
    """
    Compare the DM's stale world snapshot against current truth.
    These divergences show what the DM got wrong — and therefore what
    advice it may have given that was factually incorrect.
    """
    divergences = []

    # Key location
    s_key = stale.get("key_location")
    t_key = truth.get("key_location")
    if s_key != t_key:
        divergences.append({
            "type": "dm_stale_key_location",
            "stale": s_key,
            "truth": t_key,
            "note": "Key was picked up after DM's last observation" if s_key and not t_key else "",
        })

    # Door state
    s_door = stale.get("door_unlocked", False)
    t_door = truth.get("door_unlocked", False)
    if s_door != t_door:
        divergences.append({
            "type": "dm_stale_door_state",
            "stale": "unlocked" if s_door else "locked",
            "truth": "unlocked" if t_door else "locked",
        })

    # Agent positions
    for aid in ["A", "B"]:
        s_pos = stale.get("agent_positions", {}).get(aid)
        t_pos = truth.get("agent_positions", {}).get(aid)
        if s_pos and t_pos and list(s_pos) != list(t_pos):
            divergences.append({
                "type": f"dm_stale_agent_{aid}_position",
                "stale": s_pos,
                "truth": t_pos,
            })

    return divergences
