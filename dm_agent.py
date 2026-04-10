"""
Dungeon Master agent.

Has full board visibility but operates on a stale world snapshot — it sees
the dungeon as it was `stale_turns` turns ago. This produces the most
interesting class of failure: an explorer acts on DM advice that was true
when the DM last looked but is no longer true now.
"""

import os
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DM_SYSTEM = """You are the Dungeon Master overseeing a dungeon grid. You have full visibility of the board, but your observation is {staleness} turn(s) old — the world may have changed since then.

Your last known world state (observed {staleness} turn(s) ago):
  Agent A was at: {pos_a}
  Agent B was at: {pos_b}
  Key: {key_info}
  Door: {door_info}
  Exit: at {exit_pos}

Answer the explorer's question in 1-2 sentences. Be specific with coordinates [row, col]. Always note that your data is {staleness} turn(s) old."""


class DungeonMaster:
    def __init__(self, stale_turns: int = 5, model: str = "gpt-4o-mini"):
        self.stale_turns  = stale_turns
        self.model        = model
        # Each entry: {"turn": int, "snapshot": dict}
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # World state recording (called every game turn)
    # ------------------------------------------------------------------

    def record(self, turn: int, snapshot: dict):
        self._history.append({"turn": turn, "snapshot": snapshot})

    # ------------------------------------------------------------------
    # Stale snapshot retrieval
    # ------------------------------------------------------------------

    def get_stale_snapshot(self, current_turn: int) -> tuple[dict, int]:
        """
        Returns (snapshot, actual_staleness).
        Picks the snapshot from stale_turns ago; falls back to earliest
        available if not enough history yet.
        """
        if not self._history:
            return {}, 0

        target_turn = current_turn - self.stale_turns
        best = self._history[0]
        for entry in self._history:
            if entry["turn"] <= target_turn:
                best = entry

        actual_staleness = current_turn - best["turn"]
        return best["snapshot"], actual_staleness

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def answer(self, asker_id: str, question: str, current_turn: int) -> dict:
        """
        Generate a DM answer from the stale world snapshot.

        Returns a dict with:
          answer           — the DM's response text
          stale_snapshot   — the world state the DM used
          actual_staleness — how many turns old that snapshot is
          configured_staleness — what stale_turns is set to
          latency_ms
        """
        stale_snap, actual_staleness = self.get_stale_snapshot(current_turn)

        # Format snapshot for the prompt
        pos_a    = stale_snap.get("agent_positions", {}).get("A", "unknown")
        pos_b    = stale_snap.get("agent_positions", {}).get("B", "unknown")
        key_loc  = stale_snap.get("key_location")
        key_info = f"at {key_loc}" if key_loc else "not observed (may have been picked up)"
        door_locked = not stale_snap.get("door_unlocked", False)
        door_pos    = stale_snap.get("door_location", "unknown")
        door_info   = f"{'locked' if door_locked else 'unlocked'} at {door_pos}"
        exit_pos    = stale_snap.get("exit_location", "unknown")

        system_prompt = DM_SYSTEM.format(
            staleness=actual_staleness,
            pos_a=pos_a,
            pos_b=pos_b,
            key_info=key_info,
            door_info=door_info,
            exit_pos=exit_pos,
        )

        t0 = time.time()
        response = _client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Agent {asker_id} asks: {question}"},
            ],
            max_tokens=120,
            timeout=30,
        )
        latency_ms   = int((time.time() - t0) * 1000)
        answer_text  = response.choices[0].message.content.strip()

        return {
            "answer":                answer_text,
            "stale_snapshot":        stale_snap,
            "actual_staleness":      actual_staleness,
            "configured_staleness":  self.stale_turns,
            "latency_ms":            latency_ms,
        }
