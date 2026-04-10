from world import DungeonWorld
from agents import Agent
from tracer import Tracer
from typing import Optional, Callable

# Thresholds for cross-agent diagnostics
COORDINATION_GAP_THRESHOLD = 10   # turns of silence before flagging
STAGNATION_THRESHOLD       = 5    # turns in same position before flagging


class GameLoop:
    def __init__(
        self,
        world: DungeonWorld,
        agent_a: Agent,
        agent_b: Agent,
        tracer: Tracer,
        max_turns: int = 100,
        on_event: Optional[Callable] = None,
        dm=None,
    ):
        self.world     = world
        self.agents    = {"A": agent_a, "B": agent_b}
        self.tracer    = tracer
        self.max_turns = max_turns
        self.on_event  = on_event
        self.dm        = dm

        self._pending_messages: dict[str, list[str]] = {"A": [], "B": []}

        # Repeated-failure tracking (same failing action in a row)
        self._repeat_failures: dict[str, int]          = {"A": 0, "B": 0}
        self._last_action:     dict[str, Optional[str]] = {"A": None, "B": None}

        # Coordination gap: turns since any inter-agent message was exchanged
        self._last_message_turn: int = -1

        # Stagnation: turns each agent has spent without changing position
        self._last_positions:   dict[str, Optional[list]] = {"A": None, "B": None}
        self._turns_stationary: dict[str, int]             = {"A": 0,    "B": 0}

    # ------------------------------------------------------------------

    def run(self) -> str:
        order = ["A", "B"]
        turn  = 0

        print(f"\n{'='*40}")
        print(f"Run {self.tracer.run_id[:8]} starting")
        print(self.world.render())
        print(f"{'='*40}\n")

        while turn < self.max_turns:
            agent_id = order[turn % 2]
            agent    = self.agents[agent_id]

            # Deliver queued messages
            messages_received                  = self._pending_messages[agent_id]
            self._pending_messages[agent_id]   = []

            # Snapshot BEFORE action — this is what the agent's belief is compared against
            obs          = self.world.observable_state(agent_id)
            world_before = self.world.world_snapshot()

            # DM records state before each turn
            if self.dm:
                self.dm.record(turn, world_before)

            # Update agent's cross-turn knowledge (landmarks, loop detection)
            agent.update_knowledge(obs)

            # Agent decides
            tool_call, prompt, raw_response, latency_ms = agent.decide(obs, messages_received)

            # Execute tool
            result      = self._execute_tool(agent_id, tool_call, agent, turn)
            agent.last_result = result
            world_after = self.world.world_snapshot()

            # Sync landmark memory across both agents after key events
            if tool_call.get("tool") == "pick_up" and result.get("item") == "key":
                # Key is off the floor — both agents should stop navigating to it
                for ag in self.agents.values():
                    ag.known_landmarks.pop("key", None)
            if tool_call.get("tool") == "use_item" and result.get("success"):
                # Key consumed, door open — both agents clear key and door from landmarks.
                # They should now navigate to exit only.
                for ag in self.agents.values():
                    ag.known_landmarks.pop("key", None)
                    ag.known_landmarks.pop("door", None)

            # ── Repeated-failure tracking ──────────────────────────────
            action_key = f"{tool_call['tool']}:{tool_call.get('args', {})}"
            if not result.get("success", True) and action_key == self._last_action[agent_id]:
                self._repeat_failures[agent_id] += 1
            else:
                self._repeat_failures[agent_id] = 0
            self._last_action[agent_id] = action_key

            # ── Stagnation tracking ────────────────────────────────────
            cur_pos = world_after["agent_positions"][agent_id]
            if self._last_positions[agent_id] == cur_pos:
                self._turns_stationary[agent_id] += 1
            else:
                self._turns_stationary[agent_id] = 0
            self._last_positions[agent_id] = cur_pos

            # ── Coordination gap ───────────────────────────────────────
            coordination_gap = (
                turn - self._last_message_turn
                if self._last_message_turn >= 0
                else turn
            )

            # ── Build extra divergences ────────────────────────────────
            extra_divergences = []
            if coordination_gap >= COORDINATION_GAP_THRESHOLD:
                extra_divergences.append({
                    "type":      "coordination_gap",
                    "gap_turns": coordination_gap,
                    "note":      f"No inter-agent communication for {coordination_gap} turns",
                })
            if self._turns_stationary[agent_id] >= STAGNATION_THRESHOLD:
                extra_divergences.append({
                    "type":             "agent_stagnation",
                    "agent":            agent_id,
                    "turns_stationary": self._turns_stationary[agent_id],
                    "position":         cur_pos,
                    "note":             f"Agent {agent_id} has not moved for {self._turns_stationary[agent_id]} turns",
                })

            belief_state = {**obs, "messages_received": messages_received}

            event = self.tracer.log_turn(
                turn=turn,
                agent=agent_id,
                phase="action",
                action=tool_call,
                result=result,
                belief_state=belief_state,
                world_truth=world_before,       # ← FIXED: belief vs truth at decision time
                llm_prompt=prompt,
                llm_response=raw_response,
                latency_ms=latency_ms,
                extra_divergences=extra_divergences,
                extra={
                    "repeat_failures":    self._repeat_failures[agent_id],
                    "coordination_gap":   coordination_gap,
                    "turns_stationary":   self._turns_stationary[agent_id],
                    "messages_in_flight": {k: list(v) for k, v in self._pending_messages.items()},
                },
            )

            print(
                f"Turn {turn:3d} | Agent {agent_id} | "
                f"{tool_call['tool']}({tool_call['args']}) → {result}"
            )

            # SSE — includes positions_after separately so the SVG trail is correct
            if self.on_event:
                self.on_event({
                    "type":           "turn",
                    **event,
                    "positions_after": world_after["agent_positions"],
                    "messages_in_flight": {k: list(v) for k, v in self._pending_messages.items()},
                })

            # Win
            if self._both_at_exit():
                print(f"\nBoth agents reached the exit on turn {turn}!")
                self.tracer.log_run_end("success", turn)
                if self.on_event:
                    self.on_event({"type": "run_end", "outcome": "success",
                                   "total_turns": turn, "run_id": self.tracer.run_id})
                return "success"

            if self.world.is_at_exit(agent_id):
                print(f"  Agent {agent_id} reached the exit (waiting for partner)")

            turn += 1

        print(f"\nTurn limit ({self.max_turns}) reached.")
        self.tracer.log_run_end("turn_limit", turn)
        if self.on_event:
            self.on_event({"type": "run_end", "outcome": "turn_limit",
                           "total_turns": turn, "run_id": self.tracer.run_id})
        return "turn_limit"

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, agent_id: str, tool_call: dict, agent: Agent, turn: int = 0) -> dict:
        tool = tool_call.get("tool", "look")
        args = tool_call.get("args", {})

        if tool == "move":
            return self.world.move(agent_id, args.get("direction", "north"))

        if tool == "look":
            return {"success": True, "observation": self.world.observable_state(agent_id)}

        if tool == "pick_up":
            return self.world.pick_up(agent_id)

        if tool == "use_item":
            return self.world.use_item(agent_id, args.get("item", ""), args.get("target", ""))

        if tool == "send_message":
            text      = args.get("text", "")
            recipient = "B" if agent_id == "A" else "A"
            self._pending_messages[recipient].append(f"[Agent {agent_id}]: {text}")
            self._last_message_turn = turn          # update coordination gap clock
            return {"success": True, "delivered_to": recipient, "on_turn": "next"}

        if tool == "ask_dm":
            return self._handle_ask_dm(agent_id, args.get("question", ""), turn)

        return {"success": False, "reason": f"Unknown tool '{tool}'"}

    def _handle_ask_dm(self, agent_id: str, question: str, turn: int) -> dict:
        if not self.dm:
            return {"success": False, "reason": "No Dungeon Master in this run"}

        current_truth = self.world.world_snapshot()
        dm_result     = self.dm.answer(agent_id, question, turn)

        # Populate the agent's landmark memory directly from the DM's stale snapshot.
        # This gives the agent structured coordinates to navigate with, not just text.
        # Uses stale data intentionally — divergences are still detected by the tracer.
        snap = dm_result.get("stale_snapshot", {})
        agent = self.agents[agent_id]
        if snap.get("key_location") and "key" not in agent.known_landmarks:
            agent.known_landmarks["key"] = snap["key_location"]
        if snap.get("door_location") and "door" not in agent.known_landmarks:
            agent.known_landmarks["door"] = snap["door_location"]
        if snap.get("exit_location") and "exit" not in agent.known_landmarks:
            agent.known_landmarks["exit"] = snap["exit_location"]

        dm_event = self.tracer.log_dm_interaction(
            turn=turn,
            asker=agent_id,
            question=question,
            dm_result=dm_result,
            current_truth=current_truth,
        )

        self._pending_messages[agent_id].append(
            f"[Dungeon Master, {dm_result['actual_staleness']} turns stale]: {dm_result['answer']}"
        )

        print(f"         DM → Agent {agent_id} (staleness={dm_result['actual_staleness']}): "
              f"{dm_result['answer'][:60]}…")

        if self.on_event:
            self.on_event({
                "type":                  "dm_interaction",
                "turn":                  turn,
                "asker":                 agent_id,
                "question":              question,
                "answer":                dm_result["answer"],
                "actual_staleness":      dm_result["actual_staleness"],
                "configured_staleness":  dm_result["configured_staleness"],
                "divergences":           dm_event["divergences"],
                "latency_ms":            dm_result["latency_ms"],
            })

        return {"success": True, "queued_for": agent_id, "on_turn": "next",
                "staleness": dm_result["actual_staleness"]}

    def _both_at_exit(self) -> bool:
        return self.world.is_at_exit("A") and self.world.is_at_exit("B")
