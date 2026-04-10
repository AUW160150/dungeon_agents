from world import DungeonWorld
from agents import Agent
from tracer import Tracer
from typing import Optional, Callable


class GameLoop:
    def __init__(
        self,
        world: DungeonWorld,
        agent_a: Agent,
        agent_b: Agent,
        tracer: Tracer,
        max_turns: int = 100,
        on_event: Optional[Callable] = None,
    ):
        self.world = world
        self.agents = {"A": agent_a, "B": agent_b}
        self.tracer = tracer
        self.max_turns = max_turns
        self.on_event = on_event

        # Pending messages: dict[recipient_id] -> list of message strings
        self._pending_messages: dict[str, list[str]] = {"A": [], "B": []}
        # Track consecutive repeated failures per agent for stuck detection
        self._repeat_failures: dict[str, int] = {"A": 0, "B": 0}
        self._last_action: dict[str, Optional[str]] = {"A": None, "B": None}

    # ------------------------------------------------------------------

    def run(self) -> str:
        order = ["A", "B"]
        turn = 0

        print(f"\n{'='*40}")
        print(f"Run {self.tracer.run_id[:8]} starting")
        print(self.world.render())
        print(f"{'='*40}\n")

        while turn < self.max_turns:
            agent_id = order[turn % 2]
            agent = self.agents[agent_id]

            # Deliver pending messages to this agent
            messages_received = self._pending_messages[agent_id]
            self._pending_messages[agent_id] = []

            # Capture state BEFORE action (agent's belief at decision time)
            obs = self.world.observable_state(agent_id)

            # Agent decides
            tool_call, prompt, raw_response, latency_ms = agent.decide(obs, messages_received)

            # Execute tool
            result = self._execute_tool(agent_id, tool_call, agent)
            agent.last_result = result
            world_after = self.world.world_snapshot()

            # Detect stuck pattern: same failing tool 3+ times in a row
            action_key = f"{tool_call['tool']}:{tool_call.get('args', {})}"
            if not result.get("success", True) and action_key == self._last_action[agent_id]:
                self._repeat_failures[agent_id] += 1
            else:
                self._repeat_failures[agent_id] = 0
            self._last_action[agent_id] = action_key

            belief_state = {
                **obs,
                "messages_received": messages_received,
            }

            event = self.tracer.log_turn(
                turn=turn,
                agent=agent_id,
                phase="action",
                action=tool_call,
                result=result,
                belief_state=belief_state,
                world_truth=world_after,
                llm_prompt=prompt,
                llm_response=raw_response,
                latency_ms=latency_ms,
                extra={
                    "repeat_failures": self._repeat_failures[agent_id],
                    "messages_in_flight": {
                        k: list(v) for k, v in self._pending_messages.items()
                    },
                },
            )

            print(
                f"Turn {turn:3d} | Agent {agent_id} | "
                f"{tool_call['tool']}({tool_call['args']}) → {result}"
            )

            # Emit SSE event
            if self.on_event:
                self.on_event({
                    "type": "turn",
                    **event,
                    # Include pending message queue so UI can show in-flight messages
                    "messages_in_flight": {
                        k: list(v) for k, v in self._pending_messages.items()
                    },
                })

            # Win condition
            if self._both_at_exit():
                print(f"\nBoth agents reached the exit on turn {turn}!")
                self.tracer.log_run_end("success", turn)
                if self.on_event:
                    self.on_event({
                        "type": "run_end",
                        "outcome": "success",
                        "total_turns": turn,
                        "run_id": self.tracer.run_id,
                    })
                return "success"

            if self.world.is_at_exit(agent_id):
                print(f"  Agent {agent_id} reached the exit (waiting for partner)")

            turn += 1

        print(f"\nTurn limit ({self.max_turns}) reached.")
        self.tracer.log_run_end("turn_limit", turn)
        if self.on_event:
            self.on_event({
                "type": "run_end",
                "outcome": "turn_limit",
                "total_turns": turn,
                "run_id": self.tracer.run_id,
            })
        return "turn_limit"

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, agent_id: str, tool_call: dict, agent: Agent) -> dict:
        tool = tool_call.get("tool", "look")
        args = tool_call.get("args", {})

        if tool == "move":
            return self.world.move(agent_id, args.get("direction", "north"))

        if tool == "look":
            obs = self.world.observable_state(agent_id)
            return {"success": True, "observation": obs}

        if tool == "pick_up":
            return self.world.pick_up(agent_id)

        if tool == "use_item":
            return self.world.use_item(agent_id, args.get("item", ""), args.get("target", ""))

        if tool == "send_message":
            text = args.get("text", "")
            recipient = "B" if agent_id == "A" else "A"
            self._pending_messages[recipient].append(f"[Agent {agent_id}]: {text}")
            return {"success": True, "delivered_to": recipient, "on_turn": "next"}

        return {"success": False, "reason": f"Unknown tool '{tool}'"}

    def _both_at_exit(self) -> bool:
        return self.world.is_at_exit("A") and self.world.is_at_exit("B")
