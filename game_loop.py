from world import DungeonWorld
from agents import Agent
from tracer import Tracer
from typing import Optional


class GameLoop:
    def __init__(
        self,
        world: DungeonWorld,
        agent_a: Agent,
        agent_b: Agent,
        tracer: Tracer,
        max_turns: int = 100,
    ):
        self.world = world
        self.agents = {"A": agent_a, "B": agent_b}
        self.tracer = tracer
        self.max_turns = max_turns

        # Pending messages: dict[recipient_id] -> list of message strings
        self._pending_messages: dict[str, list[str]] = {"A": [], "B": []}

    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        Main game loop. Returns outcome string.
        Agents alternate turns: A, B, A, B, ...
        """
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

            # Get observable state BEFORE action (belief snapshot)
            obs = self.world.observable_state(agent_id)
            world_before = self.world.world_snapshot()

            # Agent decides
            tool_call, prompt, raw_response, latency_ms = agent.decide(obs, messages_received)

            # Execute tool
            result = self._execute_tool(agent_id, tool_call, agent)
            agent.last_result = result          # feed back for next turn
            world_after = self.world.world_snapshot()

            # Build belief state for this event:
            # We record what the agent could see + what messages it received
            belief_state = {
                **obs,
                "messages_received": messages_received,
            }

            self.tracer.log_turn(
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
            )

            # Print turn summary
            print(
                f"Turn {turn:3d} | Agent {agent_id} | "
                f"{tool_call['tool']}({tool_call['args']}) → {result}"
            )

            # Check win condition
            if self._both_at_exit():
                print(f"\nBoth agents reached the exit on turn {turn}!")
                self.tracer.log_run_end("success", turn)
                return "success"

            # Check individual exit arrival
            if self.world.is_at_exit(agent_id):
                print(f"  Agent {agent_id} reached the exit (waiting for partner)")

            turn += 1

        # Turn limit
        print(f"\nTurn limit ({self.max_turns}) reached.")
        self.tracer.log_run_end("turn_limit", turn)
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

    # ------------------------------------------------------------------

    def _both_at_exit(self) -> bool:
        return (
            self.world.is_at_exit("A") and
            self.world.is_at_exit("B")
        )
