import os
import time
import json
from typing import Optional
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_dm",
            "description": "Ask the Dungeon Master for information. They see the full map but their data is delayed by several turns — their answer may be outdated. Response arrives on your NEXT turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question, e.g. 'Where is the key?', 'Where is the door?', 'Where is the exit?', 'Where is Agent B?'",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move one step in a cardinal direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["north", "south", "east", "west"],
                    }
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Observe your current position and all adjacent cells.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick_up",
            "description": "Pick up an item on your current cell.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "use_item",
            "description": "Use an item from your inventory on a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item":   {"type": "string", "description": "Item name, e.g. 'key'"},
                    "target": {"type": "string", "description": "Target name, e.g. 'door'"},
                },
                "required": ["item", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to the other agent. Delivered on their next turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The message text"}
                },
                "required": ["text"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an explorer agent in a dungeon grid. Your goal: reach the exit (E).

Grid symbols:
  .  empty floor
  #  wall (impassable)
  K  key (pick it up with pick_up)
  D  locked door (use_item key door when standing adjacent to it)
  O  open door (passable)
  E  exit — move onto it to win
  A / B  the two agents

Rules:
- Fog of war: you can only see your current cell and the 8 cells around you.
- One agent must find the key (K) and unlock the door (D) before either can pass through.
- Messages arrive on the OTHER agent's NEXT turn (1-turn delay).
- The Dungeon Master (DM) sees the full map but their info is several turns old — use ask_dm when lost.
- You MUST call exactly one tool per turn.

Strategy:
- If you see K nearby, move there and pick_up.
- If you have the key and D is adjacent, use_item key door immediately.
- If you see E, move onto it.
- If lost, use ask_dm("Where is the key?") or ask_dm("Where is the exit?") — but remember the answer may be stale.
- Otherwise, MOVE to explore. Do not call look repeatedly.
- Use send_message to share discoveries with the other agent.

Your agent ID: {agent_id}
"""


class Agent:
    def __init__(self, agent_id: str, model: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.model = model
        self.last_result: Optional[dict] = None   # result of previous turn's action

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, observable_state: dict, messages_received: list[str]) -> tuple[dict, str, str, int]:
        """
        Returns (tool_call_dict, full_prompt_str, raw_response_str, latency_ms)
        """
        obs_text = _format_observation(observable_state, messages_received, self.last_result)

        # Stateless per turn — predictable context window, easier to trace
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(agent_id=self.agent_id)},
            {"role": "user",   "content": obs_text},
        ]

        start = time.time()
        response = _client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
        )
        latency_ms = int((time.time() - start) * 1000)

        raw_response = response.model_dump_json()
        choice = response.choices[0]

        # Extract tool call
        tool_call = choice.message.tool_calls[0] if choice.message.tool_calls else None
        if tool_call is None:
            # Fallback: look if no tool was chosen
            tool_dict = {"tool": "look", "args": {}}
        else:
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_dict = {"tool": tool_call.function.name, "args": args}

        return tool_dict, obs_text, raw_response, latency_ms


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_observation(obs: dict, messages: list[str], last_result: Optional[dict] = None) -> str:
    lines = [
        f"Your position: {obs['position']}",
        f"Inventory: {obs['inventory'] or 'empty'}",
        f"Door unlocked: {obs['door_unlocked']}",
        "",
        "Visible cells (row,col → cell):",
    ]
    for coord, cell in sorted(obs["visible_cells"].items()):
        lines.append(f"  {coord} → {cell}")

    if obs["other_agent_visible"]:
        lines.append(f"\nOther agent is visible at {obs['other_agent_position']}")
    else:
        lines.append("\nOther agent is not in view")

    if last_result is not None:
        success = last_result.get("success", True)
        if not success:
            lines.append(f"\nLast action FAILED: {last_result.get('reason', 'unknown reason')}. Try something different.")
        else:
            lines.append(f"\nLast action succeeded.")

    if messages:
        lines.append("\nMessages received this turn:")
        for m in messages:
            lines.append(f"  - {m}")
    else:
        lines.append("\nNo messages received this turn.")

    lines.append("\nWhat do you do? Call exactly one tool.")
    return "\n".join(lines)
