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
            "description": "Observe your current position and all adjacent cells. Only use this if you have a specific reason — prefer moving to explore.",
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

SYSTEM_PROMPT = """You are an explorer agent in a dungeon grid. Your goal: BOTH agents must reach the exit (E).

Grid symbols:
  .  empty floor      #  wall (impassable)
  K  key              D  locked door (needs key)
  O  open door        E  exit
  A / B  the agents

Rules:
- Fog of war: you can only see a 3x3 area around you.
- One agent must find K, pick it up, and use it on D to unlock it. Then both reach E.
- Messages to your partner arrive on their NEXT turn.
- The DM sees the full map but their snapshot is several turns old — useful but possibly stale.
- You MUST call exactly one tool per turn.

Strategy (follow in strict priority order):

1. TURN 0 ONLY — If you have no landmark locations known yet: ask_dm("Where is the key, door, and exit? Give row and col for each.")
   Do this exactly once. Once you have coordinates, never ask the DM the same question again — navigate instead.

2. KEY PICKUP — If you are standing on K (visible in your current cell): pick_up immediately.

3. UNLOCK — If you have the key and are adjacent to D: use_item key door immediately.

4. NAVIGATE TO TARGET — If you know a landmark's coordinates (from DM or messages), move toward it:
   Use the navigation hint provided below — it tells you exactly which direction to go.

5. SHARE DISCOVERIES — After picking up the key OR finding exit/door for the first time,
   send_message ONCE to your partner. Only share facts you have directly observed.
   Do NOT claim to have items you do not have (check your inventory first).
   Format: "KEY at row=R col=C" / "DOOR at row=R col=C" / "EXIT at row=R col=C" / "I PICKED UP THE KEY"

6. LOOPING DETECTED — If the observation shows a loop warning, follow its instruction exactly.
   If it says navigate to exit, move toward exit. If it says ask_dm, ask once then move.

7. EXPLORE — If you have no target: move in a direction not in your recent path.
   Never call look when you could move.

Your agent ID: {agent_id}
"""


class Agent:
    def __init__(self, agent_id: str, model: str = "gpt-4o-mini"):
        self.agent_id   = agent_id
        self.model      = model
        self.last_result: Optional[dict] = None

        # Cross-turn state for loop detection and landmark memory
        self.recent_positions: list  = []          # last 12 positions as (row, col) tuples
        self.known_landmarks:  dict  = {}          # {"key": [r,c], "door": [r,c], "exit": [r,c]}
        self.turn_count:       int   = 0

    def update_knowledge(self, obs: dict):
        """
        Called by the game loop before decide() each turn.
        Updates landmark memory from visible cells and tracks position history.
        """
        pos = obs.get("position")
        if pos:
            self.recent_positions.append(tuple(pos))
            if len(self.recent_positions) > 12:
                self.recent_positions.pop(0)

        # Learn landmark locations from what's currently visible
        for coord_str, cell in obs.get("visible_cells", {}).items():
            r, c = map(int, coord_str.split(","))
            if cell == "K" and "key" not in self.known_landmarks:
                self.known_landmarks["key"] = [r, c]
            if cell in ("D", "O") and "door" not in self.known_landmarks:
                self.known_landmarks["door"] = [r, c]
            if cell == "E" and "exit" not in self.known_landmarks:
                self.known_landmarks["exit"] = [r, c]

        # If we're carrying the key, it's no longer on the floor
        if "key" in obs.get("inventory", []):
            self.known_landmarks.pop("key", None)

        # If the door is already unlocked, key is gone and door no longer matters
        if obs.get("door_unlocked"):
            self.known_landmarks.pop("key", None)
            self.known_landmarks.pop("door", None)
            self.known_landmarks.pop("door_open", None)

        self.turn_count += 1

    @property
    def is_looping(self) -> bool:
        """True if the agent has been cycling through ≤3 unique cells in the last 10 moves."""
        if len(self.recent_positions) < 10:
            return False
        return len(set(self.recent_positions[-10:])) <= 3

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, observable_state: dict, messages_received: list[str]) -> tuple[dict, str, str, int]:
        """Returns (tool_call_dict, prompt_str, raw_response_str, latency_ms)"""
        obs_text = _format_observation(
            observable_state,
            messages_received,
            self.last_result,
            self.known_landmarks,
            self.recent_positions,
            self.is_looping,
            self.turn_count,
        )

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
        choice       = response.choices[0]

        tool_call = choice.message.tool_calls[0] if choice.message.tool_calls else None
        if tool_call is None:
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

def _nav_hint(from_pos: list, to_pos: list) -> str:
    """
    Compute cardinal directions to move from from_pos toward to_pos.
    Returns a human-readable hint like 'go north (3 steps) then east (5 steps)'.
    """
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    parts = []
    if dr < 0:
        parts.append(f"north ({abs(dr)} step{'s' if abs(dr) > 1 else ''})")
    elif dr > 0:
        parts.append(f"south ({dr} step{'s' if dr > 1 else ''})")
    if dc < 0:
        parts.append(f"west ({abs(dc)} step{'s' if abs(dc) > 1 else ''})")
    elif dc > 0:
        parts.append(f"east ({dc} step{'s' if dc > 1 else ''})")
    if not parts:
        return "you are already there"
    manhattan = abs(dr) + abs(dc)
    return f"go {' then '.join(parts)}  (Manhattan distance: {manhattan})"


def _format_observation(
    obs:               dict,
    messages:          list,
    last_result:       Optional[dict],
    known_landmarks:   dict,
    recent_positions:  list,
    is_looping:        bool,
    turn_count:        int,
) -> str:
    pos = obs["position"]
    r, c = pos

    door_status = "YES — door is open, head straight to the EXIT" if obs['door_unlocked'] else "NO — someone must find the key and unlock it first"
    lines = [
        f"Turn: {turn_count}",
        f"Your position: row={r}, col={c}",
        f"Inventory: {obs['inventory'] or 'empty'}",
        f"Door unlocked: {door_status}",
        "",
        "Visible cells (row,col → cell type):",
    ]
    for coord, cell in sorted(obs["visible_cells"].items()):
        lines.append(f"  {coord} → {cell}")

    # Other agent
    if obs["other_agent_visible"]:
        op = obs["other_agent_position"]
        lines.append(f"\nPartner visible at row={op[0]}, col={op[1]}")
    else:
        lines.append("\nPartner not in view")

    # Known landmarks + navigation hints
    if known_landmarks:
        lines.append("\nKnown landmark locations — USE THESE TO NAVIGATE, do not ask_dm again:")
        for name, lpos in known_landmarks.items():
            hint = _nav_hint(pos, lpos)
            lines.append(f"  {name.upper()}: row={lpos[0]}, col={lpos[1]}  → {hint}")
    else:
        lines.append("\nNo landmark locations known yet. Use ask_dm ONCE to get coordinates.")

    # Loop warning — context-aware: if exit is known, navigate there; otherwise ask DM
    if is_looping:
        unique = len(set(recent_positions[-10:]))
        if "exit" in known_landmarks:
            ep = known_landmarks["exit"]
            lines.append(
                f"\n⚠ LOOP WARNING: visiting only {unique} unique cells in last 10 moves."
                f" You already know the EXIT is at row={ep[0]}, col={ep[1]}. Navigate there NOW — do not ask_dm."
            )
        else:
            lines.append(
                f"\n⚠ LOOP WARNING: visiting only {unique} unique cells in last 10 moves."
                f" You are going in circles. Use ask_dm to get coordinates."
            )
    elif len(recent_positions) >= 4:
        recent_str = " → ".join(f"r{p[0]},c{p[1]}" for p in recent_positions[-4:])
        lines.append(f"\nRecent path (last 4): {recent_str}")

    # Last action feedback
    if last_result is not None:
        if not last_result.get("success", True):
            lines.append(f"\nLast action FAILED: {last_result.get('reason', 'unknown')}. Try something different.")
        else:
            lines.append("\nLast action succeeded.")

    # Messages
    if messages:
        lines.append("\nMessages from partner this turn:")
        for m in messages:
            lines.append(f"  {m}")
    else:
        lines.append("\nNo messages from partner this turn.")

    lines.append("\nWhat do you do? Call exactly one tool.")
    return "\n".join(lines)
