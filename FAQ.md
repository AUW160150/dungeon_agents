# FAQ

---

**What is the main algorithm?**

There is no single pathfinding algorithm driving the agents — the navigation decisions are made by an LLM (GPT-4o-mini) each turn. The Python layer computes a BFS hint over the agent's known map (every cell it has personally seen) and includes it in the prompt as "next step: move north." The LLM then calls one tool. BFS is used only to suggest a direction around known walls; the LLM decides whether to follow it or do something else (pick up an item, send a message, ask the DM, etc.).

---

**What does "stateless" mean for the agents?**

Each LLM call is independent — there is no conversation history passed between turns. Every turn the agent receives a fresh prompt describing its current position, visible cells, inventory, last action result, and any messages from its partner. It has no memory of what it decided two turns ago unless that information is re-surfaced in the current prompt.

The Python `Agent` object does hold cross-turn state (known landmark locations, a map of seen cells, recent position history for loop detection), but this is injected into the prompt each turn rather than stored in an LLM context window. The LLM itself always starts from scratch.

---

**What is the Dungeon Master (DM)?**

The DM is a third LLM agent that can see the full dungeon grid — not just a 3×3 fog-of-war window. Explorer agents can call `ask_dm` to ask it questions like "where is the key?" The DM answers in natural language.

The catch: the DM works from a rolling snapshot of the world that is N turns old (configurable, default 5). If the key was picked up 3 turns ago and the DM's snapshot is 5 turns old, the DM will confidently report the key's old location. This is the "stale DM information" failure class — reasonable agent behaviour, wrong information, emergent from system design. The tracer records the DM's staleness and whether its answer was factually correct at the time it was given.

---

**How do agents A and B communicate?**

Via `send_message`. When an agent calls `send_message(text)`, the message is queued and delivered to the other agent on their next turn — not instantly. This one-turn delay is intentional: it mimics async messaging in real distributed systems and creates interesting coordination failures (an agent acts on information that was true when sent but may be stale on arrival).

Agents are prompted to share landmark discoveries ("KEY at row=3 col=7") and key events ("I PICKED UP THE KEY") once each. In practice, coordination gaps of 20–80 turns are common — the traces record how long agents go without communicating, and the post-run analysis classifies this as an interaction failure when it causes the run to fail.

---

**What tools can the agents call, and when do they use them?**

| Tool | What it does | When agents use it |
|---|---|---|
| `move(direction)` | Move one cell north/south/east/west | Most turns — primary exploration action |
| `look()` | Return the 3×3 visible area | Rarely; the prompt discourages it in favour of moving |
| `pick_up()` | Pick up item on current cell | When standing on the key (K) |
| `use_item(item, target)` | Use an item on a target | When holding the key and adjacent to the door |
| `send_message(text)` | Queue a message for the other agent | After picking up the key or discovering a landmark |
| `ask_dm(question)` | Ask the DM a question | Once on turn 0 to get coordinates; again if looping with no known target |

Agents are required to call exactly one tool per turn (`tool_choice="required"`). If no tool call is returned, the game loop defaults to `look`.

---

**What counts as success or failure?**

**Success:** both agents reach the exit cell (E) in the same or consecutive turns, after the door has been unlocked.

**Failure — turn limit:** the run hits the max turn cap (default 100) before both agents reach the exit. This is the most common outcome.

**Failure — stopped:** the user manually stopped the run via the viewer.

The post-run LLM analysis further classifies *why* a failure happened into one of four types:

| failure_type | Meaning |
|---|---|
| `single_agent` | One agent made consistently bad decisions (e.g. looping, ignoring its inventory) |
| `interaction` | Both agents behaved reasonably in isolation but failed to coordinate (e.g. never communicated, both navigated to the same target) |
| `emergent` | Failure caused by system design rather than either agent's decisions (e.g. stale DM info sent both agents to the wrong location) |
| `success` | Both agents reached the exit |

This classification is produced by GPT after reading a compressed summary of the run — tool counts, milestones reached, messages sent, coordination gap, DM interactions, and action failures. It is not rule-based; the LLM interprets the evidence and assigns a type.

---

**What is the difference between the door and the exit?**

The door (D) and the exit (E) are two separate cells with different roles:

- **Door (D)** is a locked obstacle mid-dungeon. It blocks movement until unlocked. One agent must find the key (K), pick it up, stand adjacent to the door, and call `use_item(key, door)`. It then becomes an open door (O) that anyone can walk through.
- **Exit (E)** is the goal. Both agents must step onto it for the run to succeed. It is only reachable after the door is unlocked, since the door typically sits between the agents' starting area and the exit.

The required sequence is: **find K → pick it up → unlock D → both agents reach E.**

This two-step dependency is what makes coordination interesting — one agent usually handles the key while the other may already be near the exit, so they need to communicate to avoid one waiting forever.

---

**Why does an agent sometimes loop for 20+ turns?**

Two root causes appear in the traces:

1. **Wall oscillation:** the agent's nav hint says "go west" but a wall blocks it. It tries west, fails, tries a perpendicular direction, then the next turn the hint says "go west" again. This cycles until the loop detector fires (≤3 unique cells in the last 10 moves), at which point the observation explicitly warns the agent to break the pattern.

2. **Coordination deadlock:** both agents are waiting for the other to unlock the door, but neither has the key and they've stopped communicating. No individual agent decision is wrong; the system is stuck.

Both show up as `agent_stagnation` or `coordination_gap` divergences in the trace.
