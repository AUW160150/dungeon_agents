"""
Post-run LLM analysis.

Compresses the event log into key facts, feeds to GPT, returns a structured
diagnostic report answering: What happened? Why? What should change?
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """You are a senior engineer doing a post-mortem on a multi-agent AI system.

Context: Two explorer agents (A and B) navigate a dungeon grid with fog of war.
Goal: both agents must reach the exit (E). One must first find a key (K),
unlock a locked door (D), then both reach the exit.
A Dungeon Master (DM) can answer questions but uses world state that is several turns old.
Agents can message each other but messages are delayed by one turn.

Run trace summary:
{summary}

Respond with a JSON object using exactly these four keys:

{{
  "what_happened": "2-3 sentences. Factual narrative. What did each agent do? What was achieved or not?",
  "why_it_happened": "2-3 sentences. Root cause. Was this a single-agent decision failure, an interaction failure between agents, or emergent from the system design? Cite specific turn numbers or patterns.",
  "failure_type": "single_agent OR interaction OR emergent OR success",
  "what_should_change": "3 concrete, specific suggestions. Each on its own line starting with -. Could be prompt changes, tool design, schema improvements, or coordination mechanisms."
}}

Be specific and direct. Reference agent names and turn numbers. Do not be generic."""


def _compress(events: list) -> str:
    """
    Distil the full event list into a compact, LLM-readable summary.
    Captures: outcome, tool usage, failures, divergences, communication,
    DM interactions, stagnation, coordination gaps.
    """
    outcome      = "unknown"
    total_turns  = 0
    tool_counts  = {"A": {}, "B": {}}
    failures     = []
    divergences  = []
    milestones   = []
    messages     = []
    dm_events    = []
    max_coord_gap     = 0
    max_stationary    = {"A": 0, "B": 0}

    for e in events:
        agent  = e.get("agent", "")
        turn   = e.get("turn", 0)
        action = e.get("action") or {}
        result = e.get("result") or {}
        tool   = action.get("tool")
        extra  = e.get("extra") or {}

        if e.get("phase") == "run_end":
            outcome     = e.get("outcome", "unknown")
            total_turns = e.get("total_turns", 0)
            continue

        if agent == "DM":
            stale = extra.get("actual_staleness") or e.get("actual_staleness", "?")
            wrong = len(e.get("divergences") or []) > 0
            dm_events.append(
                f"T{turn} asked by {action.get('asker')}: "
                f"Q={action.get('question','')[:60]} | "
                f"staleness={stale} | info_wrong={wrong}"
            )
            continue

        if agent not in ("A", "B"):
            continue

        # Tool usage
        if tool:
            tool_counts[agent][tool] = tool_counts[agent].get(tool, 0) + 1

        # Failures
        if result.get("success") is False:
            failures.append(f"T{turn} {agent}: {tool} failed — {result.get('reason','')}")

        # Milestones
        if tool == "pick_up" and result.get("item"):
            milestones.append(f"T{turn}: Agent {agent} picked up {result['item']}")
        if tool == "use_item" and result.get("success"):
            milestones.append(f"T{turn}: Agent {agent} unlocked door")
        if tool == "send_message" and result.get("success"):
            messages.append(
                f"T{turn} {agent}→{result.get('delivered_to','?')}: "
                f"{action.get('args',{}).get('text','')[:80]}"
            )

        # Divergences
        for d in e.get("divergences") or []:
            dtype = d.get("type", "")
            if dtype == "coordination_gap":
                max_coord_gap = max(max_coord_gap, d.get("gap_turns", 0))
            elif dtype == "agent_stagnation":
                ag = d.get("agent", agent)
                max_stationary[ag] = max(max_stationary[ag], d.get("turns_stationary", 0))
            else:
                divergences.append(f"T{turn} {agent}: {dtype}")

        # Track from extra too
        cg = extra.get("coordination_gap", 0)
        max_coord_gap = max(max_coord_gap, cg)
        ts = extra.get("turns_stationary", 0)
        max_stationary[agent] = max(max_stationary[agent], ts)

    lines = [
        f"Outcome: {outcome} after {total_turns} turns",
        "",
        "Agent tool usage:",
        f"  A: {tool_counts['A']}",
        f"  B: {tool_counts['B']}",
        "",
        f"Key milestones: {milestones if milestones else 'NONE — key never found, door never unlocked'}",
        "",
        f"Inter-agent messages sent: {len(messages)}",
    ]

    if messages:
        lines += [f"  {m}" for m in messages[:5]]
    else:
        lines.append("  NONE — agents never communicated at all")

    lines += [
        "",
        f"Longest coordination silence: {max_coord_gap} turns without any inter-agent message",
        f"Longest stagnation — A: {max_stationary['A']} turns without moving, "
        f"B: {max_stationary['B']} turns without moving",
        "",
        f"Action failures ({len(failures)} total, sample):",
    ]
    lines += [f"  {f}" for f in failures[:8]]
    if len(failures) > 8:
        lines.append(f"  … and {len(failures) - 8} more")

    lines += [
        "",
        f"Schema divergences (belief vs truth, {len(divergences)} total, sample):",
    ]
    lines += [f"  {d}" for d in divergences[:8]]

    lines += [
        "",
        f"DM interactions: {len(dm_events)}",
    ]
    lines += [f"  {d}" for d in dm_events[:5]]

    return "\n".join(lines)


def generate(events: list) -> dict:
    """
    Run post-mortem analysis on a completed run's events.
    Returns structured dict with what_happened / why_it_happened /
    failure_type / what_should_change.
    """
    summary = _compress(events)

    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(summary=summary)}],
        response_format={"type": "json_object"},
        max_tokens=700,
    )

    result = json.loads(response.choices[0].message.content)

    # Hard override: if the run actually succeeded, failure_type must be "success"
    # regardless of LLM classification (it sometimes misreads pick_up failures etc.)
    outcome = next(
        (e.get("outcome") for e in reversed(events) if e.get("phase") == "run_end"),
        None,
    )
    if outcome == "success":
        result["failure_type"] = "success"

    result["trace_summary"] = summary   # keep what we fed the LLM — transparency
    return result
