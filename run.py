"""
Entry point — run a dungeon simulation and save the trace.

Usage:
    python run.py                    # random seed
    python run.py --seed 42          # deterministic
    python run.py --runs 3           # multiple runs
    python run.py --max-turns 50
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Callable

from world import DungeonWorld
from agents import Agent
from game_loop import GameLoop
from tracer import Tracer

RUNS_DIR = Path(__file__).parent / "runs"


def simulate(
    seed: Optional[int] = None,
    max_turns: int = 100,
    on_event: Optional[Callable] = None,
) -> tuple:
    """
    Run one simulation. Returns (run_id, outcome, json_path).
    on_event(event_dict) is called for every SSE event if provided.
    """
    if seed is None:
        seed = random.randint(0, 9999)

    world   = DungeonWorld(size=10, seed=seed)
    agent_a = Agent("A")
    agent_b = Agent("B")
    tracer  = Tracer()

    # Emit world initialisation event so the browser can draw the grid
    if on_event:
        on_event({
            "type": "world_init",
            "run_id": tracer.run_id,
            "seed": seed,
            "size": world.size,
            "grid": world.grid_layout(),
            "agent_positions": {k: list(v) for k, v in world.agent_positions.items()},
            "key_pos":  list(world.key_pos)  if world.key_pos  else None,
            "door_pos": list(world.door_pos) if world.door_pos else None,
            "exit_pos": list(world.exit_pos) if world.exit_pos else None,
        })

    loop = GameLoop(world, agent_a, agent_b, tracer, max_turns=max_turns, on_event=on_event)
    outcome = loop.run()

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"{tracer.run_id[:8]}.json"
    tracer.save(str(out_path))

    return tracer.run_id, outcome, str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--runs",      type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    for i in range(args.runs):
        seed = args.seed if args.seed is not None else random.randint(0, 9999)
        print(f"\n--- Run {i+1}/{args.runs}  seed={seed} ---")
        run_id, outcome, path = simulate(seed=seed, max_turns=args.max_turns)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
