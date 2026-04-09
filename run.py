"""
Entry point — run a dungeon simulation and save the trace.

Usage:
    python run.py                    # random seed
    python run.py --seed 42          # deterministic
    python run.py --runs 3           # multiple runs
    python run.py --max-turns 50
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional

from world import DungeonWorld
from agents import Agent
from game_loop import GameLoop
from tracer import Tracer

RUNS_DIR = Path("runs")


def run_once(seed: Optional[int], max_turns: int) -> str:
    world  = DungeonWorld(size=10, seed=seed)
    agent_a = Agent("A")
    agent_b = Agent("B")
    tracer  = Tracer()

    loop = GameLoop(world, agent_a, agent_b, tracer, max_turns=max_turns)
    outcome = loop.run()

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"{tracer.run_id[:8]}.json"
    tracer.save(str(out_path))
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int,  default=None)
    parser.add_argument("--runs",      type=int,  default=1)
    parser.add_argument("--max-turns", type=int,  default=100)
    args = parser.parse_args()

    for i in range(args.runs):
        seed = args.seed if args.seed is not None else random.randint(0, 9999)
        print(f"\n--- Run {i+1}/{args.runs}  seed={seed} ---")
        path = run_once(seed, args.max_turns)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
