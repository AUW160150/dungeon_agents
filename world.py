import random
from enum import Enum
from typing import Optional

class Cell(str, Enum):
    EMPTY   = "."
    WALL    = "#"
    KEY     = "K"
    DOOR    = "D"   # locked
    OPEN    = "O"   # door unlocked/open
    EXIT    = "E"
    AGENT_A = "A"
    AGENT_B = "B"

DIRECTIONS = {
    "north": (-1, 0),
    "south": ( 1, 0),
    "east":  ( 0, 1),
    "west":  ( 0,-1),
}

class DungeonWorld:
    def __init__(self, size: int = 10, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.grid: list[list[str]] = []
        self.agent_positions: dict[str, tuple[int,int]] = {}
        self.agent_inventories: dict[str, list[str]] = {"A": [], "B": []}
        self.door_pos: Optional[tuple[int,int]] = None
        self.key_pos: Optional[tuple[int,int]] = None
        self.exit_pos: Optional[tuple[int,int]] = None
        self.door_unlocked = False
        self._build()

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _build(self):
        s = self.size
        # Start with all empty
        self.grid = [[Cell.EMPTY for _ in range(s)] for _ in range(s)]

        # Outer walls
        for i in range(s):
            self.grid[0][i] = Cell.WALL
            self.grid[s-1][i] = Cell.WALL
            self.grid[i][0] = Cell.WALL
            self.grid[i][s-1] = Cell.WALL

        # Random interior obstacles (~15% of inner cells)
        inner_cells = [(r, c) for r in range(1, s-1) for c in range(1, s-1)]
        obstacle_count = max(1, int(len(inner_cells) * 0.15))
        obstacles = self.rng.sample(inner_cells, obstacle_count)
        for r, c in obstacles:
            self.grid[r][c] = Cell.WALL

        # Place special cells on remaining empty inner cells.
        # Filter to cells that have at least one non-wall neighbour so
        # agents are never immediately boxed in.
        def has_open_neighbour(r, c):
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < s and 0 <= nc < s and self.grid[nr][nc] == Cell.EMPTY:
                    return True
            return False

        available = [
            pos for pos in inner_cells
            if self.grid[pos[0]][pos[1]] == Cell.EMPTY and has_open_neighbour(*pos)
        ]
        self.rng.shuffle(available)

        self.key_pos  = available.pop()
        self.door_pos = available.pop()
        self.exit_pos = available.pop()
        a_start       = available.pop()
        b_start       = available.pop()

        self.grid[self.key_pos[0]][self.key_pos[1]]   = Cell.KEY
        self.grid[self.door_pos[0]][self.door_pos[1]] = Cell.DOOR
        self.grid[self.exit_pos[0]][self.exit_pos[1]] = Cell.EXIT

        self.agent_positions["A"] = a_start
        self.agent_positions["B"] = b_start

    # ------------------------------------------------------------------
    # Observation (fog of war)
    # ------------------------------------------------------------------

    def get_visible_cells(self, agent_id: str) -> dict:
        """Returns the 3x3 neighbourhood centred on the agent."""
        r, c = self.agent_positions[agent_id]
        visible = {}
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    cell_val = self.grid[nr][nc]
                    # Show agent markers
                    other = "B" if agent_id == "A" else "A"
                    if (nr, nc) == self.agent_positions.get(other):
                        cell_val = Cell(f"AGENT_{other}") if False else other
                    visible[(nr, nc)] = cell_val
                else:
                    visible[(nr, nc)] = Cell.WALL  # out of bounds = wall
        return visible

    def observable_state(self, agent_id: str) -> dict:
        pos = self.agent_positions[agent_id]
        visible = self.get_visible_cells(agent_id)
        other = "B" if agent_id == "A" else "A"
        other_pos = self.agent_positions[other]
        other_visible = other_pos in visible

        return {
            "position": list(pos),
            "inventory": list(self.agent_inventories[agent_id]),
            "visible_cells": {
                f"{r},{c}": str(v.value) if hasattr(v, 'value') else str(v)
                for (r, c), v in visible.items()
            },
            "other_agent_visible": other_visible,
            "other_agent_position": list(other_pos) if other_visible else None,
            "door_unlocked": self.door_unlocked,
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def move(self, agent_id: str, direction: str) -> dict:
        if direction not in DIRECTIONS:
            return {"success": False, "reason": f"Unknown direction '{direction}'"}

        r, c = self.agent_positions[agent_id]
        dr, dc = DIRECTIONS[direction]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return {"success": False, "reason": "Out of bounds"}

        cell = self.grid[nr][nc]

        if cell == Cell.WALL:
            return {"success": False, "reason": "Wall in the way"}

        if cell == Cell.DOOR:
            return {"success": False, "reason": "Locked door blocks the way"}

        self.agent_positions[agent_id] = (nr, nc)
        cell_str = cell.value if hasattr(cell, 'value') else str(cell)
        return {"success": True, "new_position": [nr, nc], "cell_entered": cell_str}

    def pick_up(self, agent_id: str) -> dict:
        r, c = self.agent_positions[agent_id]
        cell = self.grid[r][c]

        if cell == Cell.KEY:
            self.agent_inventories[agent_id].append("key")
            self.grid[r][c] = Cell.EMPTY
            self.key_pos = None
            return {"success": True, "item": "key"}

        return {"success": False, "reason": "Nothing to pick up here"}

    def use_item(self, agent_id: str, item: str, target: str) -> dict:
        if item != "key":
            return {"success": False, "reason": f"Don't know how to use '{item}'"}

        if target != "door":
            return {"success": False, "reason": f"Can't use key on '{target}'"}

        if "key" not in self.agent_inventories[agent_id]:
            return {"success": False, "reason": "You don't have the key"}

        r, c = self.agent_positions[agent_id]
        # Must be adjacent to door
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if (nr, nc) == self.door_pos:
                self.agent_inventories[agent_id].remove("key")
                self.grid[self.door_pos[0]][self.door_pos[1]] = Cell.OPEN
                self.door_unlocked = True
                return {"success": True, "result": "Door is now open"}

        return {"success": False, "reason": "Not adjacent to the door"}

    # ------------------------------------------------------------------
    # World truth snapshot (for divergence detection)
    # ------------------------------------------------------------------

    def world_snapshot(self) -> dict:
        return {
            "agent_positions": {k: list(v) for k, v in self.agent_positions.items()},
            "key_location": list(self.key_pos) if self.key_pos else None,
            "door_location": list(self.door_pos) if self.door_pos else None,
            "door_unlocked": self.door_unlocked,
            "exit_location": list(self.exit_pos) if self.exit_pos else None,
            "inventories": {k: list(v) for k, v in self.agent_inventories.items()},
        }

    def grid_layout(self) -> list:
        """Serialize grid as 2D list of cell characters for the browser."""
        return [
            [cell.value if hasattr(cell, 'value') else str(cell) for cell in row]
            for row in self.grid
        ]

    def is_at_exit(self, agent_id: str) -> bool:
        return self.agent_positions[agent_id] == self.exit_pos

    def render(self) -> str:
        """ASCII render of the full grid (for debugging/logs)."""
        grid_copy = [row[:] for row in self.grid]
        for agent_id, (r, c) in self.agent_positions.items():
            grid_copy[r][c] = agent_id
        return "\n".join("".join(row) for row in grid_copy)
