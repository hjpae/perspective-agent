# cear_pilot/envs/forage_grid.py
# -*- coding: utf-8 -*-
"""
ForageGridEnv (Gymnasium):
- Grid: 5x5, agent spawns at center (2,2)
- Time budget T=500
- MOVE cost 1 time
- FORAGE cost 1 time
- INSPECT cost 50 time
- Exactly 4 objects on the grid at all times: 2 GREEN (+bonus) and 2 RED (-penalty)
- Colors are hidden until INSPECT reveals current tile (or radius-1 cross if enabled).
- Goal: maximize total reward within time budget.

Actions (Discrete(6)):
0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 INSPECT, 5 FORAGE

Observation (float32, shape=(101,)):
- agent one-hot (25)
- object_present (25)
- revealed_green (25)
- revealed_red (25)
- remaining_time_fraction (1)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ForageGridConfig:
    grid_size: int = 5
    time_budget: int = 500

    move_cost: int = 1
    forage_cost: int = 1
    inspect_cost: int = 50

    n_green: int = 2
    n_red: int = 2

    forage_bonus: float = 1.0
    forage_penalty: float = 1.0

    seed: Optional[int] = None

    # If True, INSPECT reveals (self + 4-neighborhood). If False, reveals only current cell.
    inspect_reveal_radius1: bool = False


class ForageGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_INSPECT = 4
    ACTION_FORAGE = 5

    COLOR_GREEN = 1
    COLOR_RED = 2

    def __init__(self, config: Optional[ForageGridConfig] = None):
        super().__init__()
        self.cfg = config or ForageGridConfig()

        g = self.cfg.grid_size
        assert g == 5, "Requested grid_size=5."
        assert (self.cfg.n_green + self.cfg.n_red) == 4, "Requested exactly 4 objects (2 green, 2 red)."

        self.rng = np.random.default_rng(self.cfg.seed)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(101,), dtype=np.float32)

        self.agent_xy: Tuple[int, int] = (2, 2)
        self.time_left: int = int(self.cfg.time_budget)

        # objects: pos -> (color, revealed_bool)
        self.objects: Dict[Tuple[int, int], Tuple[int, bool]] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_xy = (2, 2)
        self.time_left = int(self.cfg.time_budget)
        self.objects = {}
        self._spawn_initial_objects()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action)

        reward = 0.0
        terminated = False
        truncated = False

        if self.time_left <= 0:
            truncated = True
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()

        if action in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
            self._apply_move(action)
            self.time_left -= self.cfg.move_cost

        elif action == self.ACTION_INSPECT:
            self._apply_inspect()
            self.time_left -= self.cfg.inspect_cost

        elif action == self.ACTION_FORAGE:
            reward += self._apply_forage()
            self.time_left -= self.cfg.forage_cost

        if self.time_left <= 0:
            self.time_left = 0
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        g = self.cfg.grid_size
        ax, ay = self.agent_xy
        lines = [f"time_left={self.time_left}"]
        for y in range(g):
            row = []
            for x in range(g):
                if (x, y) == (ax, ay):
                    row.append("A")
                elif (x, y) in self.objects:
                    color, revealed = self.objects[(x, y)]
                    if not revealed:
                        row.append("?")
                    else:
                        row.append("G" if color == self.COLOR_GREEN else "R")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ---------------- Internals ----------------

    def _spawn_initial_objects(self):
        empties = self._empty_tiles(exclude_agent=True)
        self.rng.shuffle(empties)
        positions = empties[:4]

        colors = [self.COLOR_GREEN] * self.cfg.n_green + [self.COLOR_RED] * self.cfg.n_red
        self.rng.shuffle(colors)

        for pos, col in zip(positions, colors):
            self.objects[pos] = (col, False)

    def _empty_tiles(self, exclude_agent: bool = True) -> List[Tuple[int, int]]:
        g = self.cfg.grid_size
        tiles: List[Tuple[int, int]] = []
        for y in range(g):
            for x in range(g):
                if exclude_agent and (x, y) == self.agent_xy:
                    continue
                if (x, y) in self.objects:
                    continue
                tiles.append((x, y))
        return tiles

    def _apply_move(self, action: int):
        x, y = self.agent_xy
        if action == self.ACTION_UP:
            y = max(0, y - 1)
        elif action == self.ACTION_DOWN:
            y = min(self.cfg.grid_size - 1, y + 1)
        elif action == self.ACTION_LEFT:
            x = max(0, x - 1)
        elif action == self.ACTION_RIGHT:
            x = min(self.cfg.grid_size - 1, x + 1)
        self.agent_xy = (x, y)

    def _apply_inspect(self):
        if not self.cfg.inspect_reveal_radius1:
            self._reveal_at(self.agent_xy)
        else:
            x, y = self.agent_xy
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                xx, yy = x + dx, y + dy
                if 0 <= xx < self.cfg.grid_size and 0 <= yy < self.cfg.grid_size:
                    self._reveal_at((xx, yy))

    def _reveal_at(self, pos: Tuple[int, int]):
        if pos in self.objects:
            color, _ = self.objects[pos]
            self.objects[pos] = (color, True)

    def _apply_forage(self) -> float:
        pos = self.agent_xy
        if pos not in self.objects:
            return 0.0

        color, _ = self.objects.pop(pos)

        if color == self.COLOR_GREEN:
            r = float(self.cfg.forage_bonus)
        else:
            r = -float(self.cfg.forage_penalty)

        # respawn SAME color elsewhere, unrevealed
        empties = self._empty_tiles(exclude_agent=True)
        if len(empties) > 0:
            new_pos = empties[self.rng.integers(0, len(empties))]
            self.objects[new_pos] = (color, False)

        return r

    def _get_obs(self) -> np.ndarray:
        g = self.cfg.grid_size
        ax, ay = self.agent_xy

        agent = np.zeros((g, g), dtype=np.float32)
        agent[ay, ax] = 1.0

        obj_present = np.zeros((g, g), dtype=np.float32)
        rev_g = np.zeros((g, g), dtype=np.float32)
        rev_r = np.zeros((g, g), dtype=np.float32)

        for (x, y), (color, revealed) in self.objects.items():
            obj_present[y, x] = 1.0
            if revealed:
                if color == self.COLOR_GREEN:
                    rev_g[y, x] = 1.0
                else:
                    rev_r[y, x] = 1.0

        tfrac = np.array([self.time_left / float(self.cfg.time_budget)], dtype=np.float32)

        obs = np.concatenate(
            [agent.reshape(-1), obj_present.reshape(-1), rev_g.reshape(-1), rev_r.reshape(-1), tfrac],
            axis=0,
        ).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        n_revealed = sum(1 for (_pos, (_c, r)) in self.objects.items() if r)
        return {
            "time_left": int(self.time_left),
            "agent_x": int(self.agent_xy[0]),
            "agent_y": int(self.agent_xy[1]),
            "n_objects": int(len(self.objects)),
            "n_revealed": int(n_revealed),
        }
