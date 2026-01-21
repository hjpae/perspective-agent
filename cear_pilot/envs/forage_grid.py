# cear_pilot/envs/forage_grid.py
# -*- coding: utf-8 -*-
"""
ForageGridEnv (Gymnasium)

Grid: 5x5, agent spawns at center (2,2)
Time budget: T=500
Actions (Discrete(6)):
  0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 INSPECT, 5 FORAGE

Objects:
  Exactly 4 objects always exist: 2 GREEN (+) and 2 RED (-).
  Colors are hidden until INSPECT reveals a tile (or cross radius-1 if enabled).
  When FORAGE on an object tile, it respawns elsewhere with the same color (unrevealed).

Rewards:
  - r_env: "true" sparse environment reward (FORAGE only).
  - r_shape: dense shaping reward for training.
  - step() returns r_train = r_env + r_shape (default training reward).
  - info includes r_env, r_shape, and shaping parts.

Observation (float32, shape=(101,)):
  agent one-hot (25)
  object_present (25)
  revealed_green (25)
  revealed_red (25)
  remaining_time_fraction (1)
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

    # -----------------------------
    # Dense shaping (training only)
    # -----------------------------
    # Reward for decreasing Manhattan distance to any GREEN object (per step improvement).
    k_green_progress: float = 1.00

    # Mild penalty for being near RED objects.
    red_threshold: int = 2
    k_red_near: float = 0.02

    # Entry bonus when stepping onto ANY object tile (once per entry).
    k_object_entry: float = 1.0

    # Strong incentive to FORAGE when on an object tile; discourage useless FORAGE on empty tiles.
    k_forage_on_object: float = 3.0
    k_forage_empty: float = 0.5

    # Mild penalty for staying on an object tile without foraging (prevents camping/ignoring FORAGE).
    k_skip_forage: float = 0.02

    # Very mild revisit / idle penalties (keep small; they can dominate otherwise).
    revisit_coef: float = 0.002
    idle_coef: float = 0.001
    idle_cap: int = 5

    # Time-cost penalty to prevent "inspect spam" and other time-wasting tricks.
    k_time_cost: float = 0.002


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
        assert g == 5, "This demo assumes grid_size=5."
        assert (self.cfg.n_green + self.cfg.n_red) == 4, "This demo assumes exactly 4 objects (2 green, 2 red)."

        self.rng = np.random.default_rng(self.cfg.seed)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(101,), dtype=np.float32)

        self.agent_xy: Tuple[int, int] = (2, 2)
        self.time_left: int = int(self.cfg.time_budget)

        # objects: pos -> (color, revealed_bool)
        self.objects: Dict[Tuple[int, int], Tuple[int, bool]] = {}

        # shaping state
        self._prev_d_green: Optional[int] = None
        self._visit_counts: Dict[Tuple[int, int], int] = {}
        self._idle_streak: int = 0
        self._was_on_object: bool = False

        # last-step reward bookkeeping
        self._last_r_env: float = 0.0
        self._last_r_shape: float = 0.0
        self._last_shape_parts: Dict[str, float] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_xy = (2, 2)
        self.time_left = int(self.cfg.time_budget)
        self.objects = {}
        self._spawn_initial_objects()

        # shaping state reset
        self._prev_d_green = self._min_manhattan_to_color(self.COLOR_GREEN)
        self._visit_counts = {}
        self._idle_streak = 0
        self._was_on_object = (self.agent_xy in self.objects)

        self._set_last_rewards(0.0, 0.0, {})

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action)

        terminated = False
        truncated = False

        if self.time_left <= 0:
            truncated = True
            self._set_last_rewards(0.0, 0.0, {})
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()

        prev_xy = self.agent_xy
        prev_dg = int(self._prev_d_green if self._prev_d_green is not None else self._min_manhattan_to_color(self.COLOR_GREEN))

        r_env = 0.0
        time_cost = 0

        # Apply action + time costs
        if action in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
            self._apply_move(action)
            time_cost = int(self.cfg.move_cost)
            self.time_left -= time_cost

        elif action == self.ACTION_INSPECT:
            self._apply_inspect()
            time_cost = int(self.cfg.inspect_cost)
            self.time_left -= time_cost

        elif action == self.ACTION_FORAGE:
            r_env += float(self._apply_forage())
            time_cost = int(self.cfg.forage_cost)
            self.time_left -= time_cost

        if self.time_left <= 0:
            self.time_left = 0
            truncated = True

        # ---------- Dense shaping ----------
        parts: Dict[str, float] = {
            "green_progress": 0.0,
            "red_near": 0.0,
            "object_entry": 0.0,
            "forage": 0.0,
            "skip_forage": 0.0,
            "revisit": 0.0,
            "idle": 0.0,
            "time_cost": 0.0,
        }

        # Time-cost penalty (prevents inspect-spam / time-wasting)
        parts["time_cost"] = -float(self.cfg.k_time_cost) * float(time_cost)

        # Idle: count blocked moves only; cap streak
        if action in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
            if self.agent_xy == prev_xy:
                self._idle_streak = min(self._idle_streak + 1, int(self.cfg.idle_cap))
            else:
                self._idle_streak = 0
        parts["idle"] = -float(self.cfg.idle_coef) * float(self._idle_streak)

        # Revisit: very mild, sublinear
        self._visit_counts[self.agent_xy] = int(self._visit_counts.get(self.agent_xy, 0) + 1)
        v = int(self._visit_counts[self.agent_xy])
        if v > 1:
            parts["revisit"] = -float(self.cfg.revisit_coef) * float(np.sqrt(v - 1.0))

        # Object tile logic
        on_obj_now = (self.agent_xy in self.objects)

        # Entry bonus: only when you ENTER an object tile
        if on_obj_now and (not self._was_on_object):
            parts["object_entry"] = float(self.cfg.k_object_entry)

        # Forage incentive shaping
        if action == self.ACTION_FORAGE:
            if on_obj_now:
                parts["forage"] = float(self.cfg.k_forage_on_object)
            else:
                parts["forage"] = -float(self.cfg.k_forage_empty)

        # If on object and NOT foraging, mild penalty to prevent camping/ignoring FORAGE
        if on_obj_now and action != self.ACTION_FORAGE:
            parts["skip_forage"] = -float(self.cfg.k_skip_forage)

        # Green progress shaping: reward distance improvements (uses true object positions)
        dg = int(self._min_manhattan_to_color(self.COLOR_GREEN))
        progress = prev_dg - dg
        if progress > 0:
            parts["green_progress"] = float(self.cfg.k_green_progress) * float(progress)

        # Red proximity penalty (uses true object positions)
        dr = int(self._min_manhattan_to_color(self.COLOR_RED))
        if dr <= int(self.cfg.red_threshold):
            closeness = float(int(self.cfg.red_threshold) - dr + 1) / float(int(self.cfg.red_threshold) + 1)
            parts["red_near"] = -float(self.cfg.k_red_near) * float(closeness)

        r_shape = float(
            parts["green_progress"]
            + parts["red_near"]
            + parts["object_entry"]
            + parts["forage"]
            + parts["skip_forage"]
            + parts["revisit"]
            + parts["idle"]
            + parts["time_cost"]
        )

        r_train = float(r_env + r_shape)

        # update shaping state for next step
        self._prev_d_green = dg
        self._was_on_object = on_obj_now

        self._set_last_rewards(r_env, r_shape, parts)

        return self._get_obs(), r_train, terminated, truncated, self._get_info()

    def render(self):
        g = self.cfg.grid_size
        ax, ay = self.agent_xy
        lines = [f"time_left={self.time_left}  r_env={self._last_r_env:+.3f}  r_shape={self._last_r_shape:+.3f}"]
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

    def _set_last_rewards(self, r_env: float, r_shape: float, parts: Dict[str, float]) -> None:
        self._last_r_env = float(r_env)
        self._last_r_shape = float(r_shape)
        self._last_shape_parts = dict(parts)

    def _spawn_initial_objects(self):
        empties = self._empty_tiles(exclude_agent=True)
        self.rng.shuffle(empties)
        positions = empties[:4]

        colors = [self.COLOR_GREEN] * int(self.cfg.n_green) + [self.COLOR_RED] * int(self.cfg.n_red)
        self.rng.shuffle(colors)

        for pos, col in zip(positions, colors):
            self.objects[pos] = (int(col), False)

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
            new_pos = empties[int(self.rng.integers(0, len(empties)))]
            self.objects[new_pos] = (color, False)

        return r

    def _min_manhattan_to_color(self, color: int) -> int:
        ax, ay = self.agent_xy
        best = 999
        for (x, y), (c, _rev) in self.objects.items():
            if int(c) != int(color):
                continue
            d = abs(ax - x) + abs(ay - y)
            if d < best:
                best = d
        if best == 999:
            # should not happen because we keep counts invariant
            best = (self.cfg.grid_size - 1) * 2
        return int(best)

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
                if int(color) == self.COLOR_GREEN:
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
        info = {
            "time_left": int(self.time_left),
            "agent_x": int(self.agent_xy[0]),
            "agent_y": int(self.agent_xy[1]),
            "n_objects": int(len(self.objects)),
            "n_revealed": int(n_revealed),
            # reward bookkeeping
            "r_env": float(self._last_r_env),
            "r_shape": float(self._last_r_shape),
        }
        # flatten shaping parts for easy logging
        for k, v in self._last_shape_parts.items():
            info[f"shape_{k}"] = float(v)
        return info
