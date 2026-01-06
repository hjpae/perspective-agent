# cear_pilot/envs/nzone_grid.py
# -*- coding: utf-8 -*-
"""
N-zone Gridworld (Gymnasium Env)

Patched for:
- active exploration (anti-stall)
- UI-friendly RGB rendering
- Phase-1 compatible intrinsic pressures only

Zones:
  3 vertical zones (0 / 1 / 2)

Actions:
  0: up, 1: down, 2: left, 3: right, 4: stay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError(
        "This environment requires gymnasium. Install with: pip install gymnasium"
    ) from e


# -------------------------
# Config
# -------------------------
@dataclass
class NZoneConfig:
    width: int = 15
    height: int = 9
    obs_dim: int = 8
    max_steps: int = 240

    # observation mean separation scale
    zone_mu_scale: float = 2.5

    # per-zone observation noise (hazard noisier)
    zone_sigma: Tuple[float, float, float] = (0.25, 0.40, 0.70)

    # include normalized (x,y) in obs tail
    include_xy: bool = False

    # ---- intrinsic pressures (NEW) ----
    step_penalty: float = 0.01      # time cost (always)
    stall_penalty: float = 0.05     # extra cost if no movement
    novel_bonus: float = 0.05       # bonus for visiting a new cell


# -------------------------
# Env
# -------------------------
class NZoneGridEnv(gym.Env):
    """
    Phase-1 gridworld:
    - no extrinsic task reward
    - intrinsic time / novelty pressures only
    - designed to prevent trivial 'stall' solutions
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, config: Optional[NZoneConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or NZoneConfig()
        self.render_mode = render_mode

        self.W = int(self.cfg.width)
        self.H = int(self.cfg.height)
        self.max_steps = int(self.cfg.max_steps)

        self.base_obs_dim = int(self.cfg.obs_dim)
        self.obs_dim = self.base_obs_dim + (2 if self.cfg.include_xy else 0)

        self.action_space = spaces.Discrete(5)

        high = np.ones((self.obs_dim,), dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng(0)

        # zone prototype means
        self._zone_mu = np.zeros((3, self.base_obs_dim), dtype=np.float32)
        self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

        # state
        self.x = 0
        self.y = 0
        self.t = 0

        # visited cells (NEW)
        self.visited = set()

        self._init_zone_prototypes(seed=0)

    # -----------------
    # Helpers
    # -----------------
    def _init_zone_prototypes(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, size=(3, self.base_obs_dim)).astype(np.float32)
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
        self._zone_mu = base * float(self.cfg.zone_mu_scale)

    def zone_id(self) -> int:
        if self.x < self.W / 3:
            return 0
        elif self.x < 2 * self.W / 3:
            return 1
        else:
            return 2

    def _observe(self) -> np.ndarray:
        zid = self.zone_id()
        mu = self._zone_mu[zid]
        sigma = float(self._zone_sigma[zid])

        obs = mu + self._rng.normal(0, sigma, size=(self.base_obs_dim,)).astype(np.float32)

        if self.cfg.include_xy:
            obs_xy = np.array(
                [self.x / max(1, self.W - 1), self.y / max(1, self.H - 1)],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, obs_xy], axis=0)

        return obs.astype(np.float32)

    # -----------------
    # Gym API
    # -----------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._init_zone_prototypes(seed=seed)

        self.x = self.W // 2
        self.y = self.H // 2
        self.t = 0

        self.visited = set()
        self.visited.add((self.x, self.y))

        obs = self._observe()
        info = {"zone_id": self.zone_id(), "x": self.x, "y": self.y, "t": self.t}
        return obs, info

    def step(self, action: int):
        old_pos = (self.x, self.y)

        # Apply action
        if action == 0:      # up
            self.y = max(0, self.y - 1)
        elif action == 1:    # down
            self.y = min(self.H - 1, self.y + 1)
        elif action == 2:    # left
            self.x = max(0, self.x - 1)
        elif action == 3:    # right
            self.x = min(self.W - 1, self.x + 1)
        elif action == 4:    # stay
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        self.t += 1
        new_pos = (self.x, self.y)
        moved = new_pos != old_pos

        obs = self._observe()

        # ---- intrinsic reward shaping (NEW) ----
        reward = -self.cfg.step_penalty

        if not moved:
            reward -= self.cfg.stall_penalty

        if moved and new_pos not in self.visited:
            reward += self.cfg.novel_bonus
            self.visited.add(new_pos)

        terminated = False
        truncated = self.t >= self.max_steps

        info = {
            "zone_id": self.zone_id(),
            "x": self.x,
            "y": self.y,
            "t": self.t,
        }
        return obs, reward, terminated, truncated, info

    # -----------------
    # Rendering
    # -----------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        else:
            self._render_ascii()

    def _render_ascii(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        grid[self.y][self.x] = "A"
        s = "\n".join("".join(row) for row in grid)
        print(s)
        print(f"t={self.t} zone={self.zone_id()} pos=({self.x},{self.y})")

    def _render_rgb(self):
        cell = 24
        img = np.zeros((self.H * cell, self.W * cell, 3), dtype=np.uint8)

        zone_colors = np.array(
            [
                [220, 220, 220],  # zone 0
                [200, 230, 255],  # zone 1
                [255, 210, 210],  # zone 2
            ],
            dtype=np.uint8,
        )

        for y in range(self.H):
            for x in range(self.W):
                zid = 0
                if x >= self.W / 3 and x < 2 * self.W / 3:
                    zid = 1
                elif x >= 2 * self.W / 3:
                    zid = 2

                y0, y1 = y * cell, (y + 1) * cell
                x0, x1 = x * cell, (x + 1) * cell
                img[y0:y1, x0:x1] = zone_colors[zid]

        # agent (black square)
        ay, ax = self.y, self.x
        y0, y1 = ay * cell, (ay + 1) * cell
        x0, x1 = ax * cell, (ax + 1) * cell
        img[y0:y1, x0:x1] = np.array([0, 0, 0], dtype=np.uint8)

        # grid lines
        img[::cell, :, :] = 0
        img[:, ::cell, :] = 0

        return img

    def close(self):
        pass


def make_env(**kwargs) -> NZoneGridEnv:
    cfg = NZoneConfig(**kwargs)
    return NZoneGridEnv(config=cfg)
