# cear_pilot/envs/nzone_grid.py
# -*- coding: utf-8 -*-
"""
N-zone Gridworld (Gymnasium Env)

- 2D grid split into 3 vertical zones (0/1/2)
- Observation: noisy vector whose mean depends on current zone_id
- No external reward (reward=0.0) by default; the point is to study latent dynamics.

Actions:
  0: up, 1: down, 2: left, 3: right, 4: stay

Gymnasium API:
  obs, info = env.reset(seed=...)
  obs, reward, terminated, truncated, info = env.step(action)
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

    # if True, include (x,y) normalized coordinates in obs tail (adds 2 dims)
    include_xy: bool = False


class NZoneGridEnv(gym.Env):
    """
    A minimal gridworld for Phase-1 pilot runs.

    The main thing you care about: zone identity induces a stable signal in observation
    space that a slow latent can form basins over.

    Reward is 0 by default (you can extend later).
    """

    metadata = {"render_modes": ["human"], "render_fps": 8}

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

        # Observation is unbounded (noise); we bound loosely for Gymnasium.
        high = np.ones((self.obs_dim,), dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng(0)

        # zone prototype means in observation space
        self._zone_mu = np.zeros((3, self.base_obs_dim), dtype=np.float32)
        self._zone_sigma = np.array(self.cfg.zone_sigma, dtype=np.float32)

        # state
        self.x = 0
        self.y = 0
        self.t = 0

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
        # split into 3 vertical zones by x coordinate
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
            obs_xy = np.array([self.x / max(1, self.W - 1), self.y / max(1, self.H - 1)], dtype=np.float32)
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
            # also re-sample zone prototypes deterministically from seed (optional but nice)
            self._init_zone_prototypes(seed=seed)

        # start in center
        self.x = self.W // 2
        self.y = self.H // 2
        self.t = 0

        obs = self._observe()
        info = {"zone_id": self.zone_id(), "x": self.x, "y": self.y, "t": self.t}
        return obs, info

    def step(self, action: int):
        # Apply action
        if action == 0:  # up
            self.y = max(0, self.y - 1)
        elif action == 1:  # down
            self.y = min(self.H - 1, self.y + 1)
        elif action == 2:  # left
            self.x = max(0, self.x - 1)
        elif action == 3:  # right
            self.x = min(self.W - 1, self.x + 1)
        elif action == 4:  # stay
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        self.t += 1

        obs = self._observe()

        # No extrinsic reward (Phase-1); keep 0.0.
        reward = 0.0

        terminated = False  # no terminal states
        truncated = self.t >= self.max_steps

        info = {
            "zone_id": self.zone_id(),
            "x": self.x,
            "y": self.y,
            "t": self.t,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # Minimal ASCII render
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        grid[self.y][self.x] = "A"
        s = "\n".join("".join(row) for row in grid)
        print(s)
        print(f"t={self.t} zone={self.zone_id()} pos=({self.x},{self.y})")

    def close(self):
        pass


# Convenience factory (optional)
def make_env(**kwargs) -> NZoneGridEnv:
    cfg = NZoneConfig(**kwargs)
    return NZoneGridEnv(config=cfg)
