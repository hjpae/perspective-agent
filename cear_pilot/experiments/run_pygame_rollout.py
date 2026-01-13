# cear_pilot/experiments/run_pygame_rollout.py
# -*- coding: utf-8 -*-
"""
Minimal pygame rollout viewer for a trained checkpoint.

Usage:
  python -m cear_pilot.experiments.run_pygame_rollout \
    --ckpt outputs/runs/<TRAIN_ID>/ckpt.pt --T 400 --greedy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig
from cear_pilot.training.pygame_viewer import PygameGridViewer


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[int(idx)] = 1.0
    return v


def build_agent_from_meta(meta: Dict[str, Any], device: str, max_steps_override: Optional[int] = None):
    # Env
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if max_steps_override is not None:
        env_cfg.max_steps = int(max_steps_override)
    env = NZoneGridEnv(config=env_cfg)

    # Agent
    agent_cfg = AgentConfig(device=device)

    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    agent = CEARAgent(agent_cfg)

    # Decoder (not strictly required for viewing, but load for completeness)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)

    return agent, decoder, env


def set_env_zone_sigma(env: NZoneGridEnv, sigma_triplet: Tuple[float, float, float]) -> None:
    # Runtime override (used elsewhere in your repo)
    env._zone_sigma = np.array(list(sigma_triplet), dtype=np.float32)


def policy_entropy_from_logits(logits: np.ndarray) -> float:
    ex = np.exp(logits - np.max(logits))
    p = ex / (np.sum(ex) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--greedy", action="store_true")

    # Optional: visualize under a fixed sigma regime
    ap.add_argument("--sigma", type=float, nargs=3, default=None)

    # Viewer
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--cell_px", type=int, default=40)

    args = ap.parse_args()

    device = args.device
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(Path(args.ckpt), map_location=device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=device, max_steps_override=args.T)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(device).eval()
    decoder.to(device).eval()

    if args.sigma is not None:
        set_env_zone_sigma(env, tuple(float(x) for x in args.sigma))

    viewer = PygameGridViewer(
        width=env.cfg.width,
        height=env.cfg.height,
        cell_px=args.cell_px,
        fps=args.fps,
        title="CEAR rollout (pygame)",
    )

    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)

    n_actions = int(env.action_space.n)
    last_action = 4  # "stay" in your setup

    try:
        for t_global in range(int(args.T)):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=bool(args.greedy), ablate_g=False)

            a_int = int(action.item())
            logits = out["logits"].squeeze(0).detach().cpu().numpy()
            g_vec = out["g"].squeeze(0).detach().cpu().numpy()

            obs, _, terminated, truncated, info2 = env.step(a_int)

            ent = policy_entropy_from_logits(logits)
            g_norm = float(np.linalg.norm(g_vec))

            # Draw. Viewer handles events; close window to stop.
            ok = viewer.draw(
                env=env,
                step=t_global,
                episode=0,
                last_action=a_int,
                loss=0.0,
                loss_pred=0.0,
                loss_smooth=0.0,
                entropy=ent,
                g_norm=g_norm,
            )
            if ok is False:
                break

            last_action = a_int
            if terminated or truncated:
                break

    finally:
        viewer.close()


if __name__ == "__main__":
    main()
