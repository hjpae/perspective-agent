# cear_pilot/experiments/run_collect.py
# -*- coding: utf-8 -*-
"""
Rollout collection with a trained checkpoint.

Outputs:
  outputs/runs/<timestamp>/
    traj.parquet (or traj.csv fallback)
    meta.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneGridEnv, NZoneConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_save_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    """
    Save to parquet if possible; otherwise csv.
    Returns actual saved path.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    parquet_path = out_path.with_suffix(".parquet")
    csv_path = out_path.with_suffix(".csv")

    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def build_agent_from_meta(meta: Dict[str, Any], device: str, zone_sigma_override=None) -> tuple[CEARAgent, ObsDecoder, NZoneGridEnv]:
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if zone_sigma_override is not None:
        env_cfg.zone_sigma = tuple(float(x) for x in zone_sigma_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    # wire dims from meta
    agent_cfg.encoder.obs_dim = enc["obs_dim"]
    agent_cfg.encoder.proprio_dim = enc["proprio_dim"]
    agent_cfg.encoder.z_dim = enc["z_dim"]
    agent_cfg.encoder.p_dim = enc["p_dim"]
    agent_cfg.encoder.hidden = enc["hidden"]
    agent_cfg.encoder.dropout = enc["dropout"]

    agent_cfg.world.z_dim = world["z_dim"]
    agent_cfg.world.p_dim = world["p_dim"]
    agent_cfg.world.g_dim = world["g_dim"]
    agent_cfg.world.g_damping = world["g_damping"]
    agent_cfg.world.layernorm = world["layernorm"]

    agent_cfg.state.z_dim = state["z_dim"]
    agent_cfg.state.p_dim = state["p_dim"]
    agent_cfg.state.g_dim = state["g_dim"]
    agent_cfg.state.s_dim = state["s_dim"]
    agent_cfg.state.hidden = state["hidden"]
    agent_cfg.state.dropout = state["dropout"]
    agent_cfg.state.g_influence = state["g_influence"]

    agent_cfg.policy.s_dim = pol["s_dim"]
    agent_cfg.policy.hidden = pol["hidden"]
    agent_cfg.policy.n_actions = pol["n_actions"]
    agent_cfg.policy.dropout = pol["dropout"]

    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)

    return agent, decoder, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt from training")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true", help="Use greedy action selection")
    ap.add_argument("--outdir", type=str, default="", help="Override output dir (default: outputs/runs/<timestamp>)")
    ap.add_argument("--ablate_g", action="store_true", help="Force g=0 (ablation baseline)")
    ap.add_argument("--zone_sigma", type=float, nargs=3, default=None,
                help="Override env zone_sigma as three floats: s0 s1 s2")
    ap.add_argument("--replay_actions", type=str, default="",
                help="Path to JSON containing action list for action-replay (forces same actions).")

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=args.device, zone_sigma_override=args.zone_sigma)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device)
    decoder.to(args.device)
    agent.eval()
    decoder.eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    # Save run meta
    run_meta = {
        "mode": "collect",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": args.episodes,
        "seed": args.seed,
        "device": args.device,
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)
    
    # ---- optional: load action-replay sequence
    replay_actions = None
    if str(args.replay_actions).strip():
        p = Path(args.replay_actions)
        obj = json.loads(p.read_text())
        if isinstance(obj, dict) and "actions" in obj:
            replay_actions = [int(a) for a in obj["actions"]]
        elif isinstance(obj, list):
            replay_actions = [int(a) for a in obj]
        else:
            raise ValueError("replay_actions JSON must be a list or a dict with key 'actions'")
        if len(replay_actions) == 0:
            raise ValueError("replay_actions is empty")

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)
        last_action = 4  # stay

        done = False
        t = 0  # step counter for replay index
        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)
        
            if replay_actions is None:
                with torch.no_grad():
                    action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
                a_int = int(action.item())
            else:
                # Action-replay: while g updates with obs, env action stays as replay
                if t >= len(replay_actions):
                    break
                a_int = int(replay_actions[t])
                with torch.no_grad():
                    out = agent.forward_step(x_t, p_t, ablate_g=args.ablate_g)

            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            # log
            g = out["g"].squeeze(0).cpu().numpy()
            s = out["s"].squeeze(0).cpu().numpy()
            z = out["z"].squeeze(0).cpu().numpy()

            row = {
                "episode": ep,
                "t": int(info2["t"]),
                "x": int(info2["x"]),
                "y": int(info2["y"]),
                "zone_id": int(info2["zone_id"]),
                "action": a_int,
            }
            # flatten obs and latents
            for i, v in enumerate(obs.astype(np.float32)):
                row[f"obs_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)

            rows.append(row)

            obs = obs_next
            last_action = a_int
            done = bool(terminated or truncated)
            t += 1

    saved_path = try_save_table(rows, run_dir / "traj")
    print(f"Saved trajectories to: {saved_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
