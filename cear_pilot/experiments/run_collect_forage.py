# cear_pilot/experiments/run_collect_forage.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import pandas as pd

from cear_pilot.envs.forage_grid import ForageGridEnv, ForageGridConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

from cear_pilot.training.pygame_viewer_forage import PygameForageViewer


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[int(idx)] = 1.0
    return v


def build_agent_from_meta(meta: Dict[str, Any], device: str):
    """
    Loads configs the same way as your default run_collect.
    Assumes meta has keys:
      meta["agent_cfg"]["encoder"/"world"/"state"/"policy"]
      meta["decoder_cfg"]
    """
    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    # encoder
    agent_cfg.encoder.obs_dim = enc["obs_dim"]
    agent_cfg.encoder.proprio_dim = enc["proprio_dim"]
    agent_cfg.encoder.z_dim = enc["z_dim"]
    agent_cfg.encoder.p_dim = enc["p_dim"]
    agent_cfg.encoder.hidden = enc["hidden"]
    agent_cfg.encoder.dropout = enc["dropout"]

    # world
    agent_cfg.world.z_dim = world["z_dim"]
    agent_cfg.world.p_dim = world["p_dim"]
    agent_cfg.world.g_dim = world["g_dim"]
    agent_cfg.world.g_damping = world["g_damping"]
    agent_cfg.world.layernorm = world["layernorm"]

    # state
    agent_cfg.state.z_dim = state["z_dim"]
    agent_cfg.state.p_dim = state["p_dim"]
    agent_cfg.state.g_dim = state["g_dim"]
    agent_cfg.state.s_dim = state["s_dim"]
    agent_cfg.state.hidden = state["hidden"]
    agent_cfg.state.dropout = state["dropout"]
    agent_cfg.state.g_influence = state["g_influence"]

    # policy
    agent_cfg.policy.s_dim = pol["s_dim"]
    agent_cfg.policy.hidden = pol["hidden"]
    agent_cfg.policy.n_actions = pol["n_actions"]
    agent_cfg.policy.dropout = pol["dropout"]

    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--ablate_g", action="store_true")

    # stance intervention (optional)
    ap.add_argument("--do_g", type=str, default="", choices=["", "shock", "swap", "zero"],
                    help="Optional do(g) at episode start. Uses agent.apply_perturbation(kind=...)")
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    # pygame
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--fps", type=int, default=10)

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    # env: ALWAYS forage
    env_cfg_dict = meta.get("env_cfg", {})
    # fall back to defaults if env_cfg absent (still usable)
    env_cfg = ForageGridConfig(**env_cfg_dict) if env_cfg_dict else ForageGridConfig(seed=args.seed)
    env = ForageGridEnv(config=env_cfg)

    agent, decoder = build_agent_from_meta(meta, device=args.device)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)

    run_meta = {
        "mode": "collect_forage",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": args.episodes,
        "seed": args.seed,
        "device": args.device,
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "do_g": args.do_g,
        "do_g_scale": args.do_g_scale,
        "env_cfg": asdict(env_cfg),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)

    viewer = None
    if args.view:
        viewer = PygameForageViewer(cell_px=90, fps=args.fps, title="Collect (Forage)")

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)

        # Optional do(g) intervention at episode start
        if args.do_g:
            agent.apply_perturbation(kind=args.do_g, scale=float(args.do_g_scale))

        last_action = 0
        done = False
        t = 0
        ep_ret = 0.0

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
            a_int = int(action.item())

            obs_next, r, terminated, truncated, info2 = env.step(a_int)
            ep_ret += float(r)

            g = out["g"].squeeze(0).cpu().numpy()
            z = out["z"].squeeze(0).cpu().numpy()
            s = out["s"].squeeze(0).cpu().numpy()

            row: Dict[str, Any] = {
                "episode": ep,
                "step": t,
                "action": a_int,
                "reward": float(r),
                "ep_return_so_far": float(ep_ret),
                "time_left": int(info2.get("time_left", -1)),
                "x": int(info2.get("agent_x", -1)),
                "y": int(info2.get("agent_y", -1)),
                "n_revealed": int(info2.get("n_revealed", -1)),
            }

            for i, v in enumerate(obs.astype(np.float32)):
                row[f"obs_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)

            rows.append(row)

            # viewer
            if viewer is not None:
                g_norm = float(np.linalg.norm(g))
                ok = viewer.draw(env, step=t, episode=ep, last_action=a_int,
                                 reward=float(r), total_reward=float(ep_ret), g_norm=g_norm)
                if ok is False:
                    print("Viewer closed; stopping collection.")
                    done = True
                    break

            obs = obs_next
            last_action = a_int
            done = bool(terminated or truncated)
            t += 1

        print(f"[collect] ep={ep} return={ep_ret:.2f}")

    df = pd.DataFrame(rows)
    out_csv = run_dir / "traj_forage.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
