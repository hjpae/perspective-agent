# cear_pilot/experiments/run_perturb.py
# -*- coding: utf-8 -*-
"""
Collect a single long episode with a perturbation to g at a specified time.
"""

from __future__ import annotations

import argparse
import json
import time
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


def try_save_table(rows, out_path: Path) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def build_agent_from_meta(meta: Dict[str, Any], device: str):
    env_cfg = NZoneConfig(**meta["env_cfg"])
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)
    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    agent_cfg.encoder.__dict__.update(enc)
    agent_cfg.world.__dict__.update(world)
    agent_cfg.state.__dict__.update(state)
    agent_cfg.policy.__dict__.update(pol)

    agent = CEARAgent(agent_cfg)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--t_perturb", type=int, default=80)
    ap.add_argument("--kind", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=args.device)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "perturb",
        "ckpt": str(Path(args.ckpt).resolve()),
        "seed": args.seed,
        "device": args.device,
        "greedy": bool(args.greedy),
        "t_perturb": args.t_perturb,
        "kind": args.kind,
        "scale": args.scale,
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)
    last_action = 4
    n_actions = int(env.action_space.n)

    rows: List[Dict[str, Any]] = []
    done = False
    while not done:
        t = int(info["t"])
        if t == args.t_perturb:
            agent.apply_perturbation(kind=args.kind, scale=args.scale)

        x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.no_grad():
            action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=False)

        a_int = int(action.item())
        obs_next, _, terminated, truncated, info2 = env.step(a_int)

        g = out["g"].squeeze(0).cpu().numpy()
        s = out["s"].squeeze(0).cpu().numpy()

        row = {
            "t": int(info2["t"]),
            "x": int(info2["x"]),
            "y": int(info2["y"]),
            "zone_id": int(info2["zone_id"]),
            "action": a_int,
            "perturbed": int(t == args.t_perturb),
        }
        for i, v in enumerate(obs.astype(np.float32)):
            row[f"obs_{i}"] = float(v)
        for i, v in enumerate(s):
            row[f"s_{i}"] = float(v)
        for i, v in enumerate(g):
            row[f"g_{i}"] = float(v)

        rows.append(row)

        obs = obs_next
        info = info2
        last_action = a_int
        done = bool(terminated or truncated)

    saved_path = try_save_table(rows, run_dir / "traj")
    print(f"Saved perturb traj to: {saved_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
