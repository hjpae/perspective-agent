# cear_pilot/experiments/run_switch_perturb.py
# -*- coding: utf-8 -*-
"""
Single-episode demo: regime switch + (optional) perturbation(s), with optional action replay.

Outputs:
  outputs/runs/<timestamp>/
    traj.parquet (or traj.csv)
    meta.json

Logged per step:
  - g, s, z
  - action, zone_id, x,y,t
  - policy stats computed from s_t: logits_act, pi_act, pi_max, pi_entropy, pi_argmax
  - flags: switched, perturbed_1, perturbed_2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

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

    # Encoder
    agent_cfg.encoder.obs_dim = enc["obs_dim"]
    agent_cfg.encoder.proprio_dim = enc["proprio_dim"]
    agent_cfg.encoder.z_dim = enc["z_dim"]
    agent_cfg.encoder.p_dim = enc["p_dim"]
    agent_cfg.encoder.hidden = enc["hidden"]
    agent_cfg.encoder.dropout = enc["dropout"]

    # World
    agent_cfg.world.z_dim = world["z_dim"]
    agent_cfg.world.p_dim = world["p_dim"]
    agent_cfg.world.g_dim = world["g_dim"]
    agent_cfg.world.g_damping = world["g_damping"]
    agent_cfg.world.layernorm = world["layernorm"]

    # State
    agent_cfg.state.z_dim = state["z_dim"]
    agent_cfg.state.p_dim = state["p_dim"]
    agent_cfg.state.g_dim = state["g_dim"]
    agent_cfg.state.s_dim = state["s_dim"]
    agent_cfg.state.hidden = state["hidden"]
    agent_cfg.state.dropout = state["dropout"]
    agent_cfg.state.g_influence = state["g_influence"]

    # Policy
    agent_cfg.policy.s_dim = pol["s_dim"]
    agent_cfg.policy.hidden = pol["hidden"]
    agent_cfg.policy.n_actions = pol["n_actions"]
    agent_cfg.policy.dropout = pol["dropout"]

    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)

    return agent, decoder, env


def compute_policy_stats(agent: CEARAgent, s_t: torch.Tensor, a_int: int) -> Dict[str, Any]:
    """
    Compute policy logits/probs/entropy from s_t.
    This works even in action-replay mode (policy is not used to act, but we can still measure it).
    """
    with torch.no_grad():
        logits = agent.policy(s_t.detach())  # [B, n_actions]
        probs = torch.softmax(logits, dim=-1)

        pi_max = float(probs.max(dim=-1).values.item())
        pi_argmax = int(probs.argmax(dim=-1).item())

        eps = 1e-8
        pi_entropy = float((-(probs * torch.log(probs + eps)).sum(dim=-1)).item())

        logits_act = float(logits[0, a_int].item())
        pi_act = float(probs[0, a_int].item())

    return {
        "logits_act": logits_act,
        "pi_act": pi_act,
        "pi_max": pi_max,
        "pi_entropy": pi_entropy,
        "pi_argmax": pi_argmax,
    }


def maybe_load_actions(path_str: str) -> Optional[List[int]]:
    if not str(path_str).strip():
        return None
    p = Path(path_str)
    obj = json.loads(p.read_text())
    if isinstance(obj, dict) and "actions" in obj:
        actions = [int(a) for a in obj["actions"]]
    elif isinstance(obj, list):
        actions = [int(a) for a in obj]
    else:
        raise ValueError("replay_actions JSON must be a list or a dict with key 'actions'")
    if len(actions) == 0:
        raise ValueError("replay_actions is empty")
    return actions


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--steps", type=int, default=240, help="Max env steps for the episode")
    ap.add_argument("--greedy", action="store_true", help="Greedy action (only when not replaying)")
    ap.add_argument("--ablate_g", action="store_true", help="Force g=0 (ablation baseline)")

    ap.add_argument("--outdir", type=str, default="", help="Override run dir")

    # Regime switch
    ap.add_argument("--zone_sigma", type=float, nargs=3, required=True, help="Sigma before switch: s0 s1 s2")
    ap.add_argument("--t_switch", type=int, default=80, help="Switch time (step index in THIS rollout)")
    ap.add_argument("--zone_sigma2", type=float, nargs=3, required=True, help="Sigma after switch: s0 s1 s2")

    # Perturbation (one or two)
    ap.add_argument("--t_perturb", type=int, default=120, help="Perturb time (step index). Set <0 to disable.")
    ap.add_argument("--kind", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--t_perturb2", type=int, default=-1, help="Optional second perturb time. Set <0 to disable.")
    ap.add_argument("--kind2", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--scale2", type=float, default=1.0)

    # Action replay (optional)
    ap.add_argument("--replay_actions", type=str, default="", help="JSON path with action list (forces same actions)")

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=args.device, zone_sigma_override=args.zone_sigma)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    # Save meta
    run_meta = {
        "mode": "switch_perturb",
        "ckpt": str(Path(args.ckpt).resolve()),
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "zone_sigma": tuple(float(x) for x in args.zone_sigma),
        "t_switch": int(args.t_switch),
        "zone_sigma2": tuple(float(x) for x in args.zone_sigma2),
        "t_perturb": int(args.t_perturb),
        "kind": args.kind,
        "scale": float(args.scale),
        "t_perturb2": int(args.t_perturb2),
        "kind2": args.kind2,
        "scale2": float(args.scale2),
        "replay_actions": str(args.replay_actions) if str(args.replay_actions).strip() else "",
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    actions_replay = maybe_load_actions(args.replay_actions)

    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)

    n_actions = int(env.action_space.n)
    last_action = 4  # stay

    rows: List[Dict[str, Any]] = []
    done = False
    t = 0

    while not done and t < args.steps:
        # Regime switch (apply BEFORE stepping at time t)
        switched = 0
        if t == args.t_switch:
            switched = 1
            # Require env to support set_zone_sigma (you already used this pattern)
            env.set_zone_sigma(args.zone_sigma2)

        # Perturb (apply BEFORE agent forward/step at time t)
        pert1 = 0
        if args.t_perturb >= 0 and t == args.t_perturb:
            pert1 = 1
            agent.apply_perturbation(kind=args.kind, scale=args.scale)

        pert2 = 0
        if args.t_perturb2 >= 0 and t == args.t_perturb2:
            pert2 = 1
            agent.apply_perturbation(kind=args.kind2, scale=args.scale2)

        x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

        if actions_replay is None:
            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
            a_int = int(action.item())
        else:
            if t >= len(actions_replay):
                break
            a_int = int(actions_replay[t])
            with torch.no_grad():
                out = agent.forward_step(x_t, p_t, ablate_g=args.ablate_g)

        # Step env with chosen/replayed action
        obs_next, _, terminated, truncated, info2 = env.step(a_int)

        # Latents
        g = out["g"].squeeze(0).detach().cpu().numpy()
        s = out["s"].squeeze(0).detach().cpu().numpy()
        z = out["z"].squeeze(0).detach().cpu().numpy()

        # Policy stats (computed from s_t)
        s_t = out["s"]
        pi_stats = compute_policy_stats(agent, s_t, a_int)

        row = {
            "t": int(info2["t"]),
            "x": int(info2["x"]),
            "y": int(info2["y"]),
            "zone_id": int(info2["zone_id"]),
            "action": int(a_int),
            "switched": int(switched),
            "perturbed_1": int(pert1),
            "perturbed_2": int(pert2),
            **pi_stats,
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

        obs = obs_next
        last_action = a_int
        done = bool(terminated or truncated)
        t += 1

    saved = try_save_table(rows, run_dir / "traj")
    print(f"[OK] Saved switch+perturb traj to: {saved}")
    print(f"[OK] Run dir: {run_dir}")


if __name__ == "__main__":
    main()
