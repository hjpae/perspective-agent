# cear_pilot/experiments/run_collect.py
# -*- coding: utf-8 -*-
"""
Rollout collection with a trained checkpoint.

Outputs:
  outputs/runs/<timestamp>/
    traj.parquet (or traj.csv fallback)
    meta.json

Features:
  - action replay (fixed env actions) to isolate latent dynamics
  - optional regime switch: change env zone_sigma at a chosen timestep
  - logs policy outputs every step (even under action replay):
      pi_max, pi_entropy, pi_argmax, logits_act_*, pi_act_*
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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


def _tuple3(x) -> Optional[Tuple[float, float, float]]:
    if x is None:
        return None
    return (float(x[0]), float(x[1]), float(x[2]))


def _load_replay_actions(path: str) -> Optional[List[int]]:
    if not str(path).strip():
        return None
    p = Path(path)
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


def build_agent_from_meta(
    meta: Dict[str, Any],
    device: str,
    zone_sigma_override: Optional[Tuple[float, float, float]] = None,
) -> tuple[CEARAgent, ObsDecoder, NZoneGridEnv]:
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if zone_sigma_override is not None:
        env_cfg.zone_sigma = tuple(float(v) for v in zone_sigma_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    # Wire dims from meta (explicit for clarity)
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


def _policy_stats_from_s(agent: CEARAgent, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """
    Compute policy logits/probs and summary stats from state s.
    Uses s.detach() to avoid any accidental gradient linkage (eval mode anyway).
    Returns:
      logits_act: (1, A)
      pi_act: (1, A)
      entropy: float
      pi_max: float
      pi_argmax: int
    """
    logits_act = agent.policy(s_t.detach())  # (1, A)
    pi_act = torch.softmax(logits_act, dim=-1)  # (1, A)
    entropy = (-torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1)).mean()
    pi_max = pi_act.max(dim=-1).values.mean()
    pi_argmax = int(torch.argmax(pi_act, dim=-1).item())
    return logits_act, pi_act, float(entropy.item()), float(pi_max.item()), pi_argmax


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

    # Regime switch options
    ap.add_argument("--t_switch", type=int, default=-1,
                    help="If >=0, switch env zone_sigma at this timestep (uses internal step counter).")
    ap.add_argument("--zone_sigma2", type=float, nargs=3, default=None,
                    help="Second sigma after switch: s0 s1 s2")

    # Optional: store full logits/pi (can be large but useful for debugging)
    ap.add_argument("--log_policy_full", action="store_true",
                    help="If set, log logits_act_* and pi_act_* columns for every action.")

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    sigma1 = _tuple3(args.zone_sigma)
    sigma2 = _tuple3(args.zone_sigma2)

    agent, decoder, env = build_agent_from_meta(meta, device=args.device, zone_sigma_override=sigma1)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    # Save run meta
    run_meta = {
        "mode": "collect",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "zone_sigma": sigma1,
        "t_switch": int(args.t_switch),
        "zone_sigma2": sigma2,
        "replay_actions": str(Path(args.replay_actions).resolve()) if str(args.replay_actions).strip() else "",
        "log_policy_full": bool(args.log_policy_full),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)

    replay_actions = _load_replay_actions(args.replay_actions)

    # Sanity check for regime switch configuration
    if args.t_switch >= 0 and sigma2 is None:
        raise ValueError("t_switch is set but zone_sigma2 is missing. Provide --zone_sigma2 s0 s1 s2.")
    if args.t_switch >= 0 and replay_actions is not None and args.t_switch >= len(replay_actions):
        print("[WARN] t_switch >= len(replay_actions). Switch may never happen.")

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)
        last_action = 4  # stay

        done = False
        t = 0
        switched = False

        while not done:
            # Regime switch before model step
            if (not switched) and args.t_switch >= 0 and sigma2 is not None and t == args.t_switch:
                env.set_zone_sigma(sigma2)  # requires env helper method
                switched = True

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            # Step model / policy
            if replay_actions is None:
                # Normal rollout: agent chooses action (greedy/stochastic)
                with torch.no_grad():
                    action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
                a_int = int(action.item())
            else:
                # Action replay: env action is forced, but g/s updates from obs
                if t >= len(replay_actions):
                    break
                a_int = int(replay_actions[t])
                with torch.no_grad():
                    out = agent.forward_step(x_t, p_t, ablate_g=args.ablate_g)

            # Compute policy outputs from s (always, even under replay)
            with torch.no_grad():
                s_t = out["s"]
                logits_act, pi_act, pi_entropy, pi_max, pi_argmax = _policy_stats_from_s(agent, s_t)

            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            # Extract latents for logging
            g = out["g"].squeeze(0).detach().cpu().numpy()
            s = out["s"].squeeze(0).detach().cpu().numpy()
            z = out["z"].squeeze(0).detach().cpu().numpy()

            row: Dict[str, Any] = {
                "episode": int(ep),
                "t": int(info2.get("t", t)),
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "zone_id": int(info2.get("zone_id", -1)),

                # The action actually executed in the env
                "action_env": int(a_int),

                # Under replay, this is the forced action; under normal, equals action_env
                "action_replay": int(a_int) if replay_actions is not None else -1,

                # What the policy would prefer at this step (given s)
                "pi_argmax": int(pi_argmax),
                "pi_max": float(pi_max),
                "pi_entropy": float(pi_entropy),

                # Regime switch flags
                "switched": int(switched),
                "t_switch": int(args.t_switch),
            }

            # Record sigma parameters (for robust downstream analysis)
            if sigma1 is not None:
                row["sigma_0"] = float(sigma1[0])
                row["sigma_1"] = float(sigma1[1])
                row["sigma_2"] = float(sigma1[2])
            if sigma2 is not None:
                row["sigma2_0"] = float(sigma2[0])
                row["sigma2_1"] = float(sigma2[1])
                row["sigma2_2"] = float(sigma2[2])

            # Optional: log full logits/pi vectors
            if args.log_policy_full:
                la = logits_act.squeeze(0).detach().cpu().numpy()
                pa = pi_act.squeeze(0).detach().cpu().numpy()
                for i in range(n_actions):
                    row[f"logits_act_{i}"] = float(la[i])
                    row[f"pi_act_{i}"] = float(pa[i])

            # Flatten obs and latents
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
