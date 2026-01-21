# cear_pilot/experiments/run_collect_forage.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from cear_pilot.envs.forage_grid import ForageGridEnv, ForageGridConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig
from cear_pilot.training.pygame_viewer_forage import PygameForageViewer


# ---------------------------
# filesystem helpers
# ---------------------------

def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# obs parsing helpers
# ---------------------------

def split_obs_101(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Split the 101-dim observation into components.
    Returns:
      agent25, obj25, revg25, revr25, tfrac
    """
    obs = obs.astype(np.float32).reshape(-1)
    assert obs.shape[0] == 101, f"Expected obs_dim=101, got {obs.shape[0]}"
    agent = obs[0:25]
    obj = obs[25:50]
    revg = obs[50:75]
    revr = obs[75:100]
    tfrac = float(obs[100])
    return agent, obj, revg, revr, tfrac


def idx_from_xy(x: int, y: int, W: int = 5) -> int:
    return int(y) * int(W) + int(x)


def action_name(a: int) -> str:
    names = ["UP", "DOWN", "LEFT", "RIGHT", "INSPECT", "FORAGE"]
    if 0 <= int(a) < len(names):
        return names[int(a)]
    return str(a)


# ---------------------------
# model rebuild from meta
# ---------------------------

def build_agent_from_meta(meta, device: str):
    """
    Rebuild CEARAgent + decoder from saved meta.

    Robust to partially missing keys in older checkpoints:
    - Only core dims are required.
    - Non-core hyperparams (hidden, dropout, etc.) fall back to defaults.
    """
    from cear_pilot.models.agent import CEARAgent, AgentConfig
    from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

    # --------- Fallback if meta incomplete ----------
    if ("agent_cfg" not in meta) or ("decoder_cfg" not in meta):
        obs_dim = 101
        n_actions = 6

        agent_cfg = AgentConfig(device=device)
        agent_cfg.encoder.obs_dim = obs_dim
        agent_cfg.encoder.proprio_dim = n_actions

        agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
        agent_cfg.world.p_dim = agent_cfg.encoder.p_dim
        agent_cfg.state.z_dim = agent_cfg.encoder.z_dim
        agent_cfg.state.p_dim = agent_cfg.encoder.p_dim
        agent_cfg.state.g_dim = agent_cfg.world.g_dim

        agent_cfg.policy.n_actions = n_actions
        agent_cfg.policy.s_dim = agent_cfg.state.s_dim

        agent = CEARAgent(agent_cfg)

        dec_cfg = DecoderConfig(
            g_dim=agent_cfg.world.g_dim,
            n_actions=n_actions,
            obs_dim=obs_dim,
            hidden=64,
            dropout=0.0,
        )
        decoder = ObsDecoder(dec_cfg)
        return agent, decoder

    # --------- Normal path (robust) ----------
    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"].get("encoder", {})
    world = meta["agent_cfg"].get("world", {})
    state = meta["agent_cfg"].get("state", {})
    policy = meta["agent_cfg"].get("policy", {})

    # Encoder: core dims required (obs_dim, proprio_dim, z_dim, p_dim)
    if "obs_dim" in enc:
        agent_cfg.encoder.obs_dim = int(enc["obs_dim"])
    if "proprio_dim" in enc:
        agent_cfg.encoder.proprio_dim = int(enc["proprio_dim"])
    if "z_dim" in enc:
        agent_cfg.encoder.z_dim = int(enc["z_dim"])
    if "p_dim" in enc:
        agent_cfg.encoder.p_dim = int(enc["p_dim"])
    # Optional hyperparams
    if "hidden" in enc:
        agent_cfg.encoder.hidden = int(enc["hidden"])

    # World: core dims (g_dim, z_dim, p_dim)
    if "g_dim" in world:
        agent_cfg.world.g_dim = int(world["g_dim"])
    if "z_dim" in world:
        agent_cfg.world.z_dim = int(world["z_dim"])
    if "p_dim" in world:
        agent_cfg.world.p_dim = int(world["p_dim"])
    # Optional hyperparams
    if "hidden" in world:
        agent_cfg.world.hidden = int(world["hidden"])

    # State: core dims (s_dim, g_dim, z_dim, p_dim)
    if "s_dim" in state:
        agent_cfg.state.s_dim = int(state["s_dim"])
    if "g_dim" in state:
        agent_cfg.state.g_dim = int(state["g_dim"])
    if "z_dim" in state:
        agent_cfg.state.z_dim = int(state["z_dim"])
    if "p_dim" in state:
        agent_cfg.state.p_dim = int(state["p_dim"])
    # Optional hyperparams
    if "hidden" in state:
        agent_cfg.state.hidden = int(state["hidden"])

    # Policy(Q-head): core dims (n_actions, s_dim)
    if "n_actions" in policy:
        agent_cfg.policy.n_actions = int(policy["n_actions"])
    if "s_dim" in policy:
        agent_cfg.policy.s_dim = int(policy["s_dim"])
    # Optional hyperparams
    if "hidden" in policy:
        agent_cfg.policy.hidden = int(policy["hidden"])

    agent = CEARAgent(agent_cfg)

    d = meta.get("decoder_cfg", {})
    # decoder core dims (g_dim, n_actions, obs_dim) — optional hidden/dropout
    dec_cfg = DecoderConfig(
        g_dim=int(d.get("g_dim", agent_cfg.world.g_dim)),
        n_actions=int(d.get("n_actions", agent_cfg.policy.n_actions)),
        obs_dim=int(d.get("obs_dim", agent_cfg.encoder.obs_dim)),
        hidden=int(d.get("hidden", 64)),
        dropout=float(d.get("dropout", 0.0)),
    )
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder



def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[int(idx)] = 1.0
    return v


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="Path to ckpt.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    # acting mode
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--ablate_g", action="store_true", help="If set, g is forced to zero in forward_step.")

    # stance intervention at episode start
    ap.add_argument(
        "--do_g",
        type=str,
        default="",
        choices=["", "shock", "swap", "zero"],
        help="Optional do(g) at episode start.",
    )
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    # stance intervention mid-episode
    ap.add_argument(
        "--do_g_t",
        type=int,
        default=-1,
        help="If >=0, apply do(g) at this step index within the episode.",
    )
    ap.add_argument(
        "--do_g_mid",
        type=str,
        default="",
        choices=["", "shock", "swap", "zero"],
        help="Optional do(g) mid-episode (requires --do_g_t >= 0).",
    )
    ap.add_argument("--do_g_mid_scale", type=float, default=1.0)

    # output
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--save_latents", action="store_true", help="Include g/z/s vectors per step (bigger CSV).")
    ap.add_argument("--save_obs", action="store_true", help="Include full 101-dim obs per step (largest CSV).")

    # viewer
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--fps", type=int, default=10)

    args = ap.parse_args()

    # ---------------------------
    # load ckpt + env
    # ---------------------------
    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    # Always forage env. Use training env_cfg if present; otherwise fall back to defaults.
    env_cfg_dict = meta.get("env_cfg", {})
    env_cfg = ForageGridConfig(**env_cfg_dict) if env_cfg_dict else ForageGridConfig(seed=args.seed)
    env = ForageGridEnv(config=env_cfg)

    agent, decoder = build_agent_from_meta(meta, device=args.device)
    agent.load_state_dict(ckpt["agent_state"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state"], strict=True)
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    for p in agent.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    # ---------------------------
    # output dir + meta
    # ---------------------------
    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)

    run_meta = {
        "mode": "collect_forage",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "do_g": args.do_g,
        "do_g_scale": float(args.do_g_scale),
        "do_g_t": int(args.do_g_t),
        "do_g_mid": args.do_g_mid,
        "do_g_mid_scale": float(args.do_g_mid_scale),
        "save_latents": bool(args.save_latents),
        "save_obs": bool(args.save_obs),
        "env_cfg": asdict(env_cfg),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    # ---------------------------
    # viewer
    # ---------------------------
    viewer = None
    if args.view:
        viewer = PygameForageViewer(cell_px=90, fps=args.fps, title="Collect (Forage)")

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)

    # ---------------------------
    # logs
    # ---------------------------
    step_rows: List[Dict[str, Any]] = []
    ep_rows: List[Dict[str, Any]] = []

    # ---------------------------
    # rollout
    # ---------------------------
    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)

        # do(g) at episode start
        if args.do_g:
            agent.apply_perturbation(kind=args.do_g, scale=float(args.do_g_scale))

        last_action = 0
        done = False
        t = 0

        # episode stats
        ep_ret = 0.0
        n_inspect = 0
        n_forage = 0
        n_move = 0
        n_pos_forage = 0
        n_neg_forage = 0
        inspect_then_forage = 0
        inspect_then_good_forage = 0

        # track whether we have "recently inspected" this tile
        # (weak proxy: we count a forage as "post-inspect" if the tile is currently revealed at forage time)
        # This is more robust than remembering coordinates because reveal is stored in obs.
        last_action_was_inspect = False

        while not done:
            # optional mid-episode do(g)
            if (args.do_g_t >= 0) and (t == int(args.do_g_t)) and args.do_g_mid:
                agent.apply_perturbation(kind=args.do_g_mid, scale=float(args.do_g_mid_scale))

            # parse obs components (for measurement fields)
            agent25, obj25, revg25, revr25, tfrac = split_obs_101(obs)
            x, y = int(info.get("agent_x", 0)), int(info.get("agent_y", 0))
            idx = idx_from_xy(x, y, W=5)
            on_object = bool(obj25[idx] > 0.5)
            is_revealed_green = bool(revg25[idx] > 0.5)
            is_revealed_red = bool(revr25[idx] > 0.5)
            is_revealed = bool(is_revealed_green or is_revealed_red)

            # agent step
            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)

            a_int = int(action.item())

            # env step
            obs_next, r, terminated, truncated, info2 = env.step(a_int)
            ep_ret += float(r)

            # update coarse action stats
            if a_int in (0, 1, 2, 3):
                n_move += 1
                last_action_was_inspect = False
            elif a_int == env.ACTION_INSPECT:
                n_inspect += 1
                last_action_was_inspect = True
            elif a_int == env.ACTION_FORAGE:
                n_forage += 1

                # count positive/negative forage events (only when reward nonzero)
                if float(r) > 0.0:
                    n_pos_forage += 1
                elif float(r) < 0.0:
                    n_neg_forage += 1

                # "post-inspect forage" proxy: tile is revealed when foraging
                if is_revealed:
                    inspect_then_forage += 1
                    if is_revealed_green and float(r) > 0.0:
                        inspect_then_good_forage += 1

                last_action_was_inspect = False
            else:
                last_action_was_inspect = False

            # latents
            g = out["g"].squeeze(0).detach().cpu().numpy()
            z = out["z"].squeeze(0).detach().cpu().numpy()
            s = out["s"].squeeze(0).detach().cpu().numpy()
            g_norm = float(np.linalg.norm(g))

            # per-step compact row (paper-friendly)
            row: Dict[str, Any] = {
                "episode": int(ep),
                "step": int(t),
                "action": int(a_int),
                "action_name": action_name(a_int),
                "reward": float(r),
                "ep_return_so_far": float(ep_ret),
                "time_left": int(info2.get("time_left", -1)),
                "x": int(info2.get("agent_x", -1)),
                "y": int(info2.get("agent_y", -1)),
                "n_revealed": int(info2.get("n_revealed", -1)),
                "on_object": int(on_object),
                "tile_revealed": int(is_revealed),
                "tile_revealed_green": int(is_revealed_green),
                "tile_revealed_red": int(is_revealed_red),
                "tfrac": float(tfrac),
                "g_norm": float(g_norm),
            }

            # optionally include latents / obs
            if args.save_latents:
                for i, v in enumerate(g.tolist()):
                    row[f"g_{i}"] = float(v)
                for i, v in enumerate(z.tolist()):
                    row[f"z_{i}"] = float(v)
                for i, v in enumerate(s.tolist()):
                    row[f"s_{i}"] = float(v)

            if args.save_obs:
                for i, v in enumerate(obs.astype(np.float32).tolist()):
                    row[f"obs_{i}"] = float(v)

            step_rows.append(row)

            # viewer
            if viewer is not None:
                ok = viewer.draw(
                    env,
                    step=t,
                    episode=ep,
                    last_action=a_int,
                    reward=float(r),
                    total_reward=float(ep_ret),
                    g_norm=float(g_norm),
                )
                if ok is False:
                    print("Viewer closed; stopping collection.")
                    done = True
                    break

            # advance
            obs = obs_next
            info = info2
            last_action = a_int
            done = bool(terminated or truncated)
            t += 1

        # episode summary row
        forage_total = max(1, n_forage)
        ep_row = {
            "episode": int(ep),
            "return": float(ep_ret),
            "steps": int(t),
            "n_move": int(n_move),
            "n_inspect": int(n_inspect),
            "n_forage": int(n_forage),
            "n_pos_forage": int(n_pos_forage),
            "n_neg_forage": int(n_neg_forage),
            "inspect_rate": float(n_inspect / max(1, t)),
            "forage_pos_rate": float(n_pos_forage / forage_total),
            "forage_neg_rate": float(n_neg_forage / forage_total),
            "postinspect_forage_frac": float(inspect_then_forage / forage_total),
            "postinspect_good_forage_frac": float(inspect_then_good_forage / max(1, inspect_then_forage)),
        }
        ep_rows.append(ep_row)

        print(
            f"[collect] ep={ep:>3}  R={ep_ret:+.2f}  "
            f"inspect={n_inspect}  forage={n_forage} (+{n_pos_forage}/-{n_neg_forage})  "
            f"postInspect={inspect_then_forage}"
        )

    # ---------------------------
    # save
    # ---------------------------
    df_steps = pd.DataFrame(step_rows)
    df_eps = pd.DataFrame(ep_rows)

    out_steps = run_dir / "traj_forage_compact.csv"
    out_eps = run_dir / "episodes_forage.csv"

    df_steps.to_csv(out_steps, index=False)
    df_eps.to_csv(out_eps, index=False)

    print(f"Saved: {out_steps}")
    print(f"Saved: {out_eps}")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
