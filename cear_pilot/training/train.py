# cear_pilot/training/train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def onehot(indices: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=n).float()


@torch.no_grad()
def make_proprio_from_last_action(last_action: int, n_actions: int, device: torch.device) -> torch.Tensor:
    a = torch.tensor([last_action], device=device)
    return onehot(a, n_actions)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_meta(run_dir: Path, meta: Dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is used later, keep this for completeness.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0, help=">=0: reproducible, <0: pick a new random seed each run")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)

    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=9)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    # actor term to make policy actually learn (internal cost, not env reward)
    ap.add_argument(
        "--w_actor",
        type=float,
        default=0.5,
        help="Weight for REINFORCE actor loss based on prediction error (internal cost).",
    )
    ap.add_argument(
        "--actor_b",
        type=float,
        default=0.98,
        help="EMA momentum for actor baseline (0 disables baseline).",
    )

    # ---- live viewer flags ----
    ap.add_argument("--view", action="store_true", help="Show live pygame viewer during training")
    ap.add_argument("--view_every", type=int, default=2, help="Render every N training steps")
    ap.add_argument("--view_fps", type=int, default=20, help="Viewer FPS cap")
    ap.add_argument("--view_cell_px", type=int, default=42, help="Cell size in pixels")

    # ---- Phase 2 env toggles ----
    ap.add_argument("--use_slip", action="store_true")
    ap.add_argument("--use_drift", action="store_true")
    ap.add_argument("--use_volatility", action="store_true")
    ap.add_argument("--use_hazard", action="store_true")

    ap.add_argument("--p_slip", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--p_drift", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--drift_vec", type=int, nargs=6, default=(0, 0, 0, 0, 0, 0))  # z0dx z0dy z1dx z1dy z2dx z2dy

    ap.add_argument("--volatile_zone", type=int, default=0)
    ap.add_argument("--volatile_period", type=int, default=40)
    ap.add_argument("--volatile_strength", type=float, default=0.0)

    ap.add_argument("--hazard_mode", type=str, default="teleport")
    ap.add_argument("--p_hazard", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--hazard_teleport_to", type=int, nargs=2, default=(0, 0))
    ap.add_argument("--hazard_blackout_steps", type=int, default=6)

    args = ap.parse_args()

    # -------------------------
    # Seed handling
    # -------------------------
    if args.seed < 0:
        # runtime seed (still recorded)
        run_seed = int(time.time_ns() % (2**31 - 1))
    else:
        run_seed = int(args.seed)

    _seed_everything(run_seed)

    device = torch.device(args.device)

    # episode-wise seed stream (keeps reproducibility for fixed run_seed, avoids "same episode forever" feel)
    ep_rng = np.random.default_rng(run_seed)

    dv = args.drift_vec
    drift_vec = ((dv[0], dv[1]), (dv[2], dv[3]), (dv[4], dv[5]))

    env_cfg = NZoneConfig(
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        use_slip=args.use_slip,
        use_drift=args.use_drift,
        use_volatility=args.use_volatility,
        use_hazard=args.use_hazard,
        p_slip=tuple(args.p_slip),
        p_drift=tuple(args.p_drift),
        drift_vec=drift_vec,
        volatile_zone=args.volatile_zone,
        volatile_period=args.volatile_period,
        volatile_strength=args.volatile_strength,
        hazard_mode=args.hazard_mode,
        p_hazard=tuple(args.p_hazard),
        hazard_teleport_to=tuple(args.hazard_teleport_to),
        hazard_blackout_steps=args.hazard_blackout_steps,
    )
    env = NZoneGridEnv(config=env_cfg)

    # first reset uses run_seed; later resets use per-episode seeds
    obs, info = env.reset(seed=run_seed)

    n_actions = int(env.action_space.n)

    # ---- agent config (wire dims)
    agent_cfg = AgentConfig(device=args.device)

    agent_cfg.encoder.obs_dim = args.obs_dim
    agent_cfg.encoder.proprio_dim = n_actions

    agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
    agent_cfg.world.p_dim = agent_cfg.encoder.p_dim

    agent_cfg.state.z_dim = agent_cfg.encoder.z_dim
    agent_cfg.state.p_dim = agent_cfg.encoder.p_dim
    agent_cfg.state.g_dim = agent_cfg.world.g_dim

    agent_cfg.policy.n_actions = n_actions
    agent_cfg.policy.s_dim = agent_cfg.state.s_dim

    agent = CEARAgent(agent_cfg).to(device)

    dec_cfg = DecoderConfig(
        g_dim=agent_cfg.world.g_dim,
        n_actions=n_actions,
        obs_dim=args.obs_dim,
        hidden=64,
        dropout=0.0,
    )
    decoder = ObsDecoder(dec_cfg).to(device)

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed_arg": int(args.seed),
        "run_seed_used": int(run_seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "device": args.device,
        "loss_weights": {"w_smooth": float(args.w_smooth), "w_entropy": float(args.w_entropy), "w_actor": float(args.w_actor)},
        "actor_b": float(args.actor_b),
        "stopgrad_actor_default": True,
        "pred_pi_fixed_uniform": True,
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
        "viewer": {
            "enabled": bool(args.view),
            "view_every": int(args.view_every),
            "view_fps": int(args.view_fps),
            "view_cell_px": int(args.view_cell_px),
        },
    }
    save_meta(run_dir, meta)

    # ---- live viewer init ----
    viewer = None
    if args.view:
        from cear_pilot.training.pygame_viewer import PygameGridViewer

        viewer = PygameGridViewer(
            width=args.width,
            height=args.height,
            cell_px=args.view_cell_px,
            fps=args.view_fps,
            title="Live Training (SPACE=Pause, Close=Stop)",
        )

    agent.reset(batch_size=1)
    last_action = 4  # stay
    g_prev = agent.get_latents()["g"].detach().clone()

    ema = None

    # actor baseline (EMA of chosen-action internal cost)
    b = None

    # policy diagnostics
    pi_prev = None
    kl_ema = None
    maxpi_ema = None

    # extra diagnostics
    logits_scale_ema = None
    err_min_ema = None
    err_max_ema = None
    err_std_ema = None

    # occupancy / action hist
    zone_counts = np.zeros(3, dtype=np.int64)
    act_counts = np.zeros(n_actions, dtype=np.int64)

    t0 = time.time()
    episode = 0

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)  # (1, n_actions)

            # forward step (world + state)
            out = agent.forward_step(x_t, p_t, ablate_g=False)
            g_t = out["g"]
            s_t = out["s"]

            # ---- STOP-GRAD DEFAULT: policy action logits use detached state
            logits_act = agent.policy(s_t.detach())  # (1, A)
            pi_act = torch.softmax(logits_act, dim=-1)  # (1, A)

            # ---- PRED POLICY: FIXED UNIFORM (prevents moving-target pred objective)
            pi_pred = torch.full((1, n_actions), 1.0 / n_actions, device=device)  # (1, A)

            # diagnostics (max(pi), KL(pi || pi_prev)) based on action policy
            with torch.no_grad():
                maxpi = float(pi_act.max(dim=-1).values.mean().item())
                if pi_prev is None:
                    kl = 0.0
                else:
                    kl_t = torch.sum(
                        pi_act * (torch.log(pi_act + 1e-9) - torch.log(pi_prev + 1e-9)),
                        dim=-1,
                    )
                    kl = float(kl_t.mean().item())
                pi_prev = pi_act.detach()

                # logits scale diagnostic
                logits_scale = float(logits_act.detach().abs().mean().item())

            maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
            kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)
            logits_scale_ema = logits_scale if (logits_scale_ema is None) else (0.98 * logits_scale_ema + 0.02 * logits_scale)

            # execute sampled action in env (exploration) -- IMPORTANT: use logits_act
            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            # logging: zone + action counts
            zid = int(info.get("zone_id", env.zone_id()))
            if 0 <= zid < 3:
                zone_counts[zid] += 1
            if 0 <= a_int < n_actions:
                act_counts[a_int] += 1

            # decoder predicts next obs for all actions
            xhat_all = decoder.predict_all_actions(g_t)  # (1, A, obs_dim)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

            # world-model style losses
            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)

            # entropy regularizer (based on action policy)
            entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

            # ---- actor loss (REINFORCE on internal "cost" = chosen-action prediction error)
            # per-action prediction error: shape (A,)
            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)  # (A,)
            e_chosen = per_a_err[a_int]  # scalar tensor

            # IMPORTANT: actor must not backprop through world model -> detach cost
            e_chosen_det = e_chosen.detach()

            # EMA baseline
            with torch.no_grad():
                e_val = float(e_chosen_det.item())
                if b is None:
                    b = e_val
                if args.actor_b > 0.0:
                    b = float(args.actor_b * b + (1.0 - args.actor_b) * e_val)
                baseline = float(b) if (args.actor_b > 0.0) else 0.0

            # prefer lower error actions: advantage is negative centered error (detached)
            adv = -(e_chosen_det - baseline)

            logp = F.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -(adv * logp)  # scalar

            # extra diagnostics: per-action error spread (EMA)
            with torch.no_grad():
                e_min = float(per_a_err.min().item())
                e_max = float(per_a_err.max().item())
                e_std = float(per_a_err.std(unbiased=False).item())

            err_min_ema = e_min if (err_min_ema is None) else (0.98 * err_min_ema + 0.02 * e_min)
            err_max_ema = e_max if (err_max_ema is None) else (0.98 * err_max_ema + 0.02 * e_max)
            err_std_ema = e_std if (err_std_ema is None) else (0.98 * err_std_ema + 0.02 * e_std)

            # total loss
            loss = (
                loss_pred
                + args.w_smooth * loss_smooth
                + args.w_actor * loss_actor
                - args.w_entropy * entropy
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # step state
            g_prev = g_t.detach().clone()
            obs = obs_next
            last_action = a_int

            # viewer draw (training-time, live)
            if viewer is not None and (step % max(1, args.view_every) == 0):
                g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())
                ok = viewer.draw(
                    env=env,
                    step=step + 1,
                    episode=episode,
                    last_action=last_action,
                    loss=float(loss.item()),
                    loss_pred=float(loss_pred.item()),
                    loss_smooth=float(loss_smooth.item()),
                    entropy=float(entropy.item()),
                    g_norm=g_norm,
                )
                if ok is False:
                    print("Viewer closed. Stopping training.")
                    break

            if truncated or terminated:
                # episode reset with new seed (still reproducible for fixed run_seed)
                ep_seed = int(ep_rng.integers(0, 2**31 - 1))
                obs, info = env.reset(seed=ep_seed)

                agent.reset(batch_size=1)
                last_action = 4
                g_prev = agent.get_latents()["g"].detach().clone()
                episode += 1

            v = float(loss.item())
            ema = v if ema is None else 0.98 * ema + 0.02 * v

            if (step + 1) % 2000 == 0:
                dt = time.time() - t0
                b_show = 0.0 if b is None else float(b)

                # normalize zone/action hist for a quick glance
                ztot = max(1, int(zone_counts.sum()))
                atot = max(1, int(act_counts.sum()))
                zfrac = zone_counts / ztot
                afrac = act_counts / atot

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"loss={v:.4f} ema={ema:.4f} pred={float(loss_pred.item()):.4f} "
                    f"smooth={float(loss_smooth.item()):.4f} "
                    f"actor={float(loss_actor.item()):.4f} b={b_show:.4f} "
                    f"H={float(entropy.item()):.3f} "
                    f"maxpi={float(maxpi_ema if maxpi_ema is not None else maxpi):.3f} "
                    f"KL={float(kl_ema if kl_ema is not None else kl):.6f} "
                    f"logits|.|={float(logits_scale_ema if logits_scale_ema is not None else logits_scale):.3f} "
                    f"e[min,max,std]={float(err_min_ema):.3f},{float(err_max_ema):.3f},{float(err_std_ema):.3f} "
                    f"zone={zfrac.round(2).tolist()} "
                    f"act={afrac.round(2).tolist()} "
                    f"(ep={episode}, {dt:.1f}s)"
                )
                t0 = time.time()

    finally:
        if viewer is not None:
            viewer.close()

    ckpt = {"agent_state": agent.state_dict(), "decoder_state": decoder.state_dict(), "meta": meta}
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()





# # cear_pilot/training/train.py
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import argparse
# import json
# import time
# from dataclasses import asdict
# from pathlib import Path
# from typing import Dict, Optional

# import numpy as np
# import torch
# import torch.nn.functional as F

# from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
# from cear_pilot.models.agent import CEARAgent, AgentConfig
# from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


# def onehot(indices: torch.Tensor, n: int) -> torch.Tensor:
#     return F.one_hot(indices.long(), num_classes=n).float()


# @torch.no_grad()
# def make_proprio_from_last_action(last_action: int, n_actions: int, device: torch.device) -> torch.Tensor:
#     a = torch.tensor([last_action], device=device)
#     return onehot(a, n_actions)


# def timestamp_id() -> str:
#     return time.strftime("%Y%m%d_%H%M%S")


# def save_meta(run_dir: Path, meta: Dict) -> None:
#     (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--steps", type=int, default=80000)
#     ap.add_argument("--lr", type=float, default=3e-4)
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--device", type=str, default="cpu")

#     ap.add_argument("--w_smooth", type=float, default=0.25)
#     ap.add_argument("--w_entropy", type=float, default=0.01)

#     ap.add_argument("--width", type=int, default=15)
#     ap.add_argument("--height", type=int, default=9)
#     ap.add_argument("--obs_dim", type=int, default=8)
#     ap.add_argument("--max_steps", type=int, default=240)

#     # actor term to make policy actually learn (internal cost, not env reward)
#     ap.add_argument(
#         "--w_actor",
#         type=float,
#         default=0.5,
#         help="Weight for REINFORCE actor loss based on prediction error (internal cost).",
#     )
#     ap.add_argument(
#         "--actor_b",
#         type=float,
#         default=0.98,
#         help="EMA momentum for actor baseline (0 disables baseline).",
#     )

#     # ---- live viewer flags ----
#     ap.add_argument("--view", action="store_true", help="Show live pygame viewer during training")
#     ap.add_argument("--view_every", type=int, default=2, help="Render every N training steps")
#     ap.add_argument("--view_fps", type=int, default=20, help="Viewer FPS cap")
#     ap.add_argument("--view_cell_px", type=int, default=42, help="Cell size in pixels")
    
#     # ---- Phase 2 env toggles ----
#     ap.add_argument("--use_slip", action="store_true")
#     ap.add_argument("--use_drift", action="store_true")
#     ap.add_argument("--use_volatility", action="store_true")
#     ap.add_argument("--use_hazard", action="store_true")
    
#     ap.add_argument("--p_slip", type=float, nargs=3, default=(0.0, 0.0, 0.0))
#     ap.add_argument("--p_drift", type=float, nargs=3, default=(0.0, 0.0, 0.0))
#     ap.add_argument("--drift_vec", type=int, nargs=6, default=(0,0, 0,0, 0,0))  # z0dx z0dy z1dx z1dy z2dx z2dy
    
#     ap.add_argument("--volatile_zone", type=int, default=0)
#     ap.add_argument("--volatile_period", type=int, default=40)
#     ap.add_argument("--volatile_strength", type=float, default=0.0)
    
#     ap.add_argument("--hazard_mode", type=str, default="teleport")
#     ap.add_argument("--p_hazard", type=float, nargs=3, default=(0.0, 0.0, 0.0))
#     ap.add_argument("--hazard_teleport_to", type=int, nargs=2, default=(0, 0))
#     ap.add_argument("--hazard_blackout_steps", type=int, default=6)


#     args = ap.parse_args()

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     device = torch.device(args.device)

#     dv = args.drift_vec
#     drift_vec = ((dv[0], dv[1]), (dv[2], dv[3]), (dv[4], dv[5]))
    
#     env_cfg = NZoneConfig(
#         width=args.width, height=args.height, obs_dim=args.obs_dim, max_steps=args.max_steps,
    
#         use_slip=args.use_slip,
#         use_drift=args.use_drift,
#         use_volatility=args.use_volatility,
#         use_hazard=args.use_hazard,
    
#         p_slip=tuple(args.p_slip),
#         p_drift=tuple(args.p_drift),
#         drift_vec=drift_vec,
    
#         volatile_zone=args.volatile_zone,
#         volatile_period=args.volatile_period,
#         volatile_strength=args.volatile_strength,
    
#         hazard_mode=args.hazard_mode,
#         p_hazard=tuple(args.p_hazard),
#         hazard_teleport_to=tuple(args.hazard_teleport_to),
#         hazard_blackout_steps=args.hazard_blackout_steps,
#     )
#     env = NZoneGridEnv(config=env_cfg)
#     obs, info = env.reset(seed=args.seed)

#     n_actions = int(env.action_space.n)

#     # ---- agent config (wire dims)
#     agent_cfg = AgentConfig(device=args.device)

#     agent_cfg.encoder.obs_dim = args.obs_dim
#     agent_cfg.encoder.proprio_dim = n_actions

#     agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
#     agent_cfg.world.p_dim = agent_cfg.encoder.p_dim

#     agent_cfg.state.z_dim = agent_cfg.encoder.z_dim
#     agent_cfg.state.p_dim = agent_cfg.encoder.p_dim
#     agent_cfg.state.g_dim = agent_cfg.world.g_dim

#     agent_cfg.policy.n_actions = n_actions
#     agent_cfg.policy.s_dim = agent_cfg.state.s_dim

#     agent = CEARAgent(agent_cfg).to(device)

#     dec_cfg = DecoderConfig(g_dim=agent_cfg.world.g_dim, n_actions=n_actions, obs_dim=args.obs_dim, hidden=64, dropout=0.0)
#     decoder = ObsDecoder(dec_cfg).to(device)

#     params = list(agent.parameters()) + list(decoder.parameters())
#     opt = torch.optim.Adam(params, lr=args.lr)

#     run_dir = Path("outputs") / "runs" / timestamp_id()
#     run_dir.mkdir(parents=True, exist_ok=True)

#     meta = {
#         "seed": args.seed,
#         "steps": args.steps,
#         "lr": args.lr,
#         "device": args.device,
#         "loss_weights": {"w_smooth": args.w_smooth, "w_entropy": args.w_entropy},
#         "env_cfg": asdict(env_cfg),
#         "agent_cfg": {
#             "encoder": asdict(agent_cfg.encoder),
#             "world": asdict(agent_cfg.world),
#             "state": asdict(agent_cfg.state),
#             "policy": asdict(agent_cfg.policy),
#         },
#         "decoder_cfg": asdict(dec_cfg),
#         "viewer": {
#             "enabled": bool(args.view),
#             "view_every": int(args.view_every),
#             "view_fps": int(args.view_fps),
#             "view_cell_px": int(args.view_cell_px),
#         },
#     }
#     save_meta(run_dir, meta)

#     # ---- live viewer init ----
#     viewer = None
#     if args.view:
#         from cear_pilot.training.pygame_viewer import PygameGridViewer

#         viewer = PygameGridViewer(
#             width=args.width,
#             height=args.height,
#             cell_px=args.view_cell_px,
#             fps=args.view_fps,
#             title="Live Training (SPACE=Pause, Close=Stop)",
#         )

#     agent.reset(batch_size=1)
#     last_action = 4  # stay
#     g_prev = agent.get_latents()["g"].detach().clone()

#     ema = None
#     pred_ema = None  # baseline for actor (EMA of pred loss)
#     pi_prev = None
#     kl_ema = None
#     maxpi_ema = None
    
#     t0 = time.time()
#     episode = 0

#     try:
#         for step in range(args.steps):
#             x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
#             p_t = make_proprio_from_last_action(last_action, n_actions, device=device)  # (1, n_actions)

#             out = agent.forward_step(x_t, p_t, ablate_g=False)
#             g_t = out["g"]
#             logits = out["logits"]
#             pi = torch.softmax(logits, dim=-1)
            
#             # diagnostics (max(pi), KL(pi || pi_prev))
#             with torch.no_grad():
#                 # max probability (greediness)
#                 maxpi = float(pi.max(dim=-1).values.mean().item())

#                 # KL between consecutive policies (per-step)
#                 if pi_prev is None:
#                     kl = 0.0
#                 else:
#                     # KL(pi || pi_prev) = sum pi * (log pi - log pi_prev)
#                     kl_t = torch.sum(
#                         pi * (torch.log(pi + 1e-9) - torch.log(pi_prev + 1e-9)),
#                         dim=-1
#                     )
#                     kl = float(kl_t.mean().item())

#                 pi_prev = pi.detach()

#             # EMA for smoother logging
#             maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
#             kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)


#             # execute sampled action in env (exploration)
#             a_t = agent.policy.sample_action(logits, greedy=False)
#             a_int = int(a_t.item())
#             obs_next, _, terminated, truncated, info = env.step(a_int)

#             x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

#             # decoder predicts next obs for all actions
#             xhat_all = decoder.predict_all_actions(g_t)               # (1, A, obs_dim)
#             xhat_exp = torch.sum(pi.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

#             # losses (world-model style)
#             loss_pred = F.mse_loss(xhat_exp, x_next)
#             loss_smooth = torch.mean((g_t - g_prev) ** 2)

#             # entropy regularizer (optional; can freeze policy to uniform if too strong)
#             entropy = -torch.sum(pi * torch.log(pi + 1e-9), dim=-1).mean()

#             # actor loss (REINFORCE on internal "cost" = prediction error)
#             # log-prob of sampled action
#             logp = F.log_softmax(logits, dim=-1).gather(-1, a_t.view(1, 1)).squeeze(-1)  # (1,)

#             # baseline to reduce variance
#             with torch.no_grad():
#                 c = float(loss_pred.detach().item())
#                 if pred_ema is None:
#                     pred_ema = c
#                 if args.actor_b > 0.0:
#                     pred_ema = float(args.actor_b * pred_ema + (1.0 - args.actor_b) * c)

#                 baseline = float(pred_ema) if (args.actor_b > 0.0) else 0.0

#             # advantage-like cost (detach so actor doesn't backprop through world model)
#             adv = (loss_pred.detach() - baseline)

#             # minimize expected cost -> loss_actor = E[ (cost - b) * logpi(a) ]
#             loss_actor = (adv * logp).mean()

#             # total loss
#             loss = (
#                 loss_pred
#                 + args.w_smooth * loss_smooth
#                 + args.w_actor * loss_actor
#                 - args.w_entropy * entropy
#             )

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(params, 1.0)
#             opt.step()

#             # step state
#             g_prev = g_t.detach().clone()
#             obs = obs_next
#             last_action = a_int

#             # viewer draw (training-time, live)
#             if viewer is not None and (step % max(1, args.view_every) == 0):
#                 g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())
#                 ok = viewer.draw(
#                     env=env,
#                     step=step + 1,
#                     episode=episode,
#                     last_action=last_action,
#                     loss=float(loss.item()),
#                     loss_pred=float(loss_pred.item()),
#                     loss_smooth=float(loss_smooth.item()),
#                     entropy=float(entropy.item()),
#                     g_norm=g_norm,
#                 )
#                 if ok is False:
#                     print("Viewer closed. Stopping training.")
#                     break

#             if truncated or terminated:
#                 obs, info = env.reset()
#                 agent.reset(batch_size=1)
#                 last_action = 4
#                 g_prev = agent.get_latents()["g"].detach().clone()
#                 episode += 1

#             v = float(loss.item())
#             ema = v if ema is None else 0.98 * ema + 0.02 * v

#             if (step + 1) % 2000 == 0:
#                 dt = time.time() - t0
#                 b = 0.0 if pred_ema is None else float(pred_ema)
#                 print(
#                     f"[{step+1:>7}/{args.steps}] "
#                     f"loss={v:.4f} ema={ema:.4f} pred={float(loss_pred.item()):.4f} "
#                     f"smooth={float(loss_smooth.item()):.4f} "
#                     f"actor={float(loss_actor.item()):.4f} b={b:.4f} "
#                     f"H={float(entropy.item()):.3f} "
#                     f"maxpi={float(maxpi_ema if maxpi_ema is not None else maxpi):.3f} "
#                     f"KL={float(kl_ema if kl_ema is not None else kl):.6f} "
#                     f"({dt:.1f}s)"
#                 )
#                 t0 = time.time()

#     finally:
#         if viewer is not None:
#             viewer.close()

#     ckpt = {"agent_state": agent.state_dict(), "decoder_state": decoder.state_dict(), "meta": meta}
#     torch.save(ckpt, run_dir / "ckpt.pt")
#     print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


# if __name__ == "__main__":
#     main()
