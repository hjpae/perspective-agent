# cear_pilot/training/train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)

    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=9)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env_cfg = NZoneConfig(width=args.width, height=args.height, obs_dim=args.obs_dim, max_steps=args.max_steps)
    env = NZoneGridEnv(config=env_cfg)
    obs, info = env.reset(seed=args.seed)

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

    dec_cfg = DecoderConfig(g_dim=agent_cfg.world.g_dim, n_actions=n_actions, obs_dim=args.obs_dim, hidden=64, dropout=0.0)
    decoder = ObsDecoder(dec_cfg).to(device)

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "loss_weights": {"w_smooth": args.w_smooth, "w_entropy": args.w_entropy},
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
    }
    save_meta(run_dir, meta)

    agent.reset(batch_size=1)
    last_action = 4  # stay
    g_prev = agent.get_latents()["g"].detach().clone()

    ema = None
    t0 = time.time()

    for step in range(args.steps):
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
        p_t = make_proprio_from_last_action(last_action, n_actions, device=device)  # (1, n_actions)

        out = agent.forward_step(x_t, p_t, ablate_g=False)
        g_t = out["g"]
        logits = out["logits"]
        pi = torch.softmax(logits, dim=-1)

        # execute sampled action in env (exploration)
        a_t = agent.policy.sample_action(logits, greedy=False)
        a_int = int(a_t.item())
        obs_next, _, terminated, truncated, info = env.step(a_int)

        x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

        # decoder predicts next obs for all actions
        xhat_all = decoder.predict_all_actions(g_t)            # (1, A, obs_dim)
        xhat_exp = torch.sum(pi.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

        # losses
        loss_pred = F.mse_loss(xhat_exp, x_next)
        loss_smooth = torch.mean((g_t - g_prev) ** 2)

        entropy = -torch.sum(pi * torch.log(pi + 1e-9), dim=-1).mean()
        loss = loss_pred + args.w_smooth * loss_smooth - args.w_entropy * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # step state
        g_prev = g_t.detach().clone()
        obs = obs_next
        last_action = a_int

        if truncated or terminated:
            obs, info = env.reset()
            agent.reset(batch_size=1)
            last_action = 4
            g_prev = agent.get_latents()["g"].detach().clone()

        v = float(loss.item())
        ema = v if ema is None else 0.98 * ema + 0.02 * v

        if (step + 1) % 2000 == 0:
            dt = time.time() - t0
            print(
                f"[{step+1:>7}/{args.steps}] "
                f"loss={v:.4f} ema={ema:.4f} pred={float(loss_pred.item()):.4f} "
                f"smooth={float(loss_smooth.item()):.4f} H={float(entropy.item()):.3f} ({dt:.1f}s)"
            )
            t0 = time.time()

    ckpt = {"agent_state": agent.state_dict(), "decoder_state": decoder.state_dict(), "meta": meta}
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
