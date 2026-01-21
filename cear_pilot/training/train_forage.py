# cear_pilot/training/train_forage.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.forage_grid import ForageGridEnv, ForageGridConfig
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


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMAMeanVar:
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = None
        self.var = None

    def update(self, x: float) -> Tuple[float, float]:
        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            m = self.mean
            self.mean = self.beta * self.mean + (1 - self.beta) * x
            self.var = self.beta * self.var + (1 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    # world-model loss weights
    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)

    # env config knobs (optional; defaults match your spec)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--inspect_cost", type=int, default=50)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--penalty", type=float, default=1.0)

    # actor term: REINFORCE on env reward (fitness)
    ap.add_argument("--w_actor", type=float, default=1.0)
    ap.add_argument("--actor_b", type=float, default=0.98)

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    # ---------------------------
    # env
    # ---------------------------
    env_cfg = ForageGridConfig(
        grid_size=5,
        time_budget=int(args.T),
        inspect_cost=int(args.inspect_cost),
        forage_bonus=float(args.bonus),
        forage_penalty=float(args.penalty),
        seed=int(args.seed),
    )
    env = ForageGridEnv(config=env_cfg)

    obs, info = env.reset(seed=args.seed)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    n_actions = int(env.action_space.n)
    obs_dim = int(np.prod(env.observation_space.shape))

    # ---------------------------
    # agent config (AUTO wiring)
    # ---------------------------
    agent_cfg = AgentConfig(device=args.device)
    agent_cfg.encoder.obs_dim = obs_dim
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
        obs_dim=obs_dim,
        hidden=64,
        dropout=0.0,
    )
    decoder = ObsDecoder(dec_cfg).to(device)

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    # warmup: learn representation/world a bit before actor turns on
    warmup_steps = max(2000, min(args.steps // 4, 20000))

    meta = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "env_name": "forage",
        "env_cfg": asdict(env_cfg),
        "loss_weights": {"w_smooth": args.w_smooth, "w_entropy": args.w_entropy, "w_actor": args.w_actor},
        "actor_b": args.actor_b,
        "warmup_steps": warmup_steps,
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
    }
    save_meta(run_dir, meta)

    # ---------------------------
    # training state
    # ---------------------------
    agent.reset(batch_size=1)
    last_action = 0  # no "stay" in this env; any valid action index is fine for proprio bootstrap
    g_prev = agent.get_latents()["g"].detach().clone()

    ema_world = None
    pi_prev = None
    kl_ema = None
    maxpi_ema = None

    b = None
    rew_stats = EMAMeanVar(beta=0.99)

    t0 = time.time()
    episode = 0

    # episode reward tracking for print
    ep_return = 0.0

    for step in range(args.steps):
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

        out = agent.forward_step(x_t, p_t, ablate_g=False)
        g_t = out["g"]
        s_t = out["s"]

        # action logits for acting (detach s to avoid actor gradients leaking into rep)
        logits_act = agent.policy(s_t.detach())
        pi_act = torch.softmax(logits_act, dim=-1)

        # prediction mixture policy is detached
        logits_pred = out["logits"]
        pi_pred = torch.softmax(logits_pred, dim=-1).detach()

        # sample action
        a_t = agent.policy.sample_action(logits_act, greedy=False)
        a_int = int(a_t.item())

        # env step (fitness reward)
        obs_next, r_env, terminated, truncated, info2 = env.step(a_int)
        ep_return += float(r_env)

        x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

        # decoder predictions
        xhat_all = decoder.predict_all_actions(g_t)  # (1, A, obs_dim)
        xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

        # world losses
        loss_pred = F.mse_loss(xhat_exp, x_next)
        loss_smooth = torch.mean((g_t - g_prev) ** 2)
        loss_world = loss_pred + args.w_smooth * loss_smooth

        # entropy (encourage diversity for latent richness)
        entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

        # actor loss: REINFORCE on env reward (normalized)
        with torch.no_grad():
            r_val = float(r_env)
            m, s = rew_stats.update(r_val)
            if b is None:
                b = r_val
            if args.actor_b > 0.0:
                b = float(args.actor_b * b + (1.0 - args.actor_b) * r_val)
            baseline = float(b) if (args.actor_b > 0.0) else 0.0

            adv = (r_val - baseline) / (s + 1e-8)
            adv = float(np.clip(adv, -5.0, 5.0))

        logp = F.log_softmax(logits_act, dim=-1)[0, a_int]
        loss_actor = -(torch.tensor(adv, device=device) * logp)

        # warmup
        w_actor_eff = 0.0 if step < warmup_steps else float(args.w_actor)

        # total loss
        loss = loss_world + w_actor_eff * loss_actor - float(args.w_entropy) * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # step state
        g_prev = g_t.detach().clone()
        obs = obs_next
        last_action = a_int

        # episode reset
        if terminated or truncated:
            obs, info = env.reset(seed=args.seed + episode + 1)
            agent.reset(batch_size=1)
            last_action = 0
            g_prev = agent.get_latents()["g"].detach().clone()

            # print episode summary
            if (episode + 1) % 10 == 0:
                print(f"[ep={episode+1}] return={ep_return:.2f}")

            ep_return = 0.0
            episode += 1

        # periodic prints
        if (step + 1) % 2000 == 0:
            dt = time.time() - t0
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

                maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
                kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)

            lw = float(loss_world.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            print(
                f"[{step+1:>7}/{args.steps}] "
                f"world={lw:.4f} w_ema={float(ema_world):.4f} pred={float(loss_pred.item()):.4f} "
                f"smooth={float(loss_smooth.item()):.4f} | "
                f"actor={float(loss_actor.item()):.4f} w_actor={w_actor_eff:.2f} "
                f"H={float(entropy.item()):.3f} maxpi={float(maxpi_ema):.3f} KL={float(kl_ema):.6f} "
                f"(ep={episode}, {dt:.1f}s)"
            )
            t0 = time.time()

    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
