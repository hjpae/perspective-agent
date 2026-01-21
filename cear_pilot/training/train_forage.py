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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.forage_grid import ForageGridEnv, ForageGridConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


# ---------------------------
# small utilities
# ---------------------------

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


class EMAMean:
    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self.v = None

    def update(self, x: float) -> float:
        if self.v is None:
            self.v = float(x)
        else:
            self.v = self.beta * self.v + (1.0 - self.beta) * float(x)
        return float(self.v)


# ---------------------------
# policy <-> vector helpers (CEM)
# ---------------------------

@torch.no_grad()
def flatten_params(module: torch.nn.Module) -> torch.Tensor:
    """Flatten parameters of a module to a 1D tensor on CPU."""
    vecs = []
    for p in module.parameters():
        vecs.append(p.detach().reshape(-1).cpu())
    return torch.cat(vecs, dim=0)


@torch.no_grad()
def set_params_from_flat(module: torch.nn.Module, flat: torch.Tensor) -> None:
    """Assign a flat parameter vector to a module (in-place)."""
    flat = flat.detach().cpu()
    offset = 0
    for p in module.parameters():
        numel = p.numel()
        chunk = flat[offset: offset + numel].view_as(p).to(p.device, dtype=p.dtype)
        p.copy_(chunk)
        offset += numel
    assert offset == flat.numel(), "Flat vector size mismatch."


@torch.no_grad()
def sample_cem_population(mu: torch.Tensor, sigma: torch.Tensor, pop: int) -> torch.Tensor:
    """
    Diagonal Gaussian sampling: theta_i = mu + sigma * eps
    mu, sigma: (D,)
    returns: (pop, D)
    """
    eps = torch.randn((pop, mu.numel()), device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(0) + eps * sigma.unsqueeze(0)


def cem_update(mu: torch.Tensor, sigma: torch.Tensor, elites: torch.Tensor, alpha: float, sigma_floor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    elites: (E, D)
    """
    elite_mu = elites.mean(dim=0)
    elite_std = elites.std(dim=0, unbiased=False)
    new_mu = (1.0 - alpha) * mu + alpha * elite_mu
    new_sigma = (1.0 - alpha) * sigma + alpha * elite_std
    new_sigma = torch.clamp(new_sigma, min=sigma_floor)
    return new_mu, new_sigma


# ---------------------------
# rollout + world loss
# ---------------------------

@torch.no_grad()
def rollout_episode(
    env: ForageGridEnv,
    agent: CEARAgent,
    decoder: ObsDecoder,
    device: torch.device,
    greedy: bool,
    ablate_g: bool = False,
) -> Tuple[float, int]:
    """
    Evaluate a single episode return using env reward ONLY.
    This is the 'fitness' for CEM.

    Returns:
      total_return_env, steps
    """
    obs, _ = env.reset()
    agent.reset(batch_size=1)

    total_return = 0.0
    steps = 0
    last_action = 0

    done = False
    while not done:
        x_t = torch.tensor(obs, device=device, dtype=torch.float32).view(1, -1)
        p_t = make_proprio_from_last_action(last_action, env.action_space.n, device)

        a_t, out = agent.step(x_t, p_t, greedy=greedy, ablate_g=ablate_g)
        a_int = int(a_t.item())

        obs, r, terminated, truncated, _ = env.step(a_int)
        done = bool(terminated or truncated)

        total_return += float(r)
        steps += 1
        last_action = a_int

    return total_return, steps


def world_update_on_episode(
    env: ForageGridEnv,
    agent: CEARAgent,
    decoder: ObsDecoder,
    opt: torch.optim.Optimizer,
    device: torch.device,
    w_smooth: float,
) -> Tuple[float, float]:
    """
    One SGD update pass using a single on-policy episode from current policy.
    Trains ONLY: encoder/world/state/decoder (policy is frozen).

    Loss:
      pred: MSE( decoder(g_t, a_t) , obs_{t+1} )
      smooth: MSE( g_t , g_{t-1} )
    """
    obs, _ = env.reset()
    agent.reset(batch_size=1)

    last_action = 0
    prev_g = agent.get_latents()["g"].detach().clone()

    pred_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []

    done = False
    while not done:
        x_t = torch.tensor(obs, device=device, dtype=torch.float32).view(1, -1)
        p_t = make_proprio_from_last_action(last_action, env.action_space.n, device)

        # forward step updates internal g
        out = agent.forward_step(x_t, p_t, ablate_g=False)
        logits = out["logits"]
        g_t = out["g"]

        # sample action from policy (policy frozen, but still used for rollouts)
        a_t = agent.policy.sample_action(logits, greedy=False)
        a_int = int(a_t.item())

        # env step
        obs_next, r, terminated, truncated, _ = env.step(a_int)
        done = bool(terminated or truncated)

        # prediction target = next obs (flattened)
        y = torch.tensor(obs_next, device=device, dtype=torch.float32).view(1, -1)

        a_oh = onehot(torch.tensor([a_int], device=device), env.action_space.n)
        y_hat = decoder(g_t, a_oh)

        pred_losses.append(F.mse_loss(y_hat, y))

        smooth_losses.append(F.mse_loss(g_t, prev_g))
        prev_g = g_t.detach()

        obs = obs_next
        last_action = a_int

    pred_loss = torch.stack(pred_losses).mean()
    smooth_loss = torch.stack(smooth_losses).mean()

    loss = pred_loss + float(w_smooth) * smooth_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.parameters()) + list(decoder.parameters()), max_norm=1.0)
    opt.step()

    return float(pred_loss.item()), float(smooth_loss.item())


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    # total env steps budget (rough; used to stop)
    ap.add_argument("--steps", type=int, default=20000)

    # SGD (world) learning
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_smooth", type=float, default=0.25)

    # CEM hyperparams
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite_frac", type=float, default=0.2)
    ap.add_argument("--cem_alpha", type=float, default=0.25)
    ap.add_argument("--sigma_init", type=float, default=0.10)
    ap.add_argument("--sigma_floor", type=float, default=0.02)

    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    # env knobs
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--inspect_cost", type=int, default=50)
    ap.add_argument("--bonus", type=float, default=1.0)
    ap.add_argument("--penalty", type=float, default=1.0)

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
    obs, _ = env.reset(seed=args.seed)

    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    n_actions = int(env.action_space.n)
    obs_dim = int(np.prod(env.observation_space.shape))

    # ---------------------------
    # agent + decoder (same wiring as before)
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

    # ---------------------------
    # IMPORTANT: freeze policy for SGD
    # (policy is optimized by CEM only)
    # ---------------------------
    for p in agent.policy.parameters():
        p.requires_grad_(False)

    sgd_params = [p for p in (list(agent.parameters()) + list(decoder.parameters())) if p.requires_grad]
    opt = torch.optim.Adam(sgd_params, lr=args.lr)

    # ---------------------------
    # run dir + meta (DO NOT BREAK META STRUCTURE)
    # ---------------------------
    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "env_name": "forage",
        "env_cfg": asdict(env_cfg),
        "loss_weights": {"w_smooth": args.w_smooth},
        # keep the SAME nested schema for collector compatibility
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
        # note: adding a new top-level key is usually safe, but if you want *zero* risk, remove this block.
        "cem": {
            "pop": args.pop,
            "elite_frac": args.elite_frac,
            "cem_alpha": args.cem_alpha,
            "sigma_init": args.sigma_init,
            "sigma_floor": args.sigma_floor,
        },
    }
    save_meta(run_dir, meta)

    # ---------------------------
    # CEM init from current policy params
    # ---------------------------
    theta0 = flatten_params(agent.policy).to(device=device, dtype=torch.float32)
    D = theta0.numel()

    mu = theta0.clone()
    sigma = torch.ones((D,), device=device, dtype=torch.float32) * float(args.sigma_init)

    pop = int(args.pop)
    elite_n = max(1, int(round(pop * float(args.elite_frac))))

    # ---------------------------
    # logging
    # ---------------------------
    ema_R = EMAMean(beta=0.98)
    ema_pred = EMAMean(beta=0.98)
    ema_smooth = EMAMean(beta=0.98)

    t0 = time.time()
    total_env_steps = 0
    gen = 0

    while total_env_steps < int(args.steps):
        gen += 1

        # 1) sample population
        population = sample_cem_population(mu, sigma, pop=pop)  # (pop, D)

        # 2) evaluate fitness
        fitness = torch.empty((pop,), device=device, dtype=torch.float32)
        steps_used = 0
        for i in range(pop):
            set_params_from_flat(agent.policy, population[i])

            R, steps = rollout_episode(
                env=env,
                agent=agent,
                decoder=decoder,
                device=device,
                greedy=False,   # stochastic evaluation for robustness
                ablate_g=False,
            )
            fitness[i] = float(R)
            steps_used += int(steps)

        total_env_steps += int(steps_used)

        # 3) select elites
        elite_idx = torch.topk(fitness, k=elite_n, largest=True).indices
        elites = population[elite_idx]  # (E, D)

        # 4) update mu/sigma
        mu, sigma = cem_update(
            mu=mu,
            sigma=sigma,
            elites=elites,
            alpha=float(args.cem_alpha),
            sigma_floor=float(args.sigma_floor),
        )

        # 5) set policy to current mean
        set_params_from_flat(agent.policy, mu)

        # 6) do ONE world SGD update on-policy (cheap & stable)
        pred_loss, smooth_loss = world_update_on_episode(
            env=env,
            agent=agent,
            decoder=decoder,
            opt=opt,
            device=device,
            w_smooth=float(args.w_smooth),
        )

        # 7) log
        best_R = float(fitness.max().item())
        avg_R = float(fitness.mean().item())
        R_ema = ema_R.update(avg_R)
        p_ema = ema_pred.update(pred_loss)
        s_ema = ema_smooth.update(smooth_loss)

        if gen % 5 == 0 or total_env_steps >= int(args.steps):
            dt = time.time() - t0
            print(
                f"[gen={gen:4d}] env_steps={total_env_steps:7d} "
                f"R_best={best_R:+.3f} R_avg={avg_R:+.3f} R_ema={R_ema:+.3f} "
                f"pred_ema={p_ema:.4f} smooth_ema={s_ema:.4f} "
                f"sigma_mean={float(sigma.mean().item()):.4f} ({dt:.1f}s)"
            )

    # final: ensure policy is at mu
    set_params_from_flat(agent.policy, mu)

    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir/'ckpt.pt'}")


if __name__ == "__main__":
    main()



#%% old code
# # cear_pilot/training/train_forage.py
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import argparse
# import json
# import os
# import random
# import time
# from dataclasses import asdict
# from pathlib import Path
# from typing import Dict, Tuple, List

# import numpy as np
# import torch
# import torch.nn.functional as F

# from cear_pilot.envs.forage_grid import ForageGridEnv, ForageGridConfig
# from cear_pilot.models.agent import CEARAgent, AgentConfig
# from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


# def timestamp_id() -> str:
#     return time.strftime("%Y%m%d_%H%M%S")


# def save_meta(run_dir: Path, meta: Dict) -> None:
#     (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# def seed_everything(seed: int, deterministic: bool = True) -> None:
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     if deterministic:
#         try:
#             torch.use_deterministic_algorithms(True)
#         except Exception:
#             pass
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# def onehot(a: int, n: int, device: torch.device) -> torch.Tensor:
#     x = torch.zeros((1, n), device=device, dtype=torch.float32)
#     x[0, int(a)] = 1.0
#     return x


# class EMAMeter:
#     def __init__(self, beta: float = 0.98):
#         self.beta = float(beta)
#         self.v = None

#     def update(self, x: float) -> float:
#         x = float(x)
#         if self.v is None:
#             self.v = x
#         else:
#             self.v = self.beta * self.v + (1.0 - self.beta) * x
#         return float(self.v)


# class ReplayS:
#     """
#     Replay buffer storing fast states only:
#       (s_t, a_t, r_train, done, s_next)
#     This avoids needing to reconstruct slow g for off-policy replay.
#     """
#     def __init__(self, capacity: int, s_dim: int, device: torch.device):
#         self.capacity = int(capacity)
#         self.device = device
#         self.s_dim = int(s_dim)

#         self.s = torch.zeros((capacity, s_dim), device=device, dtype=torch.float32)
#         self.a = torch.zeros((capacity,), device=device, dtype=torch.int64)
#         self.r = torch.zeros((capacity,), device=device, dtype=torch.float32)
#         self.d = torch.zeros((capacity,), device=device, dtype=torch.float32)
#         self.sn = torch.zeros((capacity, s_dim), device=device, dtype=torch.float32)

#         self.ptr = 0
#         self.size = 0

#     def add(self, s: torch.Tensor, a: int, r: float, done: bool, s_next: torch.Tensor) -> None:
#         i = self.ptr
#         self.s[i].copy_(s.squeeze(0))
#         self.a[i] = int(a)
#         self.r[i] = float(r)
#         self.d[i] = 1.0 if bool(done) else 0.0
#         self.sn[i].copy_(s_next.squeeze(0))

#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size: int):
#         assert self.size > 0
#         idx = torch.randint(0, self.size, (batch_size,), device=self.device)
#         return self.s[idx], self.a[idx], self.r[idx], self.d[idx], self.sn[idx]


# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--steps", type=int, default=40000)
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--device", type=str, default="cpu")

#     # env knobs
#     ap.add_argument("--T", type=int, default=500)
#     ap.add_argument("--inspect_cost", type=int, default=50)
#     ap.add_argument("--bonus", type=float, default=1.0)
#     ap.add_argument("--penalty", type=float, default=1.0)

#     # optimization
#     ap.add_argument("--lr", type=float, default=3e-4)
#     ap.add_argument("--gamma", type=float, default=0.99)

#     # dqn
#     ap.add_argument("--batch", type=int, default=64)
#     ap.add_argument("--replay", type=int, default=20000)
#     ap.add_argument("--learn_starts", type=int, default=2000)
#     ap.add_argument("--target_every", type=int, default=1000)
#     ap.add_argument("--train_every", type=int, default=1)

#     # epsilon schedule
#     ap.add_argument("--eps_start", type=float, default=0.50)
#     ap.add_argument("--eps_end", type=float, default=0.05)
#     ap.add_argument("--eps_decay_steps", type=int, default=30000)

#     # world-model loss weights
#     ap.add_argument("--w_smooth", type=float, default=0.25)
#     ap.add_argument("--w_world", type=float, default=1.0)

#     args = ap.parse_args()

#     seed_everything(args.seed, deterministic=True)
#     device = torch.device(args.device)

#     # ---------------------------
#     # env
#     # ---------------------------
#     env_cfg = ForageGridConfig(
#         grid_size=5,
#         time_budget=int(args.T),
#         inspect_cost=int(args.inspect_cost),
#         forage_bonus=float(args.bonus),
#         forage_penalty=float(args.penalty),
#         seed=int(args.seed),
#     )
#     env = ForageGridEnv(config=env_cfg)

#     obs, info = env.reset(seed=args.seed)
#     try:
#         env.action_space.seed(args.seed)
#         env.observation_space.seed(args.seed)
#     except Exception:
#         pass

#     n_actions = int(env.action_space.n)
#     obs_dim = int(np.prod(env.observation_space.shape))

#     # ---------------------------
#     # agent config (auto wiring)
#     # ---------------------------
#     agent_cfg = AgentConfig(device=args.device)
#     agent_cfg.encoder.obs_dim = obs_dim
#     agent_cfg.encoder.proprio_dim = n_actions

#     agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
#     agent_cfg.world.p_dim = agent_cfg.encoder.p_dim

#     agent_cfg.state.z_dim = agent_cfg.encoder.z_dim
#     agent_cfg.state.p_dim = agent_cfg.encoder.p_dim
#     agent_cfg.state.g_dim = agent_cfg.world.g_dim

#     agent_cfg.policy.n_actions = n_actions
#     agent_cfg.policy.s_dim = agent_cfg.state.s_dim

#     agent = CEARAgent(agent_cfg).to(device)

#     dec_cfg = DecoderConfig(
#         g_dim=agent_cfg.world.g_dim,
#         n_actions=n_actions,
#         obs_dim=obs_dim,
#         hidden=64,
#         dropout=0.0,
#     )
#     decoder = ObsDecoder(dec_cfg).to(device)

#     # Target Q network (copy of policy net)
#     q_tgt = type(agent.policy)(agent.policy.cfg).to(device)
#     q_tgt.load_state_dict(agent.policy.state_dict())
#     q_tgt.eval()

#     # Replay over s
#     replay = ReplayS(capacity=args.replay, s_dim=agent_cfg.state.s_dim, device=device)

#     # Optimizer (world + decoder + policy(Q))
#     params = list(agent.parameters()) + list(decoder.parameters())
#     opt = torch.optim.Adam(params, lr=args.lr)

#     run_dir = Path("outputs") / "runs" / timestamp_id()
#     run_dir.mkdir(parents=True, exist_ok=True)

#     meta = {
#         "seed": args.seed,
#         "steps": args.steps,
#         "lr": args.lr,
#         "device": args.device,
#         "env_name": "forage",
#         "env_cfg": asdict(env_cfg),
#         "dqn": {
#             "gamma": args.gamma,
#             "batch": args.batch,
#             "replay": args.replay,
#             "learn_starts": args.learn_starts,
#             "target_every": args.target_every,
#             "train_every": args.train_every,
#             "eps_start": args.eps_start,
#             "eps_end": args.eps_end,
#             "eps_decay_steps": args.eps_decay_steps,
#         },
#         "loss_weights": {"w_world": args.w_world, "w_smooth": args.w_smooth},
#         "agent_cfg": {
#             "encoder": asdict(agent_cfg.encoder),
#             "world": asdict(agent_cfg.world),
#             "state": asdict(agent_cfg.state),
#             "policy": asdict(agent_cfg.policy),
#         },
#         "decoder_cfg": asdict(dec_cfg),
#     }
#     save_meta(run_dir, meta)

#     # ---------------------------
#     # training state
#     # ---------------------------
#     agent.reset(batch_size=1)
#     last_action = 0
#     g_prev = agent.get_latents()["g"].detach().clone()

#     # meters
#     ema_Renv = EMAMeter(0.98)
#     ema_Rtrain = EMAMeter(0.98)
#     ema_shape = EMAMeter(0.98)
#     ema_world = EMAMeter(0.98)
#     ema_q = EMAMeter(0.98)
#     ema_maxQ = EMAMeter(0.98)
#     ema_gnorm = EMAMeter(0.98)

#     # shaping parts meters
#     part_keys = ["green_progress", "red_near", "object_entry", "forage", "skip_forage", "revisit", "idle", "time_cost"]
#     part_ema = {k: EMAMeter(0.98) for k in part_keys}

#     episode = 0
#     ep_Renv = 0.0
#     ep_Rtrain = 0.0

#     t0 = time.time()

#     def eps_at(t: int) -> float:
#         if t <= 0:
#             return float(args.eps_start)
#         frac = min(1.0, max(0.0, float(t) / float(args.eps_decay_steps)))
#         return float(args.eps_start + frac * (args.eps_end - args.eps_start))

#     for step in range(args.steps):
#         eps = eps_at(step)

#         x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
#         p_t = onehot(last_action, n_actions, device=device)

#         out = agent.forward_step(x_t, p_t, ablate_g=False)
#         g_t = out["g"]
#         s_t = out["s"]

#         # Epsilon-greedy over Q(s)
#         with torch.no_grad():
#             q_vals = agent.policy(s_t)  # treat as Q-values
#             if random.random() < eps:
#                 a_int = random.randrange(n_actions)
#             else:
#                 a_int = int(torch.argmax(q_vals, dim=-1).item())

#         # Env step returns training reward; info contains env reward
#         obs_next, r_train, terminated, truncated, info2 = env.step(a_int)
#         done = bool(terminated or truncated)

#         r_env = float(info2.get("r_env", 0.0))
#         r_shape = float(info2.get("r_shape", 0.0))

#         ep_Renv += r_env
#         ep_Rtrain += float(r_train)

#         # Next state (for replay s_next)
#         x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
#         p_next = onehot(a_int, n_actions, device=device)

#         out2 = agent.forward_step(x_next, p_next, ablate_g=False)
#         g_next = out2["g"]
#         s_next = out2["s"]

#         # Store transition in replay (fast states only)
#         replay.add(s_t.detach(), a_int, float(r_train), done, s_next.detach())

#         # -----------------------
#         # World-model + decoder loss (online)
#         # -----------------------
#         # Predict next observation from current g using predicted action distribution (softmax of Q)
#         with torch.no_grad():
#             pi_pred = torch.softmax(out["logits"], dim=-1)  # uses same net output; ok as a mixture weight

#         xhat_all = decoder.predict_all_actions(g_t)  # (1, A, obs_dim)
#         xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

#         loss_pred = F.mse_loss(xhat_exp, x_next)
#         loss_smooth = torch.mean((g_t - g_prev) ** 2)
#         loss_world = float(args.w_world) * (loss_pred + float(args.w_smooth) * loss_smooth)

#         # -----------------------
#         # DQN loss (from replay)
#         # -----------------------
#         loss_q = torch.tensor(0.0, device=device)
#         if (step >= args.learn_starts) and (replay.size >= args.batch) and ((step % args.train_every) == 0):
#             sB, aB, rB, dB, snB = replay.sample(args.batch)

#             qB_all = agent.policy(sB)
#             qB = qB_all.gather(1, aB.view(-1, 1)).squeeze(1)

#             with torch.no_grad():
#                 qn_all = q_tgt(snB)
#                 qn_max = torch.max(qn_all, dim=-1).values
#                 y = rB + (1.0 - dB) * float(args.gamma) * qn_max

#             loss_q = F.smooth_l1_loss(qB, y)

#         # Total loss
#         loss = loss_world + loss_q

#         opt.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(params, 1.0)
#         opt.step()

#         # Target update
#         if (step + 1) % int(args.target_every) == 0:
#             q_tgt.load_state_dict(agent.policy.state_dict())

#         # meters update
#         ema_Renv.update(r_env)
#         ema_Rtrain.update(float(r_train))
#         ema_shape.update(r_shape)
#         ema_world.update(float(loss_world.detach().item()))
#         ema_gnorm.update(float(torch.norm(g_t.detach(), dim=-1).mean().item()))

#         if loss_q is not None:
#             ema_q.update(float(loss_q.detach().item()))
#             with torch.no_grad():
#                 ema_maxQ.update(float(torch.max(agent.policy(s_t.detach()), dim=-1).values.mean().item()))

#         for k in part_keys:
#             part_ema[k].update(float(info2.get(f"shape_{k}", 0.0)))

#         # step state
#         g_prev = g_t.detach().clone()
#         obs = obs_next
#         last_action = a_int

#         # episode reset
#         if done:
#             episode += 1
#             if (episode % 10) == 0:
#                 print(f"[ep={episode}] return_env={ep_Renv:.2f}  return_train={ep_Rtrain:.2f}")
#             obs, info = env.reset(seed=args.seed + episode + 1)
#             agent.reset(batch_size=1)
#             last_action = 0
#             g_prev = agent.get_latents()["g"].detach().clone()
#             ep_Renv = 0.0
#             ep_Rtrain = 0.0

#         # periodic prints
#         if (step + 1) % 2000 == 0:
#             dt = time.time() - t0

#             gp = part_ema["green_progress"].v or 0.0
#             rn = part_ema["red_near"].v or 0.0
#             oe = part_ema["object_entry"].v or 0.0
#             fv = part_ema["forage"].v or 0.0
#             sf = part_ema["skip_forage"].v or 0.0
#             rv = part_ema["revisit"].v or 0.0
#             idv = part_ema["idle"].v or 0.0
#             tc = part_ema["time_cost"].v or 0.0

#             print(
#                 f"[{step+1:>7}/{args.steps}] "
#                 f"ep={episode:>4}  eps={eps:.3f}  "
#                 f"Renv_ema={ema_Renv.v:+.3f}  Rtrain_ema={ema_Rtrain.v:+.3f}  "
#                 f"shape_ema={ema_shape.v:+.3f}  "
#                 f"(gp={gp:+.3f} rn={rn:+.3f} oe={oe:+.3f} fv={fv:+.3f} sf={sf:+.3f} rv={rv:+.3f} id={idv:+.3f} tc={tc:+.3f})  "
#                 f"world_ema={ema_world.v:.4f}  q_ema={ema_q.v:.4f}  maxQ_ema={ema_maxQ.v:+.3f}  "
#                 f"||g||_ema={ema_gnorm.v:.3f}  ({dt:.1f}s)"
#             )
#             t0 = time.time()

#     ckpt = {
#         "agent_state": agent.state_dict(),
#         "decoder_state": decoder.state_dict(),
#         "meta": meta,
#     }
#     torch.save(ckpt, run_dir / "ckpt.pt")
#     print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


# if __name__ == "__main__":
#     main()
