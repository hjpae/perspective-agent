# cear_pilot/experiments/run_switch_mismatch.py
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import torch

from cear_pilot.envs.nzone_grid import make_env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

def onehot_action(a: int, n: int, device):
    x = torch.zeros((1, n), device=device)
    x[0, a] = 1.0
    return x

@torch.no_grad()
def rollout_switch_mismatch(
    env,
    agent: CEARAgent,
    decoder: ObsDecoder,
    ckpt_path: str,
    T: int = 240,
    T_switch: int = 120,
    p_before=(0.0, 0.0, 0.35),
    p_after=(0.0, 0.0, 0.70),
    greedy: bool = True,
    device: str = "cpu",
):
    device = torch.device(device)

    # load
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(device); decoder.to(device)
    agent.eval(); decoder.eval()

    # clone for imagination
    imag_agent = copy.deepcopy(agent).to(device)
    imag_agent.eval()

    obs, info = env.reset(seed=0)
    agent.reset(1)
    imag_agent.reset(1)

    n_actions = env.action_space.n
    last_action = 4

    rows = []
    g_real_prev = agent.get_latents()["g"].clone()

    for t in range(T):
        if t == T_switch:
            # switch env dynamics (you implement set_slip or runtime update)
            env.set_slip(p_after)

        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = onehot_action(last_action, n_actions, device)

        out_real = agent.forward_step(x_t, p_t, ablate_g=False)
        g_real = out_real["g"]
        logits = out_real["logits"]
        pi = torch.softmax(logits, dim=-1)

        # choose action from policy
        if greedy:
            a = int(torch.argmax(pi, dim=-1).item())
        else:
            a = int(torch.distributions.Categorical(pi).sample().item())

        # --- IMAGINE next obs from model ---
        xhat_all = decoder.predict_all_actions(g_real)     # (1,A,obs_dim)
        x_hat = xhat_all[:, a, :]                          # (1,obs_dim) pick chosen action prediction

        # advance imag_agent with imagined observation
        p_a = onehot_action(a, n_actions, device)
        out_imag = imag_agent.forward_step(x_hat, p_a, ablate_g=False)
        g_imag_next = out_imag["g"]

        # --- REAL env step ---
        obs_next, rew, term, trunc, info2 = env.step(a)
        x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

        # compute mismatch
        obs_mismatch = torch.mean((x_hat - x_next) ** 2).item()

        # g updates: real_agent will update internally when we call forward_step next iter,
        # but we can get g_real_next by manually stepping once with real obs_next now:
        out_real_next = agent.forward_step(x_next, p_a, ablate_g=False)
        g_real_next = out_real_next["g"]

        g_div = torch.norm(g_imag_next - g_real_next, dim=-1).mean().item()
        g_speed = torch.norm(g_real_next - g_real_prev, dim=-1).mean().item()
        g_real_prev = g_real_next.clone()

        rows.append({
            "t": t,
            "zone": info2.get("zone_id", -1),
            "x": info2.get("x", -1),
            "y": info2.get("y", -1),
            "reward": float(rew),
            "obs_mismatch": float(obs_mismatch),
            "g_div": float(g_div),
            "g_speed": float(g_speed),
            "H": float((-torch.sum(pi * torch.log(pi + 1e-9), dim=-1)).mean().item()),
            "maxpi": float(pi.max(dim=-1).values.mean().item()),
        })

        obs = obs_next
        last_action = a

        if term or trunc:
            break

    return pd.DataFrame(rows)
