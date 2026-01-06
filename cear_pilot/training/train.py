# cear_pilot/training/train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

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


# # -----------------------------
# # Live pygame viewer (training-time)
# # -----------------------------
# class PygameGridViewer:
#     """
#     Minimal live viewer for NZoneGridEnv during training.

#     - Draws grid with vertical 3-zone background tint
#     - Draws agent position
#     - Draws overlay text: step, ep, t, zone, last action, loss/pred/smooth/H, ||g||

#     Controls:
#       - Close window: stop training
#       - SPACE: pause/resume
#     """

#     def __init__(
#         self,
#         width: int,
#         height: int,
#         cell_px: int = 40,
#         fps: int = 12,
#         title: str = "CEAR Live Training",
#     ):
#         try:
#             import pygame  # type: ignore
#         except Exception as e:
#             raise ImportError("pygame required for --view. Install with: pip install pygame") from e

#         self.pygame = pygame
#         pygame.init()

#         self.W = int(width)
#         self.H = int(height)
#         self.cell = int(cell_px)
#         self.fps = int(fps)

#         self.pad_top = 90  # space for overlay text
#         self.screen_w = self.W * self.cell
#         self.screen_h = self.H * self.cell + self.pad_top

#         self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
#         pygame.display.set_caption(title)

#         self.clock = pygame.time.Clock()
#         self.font = pygame.font.SysFont("Arial", 18)
#         self.small = pygame.font.SysFont("Arial", 14)

#         self.paused = False

#         # palette
#         self.zone_colors = [
#             (35, 55, 90),   # zone 0 tint
#             (40, 75, 55),   # zone 1 tint
#             (85, 55, 40),   # zone 2 tint
#         ]
#         self.grid_line = (25, 25, 25)
#         self.agent_color = (230, 230, 230)
#         self.text_color = (240, 240, 240)
#         self.panel_bg = (15, 15, 15)

#     def _zone_of_x(self, x: int) -> int:
#         if x < self.W / 3:
#             return 0
#         elif x < 2 * self.W / 3:
#             return 1
#         return 2

#     def pump(self) -> Optional[bool]:
#         """Handle events. Returns:
#            - False if user requested quit
#            - True otherwise
#         """
#         pygame = self.pygame
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 return False
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     self.paused = not self.paused
#         return True

#     def wait_if_paused(self) -> Optional[bool]:
#         """While paused, keep pumping events and drawing a 'paused' label."""
#         pygame = self.pygame
#         while self.paused:
#             ok = self.pump()
#             if ok is False:
#                 return False
#             # small pause loop
#             self.clock.tick(12)
#         return True

#     def draw(
#         self,
#         env: NZoneGridEnv,
#         step: int,
#         episode: int,
#         last_action: int,
#         loss: float,
#         loss_pred: float,
#         loss_smooth: float,
#         entropy: float,
#         g_norm: float,
#     ) -> Optional[bool]:
#         pygame = self.pygame

#         ok = self.pump()
#         if ok is False:
#             return False
#         ok = self.wait_if_paused()
#         if ok is False:
#             return False

#         # background panel
#         self.screen.fill(self.panel_bg)
#         pygame.draw.rect(self.screen, self.panel_bg, (0, 0, self.screen_w, self.pad_top))

#         # overlay text
#         action_names = ["U", "D", "L", "R", "S"]
#         zid = int(env.zone_id())
#         x, y, t = int(env.x), int(env.y), int(env.t)

#         line1 = f"step={step}  ep={episode}  t={t}  zone={zid}  pos=({x},{y})  a={action_names[last_action] if 0<=last_action<5 else last_action}"
#         line2 = f"loss={loss:.4f}  pred={loss_pred:.4f}  smooth={loss_smooth:.4f}  H={entropy:.3f}  ||g||={g_norm:.3f}   (SPACE: pause/resume)"
#         txt1 = self.font.render(line1, True, self.text_color)
#         txt2 = self.small.render(line2, True, self.text_color)
#         self.screen.blit(txt1, (10, 10))
#         self.screen.blit(txt2, (10, 40))

#         if self.paused:
#             paused = self.font.render("PAUSED", True, (255, 220, 120))
#             self.screen.blit(paused, (10, 65))

#         # grid offset
#         y0 = self.pad_top

#         # draw cells with zone tints
#         for yy in range(self.H):
#             for xx in range(self.W):
#                 zid_x = self._zone_of_x(xx)
#                 col = self.zone_colors[zid_x]
#                 rect = pygame.Rect(xx * self.cell, y0 + yy * self.cell, self.cell, self.cell)
#                 pygame.draw.rect(self.screen, col, rect)

#         # grid lines
#         for xx in range(self.W + 1):
#             pygame.draw.line(
#                 self.screen,
#                 self.grid_line,
#                 (xx * self.cell, y0),
#                 (xx * self.cell, y0 + self.H * self.cell),
#                 1,
#             )
#         for yy in range(self.H + 1):
#             pygame.draw.line(
#                 self.screen,
#                 self.grid_line,
#                 (0, y0 + yy * self.cell),
#                 (self.W * self.cell, y0 + yy * self.cell),
#                 1,
#             )

#         # draw agent
#         ax = x * self.cell + self.cell // 2
#         ay = y0 + y * self.cell + self.cell // 2
#         r = max(6, self.cell // 3)
#         pygame.draw.circle(self.screen, self.agent_color, (ax, ay), r)

#         pygame.display.flip()
#         self.clock.tick(self.fps)
#         return True

#     def close(self):
#         try:
#             self.pygame.quit()
#         except Exception:
#             pass


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

    # ---- live viewer flags ----
    ap.add_argument("--view", action="store_true", help="Show live pygame viewer during training")
    ap.add_argument("--view_every", type=int, default=2, help="Render every N training steps")
    ap.add_argument("--view_fps", type=int, default=20, help="Viewer FPS cap")
    ap.add_argument("--view_cell_px", type=int, default=42, help="Cell size in pixels")

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
    t0 = time.time()

    episode = 0

    try:
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
            xhat_all = decoder.predict_all_actions(g_t)               # (1, A, obs_dim)
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
                obs, info = env.reset()
                agent.reset(batch_size=1)
                last_action = 4
                g_prev = agent.get_latents()["g"].detach().clone()
                episode += 1

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

    finally:
        if viewer is not None:
            viewer.close()

    ckpt = {"agent_state": agent.state_dict(), "decoder_state": decoder.state_dict(), "meta": meta}
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
