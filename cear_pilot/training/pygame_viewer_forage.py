# cear_pilot/training/pygame_viewer_forage.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

from cear_pilot.envs.forage_grid import ForageGridEnv


class PygameForageViewer:
    """
    Live viewer for ForageGridEnv.
    Controls:
      - Close window: stop loop
      - SPACE: pause/resume
    """

    def __init__(self, cell_px: int = 80, fps: int = 10, title: str = "Forage Viewer"):
        try:
            import pygame  # type: ignore
        except Exception as e:
            raise ImportError("pygame required for view. Install with: pip install pygame") from e

        self.pygame = pygame
        pygame.init()

        self.cell = int(cell_px)
        self.fps = int(fps)

        self.pad_top = 90
        self.text_color = (240, 240, 240)
        self.panel_bg = (15, 15, 15)
        self.grid_line = (25, 25, 25)

        self.unknown = (90, 90, 90)
        self.green = (60, 140, 80)
        self.red = (150, 70, 70)
        self.empty = (40, 40, 50)
        self.agent_color = (230, 230, 230)

        # fixed 5x5
        self.W = 5
        self.H = 5
        self.screen_w = self.W * self.cell
        self.screen_h = self.H * self.cell + self.pad_top

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(title)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.small = pygame.font.SysFont("Arial", 14)
        self.paused = False

    def pump(self) -> Optional[bool]:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def wait_if_paused(self) -> Optional[bool]:
        while self.paused:
            ok = self.pump()
            if ok is False:
                return False
            self.clock.tick(12)
        return True

    def draw(
        self,
        env: ForageGridEnv,
        step: int,
        episode: int,
        last_action: int,
        reward: float,
        total_reward: float,
        g_norm: float | None = None,
    ) -> Optional[bool]:
        pygame = self.pygame

        ok = self.pump()
        if ok is False:
            return False
        ok = self.wait_if_paused()
        if ok is False:
            return False

        self.screen.fill(self.panel_bg)

        # header
        line1 = f"ep={episode}  step={step}  time_left={env.time_left}  a={last_action}  r={reward:+.2f}  R={total_reward:+.2f}"
        if g_norm is None:
            line2 = "(SPACE: pause/resume)"
        else:
            line2 = f"||g||={g_norm:.3f}   (SPACE: pause/resume)"
        txt1 = self.font.render(line1, True, self.text_color)
        txt2 = self.small.render(line2, True, self.text_color)
        self.screen.blit(txt1, (10, 10))
        self.screen.blit(txt2, (10, 40))
        if self.paused:
            paused = self.font.render("PAUSED", True, (255, 220, 120))
            self.screen.blit(paused, (10, 65))

        y0 = self.pad_top

        # draw grid background
        for yy in range(self.H):
            for xx in range(self.W):
                rect = pygame.Rect(xx * self.cell, y0 + yy * self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, self.empty, rect)

        # draw objects
        for (x, y), (color, revealed) in env.objects.items():
            rect = pygame.Rect(x * self.cell, y0 + y * self.cell, self.cell, self.cell)
            if not revealed:
                pygame.draw.rect(self.screen, self.unknown, rect)
            else:
                pygame.draw.rect(self.screen, self.green if color == env.COLOR_GREEN else self.red, rect)

        # grid lines
        for xx in range(self.W + 1):
            pygame.draw.line(self.screen, self.grid_line, (xx * self.cell, y0), (xx * self.cell, y0 + self.H * self.cell), 1)
        for yy in range(self.H + 1):
            pygame.draw.line(self.screen, self.grid_line, (0, y0 + yy * self.cell), (self.W * self.cell, y0 + yy * self.cell), 1)

        # agent
        ax, ay = env.agent_xy
        cx = ax * self.cell + self.cell // 2
        cy = y0 + ay * self.cell + self.cell // 2
        r = max(8, self.cell // 3)
        pygame.draw.circle(self.screen, self.agent_color, (cx, cy), r)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        try:
            self.pygame.quit()
        except Exception:
            pass
