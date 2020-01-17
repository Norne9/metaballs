import pygame as pg
from pygame import surfarray
import numpy as np
import numba as nb
import os

WIDTH = 1920
HEIGHT = 1080
CORES = os.cpu_count()


@nb.jitclass([("x", nb.float32), ("y", nb.float32), ("rgb", nb.types.float32[:]), ("radius", nb.float32)])
class Ball:
    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        rgb: np.ndarray = np.array([1.0, 1.0, 0.0], dtype=np.float32),
        radius: float = 100,
    ):
        self.x = x
        self.y = y
        self.rgb = rgb
        self.radius = radius


@nb.jit(nopython=True, parallel=True)
def draw_ball(screen: np.ndarray, ball: Ball, add: bool):
    w, h = screen.shape[0], screen.shape[1]
    for start in nb.prange(CORES):
        for x in range(start, w, CORES):
            for y in range(h):
                dx, dy = x - ball.x, y - ball.y
                light = ball.radius * ball.radius / (dx * dx + dy * dy)
                for c in range(3):
                    if add:
                        screen[x, y, c] += ball.rgb[c] * light * 255.0
                    else:
                        screen[x, y, c] = ball.rgb[c] * light * 255.0


@nb.jit(nopython=True, parallel=True)
def clamp_colors(screen: np.ndarray):
    w, h = screen.shape[0], screen.shape[1]
    for start in nb.prange(CORES):
        for x in range(start, w, CORES):
            for y in range(h):
                max_color = screen[x, y].max()
                if max_color > 255:
                    screen[x, y] = screen[x, y] * 255 // max_color
                else:
                    screen[x, y] //= 2


def run():
    pg.init()
    pg.font.init()

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    font = pg.font.SysFont("Arial", 16, bold=True)

    pg.display.set_caption(f"Metaballs {WIDTH}x{HEIGHT}")

    done = False
    fps = 0
    clock = pg.time.Clock()

    balls = [
        Ball(x=500, y=500, rgb=np.array([1, 1, 0], dtype=np.float32)),
        Ball(x=800, y=500, rgb=np.array([1, 0, 1], dtype=np.float32)),
    ]
    screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int32)

    while not done:
        fps = fps * 0.97 + 1000.0 / max(clock.tick(), 1) * 0.03

        for event in pg.event.get():  # User did something
            if event.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # numpy draw
        draw_add = False
        for ball in balls:
            draw_ball(screen_arr, ball, draw_add)
            draw_add = True
        clamp_colors(screen_arr)
        surfarray.blit_array(screen, screen_arr)

        # show fps
        text = font.render(f"FPS: {fps:4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))

        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    run()
