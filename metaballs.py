import pygame as pg
from pygame import surfarray
import numpy as np
from numba import njit

WIDTH = 1920
HEIGHT = 1080


@njit
def proc_array(screen: np.ndarray, bx: float, by: float):
    w, h = screen.shape[0], screen.shape[1]
    for x in range(w):
        for y in range(h):
            dx, dy = x - bx, y - by
            dist = dx * dx + dy * dy
            radius = 100.0

            light = min(1.0, radius * radius / max(1.0, dist))
            light = light / 2.0 if light < 1 else light

            screen[x, y, 2] = light * 255


def run():
    pg.init()
    pg.font.init()

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    font = pg.font.SysFont("Arial", 16, bold=True)

    pg.display.set_caption(f"Metaballs {WIDTH}x{HEIGHT}")

    done = False
    fps = 0
    clock = pg.time.Clock()

    # coord_arr = np.array([[[x, y] for y in range(HEIGHT)] for x in range(WIDTH)], dtype=np.float)
    screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.dtype("u1"))
    while not done:
        fps = fps * 0.97 + 1000.0 / max(clock.tick(), 1) * 0.03

        for event in pg.event.get():  # User did something
            if event.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # numpy draw
        proc_array(screen_arr, 500, 500)
        surfarray.blit_array(screen, screen_arr)

        # show fps
        text = font.render(f"FPS: {fps:4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))

        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    run()
