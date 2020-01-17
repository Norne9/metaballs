import pygame as pg
from pygame import surfarray
import numpy as np
import numba as nb
import os

WIDTH = 1280
HEIGHT = 720
CORES = os.cpu_count()


# balls [x, y, r, g, b, radius, vx, vy]
def update_balls(balls: np.ndarray, dt: float):
    for b in range(balls.shape[0]):
        radius = balls[b, 5]
        # move
        balls[b, 0:2] += balls[b, 6:8] * dt * 100.0

        # bounce x
        if balls[b, 0] < radius:
            balls[b, 6] = np.abs(balls[b, 6])
        elif balls[b, 0] > WIDTH - radius:
            balls[b, 6] = -np.abs(balls[b, 6])

        # bounce y
        if balls[b, 1] < radius:
            balls[b, 7] = np.abs(balls[b, 7])
        elif balls[b, 1] > HEIGHT - radius:
            balls[b, 7] = -np.abs(balls[b, 7])

        # bounce from others
        for b2 in range(balls.shape[0]):
            if b2 == b:
                continue
            dir = balls[b, 0:2] - balls[b2, 0:2]
            dist2 = dir[0] * dir[0] + dir[1] * dir[1]
            rad2 = balls[b, 5] + balls[b2, 5]
            rad2 *= rad2

            if dist2 < rad2:
                balls[b, 6:8] = dir / np.max(np.abs(dir))


@nb.jit(nopython=True, parallel=True)
def draw_balls(screen: np.ndarray, balls: np.ndarray):
    w, h = screen.shape[0], screen.shape[1]

    for start in nb.prange(CORES):
        for x in range(start, w, CORES):
            for y in range(h):
                screen[x, y].fill(0)

    b_count = balls.shape[0]
    max_radius = balls[:, 5].max() * 2
    for b in range(b_count):
        bx, by = balls[b, 0], balls[b, 1]
        radius = balls[b, 5]
        rgb = balls[b, 2:5]

        x_start = min(w, max(0, bx - max_radius * 2))
        x_end = min(w, max(0, bx + max_radius * 2))
        y_start = min(h, max(0, by - max_radius * 2))
        y_end = min(h, max(0, by + max_radius * 2))

        for start in nb.prange(CORES):
            for x in range(x_start + start, x_end, CORES):
                for y in range(y_start, y_end):
                    dx, dy = x - bx, y - by
                    light = radius * radius / (dx * dx + dy * dy)
                    for c in range(3):
                        screen[x, y, c] += rgb[c] * light * 255.0

    for start in nb.prange(CORES):
        for x in range(start, w, CORES):
            for y in range(h):
                max_color = screen[x, y].max()
                if max_color > 255:
                    screen[x, y] = screen[x, y] * 255 // max_color
                else:
                    screen[x, y] *= screen[x, y]
                    screen[x, y] //= 500


def run():
    pg.init()
    pg.font.init()

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    font = pg.font.SysFont("Arial", 16, bold=True)

    pg.display.set_caption(f"Metaballs {WIDTH}x{HEIGHT}")

    done = False
    fps = 0
    clock = pg.time.Clock()

    balls = np.empty((5, 8), dtype=np.float32)
    for i in range(balls.shape[0]):
        # generate ball
        radius = np.random.randint(5, 10) * 5
        x, y = np.random.randint(radius, WIDTH - radius), np.random.randint(radius, HEIGHT - radius)
        color = np.random.rand(3)
        color[i % 3] = 1
        vel = np.random.rand(2)
        vel = vel / vel.max()
        # set ball
        balls[i, 0], balls[i, 1] = x, y
        balls[i, 2:5] = color
        balls[i, 5] = radius
        balls[i, 6:8] = vel

    screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int32)

    while not done:
        dt = clock.tick()
        fps = fps * 0.97 + 1000.0 / max(dt, 1) * 0.03
        dt /= 1000.0

        for event in pg.event.get():  # User did something
            if event.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # numpy
        update_balls(balls, dt)
        draw_balls(screen_arr, balls)
        surfarray.blit_array(screen, screen_arr)

        # show fps
        text = font.render(f"FPS: {fps:4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))

        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    run()
