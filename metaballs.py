import pygame as pg
import numpy as np
import numba as nb
import os

WIDTH = 800
HEIGHT = 600
CORES = os.cpu_count()
DEFAULT_N_BALLS = 6


def update_balls(balls: np.ndarray, dt: float):
    for b in range(balls.shape[0]):
        # move
        balls[b].pos += balls[b].vel * dt * 80.0

        # bounce from bounds
        screen = [WIDTH, HEIGHT]
        for axis in range(2):
            if balls[b].pos[axis] < balls[b].radius:
                balls[b].vel[axis] = np.abs(balls[b].vel[axis])
            elif balls[b].pos[axis] > screen[axis] - balls[b].radius:
                balls[b].vel[axis] = -np.abs(balls[b].vel[axis])

        # bounce from others
        for b2 in range(balls.shape[0]):
            if b2 == b:
                continue

            delta = balls[b].pos - balls[b2].pos
            dist2 = np.dot(delta, delta)
            rad2 = balls[b].radius + balls[b2].radius
            rad2 *= rad2

            if dist2 < rad2:
                balls[b].vel = delta / np.max(np.abs(delta))


@nb.njit(parallel=True, fastmath=True)
def draw_balls(screen: np.ndarray, balls: np.ndarray):
    w, h = screen.shape[0], screen.shape[1]
    b_count = balls.shape[0]

    # to use all cores
    for start in nb.prange(CORES):
        # for each pixel on screen
        for x in range(start, w, CORES):
            for y in range(h):
                screen[x, y].fill(0)  # clear pixel
                # for each ball
                for b in range(b_count):
                    # calculate value
                    dx, dy = balls[b].pos[0] - x, balls[b].pos[1] - y
                    light = balls[b].radius * balls[b].radius / (dx * dx + dy * dy)

                    # multiply value by ball color
                    for c in range(3):
                        screen[x, y, c] += balls[b].rgb[c] * light * 255.0

                # if color > max => normalize color
                max_color = screen[x, y].max()
                if max_color > 255:
                    screen[x, y] = screen[x, y] * 255 // max_color
                else:  # else => color = color * color / 2
                    screen[x, y] *= screen[x, y]
                    screen[x, y] //= 500


def create_balls(n_balls):
    """make random balls"""
    balls = np.recarray(
        (n_balls,), dtype=[("pos", ("<f4", (2,))), ("rgb", ("<f4", (3,))), ("radius", "f4"), ("vel", ("<f4", (2,)))],
    )
    for i in range(balls.shape[0]):
        # generate ball
        balls[i].radius = np.random.randint(5, 15) * 5
        balls[i].pos = (
            np.random.randint(balls[i].radius, WIDTH - balls[i].radius),
            np.random.randint(balls[i].radius, HEIGHT - balls[i].radius),
        )
        balls[i].rgb = np.random.rand(3)
        balls[i].rgb[i % 3] = 1
        balls[i].vel = np.random.rand(2)
        balls[i].vel = balls[i].vel / balls[i].vel.max()
    return balls


def run():
    # set seed for repeatability
    np.random.seed(2)
    # init pygame
    pg.init()
    pg.font.init()

    # create window
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    # set window title
    pg.display.set_caption(f"Metaballs {WIDTH}x{HEIGHT}")
    # load system font
    font = pg.font.SysFont(pg.font.get_default_font(), 24)

    # necessary variables
    n_balls = DEFAULT_N_BALLS
    done = False
    clock = pg.time.Clock()

    # show loading at center
    text = font.render("LOADING...", False, (255, 255, 255))
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
    pg.display.flip()

    balls = create_balls(n_balls)
    # numpy array for screen
    screen_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int32)

    while not done:
        dt = clock.tick() / 1000.0  # get elapsed seconds

        for event in pg.event.get():  # process events
            if event.type == pg.QUIT:  # clicked close
                done = True  # exit loop
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    balls = create_balls(n_balls)
                elif event.key == pg.K_UP:
                    n_balls += 1
                    balls = create_balls(n_balls)
                elif event.key == pg.K_DOWN:
                    n_balls = max(1, n_balls - 1)
                    balls = create_balls(n_balls)

        # numpy
        update_balls(balls, dt)  # move balls
        draw_balls(screen_arr, balls)  # draw balls in numpy array
        pg.surfarray.blit_array(screen, screen_arr)  # draw array on screen

        # show fps
        text = font.render(f"FPS: {clock.get_fps():4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))
        text = font.render(
            f"PRESS SPACE TO RELOAD. UP TO INCREASE BALL AMOUNT. DOWN TO REDUCE BALL AMOUNT", False, (255, 255, 255)
        )
        screen.blit(text, (16, HEIGHT - 40))

        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    run()
